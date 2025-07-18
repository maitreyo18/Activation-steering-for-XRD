import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer

from minigpt_utils import prompt_wrapper, generator
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Calibration Activation")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with columns: image_path,prompt,correct_answer")
    parser.add_argument("--save_path", type=str, default="./calibration_activations_6-24.pt", help="Path to save the activations.")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()
    return args

def create_custom_forward_hook(activations_dict, idx, layer):
    def custom_forward_hook(module, input, output):
        R_feat = output[0][:, -1, :]
        activations_dict[idx][layer].append(R_feat.detach().cpu().numpy())
        return output
    return custom_forward_hook

def main():
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    generator_obj = generator.Generator(model=model)
    prefix = prompt_wrapper.minigpt4_chatbot_prompt

    df = pd.read_csv(args.csv_path)
    df = df.sample(n=100, random_state=42).reset_index(drop=True)
    reference_acts = {}
    num_data_points = len(df)
    for idx in range(num_data_points):
        reference_acts[idx] = {layer: [] for layer in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]}

    for idx, row in tqdm(df.iterrows(), total=num_data_points):
        image = Image.open(row['image_path']).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        correct_answer = row['correct_answer']

        query = prefix % prompt
        prompt_wrap = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[query],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )

        # Register hooks for layers 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
        hooks = []
        for layer in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
            hook = model.llama_model.base_model.layers[layer-1].register_forward_hook(
                create_custom_forward_hook(reference_acts, idx, layer)
            )
            hooks.append(hook)

        # Generate response
        model_response, _ = generator_obj.generate(prompt_wrap)
        print(f"Model response for image {os.path.basename(row['image_path'])}: {model_response}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Save activations
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(reference_acts, args.save_path)
    print(f"Saved activations to {args.save_path}")

if __name__ == "__main__":
    main() 