import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt_utils import prompt_wrapper
import argparse

LABEL_COLS = [
    'loop_scattering', 'background_ring', 'strong_background',
    'diffuse_scattering', 'artifact', 'ice_ring', 'non_uniform_detector'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./train_classify/train_classify.csv')
    parser.add_argument('--img_dir', type=str, default='./train_classify')
    parser.add_argument('--cfg_path', type=str, default='../eval_configs/minigpt4_eval.yaml')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--layers', type=int, nargs='+', default=[12, 24, 36], help='Layers to extract activations from')
    parser.add_argument('--save_path', type=str, default='activation_train_MLP.pt')
    parser.add_argument(
        '--options',
        nargs='+',
        default=[],
        help='Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.'
    )
    return parser.parse_args()


def get_label(row):
    arr = row[LABEL_COLS].values.astype(int)
    if arr.sum() == 0:
        return 7  # no_anomalies
    else:
        return int(np.argmax(arr))


def main():
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
    prefix = prompt_wrapper.minigpt4_chatbot_prompt
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    activations_dict = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.img_dir, row['image'])
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        query = prefix % prompt
        label = get_label(row)

        print(f"\nProcessing {row['image']} | Label: {label}")
        print(f"Prompt: {prompt}")
        print(f"Wrapped prompt: {query}")

        with torch.no_grad(), torch.cuda.amp.autocast():
            prompt_wrap = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[query],
                img_prompts=[[img_tensor]],
                device=f'cuda:{args.gpu_id}'
            )
            context_embs = prompt_wrap.context_embs[0]
            outputs = model.llama_model(inputs_embeds=context_embs, output_hidden_states=True, return_dict=True)
            start_idx, end_idx = prompt_wrap.get_image_token_indices()
            print(f"Image token start index: {start_idx}, end index: {end_idx}, num tokens: {end_idx - start_idx}")
            activations_dict[idx] = {'label': label}
            for layer in args.layers:
                image_token_acts = outputs.hidden_states[layer][0, start_idx:end_idx, :].detach().cpu()
                avg_act = image_token_acts.mean(axis=0)
                print(f"Layer {layer} activation shape: {image_token_acts.shape}")
                print(f"Layer {layer} avg activation: {avg_act}")
                activations_dict[idx][layer] = avg_act  # Store as torch tensor

    torch.save(activations_dict, args.save_path)
    print(f"Saved activations and labels to {args.save_path}")

if __name__ == '__main__':
    main() 