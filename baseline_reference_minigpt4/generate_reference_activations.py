import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt_utils import prompt_wrapper, generator

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 40], help="Layer(s) to extract activations from")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV with columns: image_path,prompt,correct_answer")
    parser.add_argument("--save_path", type=str, default="./reference_activation_xrd.pt")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

def extract_option_letter(response):
    # Look for A, B, C, or D at the start of the response
    response = response.strip()
    if response.startswith(('A', 'B', 'C', 'D')):
        return response[0]
    return None

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

    # Initialize generator for model responses
    generator_obj = generator.Generator(model=model)
    prefix = prompt_wrapper.minigpt4_chatbot_prompt

    # Print the total number of layers in the model
    num_layers = len(model.llama_model.base_model.layers)
    print(f"Total number of layers in the model: {num_layers}")

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Load your dataset
    df = pd.read_csv(args.csv_path)  # columns: image_path,prompt,correct_answer
    num_data_points = len(df)  # Process all rows

    # First pass: Generate model responses and save to CSV
    print("First pass: Generating model responses...")
    results_for_csv = []
    
    for idx, row in tqdm(df.iterrows(), total=num_data_points):
        image = Image.open(row['image_path']).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        correct_answer = row['correct_answer']
        
        # Wrap prompt in template only (no instruction)
        query = prefix % prompt
        if idx == 0:
            print(f"First pass - final prompt for image {os.path.basename(row['image_path'])}:\n{query}\n")
        prompt_wrap = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[query],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        model_response, _ = generator_obj.generate(prompt_wrap)
        print(f"Model response for image {os.path.basename(row['image_path'])}: {model_response}")
        
        # Extract option letter from model response
        model_option = extract_option_letter(model_response)
        
        if model_option is None:
            print(f"Warning: No valid option found for image {os.path.basename(row['image_path'])}, skipping...")
            continue
        
        # Collect for CSV
        results_for_csv.append({
            'image_name': os.path.basename(row['image_path']),
            'prompt': prompt,
            'correct_answer': correct_answer,
            'model_response': model_response,
            'model_option': model_option
        })

    # Save first pass results
    first_pass_csv = "first_pass_responses.csv"
    pd.DataFrame(results_for_csv).to_csv(first_pass_csv, index=False)
    print(f"Saved first pass responses to {first_pass_csv}")

    # Second pass: Generate activations using the model responses
    print("\nSecond pass: Generating activations...")
    reference_acts = {}
    
    # Initialize reference_acts for the number of valid responses we have
    num_valid_responses = len(df)
    for idx in range(num_valid_responses):
        reference_acts[idx] = {
            20: [],
            40: []
        }
    
    for idx, row in tqdm(df.iterrows(), total=num_valid_responses):
        image = Image.open(row['image_path']).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        
        # Wrap prompt in template, then append answer as last token
        query = prefix % prompt
        query_with_answer = query + ' ' + row['correct_answer']
        if idx == 0:
            print(f"Second pass - final prompt for image {os.path.basename(row['image_path'])}:\n{query_with_answer}\n")
        prompt_wrap = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[query_with_answer],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        context_embs = prompt_wrap.context_embs[0]
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.llama_model(inputs_embeds=context_embs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            answer_pos = context_embs.shape[1] - 1

            # Print information
            tqdm.write("\n" + "="*50)
            tqdm.write(f"Image: {os.path.basename(row['image_path'])}")
            tqdm.write(f"Prompt: {prompt}")
            tqdm.write(f"Correct Answer: {row['correct_answer']}")
            tokens = tokenizer(query_with_answer, return_tensors="pt")
            final_token = tokenizer.decode(tokens["input_ids"][0, -1])
            tqdm.write(f"Final token: {final_token}")

            # Store activations
            for layer in [20, 40]:
                act = hidden_states[layer][0, answer_pos, :].detach().cpu().numpy()
                tqdm.write(f"Layer {layer} activation: {act}")
                reference_acts[idx][layer].append(act)

            tqdm.write("="*50 + "\n")

    # Save activations
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(reference_acts, args.save_path)
    print(f"Saved reference activations to {args.save_path}")

if __name__ == "__main__":
    main()