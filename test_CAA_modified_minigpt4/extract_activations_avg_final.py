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
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default="test_combined.csv", help="CSV with columns: image_name,prompt,answer")
    parser.add_argument("--save_dir", type=str, default="./Activations_visualize", help="Directory to save activation files")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct", help="Base directory containing images")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
    generator_obj = generator.Generator(model=model)
    prefix = prompt_wrapper.minigpt4_chatbot_prompt

    num_layers = len(model.llama_model.base_model.layers)
    print(f"Total number of layers in the model: {num_layers}")

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # Read the CSV file
    df = pd.read_csv(args.csv_path) # .head(5) for testing
    num_data_points = len(df)
    print(f"Number of data points: {num_data_points}")

    # Define layers to extract
    layers_to_extract = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    
    # Initialize activations dictionary for each layer
    activations_by_layer = {}
    for layer in layers_to_extract:
        activations_by_layer[layer] = {}
        for idx in range(num_data_points):
            activations_by_layer[layer][idx] = {
                'correct_answer': df.iloc[idx]['answer'],
                'image_tokens': None,  # Will store tensor for this layer
                'final_token': None    # Will store tensor for this layer
            }

    for idx, row in tqdm(df.iterrows(), total=num_data_points):
        # Use the combined_images directory directly
        image_path = os.path.join(args.image_dir, row['image_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue
            
        image = Image.open(image_path).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        correct_answer = row['answer']

        # Wrap in template
        query = prefix % prompt

        # Print the final prompt (same as original script)
        print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Get the wrapped prompt without correct answer for both final token and image token extraction
            prompt_wrap = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[query],
                img_prompts=[[img_tensor]],
                device=f'cuda:{args.gpu_id}'
            )
            context_embs = prompt_wrap.context_embs[0]
            outputs = model.llama_model(inputs_embeds=context_embs, output_hidden_states=True, return_dict=True)
            
            # Get image token indices from wrapped prompt (without correct answer)
            start_idx, end_idx = prompt_wrap.get_image_token_indices()
            num_image_tokens = end_idx - start_idx
            
            # Get final token index from wrapped prompt (without correct answer)
            final_token_idx = context_embs.shape[1] - 1
            
            # Print token analysis for the first image+prompt pair only (same as original script)
            if idx == 0:
                print("\n=== First Image+Prompt Token Analysis ===")
                
                # Get the input tokens from the prompt wrapper
                if hasattr(prompt_wrap, 'input_tokens'):
                    print("\nTokenized segments:")
                    for i, seg_tokens in enumerate(prompt_wrap.input_tokens):
                        print(f"Segment {i}: {tokenizer.decode(seg_tokens[0])}")
                
                print(f"\nImage token start index: {start_idx}")
                print(f"Image token end index: {end_idx}")
                print(f"Number of image tokens: {end_idx - start_idx}")
                print(f"Total context length: {context_embs.shape[1]}")
                print("=== End of First Image+Prompt Analysis ===\n")
            
            # Comprehensive logging (same format as original scripts)
            tqdm.write("\n" + "="*50)
            tqdm.write(f"Image: {row['image_name']}")
            tqdm.write(f"Prompt: {prompt}")
            tqdm.write(f"Correct Answer: {correct_answer}")
            tqdm.write(f"Image token start index: {start_idx}, end index: {end_idx}, num tokens: {num_image_tokens}")
            tqdm.write(f"Final token index: {final_token_idx}")
            tqdm.write(f"Total context length: {context_embs.shape[1]}")
            
            # Print the last few tokens with their IDs and text (same as original script)
            if hasattr(prompt_wrap, 'input_tokens') and len(prompt_wrap.input_tokens) > 0:
                last_tensor = prompt_wrap.input_tokens[-1]
                tqdm.write("\nLast tokens:")
                for i in range(min(7, last_tensor.shape[1])):
                    token_id = last_tensor[0, -(i+1)].item()
                    token_text = tokenizer.decode([token_id])
                    tqdm.write(f"Last {i+1} token ID: {token_id}, Last {i+1} token text: '{token_text}'")
            
            # Print which token activation is being extracted (same as original script)
            if hasattr(prompt_wrap, 'input_tokens') and len(prompt_wrap.input_tokens) > 0:
                last_tensor = prompt_wrap.input_tokens[-1]
                final_token_id = last_tensor[0, -1].item()
                final_token_text = tokenizer.decode([final_token_id])
                tqdm.write(f"EXTRACTING FINAL TOKEN ACTIVATION FROM: Position {final_token_idx}, Token ID {final_token_id}, Token text '{final_token_text}' (wrapped prompt without answer)")
                tqdm.write(f"EXTRACTING IMAGE TOKENS FROM: Positions {start_idx} to {end_idx-1} (average of {num_image_tokens} tokens) (wrapped prompt without answer)")

            # Extract activations for each layer
            for layer in layers_to_extract:
                # Extract average of image tokens from wrapped prompt (without correct answer)
                image_token_acts = outputs.hidden_states[layer][0, start_idx:end_idx, :].detach().cpu()
                avg_image_act = image_token_acts.mean(axis=0)
                
                # Extract final token activation from wrapped prompt (without correct answer)
                final_token_act = outputs.hidden_states[layer][0, final_token_idx, :].detach().cpu()
                
                # Store as PyTorch tensors
                activations_by_layer[layer][idx]['image_tokens'] = avg_image_act
                activations_by_layer[layer][idx]['final_token'] = final_token_act
                
                # Log activation details (same format as original scripts)
                tqdm.write(f"Layer {layer} image tokens activation shape: {image_token_acts.shape}")
                tqdm.write(f"Layer {layer} image tokens activation: {avg_image_act}")
                tqdm.write(f"Layer {layer} final token activation shape: {final_token_act.shape}")
                tqdm.write(f"Layer {layer} final token activation: {final_token_act}")
                tqdm.write("")

            tqdm.write("="*50 + "\n")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save separate .pt files for each layer
    for layer in layers_to_extract:
        save_path = os.path.join(args.save_dir, f"activations_layer_{layer}.pt")
        torch.save(activations_by_layer[layer], save_path)
        print(f"Saved activations for layer {layer} to {save_path}")
    
    print(f"Saved {len(layers_to_extract)} separate .pt files")
    print(f"Each file contains activations for one layer with structure:")
    print(f"  - Outer keys: numbers (0, 1, 2, ...)")
    print(f"  - Inner keys: 'correct_answer', 'image_tokens', 'final_token'")
    print(f"  - Values: PyTorch tensors for the specific layer")

if __name__ == "__main__":
    main() 