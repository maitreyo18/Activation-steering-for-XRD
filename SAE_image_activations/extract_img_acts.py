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
    parser.add_argument("--csv_path", type=str, default="../train_combined.csv", help="CSV with columns: image_name,prompt,answer")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save activation files")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/combined_images", help="Base directory containing images")
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
    df = pd.read_csv(args.csv_path) # head(10) for testing
    num_data_points = len(df)
    print(f"Number of data points: {num_data_points}")

    # Initialize activations dictionaries for each layer
    activations_layer_6 = {}
    activations_layer_16 = {}
    
    for idx in range(num_data_points):
        activations_layer_6[idx] = {
            'correct_answer': df.iloc[idx]['answer']
        }
        activations_layer_16[idx] = {
            'correct_answer': df.iloc[idx]['answer']
        }
        
        # Initialize patch activations (1-32)
        for patch_num in range(1, 33):
            activations_layer_6[idx][patch_num] = None
            activations_layer_16[idx][patch_num] = None

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

        # Wrap in template (NO ANSWER ADDED)
        query = prefix % prompt

        # Print the final prompt (same as original script)
        print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Get the wrapped prompt without any answer
            prompt_wrap = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[query],
                img_prompts=[[img_tensor]],
                device=f'cuda:{args.gpu_id}'
            )
            context_embs = prompt_wrap.context_embs[0]
            
            # Add debugging code to show text segments
            if idx == 0:  # Only for first image
                print("\n=== Text Segments Analysis ===")
                # Split the prompt by ImageHere
                prompt_segs = query.split('<ImageHere>')

                # Print the text segments
                print("=== Text Segments ===")
                print(f"Number of segments: {len(prompt_segs)}")

                print("\n--- Text BEFORE <ImageHere> ---")
                print(f"'{prompt_segs[0]}'")
                print(f"Length: {len(prompt_segs[0])} characters")

                print("\n--- Text AFTER <ImageHere> ---")
                print(f"'{prompt_segs[1]}'")
                print(f"Length: {len(prompt_segs[1])} characters")

                # If you want to see the tokenized versions too
                print("\n=== Tokenized Versions ===")

                print("\n--- Tokens BEFORE <ImageHere> ---")
                tokens_before = tokenizer(prompt_segs[0], return_tensors="pt")
                print(f"Token IDs: {tokens_before.input_ids[0].tolist()}")
                print(f"Decoded: '{tokenizer.decode(tokens_before.input_ids[0])}'")
                print(f"Number of tokens: {tokens_before.input_ids.shape[1]}")

                print("\n--- Tokens AFTER <ImageHere> ---")
                tokens_after = tokenizer(prompt_segs[1], return_tensors="pt")
                print(f"Token IDs: {tokens_after.input_ids[0].tolist()}")
                print(f"Decoded: '{tokenizer.decode(tokens_after.input_ids[0])}'")
                print(f"Number of tokens: {tokens_after.input_ids.shape[1]}")
                
                # Calculate expected image positions
                text1_tokens = tokens_before.input_ids.shape[1]
                img_start = text1_tokens
                img_end = text1_tokens + prompt_wrap.img_embs[0][0].shape[1]  # Use actual image embedding length
                print(f"\n=== Expected Image Positions ===")
                print(f"Image tokens should be at positions: {img_start} to {img_end}")
                print(f"Total context length: {context_embs.shape[1]}")
                print("=== End Text Segments Analysis ===\n")
            
            # Forward pass
            outputs = model.llama_model(inputs_embeds=context_embs, output_hidden_states=True, return_dict=True)
            
            # Verify that context_embs and outputs.hidden_states have the same length
            if idx == 0:  # Only for first image
                print("\n=== Length Verification ===")
                print(f"Context embeddings shape: {context_embs.shape}")
                print(f"Outputs hidden states shape: {outputs.hidden_states[0].shape}")
                print(f"Context length: {context_embs.shape[1]}")
                print(f"Hidden states length: {outputs.hidden_states[0].shape[1]}")
                
                if context_embs.shape[1] == outputs.hidden_states[0].shape[1]:
                    print("✓ Lengths match!")
                else:
                    print("✗ Lengths do not match!")
                    raise ValueError("Context embeddings and hidden states have different lengths")
                
                # Show the actual context sequence structure
                print(f"\n=== Context Sequence Analysis ===")
                print(f"Text embeddings: {len(prompt_wrap.text_embs[0])} segments")
                print(f"Image embeddings: {len(prompt_wrap.img_embs[0])} segments")
                
                # Show text segment details
                for i, text_emb in enumerate(prompt_wrap.text_embs[0]):
                    print(f"Text segment {i}: shape {text_emb.shape} (positions {text_emb.shape[1]} tokens)")
                
                # Show image segment details
                for i, img_emb in enumerate(prompt_wrap.img_embs[0]):
                    print(f"Image segment {i}: shape {img_emb.shape} (positions {img_emb.shape[1]} tokens)")
                
                # Calculate and show the interleaved structure
                print(f"\n=== Interleaved Context Structure ===")
                running_pos = 0
                for i, text_emb in enumerate(prompt_wrap.text_embs[0]):
                    text_start = running_pos
                    text_end = running_pos + text_emb.shape[1]
                    print(f"Text segment {i}: positions {text_start} to {text_end-1} (length: {text_emb.shape[1]})")
                    running_pos = text_end
                    
                    if i < len(prompt_wrap.img_embs[0]):
                        img_start = running_pos
                        img_end = running_pos + prompt_wrap.img_embs[0][i].shape[1]
                        print(f"Image segment {i}: positions {img_start} to {img_end-1} (length: {prompt_wrap.img_embs[0][i].shape[1]})")
                        running_pos = img_end
                
                print("=== End Length Verification ===\n")
            
            # Calculate image positions based on conversation.py logic
            prompt_segs = query.split('<ImageHere>')
            seg_tokens = [
                tokenizer(seg, return_tensors="pt", add_special_tokens=(i == 0)).input_ids
                for i, seg in enumerate(prompt_segs)
            ]
            
            # Calculate image positions
            running_pos = 0
            for i in range(len(seg_tokens) - 1):
                running_pos += seg_tokens[i].shape[1]
                start_idx = running_pos
                end_idx = start_idx + prompt_wrap.img_embs[0][i].shape[1]  # Use actual image embedding length
                break  # Get first image
            
            num_image_tokens = end_idx - start_idx
            
            # Print token analysis for the first image+prompt pair only
            if idx == 0:
                print("\n=== First Image+Prompt Token Analysis ===")
                
                # Print the whole wrapped prompt
                print(f"Whole wrapped prompt:")
                print(f"'{query}'")
                print()
                
                # Debug: Show what's in the context embeddings
                print(f"Context embeddings shape: {context_embs.shape}")
                print(f"Text embeddings: {len(prompt_wrap.text_embs[0])} segments")
                print(f"Image embeddings: {len(prompt_wrap.img_embs[0])} segments")
                
                for i, text_emb in enumerate(prompt_wrap.text_embs[0]):
                    print(f"  Text segment {i}: shape {text_emb.shape}")
                for i, img_emb in enumerate(prompt_wrap.img_embs[0]):
                    print(f"  Image segment {i}: shape {img_emb.shape}")
                
                print(f"\nCalculated image embedding indices: start={start_idx}, end={end_idx}")
                print(f"Number of image embeddings: {num_image_tokens}")
                print(f"Total context length: {context_embs.shape[1]}")
                
                print("=== End of First Image+Prompt Analysis ===\n")
            
            # Comprehensive logging
            tqdm.write("\n" + "="*50)
            tqdm.write(f"Image: {row['image_name']}")
            tqdm.write(f"Prompt: {prompt}")
            tqdm.write(f"Correct Answer: {correct_answer}")
            tqdm.write(f"Image token start index: {start_idx}, end index: {end_idx}, num tokens: {num_image_tokens}")
            tqdm.write(f"Total context length: {context_embs.shape[1]}")
            
            # Extract activations for layers 6 and 16 - ONLY from image tokens after forward pass
            for layer in [6, 16]:
                # Extract image token activations (keep separate, don't average)
                # This extracts activations ONLY from the 32 image token positions
                image_token_acts = outputs.hidden_states[layer][0, start_idx:end_idx, :].detach().cpu()
                
                # Store each patch activation separately
                for patch_idx in range(min(num_image_tokens, 32)):  # Ensure we don't exceed 32 patches
                    patch_num = patch_idx + 1  # 1-32 numbering
                    patch_activation = image_token_acts[patch_idx]
                    
                    if layer == 6:
                        activations_layer_6[idx][patch_num] = patch_activation
                    else:  # layer == 16
                        activations_layer_16[idx][patch_num] = patch_activation
                    
                    # Log activation details for first few patches
                    if patch_idx < 3:
                        tqdm.write(f"Layer {layer} patch {patch_num} activation shape: {patch_activation.shape}")
                        tqdm.write(f"Layer {layer} patch {patch_num} activation: {patch_activation}")
                
                # Log total number of patches extracted
                tqdm.write(f"Layer {layer} extracted {min(num_image_tokens, 32)} patch activations")
            
            tqdm.write("="*50 + "\n")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save separate .pt files for each layer
    layer_6_save_path = os.path.join(args.save_dir, "Layer_6_img_tokens_test.pt")
    layer_16_save_path = os.path.join(args.save_dir, "Layer_16_img_tokens_test.pt")
    
    torch.save(activations_layer_6, layer_6_save_path)
    torch.save(activations_layer_16, layer_16_save_path)
    
    print(f"Saved Layer 6 activations to {layer_6_save_path}")
    print(f"Saved Layer 16 activations to {layer_16_save_path}")
    
    print(f"Saved 2 separate .pt files")
    print(f"Each file contains activations for one layer with structure:")
    print(f"  - Outer keys: numbers (0, 1, 2, ...)")
    print(f"  - Inner keys: 'correct_answer' and patch numbers 1-32")
    print(f"  - Values: PyTorch tensors for each patch activation")

if __name__ == "__main__":
    main()