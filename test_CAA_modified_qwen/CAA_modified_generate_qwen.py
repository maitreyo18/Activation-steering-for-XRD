import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Simple CAA Generation with Continuous Steering (Qwen2-VL)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Steering strength multiplier")
    parser.add_argument("--steer_layer", type=int, default=20, help="Layer to apply steering")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--steering_vectors_path", type=str, default="../CAA_modified_qwen/artifact_act_diff_10-20_5_non_uniform_detector.pt", help="Path to steering vectors")
    parser.add_argument("--test_csv", type=str, default="test_generate_nu_detect.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/non_uniform_detector", help="Directory containing images")
    args = parser.parse_args()
    return args

def generate_with_steering(model, tokenizer, processor, image, prompt, steering_vector, alpha, steer_layer, max_new_tokens=150):
    """
    Autoregressive generation with KV caching and per-token steering from a specific position onward.
    
    - Applies steering vector only to tokens with position >= prompt length.
    - Prints tokens one-by-one (streaming).
    - Uses alpha for steering.
    """
    device = model.device
    steering_vector = steering_vector.to(device)

    # Create prompt using Qwen2-VL chat template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None,},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    query = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Process inputs using Qwen2-VL processor
    inputs = processor(
        text=[query], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    generated_ids = []
    step = 0  # counts total generation steps

    # Hook to steer only generated tokens (not the prompt)
    def make_hook():
        def hook_fn(module, inp, out):
            nonlocal step
            # Apply steering only to the endmost token of prompt and first three generated tokens
            if step < 3:  # 0: endmost token of prompt, 1: first generated token, 2: second generated token
                # Get the activation before steering
                original_activation = out[0][:, -1, :].clone()
                
                # Apply steering to the single token being processed
                out[0][:, -1, :] += steering_vector * alpha
                
                # Get the activation after steering
                steered_activation = out[0][:, -1, :].clone()
                
                print(f"✅ Steering applied at generation step {step} (alpha: {alpha})")
                print(f"  Original activation: {original_activation}")
                print(f"  Steered activation:  {steered_activation}")
            else:
                print(f"⏭️  No steering applied at generation step {step}")
            step += 1
            return out
        return hook_fn

    try:
        print(f"Starting generation with prompt length: {prompt_len}")
        print(f"Steering will be applied to endmost token of prompt and first 3 generated tokens only")
        print(f"Max new tokens: {max_new_tokens}")
        print("-" * 50)
        
        # Register hook BEFORE any token generation
        hook = model.model.layers[steer_layer - 1].register_forward_hook(make_hook())
        
        print("Step 1: Processing prompt embeddings...")
        
        # Generate with steering
        with torch.no_grad(), torch.cuda.amp.autocast():
            generate_ids = model.generate(
                **inputs, 
                do_sample=False, 
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
            # Extract generated tokens (excluding input tokens)
            generated_tokens = generate_ids.sequences[0, inputs["input_ids"].shape[1]:]
            generated_ids = generated_tokens.tolist()
            
            # Print tokens as they would be generated
            for i, token_id in enumerate(generated_ids):
                if i == 0:
                    print(f"Generated token: '{tokenizer.decode([token_id], skip_special_tokens=True)}' (ID: {token_id})")
                else:
                    print(f"Step {i+1}: Generating next token...")
                    print(f"Generated token: '{tokenizer.decode([token_id], skip_special_tokens=True)}' (ID: {token_id})")

    finally:
        hook.remove()
    
    # Clean the generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def process_item(item, model, tokenizer, processor, steer_vector, alpha, steer_layer, image_dir):
    """
    Process a single item from the CSV
    """
    try:
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert("RGB")
        prompt = item['prompt']
        answer = item['answer']
        
        # Generate with steering using the new method
        print(f"\n=== GENERATING WITH STEERING (Alpha: {alpha}, Layer: {steer_layer}) ===")
        print("Generated text: ", end="")
        generated_text = generate_with_steering(
            model, tokenizer, processor, image, prompt, 
            steer_vector, alpha, steer_layer, max_new_tokens=150
        )
        print()  # New line after generation
        
        # Get benign response (no steering)
        print(f"\n=== GENERATING BENIGN RESPONSE (NO STEERING) ===")
        print("Generated text: ", end="")
        
        # Create prompt for benign generation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None,},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        query = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = processor(
                text=[query], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
            generate_ids = model.generate(
                **inputs, 
                do_sample=True, 
                max_new_tokens=150,
                temperature=0.8, 
                top_p=0.9
            )
            benign_response = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
        print()  # New line after generation
        
        return {
            "image_name": item['image_name'],
            "prompt": prompt,
            "answer": answer,
            "Steered": generated_text,
            "no_steer": benign_response
        }
    except Exception as e:
        print(f"Error processing item {item['image_name']}: {e}")
        return None

def main():
    args = parse_args()
    print("=== CAA Generation with Continuous Steering (Qwen2-VL) ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Alpha: {args.alpha}, Layer: {args.steer_layer}")

    # Initialize Qwen2-VL model
    print('[Start Initialization]\n')
    model_name = '../ckpts/Qwen2-VL-7B'
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto",
        local_files_only=True
    )
    model.eval()

    # Initialize processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)

    print('[Model Initialized]\n')

    # Load steering vectors
    print("Loading steering vectors...")
    try:
        steering_data = torch.load(args.steering_vectors_path)
    except FileNotFoundError:
        print(f"Error: Steering vectors file not found: {args.steering_vectors_path}")
        return
    except Exception as e:
        print(f"Error loading steering vectors: {e}")
        return
    
    # Extract steering vector for the specified layer
    steer_acts = []
    for idx in steering_data:
        if args.steer_layer in steering_data[idx]:
            steer_acts.extend(steering_data[idx][args.steer_layer])
    
    if not steer_acts:
        raise ValueError(f"No steering vectors found for layer {args.steer_layer}")
    
    steer_acts = np.array(steer_acts)
    steer_vector = np.mean(steer_acts, axis=0)
    print(f"Mean steering vector: {steer_vector}")
    steer_vector = torch.from_numpy(steer_vector).cuda()

    print(f"Steering vector loaded (norm: {torch.norm(steer_vector, p=2):.4f})")

    # Read CSV file
    print(f"Reading CSV file: {args.test_csv}")
    try:
        df = pd.read_csv(args.test_csv) # head(5) to test
        print(f"Found {len(df)} items to process")
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.test_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Process each item
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
        print(f"\n{'='*80}")
        print(f"PROCESSING ROW {idx+1}/{len(df)}")
        print(f"Image: {row['image_name']}")
        print(f"Prompt: {row['prompt'][:100]}...")
        print(f"Expected Answer: {row['answer']}")
        print(f"{'='*80}")
        
        item = {
            'image_name': row['image_name'],
            'prompt': row['prompt'],
            'answer': row['answer']
        }
        
        try:
            result = process_item(
                item, model, tokenizer, processor, 
                steer_vector, args.alpha, args.steer_layer, 
                args.image_dir
            )
            if result:
                results.append(result)
                print(f"✓ SUCCESS: Generated text: {result['Steered']}")
            else:
                print("✗ FAILED: Could not process this item")
                
        except Exception as e:
            print(f"✗ ERROR processing item {idx+1}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    filename = f"./Qwen2VL_CAA_generation_results_layer_{args.steer_layer}_alpha_{args.alpha}.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    print(f"Processed {len(results)} items successfully")

if __name__ == "__main__":
    main() 