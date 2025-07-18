import os
import sys
sys.path.append(os.path.abspath(os.curdir))
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
    parser = argparse.ArgumentParser(description="CAA Generation with Continual Steering")
    parser.add_argument("--alpha", type=float, default=3.0, help="Initial steering strength multiplier")
    parser.add_argument("--steer_layer", type=int, default=20, help="Layer to apply steering")
    parser.add_argument('--low_resource', type=str, default='False', help='use 8-bit quantization when set to True')
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--steering_vectors_path", type=str, default="../CAA_modified/act_diff_12-20_gen_600_standby.pt", help="Path to steering vectors")
    parser.add_argument("--test_csv", type=str, default="test_generate.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/background_ring", help="Directory containing images")
    parser.add_argument(
        "--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def generate_with_continual_steering(prompt, model, tokenizer, steering_vector, alpha, steer_layer, max_new_tokens=150, stop_ids=None):
    """
    Autoregressive generation with KV caching and continual steering.
    
    - Applies provided alpha for first 5 tokens
    - Then applies decreasing values from np.linspace(1.7, 0, 200)
    - Prints tokens one-by-one (streaming).
    """
    device = model.device
    steering_vector = steering_vector.to(device)

    input_embeds = prompt.context_embs[0]  # (1, prompt_len, hidden_dim)
    prompt_len = input_embeds.shape[1]
    generated_ids = []
    past = None
    step = 0  # counts total generation steps

    # Define stopping criteria (same as original generator)
    stop_words_ids = [torch.tensor([835]).to(device),
                      torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.

    # Create decreasing alpha values
    decreasing_alphas = np.linspace(1.7, 0, 200)
    print(f"Decreasing alpha values: {decreasing_alphas[:10]}... (total: {len(decreasing_alphas)})")

    # Hook to steer with continual pattern
    def make_hook():
        def hook_fn(module, inp, out):
            nonlocal step
            # Apply provided alpha for first 5 tokens
            if step < 5:
                current_alpha = alpha
                print(f"✅ Initial steering applied at generation step {step} (alpha: {current_alpha})")
            # Then apply decreasing values
            elif step < 5 + len(decreasing_alphas):
                current_alpha = decreasing_alphas[step - 5]
                print(f"📉 Decreasing steering applied at generation step {step} (alpha: {current_alpha:.3f})")
            else:
                current_alpha = 0.0
                print(f"⏭️  No steering applied at generation step {step} (alpha: {current_alpha})")
            
            # Apply steering
            if current_alpha > 0:
                # Get the activation before steering
                original_activation = out[0][:, -1, :].clone()
                
                # Apply steering to the single token being processed
                out[0][:, -1, :] += steering_vector * current_alpha
                
                # Get the activation after steering
                steered_activation = out[0][:, -1, :].clone()
                
                print(f"  Original activation: {original_activation}")
                print(f"  Steered activation:  {steered_activation}")
            
            return out
        return hook_fn

    try:
        print(f"Starting generation with prompt length: {prompt_len}")
        print(f"Initial alpha: {alpha} for first 5 tokens")
        print(f"Then decreasing from 1.7 to 0 over 200 tokens")
        print(f"Max new tokens: {max_new_tokens}")
        print("-" * 50)
        
        # Register hook BEFORE any token generation
        hook = model.llama_model.base_model.layers[steer_layer - 1].register_forward_hook(make_hook())
        
        # Step 1: Run with image-text embedding (prompt) - STEERING APPLIED HERE TOO
        print(f"Step 1: Processing prompt embeddings...")
        out = model.llama_model(inputs_embeds=input_embeds, use_cache=True, return_dict=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(f"Generated token: '{token_text}' (ID: {token_id})")
        generated_ids.append(token_id)
        step += 1

        # Step 2+: Generate one token at a time - STEERING APPLIED HERE
        for gen_step in range(max_new_tokens - 1):
            print(f"Step {gen_step + 2}: Generating next token...")
            out = model.llama_model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            print(f"Generated token: '{token_text}' (ID: {token_id})")
            generated_ids.append(token_id)
            step += 1

            # Check stopping criteria (same as original generator)
            for stop in stop_words_ids:
                if len(generated_ids) >= len(stop) and all(
                    generated_ids[-len(stop):][i] == stop[i].item() for i in range(len(stop))
                ):
                    print(f"\n🛑 STOPPING CRITERIA MET at step {step}")
                    break
            else:
                continue
            break  # Exit the loop if stopping criteria was met

    finally:
        hook.remove()
    
    # Clean the generated text (same as original generate method)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Apply the same cleaning as the original generate method
    if generated_text.startswith('###'):
        generated_text = generated_text[3:]  # Remove leading '###'
    generated_text = generated_text.split('###')[0]  # remove the stop sign '###'
    generated_text = generated_text.split('Assistant:')[-1].strip()
    
    return generated_text

def process_item(item, model, tokenizer, processor, steer_vector, alpha, steer_layer, generator_obj, image_dir):
    """
    Process a single item from the CSV
    """
    try:
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(model.device)
        prompt = item['prompt']
        answer = item['answer']
        
        # Create prompt
        prefix = prompt_wrapper.minigpt4_chatbot_prompt
        query = prefix % prompt
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Create prompt wrapper
            prompt_wrap = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[query],
                img_prompts=[[img_tensor]],
                device=model.device
            )
            
            # Generate with continual steering using the new method
            print(f"\n=== GENERATING WITH CONTINUAL STEERING (Initial Alpha: {alpha}, Layer: {steer_layer}) ===")
            print("Generated text: ", end="")
            generated_text = generate_with_continual_steering(
                prompt_wrap, model, tokenizer, steer_vector, alpha, steer_layer, 
                max_new_tokens=generator_obj.max_new_tokens
            )
            print()  # New line after generation
            
            # Get benign response (no steering)
            print(f"\n=== GENERATING BENIGN RESPONSE (NO STEERING) ===")
            print("Generated text: ", end="")
            benign_response, _ = generator_obj.generate(prompt_wrap)
            print()  # New line after generation
            
            return {
                "image_name": item['image_name'],
                "prompt": prompt,
                "answer": answer,
                "Continual_Steered": generated_text,
                "no_steer": benign_response
            }
    except Exception as e:
        print(f"Error processing item {item['image_name']}: {e}")
        return None

def main():
    args = parse_args()
    print("=== CAA Generation with Continual Steering (Batch Mode) ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Initial Alpha: {args.alpha}, Layer: {args.steer_layer}")

    # Initialize model
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    
    if args.low_resource.lower() == 'true':
        model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}', dtype=torch.float16)
    else:
        model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Initialize generator
    generator_obj = generator.Generator(
        model=model,
        #temperature=0.5,
        #top_p=0.6,
        num_beams=1
    )

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
        df = pd.read_csv(args.test_csv).head(5)
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
                generator_obj, args.image_dir
            )
            if result:
                results.append(result)
                print(f"✓ SUCCESS: Generated text: {result['Continual_Steered']}")
            else:
                print("✗ FAILED: Could not process this item")
                
        except Exception as e:
            print(f"✗ ERROR processing item {idx+1}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    filename = f"./CAA_continual_steering_results_layer_{args.steer_layer}_alpha_{args.alpha}.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    print(f"Processed {len(results)} items successfully")

if __name__ == "__main__":
    main() 