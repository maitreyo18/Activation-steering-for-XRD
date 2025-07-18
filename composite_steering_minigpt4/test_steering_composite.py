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
    parser = argparse.ArgumentParser(description="Composite Steering Generation with Multiple Anomaly Types")
    parser.add_argument("--steer_layer", type=int, default=16, help="Layer to apply steering")
    parser.add_argument('--low_resource', type=str, default='False', help='use 8-bit quantization when set to True')
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--test_csv", type=str, default="test_generate.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/combined_images", help="Directory containing images")
    parser.add_argument(
        "--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def load_composite_steering_vectors(steer_layer):
    """
    Load all activation vectors for each anomaly type and their corresponding alpha values.
    Returns a composite steering vector computed as:
    alpha_bg_ring * v_bg_ring + alpha_ice_ring * v_ice_ring + alpha_loop_scattering * v_loop_scattering + 
    alpha_strong_background * v_strong_background + alpha_nonuniform_detector * v_nonuniform_detector
    """
    
    # ===========================================
    # PLACEHOLDER: PUT YOUR ACTIVATION PATHS HERE
    # ===========================================
    background_ring_path = "../CAA_modified_minigpt4/Bg_ring_act/act_diff_12-20_gen_600_standby.pt"
    ice_ring_path = "../CAA_modified_minigpt4/Ice_ring/ice_ring_act_diff_12-20_76_strong_background.pt"
    loop_scattering_path = "../CAA_modified_minigpt4/Loop_scattering/loop_scattering_act_diff_12-20_99_artifact.pt"
    strong_background_path = "../CAA_modified_minigpt4/Strong_background/strong_bg_act_diff_12-20_149_artifact.pt"
    nonuniform_detector_path = "../CAA_modified_minigpt4/Non-uniform_detect/act_diff_12-20_nu_detect_259_bg_ring.pt"
    
    # ===========================================
    # PLACEHOLDER: PUT YOUR ALPHA VALUES HERE
    # ===========================================
    alpha_bg_ring = -0.1
    alpha_ice_ring = 0.325
    alpha_loop_scattering = -0.1
    alpha_strong_background = -0.1
    alpha_nonuniform_detector = -0.1
    
    print("=== Loading Composite Steering Vectors ===")
    print(f"Alpha values:")
    print(f"  Background Ring: {alpha_bg_ring}")
    print(f"  Ice Ring: {alpha_ice_ring}")
    print(f"  Loop Scattering: {alpha_loop_scattering}")
    print(f"  Strong Background: {alpha_strong_background}")
    print(f"  Non-uniform Detector: {alpha_nonuniform_detector}")
    print()
    
    # Load all activation vectors
    anomaly_paths = {
        "background_ring": background_ring_path,
        "ice_ring": ice_ring_path,
        "loop_scattering": loop_scattering_path,
        "strong_background": strong_background_path,
        "nonuniform_detector": nonuniform_detector_path
    }
    
    anomaly_alphas = {
        "background_ring": alpha_bg_ring,
        "ice_ring": alpha_ice_ring,
        "loop_scattering": alpha_loop_scattering,
        "strong_background": alpha_strong_background,
        "nonuniform_detector": alpha_nonuniform_detector
    }
    
    steering_vectors = {}
    
    for anomaly_type, path in anomaly_paths.items():
        print(f"Loading {anomaly_type} from: {path}")
        try:
            steering_data = torch.load(path)
            
            # Extract steering vector for the specified layer
            steer_acts = []
            for idx in steering_data:
                if steer_layer in steering_data[idx]:
                    steer_acts.extend(steering_data[idx][steer_layer])
            
            if not steer_acts:
                print(f"Warning: No steering vectors found for layer {steer_layer} in {anomaly_type}")
                continue
            
            steer_acts = np.array(steer_acts)
            mean_vector = np.mean(steer_acts, axis=0)
            steering_vectors[anomaly_type] = torch.from_numpy(mean_vector)
            print(f"  ✓ Loaded {anomaly_type} vector (norm: {torch.norm(steering_vectors[anomaly_type], p=2):.4f})")
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {path}")
        except Exception as e:
            print(f"  ✗ Error loading {anomaly_type}: {e}")
    
    # Compute composite steering vector
    if not steering_vectors:
        raise ValueError("No steering vectors were successfully loaded!")
    
    composite_vector = torch.zeros_like(list(steering_vectors.values())[0])
    
    print("\n=== Computing Composite Steering Vector ===")
    for anomaly_type, vector in steering_vectors.items():
        alpha = anomaly_alphas[anomaly_type]
        contribution = alpha * vector
        composite_vector += contribution
        print(f"  {anomaly_type}: alpha={alpha}, contribution_norm={torch.norm(contribution, p=2):.4f}")
    
    print(f"\nFinal composite vector norm: {torch.norm(composite_vector, p=2):.4f}")
    print("=" * 50)
    
    return composite_vector

def generate_with_steering(prompt, model, tokenizer, steering_vector, steer_layer, max_new_tokens=50, stop_ids=None):
    """
    Autoregressive generation with KV caching and per-token steering from a specific position onward.
    
    - Applies steering vector only to tokens with position >= prompt length.
    - Prints tokens one-by-one (streaming).
    - Uses composite steering vector.
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

    # Hook to steer only generated tokens (not the prompt)
    def make_hook():
        def hook_fn(module, inp, out):
            nonlocal step
            # Apply steering only to the last token of prompt and first two generated tokens
            if step < 3:
                # Get the activation before steering
                original_activation = out[0][:, -1, :].clone()
                
                # Apply steering to the single token being processed
                out[0][:, -1, :] += steering_vector
                
                # Get the activation after steering
                steered_activation = out[0][:, -1, :].clone()
                
                print(f"✅ Composite steering applied at generation step {step}")
                print(f"  Original activation: {original_activation}")
                print(f"  Steered activation:  {steered_activation}")
            else:
                print(f"⏭️  No steering applied at generation step {step}")
            return out
        return hook_fn

    try:
        print(f"Starting generation with prompt length: {prompt_len}")
        print(f"Composite steering will be applied to last token of prompt and first 2 generated tokens only")
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

def process_item(item, model, tokenizer, processor, steer_vector, steer_layer, generator_obj, image_dir):
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
            
            # Generate with composite steering
            print(f"\n=== GENERATING WITH COMPOSITE STEERING (Layer: {steer_layer}) ===")
            print("Generated text: ", end="")
            generated_text = generate_with_steering(
                prompt_wrap, model, tokenizer, steer_vector, steer_layer, 
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
                "Steered": generated_text,
                "no_steer": benign_response
            }
    except Exception as e:
        print(f"Error processing item {item['image_name']}: {e}")
        return None

def main():
    args = parse_args()
    print("=== Composite Steering Generation with Multiple Anomaly Types ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Layer: {args.steer_layer}")

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

    # Load composite steering vector
    composite_steering_vector = load_composite_steering_vectors(args.steer_layer)
    composite_steering_vector = composite_steering_vector.cuda()

    # Read CSV file
    print(f"Reading CSV file: {args.test_csv}")
    try:
        df = pd.read_csv(args.test_csv).head(40)  # head(10) to test
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
                composite_steering_vector, args.steer_layer, 
                generator_obj, args.image_dir
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
    filename = f"./Composite_steering_CAA_generation_results_layer_{args.steer_layer}_artifact.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    print(f"Processed {len(results)} items successfully with composite steering")

if __name__ == "__main__":
    main() 