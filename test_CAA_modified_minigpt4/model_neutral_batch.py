import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import argparse
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm

from minigpt_utils import prompt_wrapper
from minigpt_utils.generator_batch import GeneratorBatch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Generate neutral model responses in batch")
    parser.add_argument('--low_resource', type=str, default='False', help='use 8-bit quantization when set to True')
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/background_ring", help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument(
        "--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def process_batch(batch_items, model, tokenizer, processor, generator_obj, image_dir):
    """
    Process a batch of items from the CSV and generate model responses
    """
    batch_results = []
    
    # Prepare batch data
    batch_images = []
    batch_prompts = []
    batch_answers = []
    batch_image_names = []
    valid_indices = []
    
    for i, item in enumerate(batch_items):
        try:
            image_path = os.path.join(image_dir, item['image_name'])
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
                
            image = Image.open(image_path).convert("RGB")
            img_tensor = processor(image).unsqueeze(0)  # Add batch dimension
            
            batch_images.append(img_tensor)
            batch_prompts.append(item['prompt'])
            batch_answers.append(item['answer'])
            batch_image_names.append(item['image_name'])
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error loading image {item['image_name']}: {e}")
            continue
    
    if not batch_images:
        return batch_results
    
    # Stack images into a single batch tensor
    batch_img_tensor = torch.cat(batch_images, dim=0).to(model.device)
    
    # Create prompts using the same template as model_neutral.py
    prefix = prompt_wrapper.minigpt4_chatbot_prompt
    batch_queries = [prefix % prompt for prompt in batch_prompts]
    
    print(f"Processing batch of {len(batch_queries)} items...")
    
    print(f"DEBUG EARLY: batch_image_names length: {len(batch_image_names)}")
    print(f"DEBUG EARLY: batch_prompts length: {len(batch_prompts)}")
    print(f"DEBUG EARLY: batch_answers length: {len(batch_answers)}")
    print(f"DEBUG EARLY: batch_img_tensor shape: {batch_img_tensor.shape}")
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Create individual prompt wrappers for each item in the batch
        prompt_wraps = []
        print(f"DEBUG: Starting prompt wrapper creation loop...")
        for i, (image_name, prompt, answer) in enumerate(zip(batch_image_names, batch_prompts, batch_answers)):
            print(f"DEBUG: Creating prompt wrapper {i+1}/{len(batch_image_names)}")
            try:
                # Create individual prompt wrapper for each item
                individual_img_tensor = batch_img_tensor[i:i+1]  # Take single image from batch
                individual_query = batch_queries[i]
                
                print(f"DEBUG: individual_img_tensor shape: {individual_img_tensor.shape}")
                print(f"DEBUG: individual_query: {individual_query[:100]}...")  # First 100 chars
                
                individual_prompt_wrap = prompt_wrapper.Prompt(
                    model=model,
                    text_prompts=[individual_query],
                    img_prompts=[individual_img_tensor],
                    device=model.device
                )
                print(f"DEBUG: Created prompt wrapper {i+1}")
                prompt_wraps.append(individual_prompt_wrap)
            except Exception as e:
                print(f"DEBUG ERROR: Failed to create prompt wrapper {i+1}: {e}")
                print(f"DEBUG ERROR: image_name: {image_name}")
                print(f"DEBUG ERROR: prompt: {prompt}")
                raise e  # Re-raise to see the full traceback
        
        print(f"DEBUG: Finished creating {len(prompt_wraps)} prompt wrappers")
        
        # Generate responses for the entire batch using the batch generator
        print(f"DEBUG: Number of prompt_wraps: {len(prompt_wraps)}")
        print(f"DEBUG: Number of batch_image_names: {len(batch_image_names)}")
        print(f"DEBUG: Number of batch_prompts: {len(batch_prompts)}")
        print(f"DEBUG: Number of batch_answers: {len(batch_answers)}")
        
        # Debug: Check context embedding shapes
        for i, prompt_wrap in enumerate(prompt_wraps):
            print(f"DEBUG: prompt_wrap[{i}] context_embs[0] shape: {prompt_wrap.context_embs[0].shape}")
        
        batch_results_raw = generator_obj.generate_batch(prompt_wraps)
        
        print(f"DEBUG: batch_results_raw type: {type(batch_results_raw)}")
        print(f"DEBUG: batch_results_raw length: {len(batch_results_raw)}")
        if batch_results_raw:
            print(f"DEBUG: batch_results_raw[0] type: {type(batch_results_raw[0])}")
            print(f"DEBUG: batch_results_raw[0] length: {len(batch_results_raw[0])}")
            print(f"DEBUG: batch_results_raw[0] content: {batch_results_raw[0]}")
        
        # Process results
        if len(batch_results_raw) != len(batch_image_names):
            print(f"WARNING: batch_results_raw length ({len(batch_results_raw)}) does not match batch_image_names length ({len(batch_image_names)})")
            min_len = min(len(batch_results_raw), len(batch_image_names))
        else:
            min_len = len(batch_results_raw)

        for i in range(min_len):
            response, _ = batch_results_raw[i]
            image_name = batch_image_names[i]
            prompt = batch_prompts[i]
            answer = batch_answers[i]
            print(f"  {image_name}: {response}")
            batch_results.append({
                "image_name": image_name,
                "prompt": prompt,
                "answer": answer,
                "model_response": response
            })
    
    return batch_results

def main():
    args = parse_args()
    print("=== Model Neutral Response Generation (Batch Mode) ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Batch Size: {args.batch_size}")

    # Initialize model
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    
    print(f"DEBUG MODEL: model_cls: {model_cls}")
    print(f"DEBUG MODEL: model_config.arch: {model_config.arch}")
    
    if args.low_resource.lower() == 'true':
        model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}', dtype=torch.float16)
    else:
        model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()
    
    print(f"DEBUG MODEL: model type: {type(model)}")
    print(f"DEBUG MODEL: model.encode_img: {model.encode_img}")
    print(f"DEBUG MODEL: model.encode_img.__code__.co_argcount: {model.encode_img.__code__.co_argcount}")

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    # Initialize generator (same settings as model_neutral.py)
    generator_obj = GeneratorBatch(
        model=model,
        max_new_tokens=150,
        #temperature=0.6,
        #top_p=0.7,
        num_beams=1
    )

    print('[Model Initialized]\n')

    # Read CSV file
    print(f"Reading CSV file: {args.test_csv}")
    try:
        df = pd.read_csv(args.test_csv)
        print(f"Found {len(df)} items to process")
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.test_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Process items in batches
    results = []
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    
    print(f"Processing {len(df)} items in {total_batches} batches of size {args.batch_size}")
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(df))
        
        print(f"\n{'='*80}")
        print(f"PROCESSING BATCH {batch_idx+1}/{total_batches} (items {start_idx+1}-{end_idx})")
        print(f"{'='*80}")
        
        # Prepare batch items
        batch_items = []
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            item = {
                'image_name': row['image_name'],
                'prompt': row['prompt'],
                'answer': row['answer']
            }
            batch_items.append(item)
        
        try:
            batch_results = process_batch(
                batch_items, model, tokenizer, processor, 
                generator_obj, args.image_dir
            )
            results.extend(batch_results)
            print(f"✓ SUCCESS: Processed {len(batch_results)} items in batch {batch_idx+1}")
                
        except Exception as e:
            print(f"✗ ERROR processing batch {batch_idx+1}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    filename = "model_neutral_batch.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    print(f"Processed {len(results)} items successfully out of {len(df)} total items")

if __name__ == "__main__":
    main() 