import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import argparse
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

from minigpt_utils import prompt_wrapper, generator
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Generate neutral model responses")
    parser.add_argument('--low_resource', type=str, default='False', help='use 8-bit quantization when set to True')
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/background_ring", help="Directory containing images")
    parser.add_argument(
        "--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def process_item(item, model, tokenizer, processor, generator_obj, image_dir):
    """
    Process a single item from the CSV and generate model response
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
        
        # Create prompt using the same template as CAA_simplified_full.py
        prefix = prompt_wrapper.minigpt4_chatbot_prompt
        query = prefix % prompt
        
        print(f"Processing: {item['image_name']}")
        print(f"Prompt: {prompt[:100]}...")
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Create prompt wrapper
            prompt_wrap = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[query],
                img_prompts=[[img_tensor]],
                device=model.device
            )
            
            # Generate model response (same as CAA_simplified_full.py)
            model_response, _ = generator_obj.generate(prompt_wrap)
            print(f"Model response: {model_response}")
            
            return {
                "image_name": item['image_name'],
                "prompt": prompt,
                "answer": answer,
                "model_response": model_response
            }
    except Exception as e:
        print(f"Error processing item {item['image_name']}: {e}")
        return None

def main():
    args = parse_args()
    print("=== Model Neutral Response Generation ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")

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
    
    # Initialize generator (same settings as CAA_simplified_full.py)
    generator_obj = generator.Generator(
        model=model,
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

    # Process each item
    results = []
    for idx, row in df.iterrows():
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
                generator_obj, args.image_dir
            )
            if result:
                results.append(result)
                print(f"✓ SUCCESS: Generated response")
            else:
                print("✗ FAILED: Could not process this item")
                
        except Exception as e:
            print(f"✗ ERROR processing item {idx+1}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    filename = "model_neutral.csv"
    df_results.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    print(f"Processed {len(results)} items successfully")

if __name__ == "__main__":
    main() 