import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import argparse
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Generate neutral model responses")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Test CSV file")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/background_ring", help="Directory containing images")
    args = parser.parse_args()
    return args

def process_item(item, model, tokenizer, processor, image_dir, is_first_item=False):
    """
    Process a single item from the CSV and generate model response
    """
    try:
        image_path = os.path.join(image_dir, item['image_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert("RGB")
        prompt = item['prompt']
        answer = item['answer']
        
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
        
        print(f"Processing: {item['image_name']}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Print total wrapped prompt for first item
        if is_first_item:
            print(f"\n=== TOTAL WRAPPED PROMPT (First Item) ===")
            print(f"Full wrapped prompt: {repr(query)}")
            print(f"Prompt length: {len(query)} characters")
            print("=" * 80)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Process inputs using Qwen2-VL processor
            inputs = processor(
                text=[query], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
            # Generate model response (using max_new_tokens)
            generate_ids = model.generate(
                **inputs, 
                do_sample=True, 
                max_new_tokens=256,
                temperature=0.8, 
                top_p=0.9
            )
            model_response = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
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
    print("=== Model Neutral Response Generation (Qwen2-VL) ===")
    print(f"CSV File: {args.test_csv}")
    print(f"Image Directory: {args.image_dir}")

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

    print('[Initialization Finished]\n')
    
    # Inspect chat template
    print("=== Chat Template Inspection ===")
    print("Chat template:")
    print(tokenizer.chat_template)
    
    print("\n=== Example Usage ===")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None,},
                {"type": "text", "text": "What anomaly is present in this XRD pattern?"},
            ],
        }
    ]
    
    print("Input messages:")
    print(messages)
    
    query = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("\nFormatted query:")
    print(repr(query))
    print("=" * 80)

    # Read CSV file
    print(f"Reading CSV file: {args.test_csv}")
    try:
        df = pd.read_csv(args.test_csv).head(10) # .head(10) for testing
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
                item, model, tokenizer, processor, args.image_dir, idx == 0
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