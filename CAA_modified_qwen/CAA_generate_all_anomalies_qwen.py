import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default="train_generate.csv", help="CSV with columns: image_name,prompt,answer")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/loop_scattering", help="Directory containing images")
    parser.add_argument("--correct_anomaly", type=str, default="Loop scattering", help="Correct anomaly type")
    parser.add_argument("--wrong_anomalies", type=str, nargs='+', 
                       default=["Diffuse scattering", "Ice ring", "No distinct anomalies", 
                               "Non-uniform detector", "Artifact", "Background ring", "Strong background"],
                       help="Wrong anomaly types")
    return parser.parse_args()

# Dictionary for token positions for each anomaly type
ANOMALY_TOKEN_POSITIONS = {
    "Loop scattering": 2,      # 3rd last token
    "Diffuse scattering": 3,   # 5th last token
    "Ice ring": 2,             # 2nd last token
    "No distinct anomalies": 3, # 5th last token
    "Non-uniform detector": 4, # 5th last token
    "Artifact": 1,             # 2nd last token
    "Background ring": 2,      # 2nd last token
    "Strong background": 2     # 3rd last token
}

def get_token_position(anomaly_type):
    """Get token position for a given anomaly type"""
    return ANOMALY_TOKEN_POSITIONS.get(anomaly_type, 2)  # Default to 2nd last if not found

def process_anomaly_pair(model, tokenizer, processor, anomaly_type, image, gpu_id, row):
    """Process a single anomaly type and extract activations"""
    # Use the prompt from the CSV row
    base_prompt = row['prompt']
    
    # Create prompt using Qwen2-VL chat template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None,},
                {"type": "text", "text": base_prompt},
            ],
        }
    ]
    query = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Append the anomaly type after the chat template wrapping
    query_with_anomaly = query + ' ' + anomaly_type
    
    # Print the query with anomaly
    print(f"Query with anomaly ({anomaly_type}): {repr(query_with_anomaly)}")
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Process inputs using Qwen2-VL processor
        inputs = processor(
            text=[query_with_anomaly], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        # Get input tokens for analysis
        input_ids = inputs["input_ids"]
        
        # Get token position based on anomaly type
        token_offset = get_token_position(anomaly_type)
        answer_pos = input_ids.shape[1] - token_offset
        
        # Forward pass to get hidden states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        
        # Get tokens from input_ids
        tqdm.write("\n" + "="*50)
        tqdm.write(f"Image: {row['image_name']}")
        tqdm.write(f"Anomaly type: {anomaly_type}")
        tqdm.write(f"All input tokens: {input_ids[0].tolist()}")
        
        # Print the last 7 tokens
        for i in range(7):
            if i < input_ids.shape[1]:
                token_id = input_ids[0, -(i+1)].item()
                token_text = tokenizer.decode([token_id])
                tqdm.write(f"Last {i+1} token ID: {token_id}, Last {i+1} token text: '{token_text}'")
        
        tqdm.write(f"Input sequence length: {input_ids.shape[1]}")
        tqdm.write(f"Position we're extracting from ({anomaly_type}): {answer_pos}")
        tqdm.write(f"Anomaly option we added: {anomaly_type}")
        
        # Print which token activation is being extracted
        target_token_id = input_ids[0, -token_offset].item()
        target_token_text = tokenizer.decode([target_token_id])
        tqdm.write(f"EXTRACTING ACTIVATION FROM: Position {answer_pos}, Token ID {target_token_id}, Token text '{target_token_text}'")
        
        # Extract activations for all layers
        activations = {}
        for layer in [10, 14, 16, 20]:
            act = hidden_states[layer][0, answer_pos, :].detach().cpu()
            tqdm.write(f"Layer {layer} {anomaly_type} activation: {act}")
            activations[layer] = act
        
        tqdm.write("="*50 + "\n")
        
        return activations

def main():
    args = parse_args()
    
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

    num_layers = len(model.model.layers)
    print(f"Total number of layers in the model: {num_layers}")

    df = pd.read_csv(args.csv_path).head(250) # Use .head(5) to test
    num_data_points = len(df)
    print(f"Number of data points: {num_data_points}")
    print(f"Correct anomaly: {args.correct_anomaly}")
    print(f"Wrong anomalies: {args.wrong_anomalies}")

    # Initialize reference_acts for each wrong anomaly pair
    reference_acts_pairs = {}
    for wrong_anomaly in args.wrong_anomalies:
        reference_acts_pairs[wrong_anomaly] = {}
        for idx in range(num_data_points):
            reference_acts_pairs[wrong_anomaly][idx] = {
                10: [],
                14: [],
                16: [],
                20: []
            }

    for idx, row in tqdm(df.iterrows(), total=num_data_points):
        # Use the correct anomaly type for image directory
        image_path = os.path.join(args.image_dir, row['image_name'])
        image = Image.open(image_path).convert("RGB")
        prompt = row['prompt']
        answer = row['answer']

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

        print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

        # Get model response
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
                max_new_tokens=256,
                temperature=0.8, 
                top_p=0.9
            )
            model_response = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
        print(f"Model response for image {row['image_name']}: {model_response}")

        # Process correct anomaly first
        print(f"\n=== PROCESSING CORRECT ANOMALY: {args.correct_anomaly} ===")
        correct_activations = process_anomaly_pair(
            model, tokenizer, processor, args.correct_anomaly, image, args.gpu_id, row
        )

        # Process all wrong anomalies
        all_activations = {args.correct_anomaly: correct_activations}
        
        for wrong_anomaly in args.wrong_anomalies:
            print(f"\n=== PROCESSING WRONG ANOMALY: {wrong_anomaly} ===")
            wrong_activations = process_anomaly_pair(
                model, tokenizer, processor, wrong_anomaly, image, args.gpu_id, row
            )
            all_activations[wrong_anomaly] = wrong_activations

        # Create pairs and compute difference vectors
        print(f"\n=== COMPUTING DIFFERENCE VECTORS ===")
        for wrong_anomaly in args.wrong_anomalies:
            print(f"\n--- {args.correct_anomaly} vs {wrong_anomaly} ---")
            
            for layer in [10, 14, 16, 20]:
                correct_act = all_activations[args.correct_anomaly][layer]
                wrong_act = all_activations[wrong_anomaly][layer]
                diff_vector = correct_act - wrong_act
                
                tqdm.write(f"Layer {layer} difference vector ({args.correct_anomaly} - {wrong_anomaly}): {diff_vector}")
                reference_acts_pairs[wrong_anomaly][idx][layer].append(diff_vector)

    # Save separate .pt files for each pair
    os.makedirs(".", exist_ok=True)
    
    for wrong_anomaly in args.wrong_anomalies:
        # Create filename with wrong answer name
        wrong_anomaly_clean = wrong_anomaly.lower().replace(' ', '_')
        filename = f"./nu_detector_act_diff_10-20_{num_data_points}_{wrong_anomaly_clean}.pt"
        
        # Save the data for this specific pair with exact same structure as CAA_simplified_full_1.py
        torch.save(reference_acts_pairs[wrong_anomaly], filename)
        print(f"Saved reference activations for {args.correct_anomaly} vs {wrong_anomaly} to {filename}")
    
    print(f"Saved {len(args.wrong_anomalies)} separate .pt files for all anomaly pairs")

if __name__ == "__main__":
    main() 