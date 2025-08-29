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
    parser.add_argument("--csv_path", type=str, default="train_generate.csv", help="CSV with columns: image_name,prompt,answer")
    parser.add_argument("--image_dir", type=str, default="../datasets/XRD_data/one_correct/loop_scattering", help="Directory containing images")
    parser.add_argument("--correct_anomaly", type=str, default="Loop scattering", help="Correct anomaly type")
    parser.add_argument("--wrong_anomalies", type=str, nargs='+', 
                       default=["Diffuse scattering", "Ice ring", "No distinct anomalies", 
                               "Non-uniform detector", "Artifact", "Background ring", "Strong background"],
                       help="Wrong anomaly types")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

# Dictionary for token positions for each anomaly type
ANOMALY_TOKEN_POSITIONS = {
    "Loop scattering": 3,      # 3rd last token
    "Diffuse scattering": 5,   # 5th last token
    "Ice ring": 2,             # 2nd last token
    "No distinct anomalies": 5, # 5th last token
    "Non-uniform detector": 5, # 5th last token
    "Artifact": 2,             # 2nd last token
    "Background ring": 2,      # 2nd last token
    "Strong background": 3     # 3rd last token
}

def get_token_position(anomaly_type):
    """Get token position for a given anomaly type"""
    return ANOMALY_TOKEN_POSITIONS.get(anomaly_type, 2)  # Default to 2nd last if not found

def process_anomaly_pair(prompt_wrap, model, tokenizer, anomaly_type, img_tensor, gpu_id, row):
    """Process a single anomaly type and extract activations"""
    anomaly_prompt = prompt_wrap.text_prompts[0] + ' ' + anomaly_type
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt_wrap_anomaly = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[anomaly_prompt],
            img_prompts=[[img_tensor]],
            device=f'cuda:{gpu_id}'
        )
        context_embs = prompt_wrap_anomaly.context_embs[0]
        outputs = model.llama_model(inputs_embeds=context_embs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        
        # Get token position based on anomaly type
        token_offset = get_token_position(anomaly_type)
        answer_pos = context_embs.shape[1] - token_offset
        
        # Get tokens from prompt_wrapper
        tqdm.write("\n" + "="*50)
        tqdm.write(f"Image: {row['image_name']}")
        tqdm.write(f"Anomaly type: {anomaly_type}")
        tqdm.write(f"All tokens: {prompt_wrap_anomaly.input_tokens}")
        
        # Print the last 5 tokens from the last tensor
        last_tensor = prompt_wrap_anomaly.input_tokens[-1]
        for i in range(7):
            if i < last_tensor.shape[1]:
                token_id = last_tensor[0, -(i+1)].item()
                token_text = tokenizer.decode([token_id])
                tqdm.write(f"Last {i+1} token ID: {token_id}, Last {i+1} token text: '{token_text}'")
        
        tqdm.write(f"Context embeddings length: {context_embs.shape[1]}")
        tqdm.write(f"Position we're extracting from ({anomaly_type}): {answer_pos}")
        tqdm.write(f"Anomaly option we added: {anomaly_type}")
        
        # Print which token activation is being extracted
        target_token_id = last_tensor[0, -token_offset].item()
        target_token_text = tokenizer.decode([target_token_id])
        tqdm.write(f"EXTRACTING ACTIVATION FROM: Position {answer_pos}, Token ID {target_token_id}, Token text '{target_token_text}'")
        
        # Extract activations for all layers
        activations = {}
        for layer in [12, 16, 20]:
            act = hidden_states[layer][0, answer_pos, :].detach().cpu()
            tqdm.write(f"Layer {layer} {anomaly_type} activation: {act}")
            activations[layer] = act
        
        tqdm.write("="*50 + "\n")
        
        return activations

def main():
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
    generator_obj = generator.Generator(
        model=model,
        num_beams=1
    )
    prefix = prompt_wrapper.minigpt4_chatbot_prompt

    num_layers = len(model.llama_model.base_model.layers)
    print(f"Total number of layers in the model: {num_layers}")

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    df = pd.read_csv(args.csv_path) # Use .head(5) to test
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
                12: [],
                16: [],
                20: []
            }

    for idx, row in tqdm(df.iterrows(), total=num_data_points):
        # Use the correct anomaly type for image directory
        image_path = os.path.join(args.image_dir, row['image_name'])
        image = Image.open(image_path).convert("RGB")
        img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
        prompt = row['prompt']
        answer = row['answer']

        # Wrap in template
        query = prefix % prompt

        print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

        # Get model response
        prompt_wrap = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[query],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        model_response, _ = generator_obj.generate(prompt_wrap)
        print(f"Model response for image {row['image_name']}: {model_response}")

        # Process correct anomaly first
        print(f"\n=== PROCESSING CORRECT ANOMALY: {args.correct_anomaly} ===")
        correct_activations = process_anomaly_pair(
            prompt_wrap, model, tokenizer, args.correct_anomaly, img_tensor, args.gpu_id, row
        )

        # Process all wrong anomalies
        all_activations = {args.correct_anomaly: correct_activations}
        
        for wrong_anomaly in args.wrong_anomalies:
            print(f"\n=== PROCESSING WRONG ANOMALY: {wrong_anomaly} ===")
            wrong_activations = process_anomaly_pair(
                prompt_wrap, model, tokenizer, wrong_anomaly, img_tensor, args.gpu_id, row
            )
            all_activations[wrong_anomaly] = wrong_activations

        # Create pairs and compute difference vectors
        print(f"\n=== COMPUTING DIFFERENCE VECTORS ===")
        for wrong_anomaly in args.wrong_anomalies:
            print(f"\n--- {args.correct_anomaly} vs {wrong_anomaly} ---")
            
            for layer in [12, 16, 20]:
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
        filename = f"./diffuse_scattering_act_diff_12-20_{num_data_points}_{wrong_anomaly_clean}.pt"
        
        # Save the data for this specific pair with exact same structure as CAA_simplified_full_1.py
        torch.save(reference_acts_pairs[wrong_anomaly], filename)
        print(f"Saved reference activations for {args.correct_anomaly} vs {wrong_anomaly} to {filename}")
    
    print(f"Saved {len(args.wrong_anomalies)} separate .pt files for all anomaly pairs")

if __name__ == "__main__":
    main() 