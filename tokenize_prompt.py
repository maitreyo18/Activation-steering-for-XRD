import os
import pandas as pd
import torch
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
from minigpt4.models import MiniGPT4
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def load_model():
    """Load MiniGPT-4 model and visual processor"""
    # Initialize config with correct syntax
    args = argparse.Namespace(
        cfg_path="eval_configs/minigpt4_eval.yaml",
        options=[]  # Add empty options list as required by Config
    )
    cfg = Config(args)
    
    # Get model config from the initialized Config object
    model_config = cfg.model_cfg
    
    # Load model
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    model.eval()

    # Load visual processor
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    
    return model, vis_processor

def process_image_prompt(model, vis_processor, image_path, prompt):
    """Process a single image-prompt pair"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Initialize chat with visual processor
    chat = Chat(model, vis_processor, device='cuda:0')
    chat_state = CONV_VISION.copy()
    img_list = []
    
    # Process image and prompt
    chat.upload_img(image, chat_state, img_list)
    chat.ask(prompt, chat_state)
    
    # Get response
    response = chat.answer(conv=chat_state, img_list=img_list)[0]
    return response

def main():
    # Load model
    print("Loading MiniGPT-4 model...")
    model, vis_processor = load_model()
    
    # Load tokenizer (same as ImageAttr_MINIGPT)
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load CSV
    csv_path = "datasets/harmful_corpus/advbench/train.csv"
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # MCQ instruction to append to each prompt
    mcq_instruction = "\nIMPORTANT: Please do not give any explanation. Answer with the option's letter from the given choices directly. C."
    
    # Process each image-prompt pair
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_name = row['image_name']
        prompt = row['prompt'] + mcq_instruction
        correct_answer = row['correct_answer']
        
        # Print tokens for the prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"\nPrompt for image {image_name}:")
        print(prompt)
        print("Token IDs and their corresponding text:")
        for token_id in tokens:
            token_text = tokenizer.decode([token_id])
            print(f"Token ID: {token_id:5d} | Text: {token_text}")
        
        # Construct image path
        image_path = os.path.join("datasets/adv_img_jb/minigpt/constrain_16", image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found, skipping...")
            continue
        
        try:
            # Get model response
            response = process_image_prompt(model, vis_processor, image_path, prompt)
            
            # Store results
            results.append({
                'image_name': image_name,
                'prompt': prompt,
                'correct_answer': correct_answer,
                'model_response': response
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("test_results/minigpt_xrd_responses.csv", index=False)
    print(f"Results saved to test_results/minigpt_xrd_responses.csv")

if __name__ == "__main__":
    main()  