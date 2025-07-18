import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
import json
import random
import os
from PIL import Image
import numpy as np
from minigpt_utils import prompt_wrapper, generator
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def load_images_from_folder(folder):
    images = []
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                image_names.append(filename)
    return images, image_names

# Initialize model and processor
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

data_path = "./datasets/adv_img_jb/minigpt"
data_img_path = "./results/adv_img_attr/minigpt_desc"
attack_types = ['constrain_16']

for attack_type in attack_types:
    # Load images and queries
    image_list, image_names = load_images_from_folder("{}/{}".format(data_path, attack_type))
    data_query = pd.read_csv('./results/queries/minigpt_desc_{}.csv'.format(attack_type))
    
    # Create a dictionary for quick lookup
    image_to_data = dict(zip(data_query['image_name'], zip(data_query['prompt'], data_query['correct_answer'])))
    
    # Initialize activations dictionary
    diff_attr_by_wrapper = {}
    layers = [20, 40]  # 20th and 40th layer for 13B model
    
    # Load pre-computed attribute images
    data_adv_img = []
    data_adv_img_attr = []
    data_adv_img_mask = []
    data_queries = []
    data_answers = []
    
    # Process each row in the CSV file
    for _, row in data_query.iterrows():
        index = row['image_index']  # Use the actual image index from CSV
        image_name = row['image_name']
        try:
            # Load original image and its combined attribution/mask
            data_adv_img.append(Image.open("{}/{}/img{}.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_attr.append(Image.open("{}/{}/img{}_attr.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_mask.append(Image.open("{}/{}/img{}_mask.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_queries.append(row['prompt'])
            data_answers.append(row['correct_answer'])
        except Exception as e:
            print(f"Warning: Could not load images for {image_name} (index: {index}): {str(e)}")
            continue
    
    print(f"Loaded {len(data_adv_img)} image sets")
    
    # Process each image and its corresponding query
    for i, (query, img, attr, mask) in enumerate(zip(data_queries, data_adv_img, data_adv_img_attr, data_adv_img_mask)):
        # Get the corresponding image_name and image_index from the CSV
        row = data_query.iloc[i]
        image_name = row['image_name']
        image_index = row['image_index']
        print(f"Processing image: {image_name} (index: {image_index})")
        
        # Get original image activations
        with torch.no_grad(), torch.cuda.amp.autocast():
            img = [processor(img).unsqueeze(0).to('cuda')]
            attr = [processor(attr).unsqueeze(0).to('cuda')]
            mask = [processor(mask).unsqueeze(0).to('cuda')]

            prompt_wrap = prompt_wrapper.Prompt(model=model, 
                                            text_prompts=[query]*2,
                                            img_prompts=[img, mask])
            
            output_img = model.llama_model(inputs_embeds=prompt_wrap.context_embs[0], output_hidden_states=True)
            output_mask = model.llama_model(inputs_embeds=prompt_wrap.context_embs[1], output_hidden_states=True)

        # Compute difference in activations
        diff_attr_activations = {}
        for layer in layers:
            hidden_img = output_img.hidden_states[layer].detach().cpu()
            hidden_mask = output_mask.hidden_states[layer].detach().cpu()
            diff_attr_activations[layer] = hidden_img[0, -1] - hidden_mask[0, -1]
        
        # Store activations
        diff_attr_by_wrapper[i] = {layer: [] for layer in layers}
        for layer in layers:
            diff_attr_by_wrapper[i][layer].append(diff_attr_activations[layer])
    
    # Save activations
    os.makedirs('./activations/minigpt/desc', exist_ok=True)
    torch.save(diff_attr_by_wrapper, './activations/minigpt/desc/desc_activations_{}.pt'.format(attack_type))