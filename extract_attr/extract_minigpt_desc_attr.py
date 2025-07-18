import os
import sys
sys.path.append(os.path.abspath(os.curdir))
from image_attr import ImageAttr_MINIGPT
import torch
import pandas as pd
import json
import random
import os
from PIL import Image
import numpy as np
from minigpt_utils import prompt_wrapper

# Load your prompts and correct answers with image names
data = pd.read_csv("./datasets/harmful_corpus/advbench/train.csv")
# Create a dictionary that maps each image to its prompt and answer
image_to_data = {}
for _, row in data.iterrows():
    image_to_data[row['image_name']] = (row['prompt'], row['correct_answer'])

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images[filename] = img
    return images

data_path = "./datasets/adv_img_jb/minigpt"
attack_types = ['constrain_16']
ImgAttr = ImageAttr_MINIGPT.from_pretrained()
top_k = 15

for attack_type in attack_types:
    # Load images and maintain their filenames
    images_dict = load_images_from_folder("{}/{}".format(data_path, attack_type))
    data_query = []
    save_path = "./results/adv_img_attr/minigpt_desc/{}".format(attack_type)
    os.makedirs(save_path, exist_ok=True)
    
    # Process images in the order they appear in the CSV
    for index, row in data.iterrows():
        image_name = row['image_name']
        if image_name not in images_dict:
            print(f"Warning: Image {image_name} not found in folder. Skipping.")
            continue
        
        orig_query, correct_answer = image_to_data[image_name]
        mcq_instruction = "\nIMPORTANT: Please do not give any explanation. Answer with the option's letter from the given choices directly."
        query = orig_query + mcq_instruction
        print(f"\nProcessing image: {image_name}")
        print(f"Prompt: {orig_query}")
        print(f"Correct answer: {correct_answer}")
        
        ImgAttr.init_image_query(images_dict[image_name], query, num_ablations=32, ablation_keep_prob=0.20)
        ImgAttr.partitioner.set_p(56)
        
        # Specify the correct answer before any access to response or attribution
        ImgAttr.specify_output(correct_answer)
        results = ImgAttr.get_attributions(as_dataframe=True, top_k=top_k, verbose=True)
        
        
        patch_index = ImgAttr.partitioner.patch_index
        attr_indices = []
        for _, (score, source, patch) in enumerate(zip(results['Score'], results['Source'], results['Patch'])):
            if score > 1e-6:
                attr_indices.append(patch[0]+patch[1]*patch_index)
        
        # Print the scores of the top 15 features
        print("Scores of the top 15 features:")
        for score in results['Score']:
            print(score)
        
        # Print the length of attr_indices
        print(f"Number of attributed indices: {len(attr_indices)}")
        
        if len(attr_indices) > 0:
            # Save original image
            img = ImgAttr.partitioner.get_image()
            img.save('{}/img{}.bmp'.format(save_path, index))
            
            # save combined attribution visualization
            img = ImgAttr.partitioner.visualize_attr(np.array(attr_indices))
            img.save('{}/img{}_attr.bmp'.format(save_path, index))
            
            # Generate and save combined mask visualization
            img = ImgAttr.partitioner.visualize_attr(np.array(attr_indices), flip=True)
            img.save('{}/img{}_mask.bmp'.format(save_path, index))
            
            # Print confirmation of saved images
            print(f"Saved original image, attribution, and mask for image {image_name} at index {index}.")
            
            data_query.append({
                'image_name': image_name,
                'prompt': query,
                'correct_answer': correct_answer,
                'image_index': index
            })
    
    # Save queries and correct answers
    os.makedirs('./results/queries', exist_ok=True)
    data = pd.DataFrame(data_query)
    data.to_csv('./results/queries/minigpt_desc_{}.csv'.format(attack_type), index=False) 