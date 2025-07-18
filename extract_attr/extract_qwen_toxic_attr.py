import os
import sys
sys.path.append(os.path.abspath(os.curdir))
from image_attr import ImageAttr_QWEN
import torch
import pandas as pd
import json
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
import csv

file = open("./datasets/harmful_corpus/manual_harmful_instructions.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
harmful_behavior = []
for i in range(len(data)):
    harmful_behavior.append(data[i][0])

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None and int(filename.split('_')[2])<=15:
                images.append(img)
    return images

data_path = "./datasets/adv_img_toxic/qwen"
attack_types = ['constrain_16', 'constrain_32', 'constrain_64', 'unconstrain']

top_k = 15
ImgAttr = ImageAttr_QWEN.from_pretrained()
for attack_type  in attack_types:

    image_list = load_images_from_folder("{}/{}".format(data_path, attack_type))    
    data_query = []
    save_path = "./results/adv_img_attr/qwen_toxic/{}".format(attack_type)
    for index, image in enumerate(image_list):
        for i in range(3):
            random_query = random.randint(0, len(harmful_behavior)-1)
            query = harmful_behavior[random_query]

            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None,},
                            {"type": "text", "text": query},
                        ],
                    }
                ]
            text_prompt_template = ImgAttr.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            
            ImgAttr.init_image_query(image, text_prompt_template, num_ablations=32, ablation_keep_prob=0.50) 
            ImgAttr.partitioner.set_p(56)
 
            ImgAttr.specify_output('Sure, <|im_end|>')
            results = ImgAttr.get_attributions(as_dataframe=True, top_k=top_k, verbose=True)

            patch_index = ImgAttr.partitioner.patch_index
            attr_index = []; scores = []; sources = []
            for _, (score, source, patch) in enumerate(zip(results['Score'], results['Source'], results['Patch'])):
                # print("Index: {} | Trail: {} | ".format(index, i), score, patch)
                if score>-1e-4:
                    attr_index.append(patch[0]+patch[1]*patch_index)
            
            if len(attr_index) > 0:
                img = ImgAttr.partitioner.visualize_attr(np.array(attr_index))
                img.save('{}/img{}_attr.bmp'.format(save_path, index*3+i))

                img = ImgAttr.partitioner.get_image()
                img.save('{}/img{}.bmp'.format(save_path, index*3+i))

                img = ImgAttr.partitioner.visualize_attr(np.array(attr_index), flip=True)
                img.save('{}/img{}_mask.bmp'.format(save_path, index*3+i))
            
                data_query.append(query)
            
    header = ['Query']
    data = pd.DataFrame(data_query, columns=header)
    data.to_csv('./results/queries/qwen_toxic_{}.csv'.format(attack_type), index=False)
