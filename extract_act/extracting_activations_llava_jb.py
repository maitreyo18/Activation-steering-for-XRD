import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
import argparse
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration

model_name = 'llava-hf/llava-1.5-13b-hf'
processor = LlavaProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto')

data_img_path = "./results/adv_img_attr/llava_jb"
attack_types = ['constrain_16', 'constrain_32', 'constrain_64', 'unconstrain'] # 
for attack_type in attack_types:
    data_query = pd.read_csv('./results/queries/llava_jb_{}.csv'.format(attack_type))['Query'].tolist()

    data_adv_img = []
    data_adv_img_attr = []
    data_adv_img_mask = []
    for index in range(16*3):
        try:
            data_adv_img.append(Image.open("{}/{}/img{}.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_attr.append(Image.open("{}/{}/img{}_attr.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_mask.append(Image.open("{}/{}/img{}_mask.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            print(Image.open("{}/{}/img{}.bmp".format(data_img_path, attack_type, index)).convert("RGB").shape)
        except:
            continue
    print(len(data_query), len(data_adv_img))
    assert len(data_query) == len(data_adv_img)
    # empty_image = Image.new("RGB", (224, 224), (0, 0, 0))

    layers = [20, 40] # for 13B model
    img_by_wrapper = {}; img_mask_by_wrapper = {}
    attr_by_wrapper = {}; diff_attr_by_wrapper = {}

    for index, (query, img, attr, mask) in enumerate(zip(data_query, data_adv_img, data_adv_img_attr, data_adv_img_mask)):
        query = 'USER: <image>\nASSISTANT:'
        with torch.no_grad(), torch.cuda.amp.autocast():  
            inputs = processor(text=[query]*2, images=[img, mask,], return_tensors="pt").to("cuda", torch.float16)
            output = model(**inputs, output_hidden_states=True)

        diff_attr_activations = {}
        for layer in layers:
            hidden_states = output.hidden_states[layer].detach().cpu()
            diff_attr_activations[layer] = hidden_states[0, -1]-hidden_states[1, -1]
        
        diff_attr_by_wrapper[index] = {layer: [] for layer in layers}
        for layer in layers:
            diff_attr_by_wrapper[index][layer].append(diff_attr_activations[layer])

    variable_element = 'visual_jail'
    save_path = "./activations/llava/jb"
    torch.save(diff_attr_by_wrapper, os.path.join(save_path, f"jb_diff_attr_activations_{variable_element}_{attack_type}.pt"))