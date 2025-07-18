import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
import argparse
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model_name = 'Qwen/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('[Initialization Finished]\n')

# load datasets
data_img_path = "./results/adv_img_attr/qwen_jb"
attack_types = ['constrain_16', 'constrain_32', 'constrain_64', 'unconstrain'] # 

for attack_type in attack_types:
    data_query = pd.read_csv('./results/queries/qwen_jb_{}.csv'.format(attack_type))['Query'].tolist()

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
    # assert len(data_query) == len(data_adv_img)
    # empty_image = Image.new("RGB", (224, 224), (0, 0, 0))

    layers = [14, 20, 28] # for 7B model Qwen
    diff_attr_by_wrapper = {}
    for index, (query, img, attr, mask) in enumerate(zip(data_query, data_adv_img, data_adv_img_attr, data_adv_img_mask)):
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None,},
                        {"type": "text", "text": ''},
                    ],
                }
            ]
        
        text_prompt_template = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        
        with torch.no_grad(), torch.cuda.amp.autocast(): 
            inputs = processor(text=[text_prompt_template]*2, images=[img, mask], padding=True, return_tensors="pt").to(model.device)
            output = model(**inputs, output_hidden_states=True)

        diff_attr_activations = {}
        for layer in layers:
            hidden_states = output.hidden_states[layer].detach().cpu()
            diff_attr_activations[layer] = hidden_states[0, -1]-hidden_states[1, -1]
        
        diff_attr_by_wrapper[index] = {layer: [] for layer in layers}
        for layer in layers:
            diff_attr_by_wrapper[index][layer].append(diff_attr_activations[layer])

    variable_element = 'visual_jail'
    save_path = "./activations/qwen/jb"
    torch.save(diff_attr_by_wrapper, os.path.join(save_path, f"jb_diff_attr_activations_{variable_element}_{attack_type}.pt"))