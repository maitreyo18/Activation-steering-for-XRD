import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
import argparse
from minigpt_utils import visual_attacker, prompt_wrapper, generator

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

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

data_img_path = "./results/adv_img_attr/minigpt_jb"
attack_types = ['constrain_16', 'constrain_32', 'constrain_64', 'unconstrain']

for attack_type in attack_types:

    data_query = pd.read_csv('./results/queries/minigpt_jb_{}.csv'.format(attack_type))['Query'].tolist()

    data_adv_img = []
    data_adv_img_attr = []
    data_adv_img_mask = []
    for index in range(16*3):
        try:
            data_adv_img.append(Image.open("{}/{}/img{}.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_attr.append(Image.open("{}/{}/img{}_attr.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
            data_adv_img_mask.append(Image.open("{}/{}/img{}_mask.bmp".format(data_img_path, attack_type, index)).convert("RGB"))
        except:
            continue
    print(len(data_query), len(data_adv_img))
    assert len(data_query) == len(data_adv_img)
    # empty_image = Image.new("RGB", (224, 224), (0, 0, 0))
    # empty_image = [processor(empty_image).unsqueeze(0).to('cuda')]

    layers = [20, 40] # for 13B model
    diff_attr_by_wrapper = {}
    for index, (query, img, attr, mask) in enumerate(zip(data_query, data_adv_img, data_adv_img_attr, data_adv_img_mask)):
        prefix = prompt_wrapper.minigpt4_chatbot_prompt    
        query = prefix

        with torch.no_grad(), torch.cuda.amp.autocast():
            img = [processor(img).unsqueeze(0).to('cuda')]
            attr = [processor(attr).unsqueeze(0).to('cuda')]
            mask = [processor(mask).unsqueeze(0).to('cuda')]

            prompt_wrap = prompt_wrapper.Prompt(model=model, 
                                                text_prompts=[query]*2,
                                                img_prompts=[img, mask,])
            
            output_img = model.llama_model(inputs_embeds=prompt_wrap.context_embs[0], output_hidden_states=True)
            output_mask = model.llama_model(inputs_embeds=prompt_wrap.context_embs[1], output_hidden_states=True)

        img_activations = {}; img_mask_activations = {}
        attr_activations = {}; diff_attr_activations = {}
        for layer in layers:
            hidden_img = output_img.hidden_states[layer].detach().cpu()
            hidden_mask = output_mask.hidden_states[layer].detach().cpu()
            diff_attr_activations[layer] = hidden_img[0, -1] - hidden_mask[0, -1]
        
        diff_attr_by_wrapper[index] = {layer: [] for layer in layers}
        for layer in layers:
            diff_attr_by_wrapper[index][layer].append(diff_attr_activations[layer])

    variable_element = 'visual_jail'
    save_path = "./activations/minigpt/jb"
    torch.save(diff_attr_by_wrapper, os.path.join(save_path, f"jb_diff_attr_activations_{variable_element}_{attack_type}.pt"))
