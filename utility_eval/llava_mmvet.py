import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import math
import json
import base64
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    parser.add_argument("--alpha", type=float, default=40.0, help="")
    parser.add_argument('--attack_type', default="constrain_16", 
                        choices=["constrain_16", "constrain_32", "constrain_64", "unconstrain"], 
                        type=str, help="")
    parser.add_argument('--feat_type', default="diff_attr", choices=["diff_attr"], type=str, help="")
    parser.add_argument('--variable_element', default="visual_jail", choices=["visual_jail", "text_jail"], help="")
    parser.add_argument("--threshold", type=float, default=0.5, help="")
    parser.add_argument("--steer_layer", type=int, default=20, help="")
    parser.add_argument('--eval', type=str, choices=['test', 'val'], default='val')
    parser.add_argument('--steer_vector', type=str, choices=['toxic', 'jb'], default='jb')

    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options", nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

args = parse_args()
print("Experimental Args ===", args)


print('[Start Initialization]\n')
model_name = 'llava-hf/llava-1.5-13b-hf'
processor = LlavaProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto')
print('[Initialization Finished]\n')

activations_by_wrapper = torch.load("./activations/llava/reference/reference_activations.pt")
reference_activations = []
# Concatenate activations
for wrapper, activations_per_layer in activations_by_wrapper.items():
    if args.steer_layer in activations_per_layer:
        reference_activations.extend(activations_per_layer[args.steer_layer])
reference_activations = np.array(reference_activations)
reference_activations = np.mean(reference_activations, axis=0)
norm_reference_activations = reference_activations/np.linalg.norm(reference_activations)
reference_activations = torch.from_numpy(reference_activations).cuda()
norm_reference_activations = torch.from_numpy(norm_reference_activations).cuda()

if args.steer_vector=='jb':
    activations_by_wrapper = torch.load("./activations/llava/jb/jb_{}_activations_{}_{}.pt".format(
                                args.feat_type, args.variable_element, args.attack_type)
                            )
elif args.steer_vector=='toxic':
    activations_by_wrapper = torch.load("./activations/llava/toxic/all_{}_activations_{}_{}.pt".format(
                                args.feat_type, args.variable_element, args.attack_type)
                            )
all_activations = []
for wrapper, activations_per_layer in activations_by_wrapper.items():
    if args.steer_layer in activations_per_layer:
        all_activations.extend(activations_per_layer[args.steer_layer])

if not all_activations:
    raise ValueError(f"No activations found for layer {args.steer_layer}")

all_activations = np.array(all_activations)
steer_activations = np.mean(all_activations, axis=0)
norm_steer_activations = steer_activations/np.linalg.norm(steer_activations)
steer_activations = torch.from_numpy(steer_activations).cuda()
norm_steer_activations = torch.from_numpy(norm_steer_activations).cuda()
random_steer_activations = torch.rand_like(steer_activations).cuda()

questions = json.load(open("./datasets/mm-vet/{}_mmvet.json".format(args.eval)))
print("Len of the questions: ",len(questions))
image_folder = "./datasets/mm-vet/images"

output_texts = []
for index, (key, line) in tqdm(enumerate(questions.items()), total=len(questions)):
    
    qs = line['question']; ans = line['answer']
    query = "USER: <image>\n{}\nASSISTANT:".format(qs)
    image = Image.open(os.path.join(image_folder, line["imagename"])).convert('RGB')

    with torch.no_grad(), torch.cuda.amp.autocast():    
        inputs = processor(
                    images=image, text=query, return_tensors="pt"
                ).to(device="cuda", dtype=torch.float16)
        
        def create_custom_forward_hook(steer_vector, reference_vector, steer_type, alpha):
            def custom_forward_hook(module, input, output):
                R_feat = output[0][:, -1, :]
                norm_steer_vector = torch.norm(steer_vector, p=2)
                unit_steer_vector = steer_vector / norm_steer_vector
                if steer_type=='linear':
                    R_feat += unit_steer_vector*alpha
                elif steer_type=='projection':
                    project_feat = torch.matmul(R_feat-reference_vector, steer_vector)/torch.norm(R_feat-reference_vector, p=2)/torch.norm(steer_vector, p=2)            
                    clip_proj = torch.clamp(project_feat, min=0, max=1)
                    coefficient = clip_proj*torch.norm(R_feat, p=2)*alpha
                    R_feat -= coefficient*unit_steer_vector
                elif steer_type!='no_steer':
                    raise NotImplementedError
                output[0][:, -1, :] = R_feat
                return output
            return custom_forward_hook
        
        steered_outputs = [qs]
        steer_types = ['projection',  'no_steer']
        alphas = [args.alpha, 0]
        for i, (steer_type, alpha) in enumerate(zip(steer_types, alphas)):
            custom_hook = create_custom_forward_hook(steer_activations, reference_activations, steer_type, alpha)
            hook = model.language_model.base_model.layers[args.steer_layer-1].register_forward_hook(custom_hook)
            inputs = processor(text=query, images=image, return_tensors="pt").to("cuda", torch.float16)
            generate_ids = model.generate(**inputs, do_sample=True, max_length=1024, temperature=0.2, top_p=0.9,)
            steered_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            steered_outputs.append(steered_text)
            hook.remove()
            # print("INDEX", index, " | ", steer_type, alpha, " | Generated Texts: ", steered_text)
    
        steered_outputs.append(ans)
        output_texts.append(steered_outputs)

colums = steer_types
# for index, colum in enumerate(colums):
#     if colum=='projection':
#         colums[index] = "{}_{}".format(steer_types[index], alphas[index])
colums.insert(0, 'Questions')
colums.append('Answers')
data = pd.DataFrame(output_texts, columns=colums)
data.to_csv('./results/utility/llava/mmvet/llava_{}_{}_{}_{}_alpha_{}.csv'.format(
            args.steer_vector, args.eval, args.feat_type, args.attack_type, int(args.alpha)), index=False)
