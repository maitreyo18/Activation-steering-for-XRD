import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import math
import base64
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from minigpt_utils import visual_attacker, prompt_wrapper, generator
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    parser.add_argument("--alpha", type=float, default=7.0, help="")
    parser.add_argument('--attack_type', type=str, default='constrain_16')
    parser.add_argument('--feat_type', default="diff_attr", choices=["diff_attr"], type=str, help="")
    parser.add_argument('--variable_element', default="visual_jail", choices=["visual_jail", "text_jail"], help="")
    parser.add_argument("--threshold", type=float, default=0.5, help="")
    parser.add_argument("--steer_layer", type=int, default=20, help="")
    parser.add_argument('--steer_vector', type=str, default='desc')
    parser.add_argument('--low_resource', type=str, default='False', help='use 8-bit quantization when set to True')

    parser.add_argument('--eval', type=str, default='test')
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

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def load_image_from_base64(image):
    img = Image.open(BytesIO(base64.b64decode(image)))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def main():
    args = parse_args()
    print("Experimental Args ===", args)

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    if args.low_resource.lower() == 'true':
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id), dtype=torch.float16)
    else:
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    print('[Initialization Finished]\n')

    # Load reference activations
    activations_by_wrapper = torch.load("./activations/minigpt/reference/reference_activations.pt")
    reference_activations = []
    for wrapper, activations_per_layer in activations_by_wrapper.items():
        if args.steer_layer in activations_per_layer:
            reference_activations.extend(activations_per_layer[args.steer_layer])
    reference_activations = np.array(reference_activations)
    reference_activations = np.mean(reference_activations, axis=0)
    norm_reference_activations = reference_activations/np.linalg.norm(reference_activations)
    reference_activations = torch.from_numpy(reference_activations).cuda()
    norm_reference_activations = torch.from_numpy(norm_reference_activations).cuda()

    # Load steering activations
    activations_by_wrapper = torch.load("./activations/minigpt/desc/desc_activations_{}.pt".format(args.attack_type))
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

    num_rounds = 1
    all_options = ['A', 'B', 'C', 'D']
    text_gen = generator.Generator(model=model)
    questions = pd.read_table(os.path.expanduser("./datasets/MMBench/{}_mmbench.tsv".format(args.eval)))
    print("Len of the questions: ",len(questions))

    output_texts = []
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            #hint = row['hint']
            image = load_image_from_base64(row['image'])
            #if not is_none(hint):
             #   question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            query = qs + "\nIMPORTANT: Please do not give any explanation. Answer with the option's letter from the given choices directly."

            with torch.no_grad(), torch.cuda.amp.autocast():    
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
                            R_feat += coefficient*unit_steer_vector
                        elif steer_type!='no_steer':
                            raise NotImplementedError
                        output[0][:, -1, :] = R_feat
                        return output
                    return custom_forward_hook
                
                steered_outputs = [qs]
                steer_types = ['projection', 'no_steer']
                alphas = [args.alpha, 0]
                for i, (steer_type, alpha) in enumerate(zip(steer_types, alphas)):
                    custom_hook = create_custom_forward_hook(steer_activations, reference_activations, steer_type, alpha)
                    hook = model.llama_model.base_model.layers[args.steer_layer-1].register_forward_hook(custom_hook)
                    
                    img_prompt = [processor(image).unsqueeze(0).to('cuda')]
                    prompt_wrap = prompt_wrapper.Prompt(model=model, text_prompts=[query], img_prompts=[img_prompt])
                    steered_text, _ = text_gen.generate(prompt_wrap)
                    
                    steered_outputs.append(steered_text)
                    hook.remove()

                steered_outputs.append(row['answer'])
            output_texts.append(steered_outputs)

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    colums = steer_types
    colums.insert(0, 'Questions')
    colums.append('Answers')
    data = pd.DataFrame(output_texts, columns=colums)
    os.makedirs('./results/utility/minigpt/mmbench', exist_ok=True)
    data.to_csv('./results/utility/minigpt/mmbench/minigpt_{}_{}_{}_{}_alpha_{}.csv'.format(
                args.steer_vector, args.eval, args.feat_type, args.attack_type, int(args.alpha)), index=False)

if __name__ == "__main__":
    main() 