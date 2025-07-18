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
import re

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 24], help="Layer(s) to extract activations from")
    parser.add_argument("--csv_path", type=str, default="train.csv", help="CSV with columns: image_name,prompt,correct_answer")
    parser.add_argument("--save_path", type=str, default="./activation_xrd_12-24.pt")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

args = parse_args()
cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model)
generator_obj = generator.Generator(model=model)
prefix = prompt_wrapper.minigpt4_chatbot_prompt
instruction = ""

num_layers = len(model.llama_model.base_model.layers)
print(f"Total number of layers in the model: {num_layers}")

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

df = pd.read_csv(args.csv_path).iloc[300:700].reset_index(drop=True)  # columns: image_name,prompt,correct_answer
num_data_points = len(df)  # Process all rows

reference_acts = {}
for idx in range(num_data_points * 3):  # Multiply by 3 for the three pairs per image
    reference_acts[idx] = {
        12: [],
        24: []
    }

results_for_csv = []

for idx, row in tqdm(df.iterrows(), total=num_data_points):
    if idx >= num_data_points:
        break
    image_path = os.path.join("../datasets/XRD_data/one_correct/background_ring", row['image_name'])
    image = Image.open(image_path).convert("RGB")
    img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
    prompt = row['prompt']
    correct_answer = row['correct_answer']

    # Wrap in template
    query = prefix % prompt

    # Print the final prompt
    print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

    # Extract options from the prompt
    # Find all options in the prompt (e.g., 'A. artifact', 'B. strong background', ...)
    option_pattern = r'([A-D]\.\s[^\n]+)'
    options_found = re.findall(option_pattern, query)
    options_dict = {opt[0]: opt for opt in options_found}  # {'A': 'A. artifact', ...}

    # Get the correct option (A, B, C, or D)
    correct_option = None
    for opt in ['A', 'B', 'C', 'D']:
        if correct_answer.strip().startswith(opt + '.'):
            correct_option = opt
            break
    if correct_option is None:
        raise ValueError(f"Invalid correct answer format: {correct_answer}")

    # Create wrong options based on correct option
    wrong_options = [o for o in ['A', 'B', 'C', 'D'] if o != correct_option]

    # Process correct answer first - use the full option (e.g., 'A. artifact')
    prompt_with_correct = query + ' ' + options_dict[correct_option]
    print(f"[DEBUG] Correct prompt: {prompt_with_correct}")
    tokens_correct = tokenizer(prompt_with_correct, return_tensors="pt")
    
    # Print all tokens for the first image+prompt pair only
    if idx == 0:
        print("\n=== First Image+Prompt Token Analysis ===")
        
        prompt_wrap_correct = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[prompt_with_correct],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        
        # Get the context embeddings after prompt wrapper processing
        context_embs = prompt_wrap_correct.context_embs[0]
        start_idx, end_idx = prompt_wrap_correct.get_image_token_indices()
        
        # Get the input tokens from the prompt wrapper
        if hasattr(prompt_wrap_correct, 'input_tokens'):
            print("\nTokenized segments:")
            for i, seg_tokens in enumerate(prompt_wrap_correct.input_tokens):
                print(f"Segment {i}: {tokenizer.decode(seg_tokens[0])}")
        
        print(f"\nImage token start index: {start_idx}")
        print(f"Image token end index: {end_idx}")
        print(f"Number of image tokens: {end_idx - start_idx}")
        print(f"Total context length: {context_embs.shape[1]}")
        print("=== End of First Image+Prompt Analysis ===\n")
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt_wrap_correct = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[prompt_with_correct],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        context_embs_correct = prompt_wrap_correct.context_embs[0]
        outputs_correct = model.llama_model(inputs_embeds=context_embs_correct, output_hidden_states=True, return_dict=True)
        start_idx, end_idx = prompt_wrap_correct.get_image_token_indices()
        num_image_tokens = end_idx - start_idx

        tqdm.write("\n" + "="*50)
        tqdm.write(f"Image: {row['image_name']}")
        tqdm.write(f"Prompt: {prompt}")
        tqdm.write(f"Correct Answer: {correct_answer}")
        tqdm.write(f"Image token start index: {start_idx}, end index: {end_idx}, num tokens: {num_image_tokens}")

        correct_activations = {}
        for layer in [12, 24]:
            image_token_acts = outputs_correct.hidden_states[layer][0, start_idx:end_idx, :].detach().cpu()
            avg_act = image_token_acts.mean(axis=0)
            tqdm.write(f"Layer {layer} correct activation shape: {image_token_acts.shape}")
            tqdm.write(f"Layer {layer} correct activation: {avg_act}")
            correct_activations[layer] = avg_act

        tqdm.write("="*50 + "\n")

        # Process each wrong answer
        for i, wrong_option in enumerate(wrong_options):
            pair_idx = idx * 3 + i  # 3 pairs per image
            prompt_with_wrong = query + ' ' + options_dict[wrong_option]
            print(f"[DEBUG] Wrong prompt: {prompt_with_wrong}")
            tokens_wrong = tokenizer(prompt_with_wrong, return_tensors="pt")
            prompt_wrap_wrong = prompt_wrapper.Prompt(
                model=model,
                text_prompts=[prompt_with_wrong],
                img_prompts=[[img_tensor]],
                device=f'cuda:{args.gpu_id}'
            )
            context_embs_wrong = prompt_wrap_wrong.context_embs[0]
            outputs_wrong = model.llama_model(inputs_embeds=context_embs_wrong, output_hidden_states=True, return_dict=True)
            start_idx_w, end_idx_w = prompt_wrap_wrong.get_image_token_indices()
            num_image_tokens_w = end_idx_w - start_idx_w

            tqdm.write("\n" + "="*50)
            tqdm.write(f"Wrong Answer: {wrong_option}")
            tqdm.write(f"Image token start index: {start_idx_w}, end index: {end_idx_w}, num tokens: {num_image_tokens_w}")

            for layer in [12, 24]:
                image_token_acts_wrong = outputs_wrong.hidden_states[layer][0, start_idx_w:end_idx_w, :].detach().cpu()
                avg_act_wrong = image_token_acts_wrong.mean(axis=0)
                tqdm.write(f"Layer {layer} wrong activation shape: {image_token_acts_wrong.shape}")
                tqdm.write(f"Layer {layer} wrong activation: {avg_act_wrong}")
                diff_vector = correct_activations[layer] - avg_act_wrong
                tqdm.write(f"Layer {layer} difference vector: {diff_vector}")
                reference_acts[pair_idx][layer].append(diff_vector)

            tqdm.write("="*50 + "\n")

            results_for_csv.append({
                'image_name': row['image_name'],
                'prompt': prompt,
                'correct_answer': correct_answer,
                'wrong_answer': wrong_option,
                'final_token_correct': tokenizer.decode(tokens_correct["input_ids"][0, -1]),
                'final_token_wrong': tokenizer.decode(tokens_wrong["input_ids"][0, -1]),
                'model_response_correct': prompt_with_correct,
                'model_response_wrong': prompt_with_wrong
            })

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(reference_acts, args.save_path)
print(f"Saved reference activations to {args.save_path}")

if results_for_csv:
    pd.DataFrame(results_for_csv).to_csv("test_model_responses_12-24_2.csv", index=False)
    print("Wrote model responses to test_model_responses_12-24_2.csv") 