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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default="train_generate.csv", help="CSV with columns: image_name,prompt,answer")
    parser.add_argument("--save_path", type=str, default="./act_diff_12-20_gen.pt")
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
generator_obj = generator.Generator(
    model=model,
    num_beams=1
)
prefix = prompt_wrapper.minigpt4_chatbot_prompt
#instruction = ""

num_layers = len(model.llama_model.base_model.layers)
print(f"Total number of layers in the model: {num_layers}")

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

df = pd.read_csv(args.csv_path)  # columns: image_name,prompt,answer
num_data_points = len(df)
print(f"Number of data points: {num_data_points}")

# Initialize reference_acts for layers 12, 16, 20
reference_acts = {}
for idx in range(num_data_points):  # One pair per image (A and B)
    reference_acts[idx] = {
        12: [],
        16: [],
        20: []
    }

for idx, row in tqdm(df.iterrows(), total=num_data_points):
    image_path = os.path.join("../datasets/XRD_data/one_correct/ice_ring", row['image_name'])
    image = Image.open(image_path).convert("RGB")
    img_tensor = processor(image).unsqueeze(0).to(f'cuda:{args.gpu_id}')
    prompt = row['prompt']
    answer = row['answer']

    # Wrap in template
    query = prefix % prompt

    print(f"Processed prompt for image {row['image_name']}:\n{query}\n")

    # Get model response
    prompt_wrap = prompt_wrapper.Prompt(
        model=model,
        text_prompts=[query],
        img_prompts=[[img_tensor]],
        device=f'cuda:{args.gpu_id}'
    )
    model_response, _ = generator_obj.generate(prompt_wrap)
    print(f"Model response for image {row['image_name']}: {model_response}")

    # Get the correct option (A or B)
    correct_option = 'Ice ring'
    wrong_option = 'Artifact'

    # Process correct answer
    prompt_with_correct = query + ' ' + correct_option
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt_wrap_correct = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[prompt_with_correct],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        context_embs_correct = prompt_wrap_correct.context_embs[0]
        outputs_correct = model.llama_model(inputs_embeds=context_embs_correct, output_hidden_states=True, return_dict=True)
        hidden_states_correct = outputs_correct.hidden_states
        answer_pos_correct = context_embs_correct.shape[1] - 2  # 2nd last token

        # Get tokens from prompt_wrapper instead of text-only tokenizer
        tqdm.write("\n" + "="*50)
        tqdm.write(f"Image: {row['image_name']}")
        tqdm.write(f"All tokens: {prompt_wrap_correct.input_tokens}")
        # Print the last 5 tokens from the last tensor
        last_tensor = prompt_wrap_correct.input_tokens[-1]
        for i in range(6):
            if i < last_tensor.shape[1]:
                token_id = last_tensor[0, -(i+1)].item()
                token_text = tokenizer.decode([token_id])
                tqdm.write(f"Last {i+1} token ID: {token_id}, Last {i+1} token text: '{token_text}'")
        tqdm.write(f"Context embeddings length: {context_embs_correct.shape[1]}")
        tqdm.write(f"Position we're extracting from (correct): {answer_pos_correct}")
        tqdm.write(f"Correct option we added: {correct_option}")
        
        # Print which token activation is being extracted
        second_last_token_id = last_tensor[0, -2].item()
        second_last_token_text = tokenizer.decode([second_last_token_id])
        tqdm.write(f"EXTRACTING ACTIVATION FROM: Position {answer_pos_correct}, Token ID {second_last_token_id}, Token text '{second_last_token_text}'")

        correct_activations = {}
        for layer in [12, 16, 20]:
            act = hidden_states_correct[layer][0, answer_pos_correct, :].detach().cpu().numpy()
            tqdm.write(f"Layer {layer} correct activation: {act}")
            correct_activations[layer] = act

        tqdm.write("="*50 + "\n")

        # Process wrong answer
        prompt_with_wrong = query + ' ' + wrong_option
        prompt_wrap_wrong = prompt_wrapper.Prompt(
            model=model,
            text_prompts=[prompt_with_wrong],
            img_prompts=[[img_tensor]],
            device=f'cuda:{args.gpu_id}'
        )
        context_embs_wrong = prompt_wrap_wrong.context_embs[0]
        outputs_wrong = model.llama_model(inputs_embeds=context_embs_wrong, output_hidden_states=True, return_dict=True)
        hidden_states_wrong = outputs_wrong.hidden_states
        answer_pos_wrong = context_embs_wrong.shape[1] - 2  # Second last token

        # Print tokens for wrong option
        tqdm.write("\n" + "="*50)
        tqdm.write(f"All tokens (wrong): {prompt_wrap_wrong.input_tokens}")
        # Print the last 5 tokens from the last tensor for wrong option
        last_tensor_wrong = prompt_wrap_wrong.input_tokens[-1]
        for i in range(6):
            if i < last_tensor_wrong.shape[1]:
                token_id = last_tensor_wrong[0, -(i+1)].item()
                token_text = tokenizer.decode([token_id])
                tqdm.write(f"Last {i+1} token ID (wrong): {token_id}, Last {i+1} token text (wrong): '{token_text}'")
        tqdm.write(f"Context embeddings length (wrong): {context_embs_wrong.shape[1]}")
        tqdm.write(f"Position we're extracting from: {answer_pos_wrong}")
        tqdm.write(f"Wrong option we added: {wrong_option}")
        
        # Print which token activation is being extracted for wrong option
        second_last_token_id_wrong = last_tensor_wrong[0, -2].item()
        second_last_token_text_wrong = tokenizer.decode([second_last_token_id_wrong])
        tqdm.write(f"EXTRACTING ACTIVATION FROM: Position {answer_pos_wrong}, Token ID {second_last_token_id_wrong}, Token text '{second_last_token_text_wrong}'")

        for layer in [12, 16, 20]:
            act_wrong = hidden_states_wrong[layer][0, answer_pos_wrong, :].detach().cpu().numpy()
            tqdm.write(f"Layer {layer} wrong activation: {act_wrong}")
            diff_vector = correct_activations[layer] - act_wrong
            tqdm.write(f"Layer {layer} difference vector: {diff_vector}")
            reference_acts[idx][layer].append(diff_vector)

        tqdm.write("="*50 + "\n")

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(reference_acts, args.save_path)
print(f"Saved reference activations to {args.save_path}")