# Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks

This is source code accompanying the paper of [Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks](https://arxiv.org/abs/2411.16721) by Han Wang, Gang Wang, and Huan Zhang.

![teaser](./assets/framework.jpg)
![adaptive](./assets/adaptive.jpg)

## Environment Preparation
To prepare the environment for LLaVA-v1.5 and MiniGPT-4, you can run the following commands:
```bash
conda create --name astra python==3.10.14
conda activate astra
pip install -r requirements.txt
```

To prepare the environment for Qwen2-VL, please run the following commands:
```bash
conda create --name astra_qwen python==3.10.15
conda activate astra_qwen
pip install -r requirements_qwen.txt
```

## Dataset Preparation

### Perturbation-based Attacks

We provide adversarial images for perturbation-based attack setups in `./datasets/adv_img_*`

#### Toxicity

+ Textual queries for steering vectors construction: \
we use 40 harmful instructions from [Qi et al.](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/tree/main/harmful_corpus). Please place the file `manual_harmful_instructions.csv` to `./dataset/harmful_corpus`.

+ Evaluation datasets: \
Please download the [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts) dataset and place it in `./datasets/harmful_corpus`.
Then, run the script `split_toxicity_set.py` located in `./datasets/harmful_corpus` to generate the validation and test sets.

#### Jailbreak

We mainly use text queries from AdvBench and Anthropic-HHH datasets for this setup.

+ Textual queries for steering vectors construction: \
Following the dataset split in [Schaeffer et al.](https://github.com/RylanSchaeffer/AstraFellowship-When-Do-VLM-Image-Jailbreaks-Transfer/tree/main/prompts_and_targets), we use `train.csv` in `AdvBench` to perform image attribution.

+ Evaluation datasets: \
The `eval.csv` file is equally divided to create validation and set sets. You can run the script `split_jb_set.py` in `./datasets/harmful_corpus` to generate the validatin and test sets for this setup.

##### TODO

- [ ] Release adversarial images used in the Transferability experiments

### Structured-based Attacks

Please download the [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) and place it in `./datasets`. We randomly sample 10 items from the 01-07 & 09 scenarios to construct the test set items in `./datasets/MM-SafetyBench/mmsafety_test.json`.

### Utility Evaluation

Please download the MM-Vet and MMBench datasets through [this link](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). To generate validation and test sets for MMBench dataset, run the script `split_mmbench.py` located in `./datasets/MMBench`. For the MM-Vet dataset, we provide the split items in `./dataset/mm-vet`.

## Steering Vector Construction

### Demos

To perform image attribution (e.g. in Qwen2-VL Jailbreak setup), run the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python ./extract_attr/extract_qwen_jb_attr.py
CUDA_VISIBLE_DEVICES=0 python ./extract_act/extracting_activations_qwen_jb.py
```
(Note: when performing image attribution on LLaVA-v1.5 or MiniGPT-4, please comment out line 1 in `./image_attr/__init__.py` to avoid potential bugs caused by differences in environments.)

### Activations

We provide steering vectors for each setup in `./activations/*/jb` and  `./activations/*/toxic`. Calibration activations are available in `./activations/*/reference`.

## Inference Evaluations

### Adversarial Scenarios
To evaluate the performance of adaptive steering (e.g., in Qwen2-VL), run the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python ./steer_eval/steering_qwen_toxic.py --attack_type constrain_16 --alpha 7 --eval test --steer_layer 14
CUDA_VISIBLE_DEVICES=0 python ./steer_eval/steering_qwen_jb.py --attack_type constrain_16 --alpha 7 --eval test --steer_layer 14
CUDA_VISIBLE_DEVICES=0 python ./steer_eval/steering_qwen_typo.py --alpha 7 --eval test --steer_layer 14
```
You can set `attack_type` to `constrain_16`, `constrain_32`, `constrain_64`, or `unconstrain`. Detailed option can be found in `parse_args()` function of each Python file.

To evaluate the performance of MiniGPT-4 and LLaVA-v1.5 (e.g. in the Toxicity setup), run the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python ./steer_eval/steering_minigpt_toxic.py --attack_type constrain_16 --alpha 5 --eval test
CUDA_VISIBLE_DEVICES=0 python ./steer_eval/steering_llava_toxic.py --attack_type constrain_16 --alpha 10 --eval test
```

### Benign Scenarios

To evaluate performance in the benign scenarios (e.g., with MiniGPT-4), run the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python ./utility_eval/minigpt_mmbench.py --attack_type constrain_16 --alpha 7 --eval test --steer_vector jb
CUDA_VISIBLE_DEVICES=0 python ./utility_eval/minigpt_mmvet.py --attack_type constrain_16 --alpha 7 --eval test --steer_vector jb
CUDA_VISIBLE_DEVICES=0 python ./utility_eval/minigpt_mmbench.py --attack_type constrain_16 --alpha 5 --eval test --steer_vector toxic
CUDA_VISIBLE_DEVICES=0 python ./utility_eval/minigpt_mmvet.py --attack_type constrain_16 --alpha 5 --eval test --steer_vector toxic
```
For detailed prompts to evaluate responses, see [MM-Vet](https://github.com/yuweihao/MM-Vet).

## Citation

If you find our work useful, please consider citing our paper:
```
@article{wang2024steering,
  title={Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks},
  author={Wang, Han and Wang, Gang, and Zhang, Huan},
  journal={arXiv preprint arXiv:2411.16721},
  year={2024}
}
```

Our codebase is built upon on the following work:
```
@article{cohenwang2024contextcite,
    title={ContextCite: Attributing Model Generation to Context},
    author={Cohen-Wang, Benjamin and Shah, Harshay and Georgiev, Kristian and Madry, Aleksander},
    journal={arXiv preprint arXiv:2409.00729},
    year={2024}
}
```