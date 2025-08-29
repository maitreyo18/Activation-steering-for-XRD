#!/bin/sh -l
#SBATCH -A amannodi          # amannodi from amannodi-f
#SBATCH --qos=normal         # normal/standby
#SBATCH --partition=v100     # partition that matches your available GPUs
#SBATCH --constraint=f
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --job-name=Minigpt-download

# Load required modules
module load cuda

nvidia-smi

module load conda/2024.09
conda activate /depot/amannodi/apps/ASTRA/envs/astra_minigpt

# Set up cache directories in scratch
export TRANSFORMERS_CACHE="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/models"
export HF_HOME="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/models"
export TORCH_HOME="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/torch_cache"
export HF_DATASETS_CACHE="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/datasets"
export HF_METRICS_CACHE="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/metrics"
export HF_MODULES_CACHE="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/modules"
export HF_TOKENIZERS_CACHE="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/tokenizers"

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $TORCH_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_METRICS_CACHE
mkdir -p $HF_MODULES_CACHE
mkdir -p $HF_TOKENIZERS_CACHE

#export HF_HUB_DOWNLOAD_TIMEOUT=60

# Run the dependency downloader script
PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python ./minigpt4_dependency_downloader.py --cfg_path ../eval_configs/minigpt4_eval.yaml --cache_dir ../Cached_minigpt4 --download

# Keep ../eval_configs/minigpt4_eval.yaml as 
# model:
#   arch: mini_gpt4
#   model_type: pretrain_vicuna
#   freeze_vit: True
#   freeze_qformer: True
#   max_txt_len: 160
#   end_sym: "###"
#   low_resource: True
#   device_8bit: 0
#   prompt_path: ""
#   prompt_template: '###Human: {} ###Assistant: '
#   ckpt: '/scratch/gilbreth/biswasm/ASTRA_updated/ckpts/pretrained_minigpt4.pth'
#   llama_model: '/scratch/gilbreth/biswasm/ASTRA_updated/Cached_minigpt4/models/models--Vision-CAIR--vicuna/snapshots/c80b8c2ca66c2efc88915f45a7d13f151fdace45'   # 👈 add this

# datasets:
#   cc_sbu_align:
#     vis_processor:
#       train:
#         name: "blip2_image_eval"
#         image_size: 224
#     text_processor:
#       train:
#         name: "blip_caption"

# run:
#   task: image_text_pretrain

