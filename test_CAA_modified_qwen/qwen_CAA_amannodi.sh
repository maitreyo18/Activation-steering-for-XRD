#!/bin/sh -l
#SBATCH -A amannodi-f
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --job-name Qwen2VL-CAA

# Load required modules
module load cuda

nvidia-smi

module load conda/2024.09
conda activate /depot/amannodi/apps/ASTRA/envs/astra_qwen

# Set up cache directories in scratch
export TRANSFORMERS_CACHE="/scratch/gilbreth/biswasm/huggingface_cache"
export HF_HOME="/scratch/gilbreth/biswasm/huggingface_cache"
export TORCH_HOME="/scratch/gilbreth/biswasm/torch_cache"
export HF_DATASETS_CACHE="/scratch/gilbreth/biswasm/huggingface_cache/datasets"
export HF_METRICS_CACHE="/scratch/gilbreth/biswasm/huggingface_cache/metrics"
export HF_MODULES_CACHE="/scratch/gilbreth/biswasm/huggingface_cache/modules"
export HF_TOKENIZERS_CACHE="/scratch/gilbreth/biswasm/huggingface_cache/tokenizers"

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Create cache directories
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $TORCH_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_METRICS_CACHE
mkdir -p $HF_MODULES_CACHE
mkdir -p $HF_TOKENIZERS_CACHE

# Run the CAA generation script for Qwen2-VL
PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python ../CAA_modified_qwen/CAA_generate_all_anomalies_qwen.py --csv_path ./train_generate_nu_detect.csv --image_dir "../datasets/XRD_data/one_correct/non_uniform_detector" --correct_anomaly "Non-uniform detector" --wrong_anomalies "Loop scattering" "Diffuse scattering" "Ice ring" "Artifact" "No distinct anomalies" "Background ring" "Strong background" 