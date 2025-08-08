#!/bin/sh -l
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --job-name sae_eval
#SBATCH --gres=gpu:2
#SBATCH --constraint=A100-80GB

# Load required modules
module load cuda

nvidia-smi

module load conda/2024.09
conda activate /depot/amannodi/apps/ASTRA/envs/astra_minigpt

# Set memory optimization flags
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Run the SAE trainer script
#cd /scratch/gilbreth/biswasm/ASTRA_updated/SAE_image_activations
PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python ./global_interpretation_analysis_train.py 