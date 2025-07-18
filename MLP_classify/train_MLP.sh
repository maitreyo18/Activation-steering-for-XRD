#!/bin/sh -l
#SBATCH -A standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --job-name minigpt-eval
#SBATCH --gres=gpu:2
#SBATCH --constraint=A100-80GB


# Load required modules
module load cuda

nvidia-smi

module load conda/2024.09
conda activate /depot/amannodi/apps/ASTRA/envs/astra_minigpt

python training_MLP.py