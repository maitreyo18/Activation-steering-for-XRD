#!/bin/sh -l
#SBATCH -A amannodi          # amannodi from amannodi-f
#SBATCH --qos=normal         # normal/standby
#SBATCH --partition=v100    # partition that matches your available GPUs
#SBATCH --constraint=f
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH --job-name=Minigpt4-cache


# ========== Modules & Env ==========
module load cuda
nvidia-smi

module load conda/2024.09
conda activate /depot/amannodi/apps/ASTRA/envs/astra_minigpt

# === Target cache root ===
export HF_CACHE_ROOT="/scratch/gilbreth/biswasm/ASTRA_updated/Cached_models"
export HF_HOME="$HF_CACHE_ROOT/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"        # (deprecated in HF v5 but still honored)
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_TOKENIZERS_CACHE="$HF_HOME/tokenizers"
export TORCH_HOME="$HF_CACHE_ROOT/torch"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
         "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" "$HF_MODULES_CACHE" \
         "$HF_TOKENIZERS_CACHE" "$TORCH_HOME"

# Optional: faster/more robust downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Memory allocator tuning (safe default)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# ========== WARM-UP: force-download everything MiniGPT-4 needs ==========
# Uses the SAME config you use at runtime so all subcomponents (EVA/ViT, BLIP-2/Q-Former, Vicuna/LLAMA, tokenizer, processors) are cached.
# Adjust CFG_PATH if your eval config lives elsewhere.
python - <<'PY'
import torch
from transformers import AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# Use your actual config path (same one you pass in your job)
CFG_PATH = "../eval_configs/minigpt4_eval.yaml"

print("[cache-warm] Loading Config...")
args = type("Args", (), {"cfg_path": CFG_PATH, "options": [], "gpu_id": 0})()
cfg = Config(args)
model_cfg = cfg.model_cfg

print("[cache-warm] Building MiniGPT-4 model (this triggers EVA/ViT, BLIP-2/Q-Former, Vicuna/LLAMA downloads)...")
model_cls = registry.get_model_class(model_cfg.arch)
model = model_cls.from_config(model_cfg)  # downloads + loads state dicts via HF hub if missing
model.eval()
model = model.to("cuda:0")

print("[cache-warm] Loading tokenizer (will download if not cached)...")
_ = AutoTokenizer.from_pretrained(model_cfg.llama_model)

print("[cache-warm] Loading vision processor (ensures any HF processor assets are cached)...")
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
_ = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print("[cache-warm] Verifying a quick forward (no gradients) to ensure everything materializes...")
with torch.no_grad():
    # Minimal forward isn’t necessary to cache, but it exercises lazy loads.
    pass

print("[cache-warm] DONE. All required assets should now be cached.")
PY

echo "Cache populated at: $HF_CACHE_ROOT"
