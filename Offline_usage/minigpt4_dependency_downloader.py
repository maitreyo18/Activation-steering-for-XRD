#!/usr/bin/env python3
"""
MiniGPT-4 Dependency Downloader
This script investigates what models are being downloaded and downloads them to local cache,
using the same codepath as the anomaly extractor (registry + config).
"""

import os
import sys
from pathlib import Path
import argparse
from transformers import AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Download MiniGPT-4 dependencies")
    parser.add_argument("--cfg_path", default="../eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--cache_dir", default="../Cached_minigpt4")
    parser.add_argument("--download", action="store_true", help="Actually download the models")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def setup_cache_directories(cache_dir):
    """Set up cache directories and environment variables"""
    cache_path = Path(cache_dir).resolve()

    # Create cache directories
    cache_path.mkdir(exist_ok=True)
    (cache_path / "models").mkdir(exist_ok=True)
    (cache_path / "tokenizers").mkdir(exist_ok=True)
    (cache_path / "processors").mkdir(exist_ok=True)

    # Set environment variables
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "models")
    os.environ["TORCH_HOME"] = str(cache_path / "torch_cache")

    print(f"Cache directories created at: {cache_path}")
    print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")

    return cache_path


def investigate_model_loading(cfg_path):
    """Investigate what models will be loaded"""
    print("=" * 60)
    print("INVESTIGATING MINIGPT-4 MODEL LOADING")
    print("=" * 60)

    args = argparse.Namespace(cfg_path=cfg_path, options=[])
    cfg = Config(args)
    model_config = cfg.model_cfg

    print(f"Model architecture: {model_config.arch}")
    print(f"Model type: {getattr(cfg, 'model_type', 'Not specified')}")

    return cfg, model_config


def download_minigpt_dependencies(cfg_path, gpu_id=0):
    """
    Download all MiniGPT-4 dependencies by instantiating the model the same
    way the anomaly extractor does (registry + config). This ensures the
    vision tower, Q-Former, LLaMA/Vicuna, tokenizer, and processors are
    fetched into cache, but without blowing GPU RAM.
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING MINIGPT-4 DEPENDENCIES")
    print("=" * 60)

    # Load config
    args = argparse.Namespace(cfg_path=cfg_path, options=[], gpu_id=gpu_id)
    cfg = Config(args)
    model_config = cfg.model_cfg

    # Build model (triggers downloads into cache)
    print("[download] Building MiniGPT-4 model via registry...")
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    model.eval()

    # Tokenizer
    print("[download] Loading tokenizer...")
    _ = AutoTokenizer.from_pretrained(model_config.llama_model)

    # Vision processor
    print("[download] Loading vision processor...")
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    _ = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    print("[download] Done. All MiniGPT-4 dependencies should now be cached.")


def main():
    args = parse_args()

    print("MiniGPT-4 Dependency Downloader")
    print("=" * 60)

    # Setup cache directories
    cache_path = setup_cache_directories(args.cache_dir)

    # Investigate model loading
    cfg, model_config = investigate_model_loading(args.cfg_path)

    if args.download:
        download_minigpt_dependencies(args.cfg_path, gpu_id=args.gpu_id)

        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"All models cached at: {cache_path}")
        print("\nTo use local models offline, set:")
        print("  export HF_HUB_OFFLINE=1")
        print("  export TRANSFORMERS_OFFLINE=1")
        print("and point your jobs to the same --cache_dir")
    else:
        print(f"\nTo actually download the models, run with --download flag:")
        print(f"  python {sys.argv[0]} --download")


if __name__ == "__main__":
    main()

