#!/usr/bin/env bash
set -euo pipefail

/root/miniconda3/envs/BoQ/bin/python scripts/xl_rerank.py \
  --checkpoint '/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[17].ckpt' \
  --top-k 50 \
    --datasets tokyo247 \
  --global-weight 0.7 \
  --similarity-threshold 0.5 \
  --spatial-sigma 0.15 \
  --spatial-weight 0.3 \
  --min-matches 4 \
  --rerank-device cuda:0 
