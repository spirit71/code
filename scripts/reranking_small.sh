#!/usr/bin/env bash
set -euo pipefail

/root/miniconda3/envs/BoQ/bin/python scripts/xl_rerank.py \
  --checkpoint '/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[17].ckpt' \
  --datasets nordland \
  --top-k 50 \
  --debug-num-queries 50 \
  --rerank-device cuda:0
  2>&1 | tee reranking_$(date +%Y%m%d_%H%M%S).log
