#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录执行；数据集目录结构需与 README / VPR benchmark 一致。
# 用法: DATASETS_FOLDER=/path/to/datasets ./scripts/train_sage.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"


python3 train.py \
  --resize 322 322 \
  --mining sage \
  --cache_refresh_rate 1000 \
  --queries_per_epoch 5000 \
  --train_batch_size 16 \
  --infer_batch_size 16 \
  --epochs_num 50 \
  --patience 20 \
  --lr 0.0001 \
  --optim adamw \
  --negs_num_per_query 10 \
  --neg_samples_num 1000 \
  --num_workers 4 \
  --crossimage_encoder \
  --save_dir sage_train_run \
  "$@"
