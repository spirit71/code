#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# python train.py --quick 2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt
# python train.py --epochs 1 2>&1 | tee one_epoch_full_eval.log
python train.py --epochs 1 