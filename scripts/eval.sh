#!/usr/bin/env bash
set -euo pipefail

# Consolidated comparison for one version (all checkpoints).
python scripts/eval_all_checkpoints.py \
  --run_name dinov2_vitb14 \
  --version version_14 \
  --backbone dinov2_vitb14 \
  --primary_k 1 \
  --report_dir ./logs/eval_reports/baseline \
  --skip_existing \
  --cleanup_single_reports