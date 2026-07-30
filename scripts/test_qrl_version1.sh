#!/usr/bin/env bash
set -euo pipefail

# 测试 version_1 中所有逐轮 QRL-BoQ checkpoint。
# --skip-existing 会跳过已经具有全部测试集指标的 epoch，支持中断后续跑。

PROJECT_ROOT="/home/code_qy_7_28/04_BoQ_cross_view_v0260720"
PYTHON_BIN="/root/miniconda3/envs/BoQ/bin/python"
VERSION_DIR="${PROJECT_ROOT}/logs/dinov2_vitb14/version_1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/root/qrl_version1_checkpoint_test_${TIMESTAMP}.log"

cd "${PROJECT_ROOT}"

echo "Checkpoint directory: ${VERSION_DIR}"
echo "Test log: ${LOG_FILE}"

"${PYTHON_BIN}" scripts/test_checkpoints_in_dir.py \
  --version-dir "${VERSION_DIR}" \
  --skip-existing \
  --query-reliability \
  --reliability-mode learned \
  --reliability-gate centered_residual \
  --rel-target-mode rank \
  --rel-loss-type smooth_l1 \
  --eval-bs 512 \
  --nw 8 \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
