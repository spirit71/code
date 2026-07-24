#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${QASSR_PYTHON:-/root/miniconda3/envs/BoQ/bin/python}"
CHECKPOINT="${QASSR_CHECKPOINT:-/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[17].ckpt}"
DATASET="${1:-sped}"
MODE="${2:-quick}"
DEVICE="${QASSR_DEVICE:-cuda:0}"
EVAL_BS="${QASSR_EVAL_BS:-128}"
NUM_WORKERS="${QASSR_NUM_WORKERS:-8}"

# V2 使用独立根目录和 v2_ 前缀；不会覆盖 /root/data/qassr_outputs 的第一版结果。
CACHE_DIR="${QASSR_CACHE_DIR:-/root/qassr_cache}"
OUTPUT_DIR="${QASSR_V2_OUTPUT_DIR:-/root/data/qassr_outputs_v2}"
LOG_DIR="$OUTPUT_DIR/run_logs_v2"

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_DIR" "$LOG_DIR"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

run_one() {
  local short_name="$1"
  shift
  local experiment="v2_${DATASET}_${short_name}"
  echo "[QASSR-V2] dataset=$DATASET experiment=$experiment"
  "$PYTHON_BIN" scripts/qassr.py \
    --checkpoint "$CHECKPOINT" \
    --config configs/qassr/default.yaml \
    --datasets "$DATASET" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --eval-bs "$EVAL_BS" \
    --nw "$NUM_WORKERS" \
    --allow-test-overrides \
    --set dataset_split=test \
    --set "experiment_name=$experiment" \
    "$@" 2>&1 | tee "$LOG_DIR/${experiment}.log"
}

case "$MODE" in
  # quick 先在小数据集验证四项改动；feature 是与第一版同协议的对照。
  quick)
    run_one "attention64_feature" --set token_selection=attention_max --set token_count=64 --set matching=feature_only --set spatial=none --set gate=all-query
    run_one "attention64_residual025" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=none --set gate=all-query
    run_one "attention64_residual025_spatial" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set spatial_sigma=0.15 --set gate=all-query
    run_one "attention64_residual025_early_global" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set spatial_sigma=0.15 --set gate=global_margin
    ;;
  role)
    run_one "attention64_feature" --set token_selection=attention_max --set token_count=64 --set matching=feature_only --set spatial=none --set gate=all-query
    run_one "attention64_additive025" --set token_selection=attention_max --set token_count=64 --set matching=additive --set role_weight=0.25 --set spatial=none --set gate=all-query
    run_one "attention64_residual025" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=none --set gate=all-query
    ;;
  spatial)
    run_one "attention64_residual025_no_spatial" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=none --set gate=all-query
    run_one "attention64_residual025_translation" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_median --set spatial_sigma=0.15 --set gate=all-query
    run_one "attention64_residual025_inlier" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set spatial_sigma=0.15 --set gate=all-query
    ;;
  gate)
    run_one "attention64_residual025_all" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set gate=all-query
    run_one "attention64_residual025_early_global" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set gate=global_margin
    run_one "attention64_residual025_early_combined" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set gate=combined
    ;;
  best)
    # 用 SPED quick 中胜出的 V2 配置做跨数据集确认。
    run_one "attention64_residual025_spatial" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set spatial_sigma=0.15 --set gate=all-query
    ;;
  combined)
    # 单独验证“重排 Top-1 指标 + early global”修正后的 combined gate。
    run_one "attention64_residual025_early_combined" --set token_selection=attention_max --set token_count=64 --set matching=residual_role --set role_weight=0.25 --set spatial=translation_inlier --set spatial_sigma=0.15 --set gate=combined
    ;;
  *)
    echo "Unknown V2 mode: $MODE (use quick|role|spatial|gate|best|combined)" >&2
    exit 2
    ;;
esac
