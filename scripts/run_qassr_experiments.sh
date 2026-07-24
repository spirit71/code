#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 以下变量都允许通过同名 QASSR_* 环境变量覆盖，默认值适配当前机器。
PYTHON_BIN="${QASSR_PYTHON:-/root/miniconda3/envs/BoQ/bin/python}"
CHECKPOINT="${QASSR_CHECKPOINT:-/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[17].ckpt}"
DATASET="${1:-sped}"
MODE="${2:-suite}"
DEVICE="${QASSR_DEVICE:-cuda:0}"
EVAL_BS="${QASSR_EVAL_BS:-128}"
NUM_WORKERS="${QASSR_NUM_WORKERS:-8}"
CACHE_DIR="${QASSR_CACHE_DIR:-/root/qassr_cache}"
OUTPUT_DIR="${QASSR_OUTPUT_DIR:-/root/qassr_outputs}"

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR/run_logs"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

case "$DATASET" in
  # 在真正运行前预留安全磁盘空间；大数据集的 x2/attention 缓存可达数十 GiB。
  sped) required_gib=2 ;;
  amstertime) required_gib=4 ;;
  pitts30k-test) required_gib=15 ;;
  nordland) required_gib=45 ;;
  svox-all) required_gib=35 ;;
  tokyo247) required_gib=60 ;;
  *) required_gib=15 ;;
esac
available_kib="$(df -Pk "$CACHE_DIR" | awk 'NR==2 {print $4}')"
if (( available_kib < required_gib * 1024 * 1024 )); then
  echo "Insufficient free space under $CACHE_DIR: need at least ${required_gib} GiB" >&2
  exit 3
fi

run_one() {
  # 所有消融最终都调用同一个 Python 入口，区别只来自明确记录的 --set 参数。
  local name="$1"
  shift
  echo "[QASSR] dataset=$DATASET experiment=$name"
  # 此脚本用于 test 上的诊断消融，所以显式允许覆盖；不能冒充正式冻结测试。
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
    --set "experiment_name=$name" \
    "$@" 2>&1 | tee "$OUTPUT_DIR/run_logs/${name}.log"
}

case "$MODE" in
  baseline)
    run_one "${DATASET}_all529_mutual" --set token_selection=all --set token_count=529 --set matching=feature_only --set spatial=none --set gate=all-query
    ;;
  spatial)
    run_one "${DATASET}_all529_translation" --set token_selection=all --set token_count=529 --set matching=feature_only --set spatial=translation_median --set gate=all-query
    run_one "${DATASET}_all529_pairwise" --set token_selection=all --set token_count=529 --set matching=feature_only --set spatial=pairwise_relative --set gate=all-query
    ;;
  sparse)
    run_one "${DATASET}_random64_seed1" --set token_selection=random --set token_count=64 --set random_seed=1
    run_one "${DATASET}_random64_seed42" --set token_selection=random --set token_count=64 --set random_seed=42
    run_one "${DATASET}_random64_seed3" --set token_selection=random --set token_count=64 --set random_seed=3
    run_one "${DATASET}_norm64" --set token_selection=feature_norm --set token_count=64
    run_one "${DATASET}_attention64" --set token_selection=attention_max --set token_count=64
    ;;
  role)
    run_one "${DATASET}_attention64_feature" --set token_selection=attention_max --set token_count=64 --set matching=feature_only
    run_one "${DATASET}_attention64_product_role" --set token_selection=attention_max --set token_count=64 --set matching=product_relu
    run_one "${DATASET}_attention64_add_role" --set token_selection=attention_max --set token_count=64 --set matching=additive
    ;;
  gate)
    run_one "${DATASET}_attention64_all_query" --set token_selection=attention_max --set token_count=64 --set matching=product_relu --set gate=all-query
    run_one "${DATASET}_attention64_global_gate" --set token_selection=attention_max --set token_count=64 --set matching=product_relu --set gate=global_margin
    run_one "${DATASET}_attention64_combined_gate" --set token_selection=attention_max --set token_count=64 --set matching=product_relu --set gate=combined
    ;;
  suite)
    run_one "${DATASET}_all529_mutual" --set token_selection=all --set token_count=529 --set matching=feature_only --set spatial=none --set gate=all-query
    run_one "${DATASET}_random64_seed1" --set token_selection=random --set token_count=64 --set random_seed=1
    run_one "${DATASET}_attention64_feature" --set token_selection=attention_max --set token_count=64 --set matching=feature_only
    run_one "${DATASET}_attention64_product_role" --set token_selection=attention_max --set token_count=64 --set matching=product_relu
    run_one "${DATASET}_attention64_combined_gate" --set token_selection=attention_max --set token_count=64 --set matching=product_relu --set gate=combined
    ;;
  *)
    echo "Unknown mode: $MODE (use baseline|spatial|sparse|role|gate|suite)" >&2
    exit 2
    ;;
esac
