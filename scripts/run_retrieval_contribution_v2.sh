#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/code_qy_7_28/04_BoQ_cross_view_v0260720"
PYTHON_BIN="/root/miniconda3/envs/BoQ/bin/python"
OUTPUT_ROOT="${QRL_CONTRIB_ROOT:-/root/qrl_retrieval_contribution_v2}"
DATASET="${1:-}"

if [[ -z "${DATASET}" ]]; then
  echo "Usage: $0 DATASET [e0|e2_gateoff|e2_gateon ...]" >&2
  exit 2
fi
shift
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(e0 e2_gateoff e2_gateon)
fi

E0_CKPT="/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/epoch[10]_R@1[0.9257]_R@5[0.9622]_R@10[0.0000]_R@20[0.0000].ckpt"
E2_CKPT="/root/qrl_e_matrix/E2/dinov2_vitb14/version_0/checkpoints/epoch[15]_R@1[0.9284]_R@5[0.9649]_R@10[0.0000]_R@20[0.0000].ckpt"

cd "${PROJECT_ROOT}"
test -x "${PYTHON_BIN}"
test -f "${E0_CKPT}"
test -f "${E2_CKPT}"
mkdir -p "${OUTPUT_ROOT}"

df -h /root /home
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"${PYTHON_BIN}" -m py_compile \
  scripts/analyze_query_retrieval_contribution.py \
  scripts/summarize_retrieval_contribution_v2.py

for variant in "${VARIANTS[@]}"; do
  case "${variant}" in
    e0)
      checkpoint="${E0_CKPT}"
      model_variant="e0"
      gate_mode="off"
      ;;
    e2_gateoff)
      checkpoint="${E2_CKPT}"
      model_variant="e2"
      gate_mode="off"
      ;;
    e2_gateon)
      checkpoint="${E2_CKPT}"
      model_variant="e2"
      gate_mode="on"
      ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      exit 2
      ;;
  esac

  output_dir="${OUTPUT_ROOT}/${DATASET}/${variant}"
  if [[ -s "${output_dir}/summary.json" ]]; then
    echo "[SKIP] completed: ${output_dir}"
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "[STOP] incomplete output exists; inspect before resume: ${output_dir}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" -u scripts/analyze_query_retrieval_contribution.py \
    --checkpoint "${checkpoint}" \
    --model-variant "${model_variant}" \
    --gate-mode "${gate_mode}" \
    --positive-mode both \
    --dataset "${DATASET}" \
    --output-dir "${output_dir}" \
    --batch-size 128 \
    --loo-query-batch-size 32 \
    --num-workers 8 \
    --device cuda:0 \
    --seed 42 \
    --random-repeats 10
done

"${PYTHON_BIN}" scripts/summarize_retrieval_contribution_v2.py --root "${OUTPUT_ROOT}"
