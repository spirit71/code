#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/code_qy_7_28/04_BoQ_cross_view_v0260720"
CONTRIB_ROOT="/root/qrl_retrieval_contribution_v2"
OUTPUT_ROOT="${QRL_MECHANISM_ROOT:-/root/qrl_utility_mechanisms}"
PYTHON_BIN="/root/miniconda3/envs/BoQ/bin/python"
E0_CKPT="/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/epoch[10]_R@1[0.9257]_R@5[0.9622]_R@10[0.0000]_R@20[0.0000].ckpt"
E2_CKPT="/root/qrl_e_matrix/E2/dinov2_vitb14/version_0/checkpoints/epoch[15]_R@1[0.9284]_R@5[0.9649]_R@10[0.0000]_R@20[0.0000].ckpt"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/features" "${OUTPUT_ROOT}/analysis"
df -h /root /home
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"${PYTHON_BIN}" -m py_compile \
  scripts/cache_query_utility_features.py \
  scripts/analyze_query_utility_mechanisms.py

for dataset in amstertime sped; do
  for label in e0 e2_gateoff e2_gateon; do
    case "${label}" in
      e0)
        checkpoint="${E0_CKPT}"
        variant="e0"
        gate="off"
        experiment_label="e0_gateoff"
        [[ "${dataset}" == "sped" ]] && experiment_label="e0"
        ;;
      e2_gateoff)
        checkpoint="${E2_CKPT}"
        variant="e2"
        gate="off"
        experiment_label="e2_gateoff"
        ;;
      e2_gateon)
        checkpoint="${E2_CKPT}"
        variant="e2"
        gate="on"
        experiment_label="e2_gateon"
        ;;
    esac
    experiment_dir="${CONTRIB_ROOT}/${dataset}/${experiment_label}"
    cache="${OUTPUT_ROOT}/features/${dataset}_${label}.pt"
    analysis="${OUTPUT_ROOT}/analysis_v2/${dataset}_${label}"
    test -s "${experiment_dir}/summary.json"

    if [[ ! -s "${cache}" ]]; then
      "${PYTHON_BIN}" -u scripts/cache_query_utility_features.py \
        --checkpoint "${checkpoint}" \
        --model-variant "${variant}" \
        --gate-mode "${gate}" \
        --dataset "${dataset}" \
        --output "${cache}" \
        --batch-size 128 \
        --num-workers 8 \
        --device cuda:0
    else
      echo "[SKIP] feature cache exists: ${cache}"
    fi

    if [[ -s "${analysis}/summary.json" ]]; then
      echo "[SKIP] analysis exists: ${analysis}"
    elif [[ -e "${analysis}" ]]; then
      echo "[STOP] incomplete analysis directory exists: ${analysis}" >&2
      exit 1
    else
      "${PYTHON_BIN}" -u scripts/analyze_query_utility_mechanisms.py \
        --experiment-dir "${experiment_dir}" \
        --feature-cache "${cache}" \
        --output-dir "${analysis}" \
        --eps-scale 0.25 \
        --redundancy-batch-size 256 \
        --device cuda:0
    fi
  done
done
