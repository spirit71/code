#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: ./scripts/eval_all.sh <RUN_DIR>}"

# 读取 best checkpoint
if [ -f "${RUN_DIR}/best_ckpt.txt" ]; then
  CKPT="$(cat ${RUN_DIR}/best_ckpt.txt)"
else
  echo "[ERROR] ${RUN_DIR}/best_ckpt.txt not found."
  exit 1
fi

OUT_DIR="${RUN_DIR}/metrics"
mkdir -p "${OUT_DIR}"

# ====== 定义数据集列表（按你现有命令整理） ======
# 格式：name|folder|dataset_name
DATASETS=(
  "amstertime|/home/code_qy_7_28/VPR-datasets-downloader/datasets|amstertime/images"
  "sped|/home/code_qy_7_28/VPR-datasets-downloader/datasets|sped/images"
  "tokyo247|/root/data|Tokyo247/images"
  "pitts30k|/root/data/Pittsburgh|pitts30k"
  "nordland|/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/|images_winter_as_quries"
  "svox|/home/code_qy_7_28/VPR-datasets-downloader/datasets|svox/images"
  "msls|/home/code_qy_7_28/VPR-datasets-downloader|msls"

)

for item in "${DATASETS[@]}"; do
  IFS="|" read -r NAME FOLDER DNAME <<< "${item}"
  echo "=== Evaluating ${NAME} with ${CKPT} ==="
  python3 eval.py \
    --eval_datasets_folder="${FOLDER}" \
    --eval_dataset_name="${DNAME}" \
    --resume="${CKPT}" \
    2>&1 | tee "${OUT_DIR}/eval_${NAME}.log"
done
