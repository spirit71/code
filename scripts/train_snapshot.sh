#!/usr/bin/env bash
set -euo pipefail

# ====== 你需要按实际情况改的参数 ======
SEED="${SEED:-42}"
EXP_NAME="${EXP_NAME:-edtf_repro}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-25}"

EVAL_DATASETS_FOLDER="${EVAL_DATASETS_FOLDER:-/root/data/Pittsburgh}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-pitts30k}"

FOUNDATION_MODEL="${FOUNDATION_MODEL:-./dinov2_vitb14_pretrain.pth}"

# ====== 生成 RunID & 目录 ======
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_${EXP_NAME}_seed${SEED}"
RUN_DIR="./logs/default/${RUN_ID}"
META_DIR="${RUN_DIR}/meta"
CKPT_DIR="${RUN_DIR}/ckpts"
mkdir -p "${META_DIR}" "${CKPT_DIR}" "${RUN_DIR}/metrics"

# ====== 保存元信息：命令、git、环境 ======
echo "python3 train.py $*" > "${META_DIR}/cmd_train.txt"

# 保存“等价配置”（先用 args 文本，后续你再升级成 yaml）
cat > "${META_DIR}/config_train.args" <<EOF
--seed=${SEED}
--epochs_num=${EPOCHS}
--patience=${PATIENCE}
--foundation_model_path=${FOUNDATION_MODEL}
--eval_datasets_folder=${EVAL_DATASETS_FOLDER}
--eval_dataset_name=${EVAL_DATASET_NAME}
EOF

git rev-parse HEAD > "${META_DIR}/git_commit.txt" 2>/dev/null || true
git status --porcelain > "${META_DIR}/git_status.txt" 2>/dev/null || true
git diff > "${META_DIR}/git_diff.patch" 2>/dev/null || true
python3 -m pip freeze > "${META_DIR}/pip_freeze.txt" || true

# 若你有 split 文件（例如 splits/*.txt 或 *.json），拷贝进来（没有就先跳过）
mkdir -p "${META_DIR}/splits"
cp -a ./splits/* "${META_DIR}/splits/" 2>/dev/null || true

# ====== 开始训练（日志重定向） ======
# 强烈建议：train.py 支持 --seed 与 --save_dir（如果现在没有就先不加）
python3 train.py \
  --eval_datasets_folder="${EVAL_DATASETS_FOLDER}" \
  --eval_dataset_name="${EVAL_DATASET_NAME}" \
  --foundation_model_path="${FOUNDATION_MODEL}" \
  --epochs_num="${EPOCHS}" \
  --patience="${PATIENCE}" \
  --seed="${SEED}" \
  --save_dir="${RUN_DIR}" \
  2>&1 | tee "${RUN_DIR}/train.log"

# ====== 训练结束后：写一个“best checkpoint 指针” ======
# 你需要在 train.py 里保证：best 模型会保存为 ${RUN_DIR}/best_model.pth 或者写入 best_ckpt.txt
# 如果你目前保存格式是 model_epoch_xx.pth，那你至少要让 train.py 在更新 best 时写 best_ckpt.txt。
if [ -f "${RUN_DIR}/best_ckpt.txt" ]; then
  echo "[OK] Best ckpt recorded: $(cat ${RUN_DIR}/best_ckpt.txt)"
else
  echo "[WARN] best_ckpt.txt not found. Please modify train.py to write it when val improves."
fi
