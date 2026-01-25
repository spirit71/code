#!/bin/bash
# 文件名: batch.sh

# ================= 配置区域 =================

# 1. 实验标识符 (这将是 eval_results 下的一级子目录名)
# 建议每次跑新模型时修改这里，例如: "dinov2_L_seed42" 或 "baseline_epoch20"
# EXP_NAME="dinov2_B_experiment_01_seed42"
EXP_NAME="dinov2_B_experiment_01_seed0"

# 2. 指定模型文件夹路径 (在此处修改为你实际的模型路径)
TARGET_MODEL_DIR="./logs/default/2026-01-15_10-05-18"

# 3. 基础结果输出目录
BASE_RESULTS_DIR="./eval_results"

# ===========================================

# 构建当前实验的总目录: ./eval_results/dinov2_L_experiment_01
CURRENT_EXP_ROOT="${BASE_RESULTS_DIR}/${EXP_NAME}"

# 检查 run_eval.sh 是否存在
SCRIPT="./scripts/run_eval.sh"
if [ ! -x "$SCRIPT" ]; then
    echo "错误: $SCRIPT 不存在或没有执行权限。"
    exit 1
fi

echo "开始批量测试..."
echo "实验名称: $EXP_NAME"
echo "结果将保存至: $CURRENT_EXP_ROOT"
echo "----------------------------------------"

# 定义函数
run_dataset_eval() {
    NAME=$1
    CMD_FLAGS=$2
    
    # 处理名称斜杠
    SAFE_NAME=${NAME//\//_} 
    
    # 【关键修改】构建多级目录路径
    # 格式: eval_results / 实验名 / 数据集名
    RESULTS_DIR="${CURRENT_EXP_ROOT}/${SAFE_NAME}"
    
    # 构建完整命令
    FULL_CMD="python3 eval.py ${CMD_FLAGS}"
    
    # 调用 run_eval.sh (它会自动创建多级目录 mkdir -p)
    ./scripts/run_eval.sh "$FULL_CMD" "$TARGET_MODEL_DIR" "$RESULTS_DIR"
}

# ================= 任务列表 =================

# 5. nordLand
run_dataset_eval "nordland" \
"--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets --eval_dataset_name=Nordland/images_winter_as_quries"


# # # 2. SPED
# run_dataset_eval "sped" \
# "--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets --eval_dataset_name=sped/images"

# # # 1. amstertime
# run_dataset_eval "amstertime" \
# "--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets --eval_dataset_name=amstertime/images"

# # # 3. Tokyo247
# run_dataset_eval "tokyo247" \
# "--eval_datasets_folder=/root/data --eval_dataset_name=Tokyo247/images"

# # # 4. pitts30k
# run_dataset_eval "pitts30k" \
# "--eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k"

# # 5. nordLand
# run_dataset_eval "nordland" \
# "--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets/nordland --eval_dataset_name=images_winter_as_quries"


# # 6. SVOX
# run_dataset_eval "svox" \
# "--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets --eval_dataset_name=svox/images"

# #7. msls
# python3 eval.py  "msls"  \
# "--eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader  --eval_dataset_name=msls"

echo "========================================"
echo "所有数据集评估完成！"
echo "结果位于: $CURRENT_EXP_ROOT"