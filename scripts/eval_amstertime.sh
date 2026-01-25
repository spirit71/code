#!/bin/bash

# 基础配置
BASE_CMD="python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=amstertime/images"
#diniv2_L
MODEL_DIR="./logs/default/2025-10-15_13-47-05"
RESULTS_DIR="./eval_results/amstertime_results_dinov2L"

mkdir -p "$RESULTS_DIR"

# 定义评估函数
evaluate_model() {
    local model_path=$1
    local model_file=$(basename "$model_path")
    echo "评估模型: $model_file"
    
    CMD="$BASE_CMD --resume=$model_path"
    RESULT_FILE="$RESULTS_DIR/${model_file%.pth}.log"
    
    echo "执行: $CMD" | tee "$RESULT_FILE"
    $CMD 2>&1 | tee -a "$RESULT_FILE"
    
    echo "完成: $model_file -> $RESULT_FILE"
    echo "----------------------------------------"
}

# 第一阶段：优先评估 best_model 和 last_model
echo "=== 第一阶段：优先评估 best_model 和 last_model ==="

# 评估 best_model.pth
if [ -f "$MODEL_DIR/best_model.pth" ]; then
    evaluate_model "$MODEL_DIR/best_model.pth"
else
    echo "警告: $MODEL_DIR/best_model.pth 不存在"
fi

# 评估 last_model.pth
if [ -f "$MODEL_DIR/last_model.pth" ]; then
    evaluate_model "$MODEL_DIR/last_model.pth"
else
    echo "警告: $MODEL_DIR/last_model.pth 不存在"
fi

# 第二阶段：按 epoch 编号降序评估其他模型
echo "=== 第二阶段：按 epoch 编号降序评估其他模型 ==="

# 使用find命令发现所有.pth文件，排除best和last，然后按epoch降序排序
find "$MODEL_DIR" -name "model_epoch_*.pth" | \
grep -v "best_model.pth" | grep -v "last_model.pth" | \
sort -V  | \
while read model_path; do
    evaluate_model "$model_path"
done

echo "所有模型评估完成！"