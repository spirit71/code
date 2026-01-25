#!/bin/bash

# 基础配置
BASE_CMD="python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets   --eval_dataset_name=svox/images"
MODEL_DIR="./logs/default/2025-10-03_10-30-41"
RESULTS_DIR="./eval_results/svox_results"
START_EPOCH=${1:-0}  # 从命令行参数获取起始epoch，默认为0

mkdir -p "$RESULTS_DIR"

# 定义评估函数
evaluate_model() {
    local model_path=$1
    local model_file=$(basename "$model_path")
    echo "评估模型: $model_file"
    
    CMD="$BASE_CMD --resume=$model_path"
    RESULT_FILE="$RESULTS_DIR/${model_file%.pth}.log"
    
    # 检查结果文件是否已存在且非空
    if [ -s "$RESULT_FILE" ]; then
        echo "跳过已评估的模型: $model_file (结果文件已存在)"
        return 0
    fi
    
    echo "执行: $CMD" | tee "$RESULT_FILE"
    $CMD 2>&1 | tee -a "$RESULT_FILE"
    
    echo "完成: $model_file -> $RESULT_FILE"
    echo "----------------------------------------"
}

# 检查起始epoch参数
if ! [[ "$START_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "错误: 起始epoch必须是数字"
    echo "用法: $0 [起始epoch]"
    echo "示例: $0 10    # 从epoch 10开始测试"
    exit 1
fi

echo "=== 开始评估，起始epoch: $START_EPOCH ==="

# 第一阶段：优先评估 best_model 和 last_model（如果未评估过）
echo "=== 第一阶段：检查并评估 best_model 和 last_model ==="

# 评估 best_model.pth（如果结果文件不存在）
if [ -f "$MODEL_DIR/best_model.pth" ]; then
    if [ ! -s "$RESULTS_DIR/best_model.log" ]; then
        evaluate_model "$MODEL_DIR/best_model.pth"
    else
        echo "跳过已评估的模型: best_model.pth (结果文件已存在)"
    fi
else
    echo "警告: $MODEL_DIR/best_model.pth 不存在"
fi

# 评估 last_model.pth（如果结果文件不存在）
if [ -f "$MODEL_DIR/last_model.pth" ]; then
    if [ ! -s "$RESULTS_DIR/last_model.log" ]; then
        evaluate_model "$MODEL_DIR/last_model.pth"
    else
        echo "跳过已评估的模型: last_model.pth (结果文件已存在)"
    fi
else
    echo "警告: $MODEL_DIR/last_model.pth 不存在"
fi

# 第二阶段：按 epoch 编号升序评估其他模型，从指定epoch开始
echo "=== 第二阶段：从epoch $START_EPOCH 开始评估模型 ==="

# 使用find命令发现所有.pth文件，排除best和last，然后按epoch升序排序
find "$MODEL_DIR" -name "model_epoch_*.pth" | \
grep -v "best_model.pth" | grep -v "last_model.pth" | \
sort -V | \
while read model_path; do
    model_file=$(basename "$model_path")
    
    # 提取epoch编号（处理00, 01, 02等格式）
    epoch_num=$(echo "$model_file" | grep -oP 'model_epoch_\K\d+')
    
    # 去除前导零，使其成为十进制数字
    epoch_num=$((10#$epoch_num))
    
    # 只有当epoch编号 >= START_EPOCH时才进行评估
    if [ "$epoch_num" -ge "$START_EPOCH" ]; then
        evaluate_model "$model_path"
    else
        echo "跳过epoch $epoch_num (小于起始epoch $START_EPOCH)"
    fi
done

echo "所有模型评估完成！"
echo "结果保存在: $RESULTS_DIR"