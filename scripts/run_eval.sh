#!/bin/bash
# 文件名: run_eval.sh

# 检查参数个数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <BASE_CMD> <MODEL_DIR> <RESULTS_DIR>"
    exit 1
fi

# 1. 接收传入的参数
BASE_CMD="$1"
MODEL_DIR="$2"
RESULTS_DIR="$3"

echo "========================================"
echo "开始评估任务"
echo "数据集命令: $BASE_CMD"
echo "模型路径: $MODEL_DIR"
echo "结果输出: $RESULTS_DIR"
echo "========================================"

mkdir -p "$RESULTS_DIR"

# 定义评估函数
evaluate_model() {
    local model_path=$1
    local model_file=$(basename "$model_path")
    echo "评估模型: $model_file"
    
    # 拼接 resume 参数
    CMD="$BASE_CMD --resume=$model_path"
    RESULT_FILE="$RESULTS_DIR/${model_file%.pth}.log"
    
    echo "执行: $CMD" | tee "$RESULT_FILE"
    
    # 执行命令并记录日志
    $CMD 2>&1 | tee -a "$RESULT_FILE"
    
    echo "完成: $model_file -> $RESULT_FILE"
    echo "----------------------------------------"
}

# --- 以下逻辑保持原样 ---

# 第一阶段：优先评估 best_model 和 last_model
echo "=== 第一阶段：优先评估 best_model 和 last_model ==="

if [ -f "$MODEL_DIR/best_model.pth" ]; then
    evaluate_model "$MODEL_DIR/best_model.pth"
else
    echo "警告: $MODEL_DIR/best_model.pth 不存在"
fi

if [ -f "$MODEL_DIR/last_model.pth" ]; then
    evaluate_model "$MODEL_DIR/last_model.pth"
else
    echo "警告: $MODEL_DIR/last_model.pth 不存在"
fi

# 第二阶段：按 epoch 编号升序评估其他模型
echo "=== 第二阶段：按 epoch 编号升序评估其他模型 ==="

find "$MODEL_DIR" -name "model_epoch_*.pth" | \
grep -v "best_model.pth" | grep -v "last_model.pth" | \
sort -n | \
while read model_path; do
    evaluate_model "$model_path"
done

echo "当前数据集所有模型评估完成！"