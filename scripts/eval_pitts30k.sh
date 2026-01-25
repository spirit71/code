#!/bin/bash

# 基础配置
BASE_CMD="python3 eval.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k"
# MODEL_DIR="./logs/default/2025-10-03_10-30-41"
MODEL_DIR="./logs/default/2025-10-15_13-47-05"
RESULTS_DIR="./pitts30k_results_dinov2_L"

mkdir -p "$RESULTS_DIR"

# 定义评估函数
evaluate_model() {
    local model_path="$1"
    local model_file=$(basename "$model_path")
    local log_file="$RESULTS_DIR/${model_file%.pth}.log"
    
    echo "评估模型: $model_file"
    echo "执行: $BASE_CMD --resume=$model_path"
    
    # 执行命令并处理输出
    {
        # 打印命令和初始信息
        echo "评估模型: $model_file"
        echo "执行: $BASE_CMD --resume=$model_path"
        
        # 执行命令并过滤输出
        $BASE_CMD --resume="$model_path" 2>&1 | awk '
            # 保留所有警告信息
            /Warning/ {print}
            
            # 保留关键配置信息
            /Arguments: Namespace/ {print}
            /The outputs are being saved in/ {print}
            /using MLP layer as FFN/ {print}
            /Resuming model from/ {print}
            /Test set:/ {print}
            /Extracting .* features for evaluation/ {print}
            
            # 保留所有完整的进度条
            /100%/ {print}
            
            # 保留所有评估结果
            /Calculating recalls/ {print}
            /Recalls for queries:/ {print}
            /Combined recalls:/ {print}
            /Recalls on .*:/ {print}
            /Finished evaluation/ {print}
        '
        
        echo "完成: $model_file -> $log_file"
        echo "----------------------------------------"
    } | tee "$log_file"
}

# 按数字顺序处理model_epoch文件，然后处理其他文件
for category in "epoch" "other" "special"; do
    case $category in
        "epoch")
            # 处理model_epoch文件，使用sort -V进行版本排序
            find "$MODEL_DIR" -name "model_epoch_*.pth" | sort -t_ -k3 -V | while read model_path; do
                evaluate_model "$model_path"
            done
            ;;
        "other")
            # 处理其他.pth文件（除了best和last）
            find "$MODEL_DIR" -name "*.pth" ! -name "model_epoch_*" ! -name "best_model.pth" ! -name "last_model.pth" | while read model_path; do
                evaluate_model "$model_path"
            done
            ;;
        "special")
            # 最后处理best和last模型
            for special_model in "best_model.pth" "last_model.pth"; do
                if [[ -f "$MODEL_DIR/$special_model" ]]; then
                    evaluate_model "$MODEL_DIR/$special_model"
                fi
            done
            ;;
    esac
done