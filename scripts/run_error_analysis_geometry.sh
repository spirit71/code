#!/bin/bash
# 筛选 amstertime / tokyo247 / Nordland 的错误样本图片，并生成最终层 queries 注意力热图
# 用于分析 Geometry-Constrained-Assignment 在部分数据集提升、Nordland 下降的原因

set -e
cd "$(dirname "$0")/.."
RESUME="${1:-./logs/default/2026-01-31_10-39-54/best_model.pth}"
SAVE_DIR="${2:-./logs/geometry_error_analysis_0306}"

mkdir -p "$SAVE_DIR"
PYTHONPATH=. python scripts/analyze_errors_attnmap.py \
  --geometry_datasets \
  --resume "$RESUME" \
  --save_dir "$SAVE_DIR" \
  --eval_datasets_folder /root/data

echo "结果目录: $SAVE_DIR"
echo "  - error_sample_images_pitts30k/  错误样本查询图 (pitts30k)"
# echo "  - error_sample_images_amstertime/  错误样本查询图 (amstertime)"
# echo "  - error_sample_images_tokyo247/     错误样本查询图 (tokyo247)"
# echo "  - error_sample_images_nordland/    错误样本查询图 (Nordland)"
echo "  - error_visualization_*/            各数据集错误样本的叠加热力图与最终层热图"
echo "  - retrieval_errors_*/               各数据集错误样本索引与统计"
