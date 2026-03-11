#!/bin/bash
# pitts30k 检索结果保存与改进前后对比 - 运行示例
# 使用前请根据实际路径修改 --eval_datasets_folder 和 --resume

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# ---------- 1) 运行检索并保存 JSON + 错误样本图（默认执行） ----------
# 保存每个 query 的 top-k 结果到 retrieval_results.json，并保存 top1 错误样本并排图
# python scripts/pitts30k_retrieval_save_and_compare_1.py \
#   --resume /home/code_qy_7_28/EDTformer/logs/logs/default/2026-01-25_14-18-27_edtf_repro_seed42/2026-01-25_14-18-33/best_model.pth \
#   --eval_datasets_folder /root/data/Pittsburgh \
#   --eval_dataset_name pitts30k \
#   --save_dir ./logs/pitts30k_retrieval_baseline \
#   --top_k 20 \
#   --output_json retrieval_results.json \
#   --save_error_images \
#   --max_error_images 2000
# --resume ./logs/default/2026-02-02_13-37-12/best_model.pth \ 

# ---------- 2) 改进后再跑一次，保存到另一目录 ----------
# python scripts/pitts30k_retrieval_save_and_compare.py \
#   --resume ./logs/default/YOUR_IMPROVED_MODEL/best_model.pth \
#   --eval_datasets_folder /root/data/Pittsburgh \
#   --eval_dataset_name pitts30k \
#   --save_dir ./logs/pitts30k_retrieval_improved \
#   --output_json retrieval_results.json \
#   --save_error_images \
#   --max_error_images 500

# ---------- 3) 对比改进前后：fixed / still wrong / new wrong ----------
# 先跑完 1 和 2 后，再执行下面命令。输出 comparison_report.txt、comparison_summary.json 及 fixed/ still_wrong/ new_wrong/ 对比图
# python scripts/pitts30k_retrieval_save_and_compare.py \
#   --comparison \
#   --baseline_json ./logs/pitts30k_retrieval_baseline/retrieval_results.json \
#   --improved_json ./logs/pitts30k_retrieval_improved/retrieval_results.json \
#   --eval_datasets_folder /root/data/Pittsburgh \
#   --eval_dataset_name pitts30k \
#   --comparison_dir ./logs/pitts30k_comparison
# echo "Done. Check comparison_report.txt and fixed/ still_wrong/ new_wrong/ under comparison_dir."

python scripts/analyze_comparison.py \
  --baseline /home/code_qy_7_28/EDTformer/logs/pitts30k_retrieval_baseline/retrieval_results.json \
  --improved ./logs/pitts30k_retrieval_improved_02/retrieval_results.json \
  --out_dir ./logs/pitts30k_comparison_report