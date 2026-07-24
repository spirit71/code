QASSR_CKPT='/home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[17].ckpt'

mkdir -p outputs/run_logs

python scripts/qassr.py \
  --checkpoint "$QASSR_CKPT" \
  --config configs/qassr/default.yaml \
  --datasets sped \
  --device cuda:0 \
  --eval-bs 32 \
  --nw 8 \
  --set dataset_split=test \
  --set experiment_name=sped_mutual_baseline \
  2>&1 | tee outputs/run_logs/sped_mutual_baseline.log