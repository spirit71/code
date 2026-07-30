# python train.py 2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt
# python train.py --test-every-epoch  2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt 
python train.py \
    --epochs 60 \
  --query-reliability \
  --reliability-mode learned \
  --reliability-gate centered_residual \
  --rel-target-mode rank \
  --rel-loss-type smooth_l1 \
   2>&1 | tee corss_view_allcommand_$(date +%Y%m%d_%H%M%S).log 