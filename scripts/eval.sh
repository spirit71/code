# python train.py 2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt
# python train.py --test-every-epoch  2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt 
# python train.py --test-only --checkpoint /home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_11/checkpoints/epoch[27]_R@1[0.9311]_R@5[0.9649]_R@10[0.0000]_R@20[0.0000].ckpt 2>&1 | tee baseline_test_version11_$(date +%Y%m%d_%H%M%S).log 
python scripts/test_checkpoints_in_dir.py --skip-existing  2>&1 | tee Baseline_version16_$(date +%Y%m%d_%H%M%S).log