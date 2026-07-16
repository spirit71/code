# python train.py 2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt
# python train.py --test-every-epoch  2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt 
python train.py --test-every-epoch --epoch 60 --checkpoint /home/code_qy_7_28/00_Baseline_BoQ_260711/logs/dinov2_vitb14/version_16/checkpoints/archive-epoch[36].ckpt  2>&1 | tee baseline_difference_test_every_epoch_$(date +%Y%m%d_%H%M%S).log 