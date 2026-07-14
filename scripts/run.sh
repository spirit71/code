# python train.py 2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt
# python train.py --test-every-epoch  2>&1 | tee log_$(date +%Y%m%d_%H%M%S).txt 
python train.py --test-every-epoch  2>&1 | tee baseline_test_every_epoch_$(date +%Y%m%d_%H%M%S).log 