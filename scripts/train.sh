# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
## pitts30k

python3 ./train.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k --foundation_model_path=./weights/dinov2_vitb14_pretrain.pth --epochs_num=100  --patience=25 
#  --resume ./logs/default/2025-12-04_02-00-50/last_model.pth
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k --foundation_model_path=./dinov2_vitl14_reg4_pretrain.pth --epochs_num=100  --patience=100
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k --foundation_model_path=dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --epochs_num=100  --patience=100
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k --foundation_model_path=dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --epochs_num=150  --patience=100  --resume  ./logs/default/2025-11-07_14-42-21/model_epoch_37.pth
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k --foundation_model_path=dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --epochs_num=150  --patience=100  

## pitts30k
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=30   --resume /home/code_qy_7_28/EDTformer/logs/default/2025-09-23_14-26-17/last_model.pth    # 建议保持相同保存路径/home/code_qy_7_28/EDTformer/logs/default/2025-09-23_14-26-17

##svox
# python3 train.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets/ --eval_dataset_name=svox/images --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15


