## pitts30k
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15
python3 train.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=100  --patience=60   
##svox
# python3 train.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets/ --eval_dataset_name=svox/images --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15

##msls
# python3 train.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader --eval_dataset_name=msls --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15

