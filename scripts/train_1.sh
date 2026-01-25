## pitts30k
python3 train_onnx.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k --foundation_model_path=./weights/dinov2_vitb14_pretrain.pth --epochs_num=15
# python3 train.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k --foundation_model_path=./weights/dinov2_vitb14_pretrain.pth --epochs_num=150  --patience=150   --resume /home/code_qy_7_28/EDTformer/logs/default/2025-09-28_04-33-24/last_model.pth   # 建议保持相同保存路径/home/code_qy_7_28/EDTformer/logs/default/2025-09-23_14-26-17
##svox
# python3 train.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets/ --eval_dataset_name=svox/images --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15

##msls
# python3 train.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader --eval_dataset_name=msls --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=15

