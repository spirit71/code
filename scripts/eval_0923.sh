##amstertime
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=amstertime/images --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth

##SPED
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=sped/images --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth

##Tokyo247
python3 eval.py --eval_datasets_folder=/root/data --eval_dataset_name=Tokyo247/images --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth

##pitts30k
python3 eval.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k  --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth

##nordland
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/nordland_data  --eval_dataset_name=images_summer_as_quries --resume=./logs/default/2025-12-01_02-38-54/model_epoch_20.pth

##msls
# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader  --eval_dataset_name=msls --resume=./logs/default/2025-09-07_05-58-57/best_model.pth

##SVOX 
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets   --eval_dataset_name=svox/images --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth