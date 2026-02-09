##pitts30k
python3 eval.py --eval_datasets_folder=/root/data/Pittsburgh --eval_dataset_name=pitts30k  --resume=./logs/default/2026-01-31_10-39-54/best_model.pth
##pitts30k
# python3 eval.py --eval_datasets_folder=/root/data/Pittsburgh250k --eval_dataset_name=pitts30k  --resume=./logs/default/2025-09-23_14-26-17/best_model.pth

##SPED
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=sped/images --resume=./logs/default/2026-01-31_10-39-54/best_model.pth
# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=sped/images --resume=./logs/default/2025-09-23_14-26-17/best_model.pth

##amstertime
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=amstertime/images --resume=./logs/default/2026-01-31_10-39-54/best_model.pth
# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=amstertime/images --resume=./logs/default/2025-09-23_14-26-17/best_model.pth

##nordland
# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets  --eval_dataset_name=nordland/images --resume=./logs/default/2026-01-31_10-39-54/best_model.pth

##nordland
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland  --eval_dataset_name=images_winter_as_quries --resume=./logs/default/2026-01-31_10-39-54/best_model.pth

# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/nordland_data  --eval_dataset_name=images_summer_as_quries --resume=./logs/default/2025-09-28_04-33-24/model_epoch_15.pth
# python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/nordland_data  --eval_dataset_name=images_summer_as_quries --resume=/home/code_qy_7_28/EDTformer/logs/default/2025-10-03_10-30-41/best_model.pth

##Mapillary
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader  --eval_dataset_name=msls --resume=./logs/default/2026-01-31_10-39-54/best_model.pth

##SVOX 
python3 eval.py --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets   --eval_dataset_name=svox/images --resume=./logs/default/2026-01-31_10-39-54/best_model.pth

##Tokyo247
python3 eval.py --eval_datasets_folder=/root/data --eval_dataset_name=Tokyo247/images --resume=./logs/default/2026-01-31_10-39-54/best_model.pth
##Tokyo247
# python3 eval.py --eval_datasets_folder=/root/data --eval_dataset_name=Tokyo247/images --resume=./logs/default/2025-09-23_14-26-17/best_model.pth

PYTHONPATH=. python scripts/analyze_errors_attnmap.py   --resume ./logs/default/2026-01-31_10-39-54/best_model.pth   --save_dir ./logs/default/2026-01-31_10-39-54 --eval_datasets_folder=/home/code_qy_7_28/VPR-datasets-downloader/datasets