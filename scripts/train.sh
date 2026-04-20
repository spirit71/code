export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python train.py \
  --train_datasets_folder /home/code_qy_7_28/VPR-datasets-downloader/datasets \
  --train_dataset_name pitts30k \
  --save_dir pitts30k_sage_try \
  --mining sage \
  --selection_metric r5 \
  --train_positives_dist_threshold 25 \
  --crossimage_encoder  \
  