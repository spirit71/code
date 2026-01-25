
import re
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from os.path import join
from sklearn.decomposition import PCA

import datasets_ws

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))

def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device,weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

# def resume_train(args, model, optimizer=None, strict=False):
#     """Load model, optimizer, and other training parameters"""
#     logging.debug(f"Loading checkpoint: {args.resume}")
#     checkpoint = torch.load(args.resume,weights_only=False)
#     start_epoch_num = checkpoint["epoch_num"]+1
#     model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
#     if optimizer:
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     best_r1 = checkpoint["best_r1"]
#     not_improved_num = checkpoint["not_improved_num"]
#     logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
#                   f"current_best_R@1 = {best_r1:.1f}")
#     if args.resume.endswith("best_model.pth"):  # Copy best model to current save_dir
#         shutil.copy(args.resume.replace("best_model.pth", "best_model.pth"), args.save_dir)
#     return model, optimizer, best_r1, start_epoch_num, not_improved_num

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, weights_only=False)
    
    # 检查检查点类型
    if "epoch_num" in checkpoint:
        # 完整的训练检查点
        start_epoch_num = checkpoint["epoch_num"] + 1
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_r1 = checkpoint["best_r1"]
        not_improved_num = checkpoint["not_improved_num"]
        logging.debug(f"Loaded training checkpoint: start_epoch_num = {start_epoch_num}, "
                      f"current_best_R@1 = {best_r1:.1f}")
    else:
        # 只有模型权重的检查点
        model.load_state_dict(checkpoint, strict=strict)
        
        # 从文件名推断epoch号，从下一轮开始
        import re
        match = re.search(r'epoch_(\d+)', args.resume)
        if match:
            epoch_from_file = int(match.group(1))
            start_epoch_num = epoch_from_file + 1  # 从下一轮开始
            best_r1 = 44.677  # 使用之前训练的最佳R@1值
            not_improved_num = 6
            logging.debug(f"Loaded model weights from epoch {epoch_from_file}, starting from epoch {start_epoch_num} with best R@1 = {best_r1}")
        else:
            # 如果无法从文件名推断，使用默认值
            start_epoch_num = 38  # 直接从第38轮开始
            best_r1 =  44.677
            not_improved_num = 6
            logging.debug(f"Starting from epoch {start_epoch_num} with best R@1 = {best_r1}")
    
    if args.resume.endswith("best_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("best_model.pth", "best_model.pth"), args.save_dir)
    
    return model, optimizer, best_r1, start_epoch_num, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.eval_datasets_folder, pca_dataset_folder)
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca
