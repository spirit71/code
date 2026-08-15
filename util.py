
import os
import re
import torch
import shutil
import logging
import numpy as np
import csv
from collections import OrderedDict
from pathlib import Path
from os.path import join
from sklearn.decomposition import PCA

import datasets_ws

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def save_topk_model_checkpoint(args, model, epoch_num, recalls, score, top_k=25):
    """Keep only the top-k model weights ranked by the validation score."""
    topk_dir = Path(args.save_dir) / "top25"
    topk_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for path in topk_dir.glob("epoch_*_score_*.pth"):
        try:
            saved_epoch = int(path.name.split("_")[1])
            saved_score = float(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        entries.append((saved_score, saved_epoch, path))

    if len(entries) >= top_k and score <= min(entries, key=lambda item: item[0])[0]:
        _write_topk_manifest(topk_dir, entries)
        return None

    model_path = topk_dir / f"epoch_{epoch_num:02d}_score_{score:.4f}.pth"
    torch.save({
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "recalls": np.asarray(recalls).tolist(),
        "selection_metric": "msls_val_r1_plus_r5",
        "selection_score": float(score),
    }, model_path)
    entries.append((float(score), int(epoch_num), model_path))

    while len(entries) > top_k:
        worst = min(entries, key=lambda item: item[0])
        worst[2].unlink()
        entries.remove(worst)

    _write_topk_manifest(topk_dir, entries)
    return model_path


def _write_topk_manifest(topk_dir, entries):
    manifest_path = topk_dir / "top25_ranking.csv"
    with manifest_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "epoch", "selection_score", "checkpoint"])
        for rank, (score, epoch, path) in enumerate(
                sorted(entries, key=lambda item: item[0], reverse=True), start=1):
            writer.writerow([rank, epoch, f"{score:.4f}", path.name])


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
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
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(state_dict)
    return model


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    current_score = float(np.asarray(checkpoint.get("recalls", [0, 0]))[:2].sum())
    best_r1r5 = max(float(checkpoint["best_r5"]), current_score)
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r1r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        best_source = args.resume.replace("last_model.pth", "best_model.pth")
        best_destination = join(args.save_dir, "best_model.pth")
        if os.path.abspath(best_source) != os.path.abspath(best_destination):
            shutil.copy(best_source, best_destination)
    return model, optimizer, best_r1r5, start_epoch_num + 1, not_improved_num


def compute_pca(args, model, pca_dataset_folder, full_features_dim):
    model = model.eval()
    pca_ds = datasets_ws.PCADataset(args, args.datasets_folder, pca_dataset_folder)
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
