
import os
import re
import random
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


def restore_scheduler_state(scheduler, optimizer, checkpoint, completed_steps):
    """【相对 Baseline 新增】恢复逐 step 调度器，且不重复衰减已恢复学习率。"""
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        return "checkpoint"
    scheduler.last_epoch = int(completed_steps)
    scheduler._step_count = int(completed_steps) + 1
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return "reconstructed"


def capture_rng_state(include_cuda=True):
    """【相对 Baseline 新增】保存各随机源状态，使数据采样和增强可以精确续接。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """【相对 Baseline 新增】初始化完成后恢复 Python、NumPy、CPU 和 CUDA 随机状态。"""
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG state count does not match visible GPUs: "
                f"{len(cuda_states)} != {torch.cuda.device_count()}")
        torch.cuda.set_rng_state_all(cuda_states)
    return True


def atomic_torch_save(state, destination):
    """【相对 Baseline 新增】先写临时文件再原子替换，避免中断留下损坏 checkpoint。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source, destination):
    """【相对 Baseline 新增】原子更新 best_model，避免复制中断破坏已有最佳权重。"""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    atomic_torch_save(state, model_path)
    if is_best:
        atomic_copy(model_path, join(args.save_dir, "best_model.pth"))


def save_epoch_model_checkpoint(args, model, epoch_num, pitts_recalls,
                                msls_recalls, score, is_best):
    """【相对 Baseline 新增】每个 epoch 保存仅含可部署 Student 的权重和验证指标。"""
    epochs_dir = Path(args.save_dir) / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    model_path = epochs_dir / f"epoch_{epoch_num:02d}_score_{score:.4f}.pth"
    pitts_recalls = np.asarray(pitts_recalls).tolist()
    msls_recalls = np.asarray(msls_recalls).tolist()
    atomic_torch_save({
        "epoch_num": int(epoch_num),
        "model_state_dict": model.state_dict(),
        "pitts_recalls": pitts_recalls,
        "recalls": msls_recalls,
        "selection_metric": "msls_val_r1_plus_r5",
        "selection_score": float(score),
        "is_best_at_save_time": bool(is_best),
    }, model_path)

    manifest_path = epochs_dir / "epoch_metrics.csv"
    rows = {}
    if manifest_path.is_file():
        with manifest_path.open(newline="") as file:
            for row in csv.DictReader(file):
                rows[int(row["epoch"])] = row
    rows[int(epoch_num)] = {
        "epoch": int(epoch_num),
        "selection_score": f"{score:.6f}",
        "pitts_r1": f"{pitts_recalls[0]:.6f}",
        "pitts_r5": f"{pitts_recalls[1]:.6f}",
        "msls_r1": f"{msls_recalls[0]:.6f}",
        "msls_r5": f"{msls_recalls[1]:.6f}",
        "is_best_at_save_time": int(bool(is_best)),
        "checkpoint": model_path.name,
    }
    fieldnames = [
        "epoch", "selection_score", "pitts_r1", "pitts_r5",
        "msls_r1", "msls_r5", "is_best_at_save_time", "checkpoint",
    ]
    with manifest_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[index] for index in sorted(rows))
    return model_path


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
    atomic_torch_save({
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


def resume_train(args, model, optimizer=None, strict=False, checkpoint=None):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    if checkpoint is None:
        checkpoint = torch.load(args.resume, map_location="cpu")
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    current_score = float(np.asarray(checkpoint.get("recalls", [0, 0]))[:2].sum())
    expected_policy = "run_msls_val_r1_plus_r5"
    is_legacy_multiview_checkpoint = (
        bool(getattr(args, "multiview_distill", False))
        and checkpoint.get("best_selection_policy") != expected_policy)
    reset_best_tracking = (
        bool(getattr(args, "reset_best_tracking_on_resume", False))
        or is_legacy_multiview_checkpoint)
    if reset_best_tracking:
        best_r1r5 = current_score
        not_improved_num = 0
        logging.info(
            "Reset best-model tracking from resumed checkpoint score %.4f "
            "(legacy_multiview_checkpoint=%s)",
            current_score, is_legacy_multiview_checkpoint)
    else:
        best_r1r5 = max(float(checkpoint["best_r5"]), current_score)
        not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r1r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        best_source = (args.resume if reset_best_tracking else
                       args.resume.replace("last_model.pth", "best_model.pth"))
        best_destination = join(args.save_dir, "best_model.pth")
        if os.path.abspath(best_source) != os.path.abspath(best_destination):
            atomic_copy(best_source, best_destination)
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
