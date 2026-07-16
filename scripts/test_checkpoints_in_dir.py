#!/usr/bin/env python
"""Scan a log version directory and test its checkpoints on all test sets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



DEFAULT_VERSION_DIR = PROJECT_ROOT / "logs" / "dinov2_vitb14" / "version_16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints in a log version directory on configured test sets.",
    )
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=DEFAULT_VERSION_DIR,
        help=f"Log version directory that contains checkpoints/ (default: {DEFAULT_VERSION_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="archive-epoch[[]*[]].ckpt",
        help="Checkpoint glob under checkpoints/. Default scans per-epoch archive checkpoints only.",
    )
    parser.add_argument(
        "--all-ckpt",
        action="store_true",
        help="Scan every *.ckpt file, including last.ckpt and best/top-k checkpoints.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints whose evaluation_summary_epoch_XX.json already contains complete test metrics.",
    )
    parser.add_argument("--backbone", default="dinov2_vitb14", help="Backbone name used by the checkpoint.")
    parser.add_argument("--unfreeze-n", type=int, default=2, help="Number of unfrozen backbone blocks.")
    parser.add_argument("--output-dim", type=int, default=8192, help="BoQ output dimensionality.")
    parser.add_argument("--num-queries", type=int, default=64, help="Number of BoQ queries.")
    parser.add_argument("--eval-bs", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--nw", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=int, default=0, help="GPU device index.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU.")
    parser.add_argument("--silent", action="store_true", help="Disable progress bar and rich recall tables.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def epoch_index_from_checkpoint(ckpt_path: Path) -> int | None:
    match = re.search(r"epoch\[(\d+)\]", ckpt_path.name)
    if match is None:
        return None
    return int(match.group(1)) + 1


def sort_key(ckpt_path: Path) -> tuple[int, int | str]:
    epoch_index = epoch_index_from_checkpoint(ckpt_path)
    if epoch_index is None:
        return (1, ckpt_path.name)
    return (0, epoch_index)


def find_checkpoints(version_dir: Path, pattern: str, all_ckpt: bool) -> list[Path]:
    checkpoint_dir = version_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    glob_pattern = "*.ckpt" if all_ckpt else pattern
    checkpoint_paths = sorted(checkpoint_dir.glob(glob_pattern), key=sort_key)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched {glob_pattern!r} under {checkpoint_dir}")
    return checkpoint_paths


def summary_has_complete_test_metrics(summary_path: Path, expected_test_sets: dict) -> bool:
    if not summary_path.exists():
        return False

    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    test_recalls = summary.get("test")
    if not isinstance(test_recalls, dict):
        return False

    required_metrics = {"R@1", "R@5", "R@10", "R@20"}
    for dataset_name in expected_test_sets:
        dataset_recalls = test_recalls.get(dataset_name)
        if not isinstance(dataset_recalls, dict):
            return False
        if not required_metrics.issubset(dataset_recalls):
            return False
    return True


def build_model_and_datamodule(hparams):
    from src.backbones import DinoV2, ResNet
    from src.boq import BoQ
    from src.dataloaders.datamodule import VPRDataModule
    from src.model import BoQModel

    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(
            backbone_name=hparams.backbone_name,
            unfreeze_n_blocks=hparams.unfreeze_n_blocks,
        )
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name
    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(
            backbone_name=hparams.backbone_name,
            unfreeze_n_blocks=hparams.unfreeze_n_blocks,
            crop_last_block=True,
        )
        train_img_size = (320, 320)
        val_img_size = (384, 384)
    else:
        raise ValueError(f"Backbone {hparams.backbone_name!r} is not supported")

    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim // hparams.channel_proj,
    )
    model = BoQModel(
        backbone,
        aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        test_sets=hparams.test_sets,
        periodic_test=False,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        eval_batch_size=hparams.eval_batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )
    return model, datamodule


def logger_for_version_dir(version_dir: Path, backbone_name: str):
    from lightning.pytorch.loggers import TensorBoardLogger

    version_dir = version_dir.resolve()
    version_name = version_dir.name
    name = version_dir.parent.name
    save_dir = version_dir.parent.parent

    if name != backbone_name:
        print(f"[WARN] Version directory is under {name!r}, but --backbone is {backbone_name!r}.")

    version_match = re.fullmatch(r"version_(\d+)", version_name)
    version: int | str = int(version_match.group(1)) if version_match else version_name
    return TensorBoardLogger(save_dir=str(save_dir), name=name, version=version, default_hp_metric=False)


def main() -> None:
    args = parse_args()
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import RichProgressBar
    from train import HyperParams

    version_dir = args.version_dir.resolve()
    checkpoint_paths = find_checkpoints(version_dir, args.pattern, args.all_ckpt)

    hparams = HyperParams()
    hparams.backbone_name = args.backbone
    hparams.unfreeze_n_blocks = args.unfreeze_n
    hparams.output_dim = args.output_dim
    hparams.num_queries = args.num_queries
    hparams.eval_batch_size = args.eval_bs
    hparams.num_workers = args.nw
    hparams.silent = args.silent
    hparams.seed = args.seed

    seed_everything(hparams.seed, workers=True)
    model, datamodule = build_model_and_datamodule(hparams)
    logger = logger_for_version_dir(version_dir, hparams.backbone_name)

    trainer_kwargs = {
        "logger": logger,
        "callbacks": [] if args.silent else [RichProgressBar()],
        "enable_progress_bar": not args.silent,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 10,
    }
    if args.cpu:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1, "precision": "32-true"})
    else:
        trainer_kwargs.update({"accelerator": "gpu", "devices": [args.device], "precision": "16-mixed"})

    trainer = Trainer(**trainer_kwargs)
    datamodule.setup(stage="test")

    print(f"Testing {len(checkpoint_paths)} checkpoint(s) from {version_dir}")
    print(f"Results will be written to {logger.log_dir}")

    for ckpt_path in checkpoint_paths:
        epoch_index = epoch_index_from_checkpoint(ckpt_path)
        if epoch_index is None:
            print(f"\n[SKIP] Cannot parse epoch index from {ckpt_path.name}")
            continue

        summary_path = version_dir / f"evaluation_summary_epoch_{epoch_index:02d}.json"
        if args.skip_existing and summary_has_complete_test_metrics(summary_path, hparams.test_sets):
            print(f"[SKIP] epoch {epoch_index:02d}: {summary_path.name} already has complete test metrics")
            continue

        model.forced_summary_epoch_index = epoch_index
        print(f"\n[TEST] epoch {epoch_index:02d}: {ckpt_path.name}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path), verbose=False)

    model.forced_summary_epoch_index = None
    print("\nDone.")


if __name__ == "__main__":
    main()
