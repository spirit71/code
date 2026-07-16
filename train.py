# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
from pathlib import Path
import re

import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2, ResNet
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.datamodule import VPRDataModule


class HyperParams:
    # 这个类把训练、验证、测试相关超参数集中放在一起，
    # 这样 train()、test-only、日志保存都共用同一份配置。
    def __init__(self):
        ## Backbone config:  主干网络相关设置
        self.backbone_name: str = "dinov2_vitb14"    # resnet18, resnet50, dinov2_vits14, dinov2_vitl14
        self.unfreeze_n_blocks: int = 2              # number of blocks to unfreeze in the backbone

        ## BoQ config:  BoQ 聚合器相关设置
        self.channel_proj: int = 512
        self.num_queries: int = 64
        self.num_layers: int = 2
        self.output_dim: int = 8192

        ## Datasets:  数据集路径设置
        # NOTE: if you already have OpenVPRLab, you can set the path to the datasets from there.
        # Otherwise use the download scripts in `scripts/` to download to the local data folder.
        self.gsv_cities_path: str = "/root/data/gsv_cities"

        self.cities: str | list = "all"  # 默认使用全部城市训练

        self.val_sets: dict = {
            "msls-val": "/home/code_qy_7_28/01code1/code/data/val/msls-val",
            "pitts30k-val": "/home/code_qy_7_28/01code1/code/data/val/pitts30k-val",
        }
        self.test_sets: dict = {
            "pitts30k-test": "/root/data/Pittsburgh/pitts30k",
            "nordland": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/images",
            "sped": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/sped/images",
            "amstertime": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/amstertime/images",
            "tokyo247": "/root/data/Tokyo247/images",
            "svox-all": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/svox/images",
        }

        ## Training config:  训练超参数
        self.batch_size: int = 128           # batch size is the number of places per batch
        self.eval_batch_size: int = 128      # eval-only batch size; can be larger than train if memory allows
        self.img_per_place: int = 4          # number of images per place
        self.max_epochs: int = 60
        self.warmup_epochs: int = 10         # number of linear warmup epochs (not iterations)
        self.lr: float = 1e-4                # learning rate
        self.weight_decay: float = 1e-4
        self.lr_mul: float = 0.1
        self.milestones: list = [10, 20]
        self.num_workers: int = 8

        ## Evaluation / logging config
        # 为 True 时，训练阶段会保存每个 epoch 的 checkpoint；
        # fit 结束后再统一评估所有 checkpoint，并导出逐轮 val/test 结果。
        self.test_every_epoch: bool = False

        ## misc
        self.silent: bool = False
        self.compile: bool = False
        self.seed: int = 42


def train(hparams, dev_mode=False, run_fit=True, run_test=True, test_ckpt_path="best", resume_ckpt_path=None):
    # Lightning 的 fit() 本身负责“训练 + 验证”。
    # 这里额外支持把 test set 拼进每轮验证阶段，实现“每个 epoch 后都全测一次”。
    seed_everything(hparams.seed, workers=True)

    # 先根据 backbone 类型决定网络和输入分辨率。
    # 不同 backbone 对训练/验证分辨率要求不一样。
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(
            backbone_name=hparams.backbone_name,
            unfreeze_n_blocks=hparams.unfreeze_n_blocks,
            crop_last_block=True,
        )
        train_img_size = (320, 320)
        val_img_size = (384, 384)
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
    else:
        raise ValueError(f"backbone {hparams.backbone_name} not recognized or not implemented!")

    # BoQ 聚合器把 backbone 输出的特征图压成最终全局描述子。
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim // hparams.channel_proj,
    )

    # BoQModel 是 Lightning 封装后的训练/验证/测试总入口。
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

    if hparams.compile:
        model = torch.compile(model)

    # 训练阶段默认只跑验证集，避免把全部测试集塞进每个 epoch 导致训练严重变慢。
    # 如果需要每轮 checkpoint 的全量测试结果，训练结束后再统一扫全部 checkpoint。
    periodic_test = False

    # DataModule 统一管理 train / val / test 三类数据集和 dataloader。
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        test_sets=hparams.test_sets,
        periodic_test=periodic_test,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        eval_batch_size=hparams.eval_batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )

    # 如果打开 periodic_test，这里也会提前构建测试集，
    # 这样启动时打印的数据统计里就能看到今晚到底会测哪些 benchmark。
    if not hparams.silent:
        datamodule.setup(stage="fit")
        display_datasets_stats(datamodule)

    # TensorBoardLogger 会创建 logs/<backbone>/version_x 目录，
    # checkpoint、summary、history 等文件后续都会落在附近。
    tensorboard_logger = TensorBoardLogger(
        save_dir="./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False,
    )
    tensorboard_logger.log_hyperparams(hparams.__dict__)

    # best checkpoint 仍然只按验证集 msls-val/R@1 来选，不看测试集指标。
    checkpointing = callbacks.ModelCheckpoint(
        monitor="msls-val/R@1",
        filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]_R@10[{msls-val/R@10:.4f}]_R@20[{msls-val/R@20:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        save_last=True,
        mode="max",
    )
    epoch_checkpointing = callbacks.ModelCheckpoint(
        filename="archive-epoch[{epoch:02d}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=-1,
        every_n_epochs=1,
    )

    program_bar = callbacks.RichProgressBar()
    callback_list = [checkpointing, epoch_checkpointing]
    if not hparams.silent:
        callback_list.append(program_bar)

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        precision="16-mixed",
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
    )

    if run_fit:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)

    if run_fit and hparams.test_every_epoch:
        evaluate_saved_checkpoints(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            silent=hparams.silent,
        )

    should_run_postfit_test = run_test and hparams.test_sets and (not run_fit or not hparams.test_every_epoch)
    if should_run_postfit_test:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt_path)


def _extract_epoch_index_from_checkpoint_name(ckpt_path: Path) -> int:
    match = re.search(r"epoch\[(\d+)\]", ckpt_path.name)
    if match is None:
        raise ValueError(f"Unable to parse epoch index from checkpoint name: {ckpt_path.name}")
    return int(match.group(1)) + 1


def evaluate_saved_checkpoints(trainer, model, datamodule, silent=False):
    logger = getattr(trainer, "logger", None)
    log_dir = getattr(logger, "log_dir", None)
    if not log_dir:
        return

    checkpoint_dir = Path(log_dir) / "checkpoints"
    checkpoint_paths = sorted(
        checkpoint_dir.glob("archive-epoch[[]*[]].ckpt"),
        key=_extract_epoch_index_from_checkpoint_name,
    )
    if not checkpoint_paths:
        return

    datamodule.setup(stage="test")

    for ckpt_path in checkpoint_paths:
        epoch_index = _extract_epoch_index_from_checkpoint_name(ckpt_path)
        model.forced_summary_epoch_index = epoch_index
        if not silent:
            print(f"\nEvaluating checkpoint epoch {epoch_index:02d}: {ckpt_path.name}")
        if datamodule.test_sets:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path), verbose=False)

    model.forced_summary_epoch_index = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--silent", action="store_true", help="Disable console output.")
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile().")

    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--bs", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--wd", type=float, help="Weight decay.")

    parser.add_argument("--epochs", type=int, help="Maximum number of epochs.")
    parser.add_argument("--warmup", type=int, help="Number of warmup epochs.")
    parser.add_argument("--nw", type=int, help="Number of workers.")
    parser.add_argument("--eval-bs", type=int, help="Evaluation batch size. Increase this to speed up val/test if GPU memory allows.")

    parser.add_argument("--backbone", type=str, help="Backbone model name [resnet50, dinov2].")
    parser.add_argument("--unfreeze_n", type=int, help="Number of blocks to unfreeze in the backbone.")
    parser.add_argument("--dim", type=int, help="Output dimensionality.")
    parser.add_argument("--output-dim", type=int, dest="output_dim", help="Output dimensionality. Alias of --dim.")
    parser.add_argument("--num-queries", type=int, help="Number of learnable BoQ queries.")

    parser.add_argument("--quick", action="store_true", help="Run a fast 1-epoch smoke test with a small training subset and only msls-val.")
    parser.add_argument("--no-test", action="store_true", help="Skip all test-set evaluation.")
    parser.add_argument("--test-only", action="store_true", help="Run only test-set evaluation without training.")
    parser.add_argument("--test-every-epoch", action="store_true", help="Save every epoch checkpoint during training, then after fit evaluate all checkpoints on test sets and export per-epoch test results.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for --test-only, or override the checkpoint used for testing.")
    parser.add_argument("--resume-from", type=str, help="Resume interrupted training from a checkpoint path, including optimizer/scheduler/global_step state.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test_only and args.quick:
        raise ValueError("--test-only and --quick cannot be used together")
    if args.test_only and not args.checkpoint:
        raise ValueError("--test-only requires --checkpoint")
    if args.test_only and args.resume_from:
        raise ValueError("--test-only and --resume-from cannot be used together")

    hparams = HyperParams()

    # quick 模式只用于快速 smoke check。
    # 它会缩小训练集、只保留一个验证集、关闭 test set，不适合正式 baseline 记录。
    if args.quick:
        hparams.cities = ["Bangkok", "Boston", "PRS"]
        hparams.val_sets = {"msls-val": hparams.val_sets["msls-val"]}
        hparams.test_sets = {}
        hparams.batch_size = 32
        hparams.eval_batch_size = 32
        hparams.num_workers = 4
        hparams.max_epochs = 1
        hparams.warmup_epochs = 1

    if args.seed:
        hparams.seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.bs:
        hparams.batch_size = args.bs
        hparams.eval_batch_size = args.bs
    if args.eval_bs:
        hparams.eval_batch_size = args.eval_bs
    if args.lr:
        hparams.lr = args.lr
    if args.wd:
        hparams.weight_decay = args.wd
    if args.epochs:
        hparams.max_epochs = args.epochs
    if args.warmup:
        hparams.warmup_epochs = args.warmup
    if args.nw:
        hparams.num_workers = args.nw
    if args.backbone:
        hparams.backbone_name = args.backbone
    if args.unfreeze_n:
        hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim and args.output_dim and args.dim != args.output_dim:
        raise ValueError("--dim and --output-dim must match when both are provided")

    requested_output_dim = args.output_dim if args.output_dim else args.dim
    if requested_output_dim:
        hparams.output_dim = requested_output_dim
    if args.num_queries:
        hparams.num_queries = args.num_queries

    recommended_output_dims = {96: 12288}
    if args.num_queries and requested_output_dim is None and args.num_queries in recommended_output_dims:
        recommended_dim = recommended_output_dims[args.num_queries]
        print(
            f"[INFO] Recommended --output-dim for --num-queries {args.num_queries}: {recommended_dim} "
            f"(current default: {hparams.output_dim})"
        )
    if args.test_every_epoch:
        hparams.test_every_epoch = True

    run_fit = not args.test_only
    run_test = (not args.no_test) or args.test_only
    test_ckpt_path = args.checkpoint if args.checkpoint else "best"
    resume_ckpt_path = args.resume_from if args.resume_from else None

    train(
        hparams,
        dev_mode=args.dev,
        run_fit=run_fit,
        run_test=run_test,
        test_ckpt_path=test_ckpt_path,
        resume_ckpt_path=resume_ckpt_path,
    )
