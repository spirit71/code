# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2, DinoV3, ResNet
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.datamodule import VPRDataModule
from src.eval_reporting import build_eval_report, save_eval_report

class HyperParams:
    def __init__(self):
        ## Backbone config:
        self.backbone_name: str = "dinov2_vitb14"    # resnet50, dinov2_vitb14, dinov3_vitb16
        self.unfreeze_n_blocks: int = 2              # number of blocks to unfreeze in the backbone
        self.dino_weights: str | None = None         # optional torch.hub weights name for dino backbones
        
        ## BoQ config:
        self.channel_proj: int = 512
        self.num_queries: int = 64
        self.num_layers: int = 2
        self.output_dim: int = 8192
        
        ## Datasets:
        # NOTE: if you already have OpenVPRLab, you can set the path to the datasets from there
        # otherwise use the dowload scripts in `scripts/` to download to `data/` folder 
        # self.gsv_cities_path: str = "../OpenVPRLab/data/train/gsv-cities"    # path to gsv-cities in OpenVPRLab
        self.gsv_cities_path: str = "/root/data/gsv_cities"    # path to gsv-cities in OpenVPRLab
        # gsv_cities_path: str = "./data/train/gsv-cities"                   # or path to gsv-cities in this project
        
        self.cities: str | list = "all" # train on all cities
        # self.cities: str | list = ["Bangkok", "Boston", "PRS"] # train on a subset of cities (check the gsv-cities folder)
        
        self.val_sets: dict = {
            "pitts30k-val": "./data/val/pitts30k-val",          # path to the pitts30k-val dataset
            # "msls-val":     "./data/val/msls-val",              # path to the msls-val dataset
            
        }
        # 2. [新增] 定义测试集 (训练结束后或测试模式下运行)
        # 这里的路径需要根据你实际存放数据集的位置修改
        self.test_sets: dict = {
            "pitts30k-test": "/root/data/Pittsburgh/pitts30k/",
            "msls-val":      "./data/val/msls-val",       # MSLS通常用Val作为Test
            "nordland":      "/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/images",
            "sped":          "/home/code_qy_7_28/VPR-datasets-downloader/datasets/sped/images",
            "amstertime":    "/home/code_qy_7_28/VPR-datasets-downloader/datasets/amstertime/images",
            "tokyo247": "/root/data/Tokyo247/images",
            # "eynsham":       "./data/test/eynsham",
            # "st_lucia":      "./data/test/st_lucia",
            "svox":          "/home/code_qy_7_28/VPR-datasets-downloader/datasets/svox/images", 
        }
        ## Metrics Config
        # 3. [新增] 定义需要计算的 Recall K 值
        # 注意：这需要传入 Model，且 Model 内部需要支持接收此参数
        self.recall_ks: list = [1, 5, 10, 20]
        ## Training config:
        self.batch_size: int = 128           # batch size is the number of places per batch
        self.img_per_place: int = 4          # number of images per place
        self.max_epochs: int = 60
        self.warmup_epochs: int =5    # number of linear warmup epochs (not iterations)
        self.lr: float = 1e-4                # learning rate
        self.weight_decay: float = 1e-4
        self.lr_mul: float = 0.1  #0.1 0.25 0.5
        self.scheduler_gamma = 0.1
        self.milestones: list = [20, 30]
        self.num_workers: int = 8
        self.top1_monitor: str = "pitts30k-val/R@1"  # metric used for checkpointing/early stopping
        self.early_stop_patience: int = 15            # stop if no Top-1 improvement for N val epochs
        self.early_stop_min_delta: float = 0.0       # minimum improvement to qualify as progress
        self.enable_early_stopping: bool = True
        self.eval_report_dir: str = "./logs/eval_reports"
        self.enable_console_file_log: bool = True  # mirror console stdout/stderr to log file 将控制台输出/错误输出镜像到日志文件中
        self.use_domain_routing = True
        self.num_query_banks = 4
        self.routing_balance_weight = 0.001
        self.routing_type= "delta"

        ## misc
        self.silent: bool = False            # disable console output
        self.compile: bool = False           # compile the model using torch.compile() [experimental]
        self.seed: int = 42                # random seed for reproducibility

def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)
    
    # Instantiate the backbone and define the image size for training and validation
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(
            backbone_name=hparams.backbone_name,
            unfreeze_n_blocks=hparams.unfreeze_n_blocks,
            weights=hparams.dino_weights,
        )
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name # in case the user passed dinov2 without the version
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size

    elif "dinov3" in hparams.backbone_name:
        backbone = DinoV3(
            backbone_name=hparams.backbone_name,
            unfreeze_n_blocks=hparams.unfreeze_n_blocks,
            weights=hparams.dino_weights,
        )
        train_img_size = (224, 224)
        val_img_size = (336, 336)
        hparams.backbone_name = backbone.backbone_name
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
        
    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, crop_last_block=True)
        train_img_size = (320, 320)
        val_img_size = (384, 384)
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
        
    else:
        raise ValueError(f"backbone {hparams.backbone_name} not recognized or not implemented!") 
    
    
    # Instantiate BoQ aggregator
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim//hparams.channel_proj,
        use_domain_routing=hparams.use_domain_routing,
        num_query_banks=hparams.num_query_banks,
        routing_type=hparams.routing_type,
    )
    
    # Define the entire Lightning model for training and validation
    model = BoQModel(
        backbone,
        aggregator,
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
        recall_ks=hparams.recall_ks,
        routing_balance_weight=hparams.routing_balance_weight,
    )
    # 冻结检查函数，防止“以为冻结了但其实没冻结
    def print_trainable_parameters(model):
        total = 0
        trainable = 0

        print("\n========== Trainable Parameter Check ==========")
        for name, param in model.named_parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
                print(f"[Trainable] {name}: {tuple(param.shape)}")

        print(f"Total params:     {total / 1e6:.2f} M")
        print(f"Trainable params: {trainable / 1e6:.2f} M")
        print(f"Trainable ratio:  {100 * trainable / total:.4f}%")
        print("================================================\n")


    def assert_backbone_frozen(backbone):
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                raise RuntimeError(
                    f"Backbone is not fully frozen! Trainable parameter found: {name}"
                )
        print("[OK] Backbone is fully frozen. Only aggregator should be trainable.")
    
    if hparams.compile:
        model = torch.compile(model)
    
    if "dinov3" in hparams.backbone_name and hparams.unfreeze_n_blocks == 0:
        assert_backbone_frozen(backbone)

    print_trainable_parameters(model)
    
    # Define the datamodule for handling training and validation datasets
    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        test_sets=hparams.test_sets,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )
    
    # If you want to display the datasets and training configs
    if not hparams.silent:
        datamodule.setup(stage="fit")       # first init train/val datasets only
        display_datasets_stats(datamodule)  # then display the stats
    
    # we use Tensorboard for logging (integrated with PyTorch Lightning)
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}",
        default_hp_metric=False
    )

    class _StreamTee:
        def __init__(self, stream, file_obj):
            self.stream = stream
            self.file_obj = file_obj

        def write(self, data):
            self.stream.write(data)
            self.file_obj.write(data)

        def flush(self):
            self.stream.flush()
            self.file_obj.flush()

        def isatty(self):
            return self.stream.isatty()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def _setup_console_file_logging():
        if not hparams.enable_console_file_log:
            return None, None
        log_dir = Path(tensorboard_logger.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        console_log_path = log_dir / "console.log"
        file_obj = open(str(console_log_path), "a", encoding="utf-8", buffering=1)
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamTee(orig_stdout, file_obj)
        sys.stderr = _StreamTee(orig_stderr, file_obj)
        print(f"[ConsoleLog] Mirroring console output to: {str(console_log_path)}")

        def _restore():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            file_obj.close()

        return _restore, str(console_log_path)
    
    # let's save all the hyperparameters to the the log file
    # this will be saved in the logs folder
    # e.g. ./logs/dinov2_vitb14/version_0/hparams.yaml
    tensorboard_logger.log_hyperparams(hparams.__dict__) 
    
    # Define the checkpointing callback
    # checkpointing = callbacks.ModelCheckpoint(
    #     monitor="msls-val/R@1",  # <==== monitor the Recall@1 on the msls-val dataset
    #     filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
    #     auto_insert_metric_name=False,
    #     save_weights_only=False,
    #     save_top_k=3,
    #     mode="max",
    # )
    checkpointing = callbacks.ModelCheckpoint(
        monitor=hparams.top1_monitor,  # monitor Top-1 recall
        filename="epoch[{epoch:02d}]_R@1[{pitts30k-val/R@1:.4f}]_R@5[{pitts30k-val/R@5:.4f}]_R@10[{pitts30k-val/R@10:.4f}]_R@20[{pitts30k-val/R@20:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        save_last=True,  # always keep a last.ckpt for resume
        mode="max",
    )
# 训练会在“指定轮次内 Top-1 无提升”时提前停止，核心是 EarlyStopping 监控 pitts30k-val/R@1（默认）
    early_stopping = None
    if hparams.enable_early_stopping and hparams.early_stop_patience > 0:
        early_stopping = callbacks.EarlyStopping(
            monitor=hparams.top1_monitor,
            mode="max",
            patience=hparams.early_stop_patience,
            min_delta=hparams.early_stop_min_delta,
            strict=False,
            check_on_train_epoch_end=False,
            verbose=not hparams.silent,
        )
    
    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()
    
    # Lightning Trainer will take a list of callbacks
    callback_list = [checkpointing]
    if early_stopping is not None:
        callback_list.append(early_stopping)
    if not hparams.silent:
        callback_list.append(program_bar)
    
    # Define the trainer
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
        # accumulate_grad_batches=4,
    )

    def _write_eval_report(requested_ckpt_path: str, resolved_ckpt_path: str):
        dataset_recalls = getattr(model, "last_test_recalls", {}) or {}
        report = build_eval_report(
            ckpt_path=resolved_ckpt_path,
            requested_ckpt_path=requested_ckpt_path,
            dataset_recalls=dataset_recalls,
            hparams_dict=hparams.__dict__,
            cli_args_dict=vars(args),
        )
        json_path, md_path = save_eval_report(report, report_dir=hparams.eval_report_dir)
        if not hparams.silent:
            print(f"[EvalReport] JSON: {json_path}")
            print(f"[EvalReport] Markdown: {md_path}")
    
    # # Train the model
    # trainer.fit(model=model, datamodule=datamodule)
# --- Execution Logic (Train or Test) ---

    restore_console, _ = _setup_console_file_logging()
    try:
        if args.test_only:
            # 5. 纯测试模式
            if not args.ckpt_path:
                raise ValueError("Please provide --ckpt_path when using --test_only")
            print(f"Starting testing using checkpoint: {args.ckpt_path}")
            trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
            _write_eval_report(
                requested_ckpt_path=args.ckpt_path,
                resolved_ckpt_path=args.ckpt_path,
            )
            
        else:
            # 6. 正常训练模式
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_ckpt)
            
            # 训练结束后，加载最好的模型进行测试
            print("Training finished. Starting testing with best checkpoint...")
            trainer.test(model, datamodule=datamodule, ckpt_path="best")
            best_path = checkpointing.best_model_path if checkpointing.best_model_path else "best"
            _write_eval_report(
                requested_ckpt_path="best",
                resolved_ckpt_path=best_path,
            )
    finally:
        if restore_console is not None:
            restore_console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")

    parser.add_argument("--dev",      action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--silent",   action="store_true", help="Disable console output.")
    parser.add_argument('--compile',  action='store_true', help='Compile the model using torch.compile()')
    
    parser.add_argument("--seed",   type=int,   help="Random seed for reproducibility.")
    
    parser.add_argument("--bs",     type=int,   help="Batch size.")
    parser.add_argument("--lr",     type=float, help="Learning Rate.")
    parser.add_argument("--wd",     type=float, help="Weight Decay.")
    
    parser.add_argument('--epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--warmup', type=int, help='Number of warmup epochs')
    parser.add_argument("--nw",     type=int, help="Numbers of workers.")

    parser.add_argument('--backbone',   type=str, help='Backbone model name [resnet50, dinov2_vitb14, dinov3_vitb16]')
    parser.add_argument('--dino_weights', type=str, default=None, help='Optional torch.hub weights argument for dino backbones.')
    parser.add_argument('--unfreeze_n', type=int, help='Number of blocks to unfreeze in the backbone.')
    parser.add_argument("--dim",        type=int, help="Output dimensionality.")
    parser.add_argument("--monitor", type=str, default=None, help="Metric name to monitor for checkpoint/early stop, e.g. pitts30k-val/R@1")
    parser.add_argument("--es_patience", type=int, default=None, help="Early stop patience in epochs with no Top-1 improvement.")
    parser.add_argument("--es_min_delta", type=float, default=None, help="Minimum Top-1 improvement to reset early-stop patience.")
    parser.add_argument("--es_disable", action="store_true", help="Disable early stopping.")
    parser.add_argument("--eval_report_dir", type=str, default=None, help="Directory to save evaluation reports (JSON + Markdown + index).")
    parser.add_argument("--no_console_file_log", action="store_true", help="Disable writing console output to logs/<run>/<version>/console.log")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Checkpoint path to resume training, e.g. .../checkpoints/last.ckpt")
     # 7. [新增] 命令行参数
    parser.add_argument("--test_only", action="store_true", help="Skip training and run testing only.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for testing.")
    parser.add_argument(
        "--use_domain_routing",
        action="store_true",
        help="Enable domain-routed query banks in BoQ."
    )

    parser.add_argument(
        "--num_query_banks",
        type=int,
        default=None,
        help="Number of query banks for domain-routed BoQ."
    )
    parser.add_argument(
    "--routing_type",
    type=str,
    default=None,
    choices=["delta"],
    help="Routing type. Currently support: delta."
    )

    parser.add_argument(
        "--routing_balance_weight",
        type=float,
        default=None,
        help="Weight for routing balance regularization loss."
    )
    parser.add_argument(
    "--milestones",
    type=int,
    nargs="+",
    default=None,
    help="Milestones for MultiStepLR, e.g. --milestones 20 30"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()
    
    if args.seed is not None:
        hparams.seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.bs is not None:
        hparams.batch_size = args.bs
    if args.lr is not None:
        hparams.lr = args.lr
    if args.wd is not None:
        hparams.weight_decay = args.wd
    if args.epochs is not None:
        hparams.max_epochs = args.epochs
    if args.warmup is not None:
        hparams.warmup_epochs = args.warmup
    if args.nw is not None:
        hparams.num_workers = args.nw
    if args.backbone:
        hparams.backbone_name = args.backbone
    if args.unfreeze_n is not None:
        hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim is not None:
        hparams.output_dim = args.dim
    if args.dino_weights:
        hparams.dino_weights = args.dino_weights
    if args.monitor:
        hparams.top1_monitor = args.monitor
    if args.es_patience is not None:
        hparams.early_stop_patience = args.es_patience  #（小于该提升不算进步）
    if args.es_min_delta is not None:
        hparams.early_stop_min_delta = args.es_min_delta
    if args.es_disable:
        hparams.enable_early_stopping = False  #关闭早停（只按 max_epochs 跑）
    if args.eval_report_dir:
        hparams.eval_report_dir = args.eval_report_dir
    if args.no_console_file_log:
        hparams.enable_console_file_log = False
    if args.use_domain_routing is not None:
        hparams.use_domain_routing = True

    if args.num_query_banks is not None:
        hparams.num_query_banks = args.num_query_banks
    if args.routing_type is not None:
        hparams.routing_type = args.routing_type

    if args.routing_balance_weight is not None:
        hparams.routing_balance_weight = args.routing_balance_weight
    if args.milestones is not None:
        hparams.milestones = args.milestones
    
    train(hparams, dev_mode=args.dev)
