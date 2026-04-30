# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse
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
    def __init__(self):
        ## Backbone config:
        self.backbone_name: str = "dinov2_vitb14"    # resnet18, resnet50, dinov2_vits14, dinov2_vitl14
        self.unfreeze_n_blocks: int = 2              # number of blocks to unfreeze in the backbone
        
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
        self.max_epochs: int = 40
        self.warmup_epochs: int = 10    # number of linear warmup epochs (not iterations)
        self.lr: float = 1e-4                # learning rate
        self.weight_decay: float = 1e-4
        self.lr_mul: float = 0.1
        self.milestones: list = [10, 20]
        self.num_workers: int = 8
        
        ## misc
        self.silent: bool = False            # disable console output
        self.compile: bool = False           # compile the model using torch.compile() [experimental]
        self.seed: int = 42                # random seed for reproducibility

def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)
    
    # Instantiate the backbone and define the image size for training and validation
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name # in case the user passed dinov2 without the version
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
    )
    
    if hparams.compile:
        model = torch.compile(model)
    
    
    
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
        monitor="pitts30k-val/R@1",  # <==== monitor the Recall@1 on the msls-val dataset
        filename="epoch[{epoch:02d}]_R@1[{pitts30k-val/R@1:.4f}]_R@5[{pitts30k-val/R@5:.4f}]_R@10[{pitts30k-val/R@10:.4f}]_R@20[{pitts30k-val/R@20:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=3,
        mode="max",
    )
    
    # Define the progress bar callback
    program_bar = callbacks.RichProgressBar()
    
    # Lightning Trainer will take a list of callbacks
    callback_list = [checkpointing]
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
    
    # # Train the model
    # trainer.fit(model=model, datamodule=datamodule)
# --- Execution Logic (Train or Test) ---
    
    if args.test_only:
        # 5. 纯测试模式
        if not args.ckpt_path:
            raise ValueError("Please provide --ckpt_path when using --test_only")
        print(f"Starting testing using checkpoint: {args.ckpt_path}")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        
    else:
        # 6. 正常训练模式
        trainer.fit(model=model, datamodule=datamodule)
        
        # 训练结束后，加载最好的模型进行测试
        print("Training finished. Starting testing with best checkpoint...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

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

    parser.add_argument('--backbone',   type=str, help='Backbone model name [resnet50, dinov2]')
    parser.add_argument('--unfreeze_n', type=int, help='Number of blocks to unfreeze in the backbone.')
    parser.add_argument("--dim",        type=int, help="Output dimensionality.")
     # 7. [新增] 命令行参数
    parser.add_argument("--test_only", action="store_true", help="Skip training and run testing only.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for testing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()
    
    if args.seed:
        hparams.seed = args.seed
    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.bs:
        hparams.batch_size = args.bs
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
    if args.dim:
        hparams.output_dim = args.dim
    
    train(hparams, dev_mode=args.dev)
