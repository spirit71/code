import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup
from torchinfo import summary
import torchvision.models as models
import util
import test
import parser
import commons
import datasets_ws
import network as network
# import network_dinov2_l as network
# import network_copy as network

from loss import loss_function
from dataloaders.GSVCities import get_GSVCities

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import json

def extract_r_at_k(recalls: dict) -> dict:
    """
    Extract R@1/5/10/20 for the 'queries' entry and return Python-float metrics.
    This avoids JSON serialization issues (e.g., numpy.ndarray).
    Assumes recalls['queries'][0] is an array-like: [R@1, R@5, R@10, R@20, ...].
    """
    q = recalls.get("queries", None)
    if q is None:
        return {}

    # q could be list/ndarray; ensure we index safely
    q0 = q[0]

    def _to_float(x):
        # numpy scalar / torch scalar / python number -> float
        try:
            return float(x)
        except Exception:
            # e.g., torch tensor with one element
            try:
                return float(x.item())
            except Exception:
                return None

    metrics = {}
    # Index mapping based on typical VPR recalls order: 1, 5, 10, 20
    if len(q0) > 0: metrics["R@1"]  = _to_float(q0[0])
    if len(q0) > 1: metrics["R@5"]  = _to_float(q0[1])
    if len(q0) > 2: metrics["R@10"] = _to_float(q0[2])
    if len(q0) > 3: metrics["R@20"] = _to_float(q0[3])

    return metrics

#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.VPRNet(pretrained_foundation=True, foundation_model_path=args.foundation_model_path)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

args.features_dim = 4096

# Freeze parameters except adapter
for name, param in model.module.backbone.named_parameters():
    if "adapter" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# initialize Adapter
# 定位模型中所有 adapter模块中的 D_fc2线性层。

# 将其权重和偏置强制初始化为 0，通常是为了控制适配器在训练初期的行为（如残差连接的初始无扰动状态）。

# 这是一种​​特定场景下的初始化策略​​，常见于迁移学习或模块化神经网络设计。

for n, m in model.named_modules():# 遍历模型的所有子模块
    if 'adapter' in n:  # 如果子模块名称包含 'adapter'
        for n2, m2 in m.named_modules():# 进一步遍历该 adapter 的子模块
            if 'D_fc2' in n2: # 如果子模块名称包含 'D_fc2'
                if isinstance(m2, nn.Linear):# 确认该子模块是线性层
                    nn.init.constant_(m2.weight, 0.)# 权重初始化为0 ，暂时禁用？
                    nn.init.constant_(m2.bias, 0.)# 偏置初始化为0

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Resume model, optimizer, and other training parameters
# args.resume通常由用户在启动训练脚本时通过命令行参数（例如 --resume）指定，用以指示是否从之前的检查点恢复训练
if args.resume:
    model, optimizer, best_r1, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1 {best_r1:.1f}")
else:
    best_r1 = start_epoch_num = not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Getting GSVCities
train_dataset = get_GSVCities()

train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    #这个参数决定当数据集的大小不能被 batch_size整除时，是否​​丢弃最后一个不完整的批次​​（样本数少于 batch_size）
    'drop_last': False,
    #设置为 True时，​​可以加速数据从CPU内存到GPU显存的传输​​（因为固定内存允许更快的DMA拷贝），这在利用GPU训练时通常是一个好的实践，能提升训练效率
    'pin_memory': True,
    'shuffle': False}

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.7, last_epoch=-1)
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds):       
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w) #torch.Size([72, 4, 3, 224, 224]) 
        labels = place_id.view(-1)

        descriptors = model(images.to(args.device))     #backbone返回的x ([288, 4096])
        descriptors = descriptors.cuda()
        loss = loss_function(descriptors, labels) # Call the loss_function we defined above
        del descriptors

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        # ===================== Validation =====================
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    # Extract stable scalar metrics for logging/JSON
    val_metrics = extract_r_at_k(recalls)
    current_r1 = val_metrics.get("R@1", None)
    if current_r1 is None:
        # fallback (should not happen unless recalls format changed)
        current_r1 = float(recalls["queries"][0][0])

    is_best = current_r1 > best_r1

    # ===================== Update best & early-stop counter FIRST =====================
    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r1:.3f}, current R@1 = {current_r1:.3f}")
        best_r1 = current_r1
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.3f}, current R@1 = {current_r1:.3f}"
        )

    # ===================== Save checkpoint (now best_r1 is consistent) =====================
    util.save_checkpoint(
        args,
        {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,                  # 放在checkpoint里没问题（torch.save能存numpy/tensor）
            "best_r1": best_r1,                  # 注意：这里是更新后的 best
            "not_improved_num": not_improved_num
        },
        is_best,
        filename="last_model.pth"
    )

    # ===================== Save per-epoch weights (your existing behavior) =====================
    epoch_model_filename = f"model_epoch_{epoch_num:02d}.pth"
    epoch_model_path = join(args.save_dir, epoch_model_filename)
    torch.save(model.state_dict(), epoch_model_path)
    logging.info(f"Model weights for epoch {epoch_num:02d} saved to {epoch_model_path}")

    # ===================== Write best checkpoint pointer (NEW) =====================
    if is_best:
        # util.save_checkpoint typically creates args.save_dir/best_model.pth
        best_path = join(args.save_dir, "best_model.pth")
        with open(join(args.save_dir, "best_ckpt.txt"), "w") as f:
            f.write(best_path + "\n")
        with open(join(args.save_dir, "best_epoch.txt"), "w") as f:
            f.write(str(epoch_num) + "\n")

    # ===================== Write JSON metrics (NEW, safe serialization) =====================
    metrics_dir = join(args.save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    val_json_path = join(metrics_dir, "val_pitts30k.json")

    payload = {
        "dataset": str(args.eval_dataset_name),
        "split": "val",
        "epoch": int(epoch_num),
        "is_best": bool(is_best),
        "best_r1_so_far": float(best_r1),
        "metrics": val_metrics,         # 只包含 Python float
        "recalls_str": recalls_str
    }
    with open(val_json_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ===================== Early stop =====================
    if not is_best and not_improved_num >= args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break
    logging.info(f"[DEBUG] current_r1 scalar = {current_r1}, recalls_str = {recalls_str}")

logging.info(f"Best R@1: {best_r1:.3f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"), weights_only=False)["model_state_dict"]
model.load_state_dict(best_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

#### Test last model on test set
logging.info("Test *last* model on test set")
last_model_state_dict = torch.load(join(args.save_dir, "last_model.pth"), weights_only=False)["model_state_dict"]
model.load_state_dict(last_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
print(type(recalls), type(recalls["queries"]), type(recalls["queries"][0]))
logging.info(f"Recalls on {test_ds}: {recalls_str}")