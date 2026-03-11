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
from scripts import analyze_adapters
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities
from torch.utils.tensorboard import SummaryWriter
import tensorboard
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")
writer = SummaryWriter(log_dir=args.save_dir)
logging.info(f"TensorBoard logs are being saved in {args.save_dir}")

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
        param.requires_grad = True   # 只有Adapter层的参数参与训练
    else:
        param.requires_grad = False  # 基础网络的参数被冻结

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
    #初始化用于记录最佳性能 best_r1、起始epoch数 start_epoch_num和“性能未提升计数” not_improved_num的变量，均设置为0。
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

# 用于记录统计信息
global_step = 0
log_interval = 50  # 每50个batch记录一次统计信息

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    # 用于累积统计信息
    epoch_stats = {
        'content_sim_mean': [],
        'geometry_bias_mean': [],
        'scale_ratio': [],
        'attn_entropy': [],
    }
          
    model = model.train()
    # 启用统计信息返回（每隔一定batch记录一次）
    model.module._return_stats = True
    
    epoch_losses=[]
    for batch_idx, (images, place_id) in enumerate(tqdm(ds)):       
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w) #torch.Size([72, 4, 3, 224, 224]) 
        labels = place_id.view(-1)          #place_id[[72, 4]]  #labels =[288]

        descriptors = model(images.to(args.device))     #backbone返回的x ([288, 4096])
        descriptors = descriptors.cuda()
        loss = loss_function(descriptors, labels) # Call the loss_function we defined above
        
        # 记录统计信息到TensorBoard
        if hasattr(model.module, '_decoder_stats') and model.module._decoder_stats is not None:
            if batch_idx % log_interval == 0:
                for layer_idx, layer_stats in enumerate(model.module._decoder_stats):
                    if layer_stats is not None:
                        # 记录各层统计信息
                        for key, value in layer_stats.items():
                            if key != 'layer_idx' and key != 'attn_weights_sample' and isinstance(value, (int, float)):
                                writer.add_scalar(f'GeometryAttention/Layer{layer_idx}/{key}', value, global_step)
                                # 累积到epoch统计
                                if key in epoch_stats:
                                    epoch_stats[key].append(value)
                        
                        # 记录attention权重热力图（每100个batch记录一次）
                        if batch_idx % (log_interval * 2) == 0 and 'attn_weights_sample' in layer_stats:
                            attn_weights = layer_stats['attn_weights_sample']
                            if attn_weights is not None and attn_weights.numel() > 0:
                                # 转换为numpy并归一化到[0,1]用于可视化
                                attn_np = attn_weights.numpy()
                                attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
                                writer.add_image(f'AttentionHeatmap/Layer{layer_idx}', 
                                               attn_np.reshape(1, *attn_np.shape), global_step, dataformats='CHW')
        
        del descriptors

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        
        # 记录loss和learning rate
        writer.add_scalar('Train/Loss', batch_loss, global_step)
        writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
        
        global_step += 1
        del loss
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # 记录epoch级别的统计信息
    writer.add_scalar('Epoch/Loss', epoch_losses.mean(), epoch_num)
    for key, values in epoch_stats.items():
        if len(values) > 0:
            writer.add_scalar(f'Epoch/GeometryAttention/{key}', np.mean(values), epoch_num)

    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    current_r1 = recalls['queries'][0][0]
    
    # 记录验证集指标
    writer.add_scalar('Val/Recall@1', current_r1, epoch_num)
    if len(recalls['queries']) > 0:
        for i, r in enumerate(recalls['queries'][0]):
            writer.add_scalar(f'Val/Recall@{i+1}', r, epoch_num)
    # is_best = recalls[0] > best_r1
    is_best = current_r1 > best_r1

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
                                "not_improved_num": not_improved_num
                                }, is_best, filename="last_model.pth")
    # --- 新增：在每个epoch后都保存模型权重 ---
    # 构建按epoch命名的模型权重文件名，例如 model_epoch_00.pth, model_epoch_01.pth
    epoch_model_filename = f"model_epoch_{epoch_num:02d}.pth"
    epoch_model_path = join(args.save_dir, epoch_model_filename)
    # 只保存模型的 state_dict，文件较小且灵活[1,4](@ref)
    torch.save(model.state_dict(), epoch_model_path)
    logging.info(f"Model weights for epoch {epoch_num:02d} saved to {epoch_model_path}")
    # --- 新增结束 ---

    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r1:.3f}, current R@1 = {current_r1:.3f}")
        best_r1 = current_r1
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.3f}, current R@1 = {current_r1:.3f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

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
logging.info(f"Recalls on {test_ds}: {recalls_str}")