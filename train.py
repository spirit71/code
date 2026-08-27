
import math
import copy
import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model

import warnings
warnings.filterwarnings("ignore")
import os

# 【相对 Baseline 新增】蒸馏目标版本用于阻止不兼容 checkpoint 混合续训。
MULTIVIEW_DISTILLATION_VERSION = 4


def _multiview_training_config(parsed_args):
    """Return trajectory-critical settings that must match on resume."""
    return {
        "backbone": parsed_args.backbone,
        "aggregation": parsed_args.aggregation,
        "training_dataset": parsed_args.training_dataset,
        "train_batch_size": parsed_args.train_batch_size,
        "resize": list(parsed_args.resize),
        "optimizer": parsed_args.optim,
        "learning_rate": parsed_args.lr,
        "seed": parsed_args.seed,
        "num_workers": parsed_args.num_workers,
        "patience": parsed_args.patience,
        "visible_gpu_count": torch.cuda.device_count(),
        "views_per_place": 4,
        "shuffle_training_places": False,
        "teacher_ema_decay": parsed_args.teacher_ema_decay,
        "teacher_view_temperature": parsed_args.teacher_view_temperature,
        "teacher_reliability_threshold": parsed_args.teacher_reliability_threshold,
        "teacher_margin_threshold": parsed_args.teacher_margin_threshold,
        "teacher_max_active_fraction": parsed_args.teacher_max_active_fraction,
        "teacher_reliability_power": parsed_args.teacher_reliability_power,
        "distill_temperature": parsed_args.distill_temperature,
        "distill_relation_topk": parsed_args.distill_relation_topk,
        "distill_warmup_epochs": parsed_args.distill_warmup_epochs,
        "distill_ramp_epochs": parsed_args.distill_ramp_epochs,
        "distill_weight": parsed_args.distill_weight,
        "distill_prototype_weight": parsed_args.distill_prototype_weight,
        "distill_relational_weight": parsed_args.distill_relational_weight,
        "distill_max_loss_ratio": parsed_args.distill_max_loss_ratio,
        "distill_max_grad_ratio": parsed_args.distill_max_grad_ratio,
    }


#### Initial setup: parser, logging...
args = parser.parse_arguments()
if args.multiview_distill and args.reset_best_tracking_on_resume:
    raise ValueError(
        "Safe multi-view v4 preserves the original run best-model history; "
        "--reset_best_tracking_on_resume is not allowed.")
resume_checkpoint = None
resume_rng_state = None
training_config = (
    _multiview_training_config(args) if args.multiview_distill else None)
# 【相对 Baseline 新增：严格续训】统一恢复 Student、Optimizer、Scheduler、AMP、EMA Teacher 和随机状态。
if args.resume:
    resume_checkpoint = torch.load(args.resume, map_location="cpu")
    if args.multiview_distill:
        checkpoint_version = resume_checkpoint.get("multiview_distillation_version")
        if checkpoint_version != MULTIVIEW_DISTILLATION_VERSION:
            raise ValueError(
                "Safe multi-view v4 requires a fresh run or a v4 resume "
                "checkpoint; supplied objective version is "
                f"{checkpoint_version!r}.")
        saved_config = resume_checkpoint.get("training_config")
        if saved_config is None:
            raise ValueError("Safe multi-view v4 checkpoint has no training_config")
        differing_keys = [
            key for key in training_config
            if saved_config.get(key) != training_config[key]]
        extra_keys = sorted(set(saved_config) - set(training_config))
        if differing_keys or extra_keys:
            details = {
                key: (saved_config.get(key), training_config.get(key))
                for key in differing_keys}
            raise ValueError(
                "Resume settings differ from the checkpoint and would change "
                f"the training trajectory: {details}; extra_keys={extra_keys}")
        resume_rng_state = resume_checkpoint.get("rng_state")
        if resume_rng_state is None:
            raise ValueError("Safe multi-view v4 checkpoint has no RNG state")
    if int(resume_checkpoint["epoch_num"]) + 1 >= args.epochs_num:
        print(
            f"Checkpoint already completed {int(resume_checkpoint['epoch_num']) + 1} "
            f"epochs (requested {args.epochs_num}); no training was started. "
            "Use the evaluation script to test its best_model.pth.")
        raise SystemExit(0)
start_time = datetime.now()
if args.resume and args.resume.endswith("last_model.pth"):
    args.save_dir = os.path.dirname(os.path.abspath(args.resume))
else:
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(
    args.save_dir,
    allow_existing=bool(args.resume and args.resume.endswith("last_model.pth")))
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Validation Datasets
val_ds0 = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "val")
logging.info(f"Val set0: {val_ds0}")
val_ds1 = datasets_ws.BaseDataset(args, args.datasets_folder, "msls", "val")
logging.info(f"Val set1: {val_ds1}")

#### Initialize model
model = network.GeoLocalizationNet(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

for name, param in model.module.backbone.named_parameters():
    if "adapter" not in name:
        param.requires_grad = False

## initialize Adapter
for n, m in model.named_modules():
    if 'adapter' in n:
        for n2, m2 in m.named_modules():
            if 'D_fc2' in n2:
                if isinstance(m2, nn.Linear):
                    nn.init.constant_(m2.weight, 0.)
                    nn.init.constant_(m2.bias, 0.)
        for n2, m2 in m.named_modules():
            if 'conv' in n2:
                if isinstance(m2, nn.Conv2d):
                    nn.init.constant_(m2.weight, 0.00001)
                    nn.init.constant_(m2.bias, 0.00001)

total = sum([param.nelement() for param in model.module.parameters()])
print("Number of model parameter: %.2fM" % (total/1e6))

total1 = sum([param.nelement() for param in model.module.backbone.parameters()])
print("Number of model backbone parameter: %.2fM" % (total1/1e6))
print("difference: %.2fM" % ((total-total1)/1e6))

# total2 = sum([param.nelement() for param in model.module.aggregation.parameters()])
# print("Number of aggregation parameter: %.2fM" % (total2/1e6))

total3 = sum([param.nelement() for param in model.module.parameters() if param.requires_grad])
print("Number of trainable parameter: %.2fM" % (total3/1e6))

total4 = sum([param.nelement() for param in model.module.backbone.parameters() if param.requires_grad])
print("Numer of trainable parameter introduced by fine-tuning/adaptation: %.2fM" % (total4/1e6))

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r1r5, start_epoch_num, not_improved_num = util.resume_train(
        args, model, optimizer, checkpoint=resume_checkpoint)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1+recall@5 {best_r1r5:.1f}")
else:
    best_r1r5 = start_epoch_num = not_improved_num = 0

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

if args.backbone == "dinov2-base" and args.aggregation == "gem":
    args.features_dim=2048
elif args.backbone == "dinov2-large" and args.aggregation == "gem":
    args.features_dim=4096
elif args.aggregation == "boq":
    args.features_dim=12288
elif args.aggregation == "salad":
    args.features_dim=8448
logging.info(f"Output dimension of the model is {args.features_dim}")

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}
if args.training_dataset == "gsv_cities":
    TRAIN_CITIES = [
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
else:
    TRAIN_CITIES = [
        "SFXL",
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
    citylist = [
        "Trondheim", 
        "Amsterdam",
        "Helsinki",
        "Tokyo",
        "Toronto",
        "Saopaulo",
        "Moscow",
        "Zurich",
        "Paris",
        "Budapest",
        "Austin",
        "Berlin",
        "Ottawa",
        "Goa",
        "Amman",
        "Nairobi",
        "Manila",
        "bangkok",
        "boston",
        "london",
        "melbourne",
        "phoenix",
        "Pitts30k",
    ]

    newcitylist = []
    for i in range(18):
        for cityname in citylist:
            if i==17 and (cityname=="Amman" or cityname=="Nairobi"):
                continue
            else:
                newcitylist.append(cityname+str(i))
    TRAIN_CITIES = TRAIN_CITIES + newcitylist

batch_size=args.train_batch_size
img_per_place=4
min_img_per_place=4
# Keep the original float-baseline place order. Changing this alters both the
# metric miner and every in-batch teacher relation.
shuffle_all=True
image_size=tuple(args.resize)
num_workers=args.num_workers
cities=TRAIN_CITIES
mean_std=IMAGENET_MEAN_STD
random_sample_from_each_place=True
mean_dataset = mean_std['mean']
std_dataset = mean_std['std']
train_transform = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
    T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_dataset, std=std_dataset),
])
teacher_transform = None
if args.multiview_distill:
    # 【相对 Baseline 新增】Teacher 使用 Resize+Normalize 弱增强；Student 继续使用 RandAugment 强增强。
    teacher_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_dataset, std=std_dataset),
    ])

train_loader_config = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': shuffle_all}

train_dataset = GSVCitiesDataset(
            cities=cities,
            img_per_place=img_per_place,
            min_img_per_place=min_img_per_place,
            random_sample_from_each_place=random_sample_from_each_place,
            transform=train_transform,
            teacher_transform=teacher_transform)
logging.info(f"Training dataset root: {train_dataset.base_path}")
logging.info(f"Training dataset shards: {len(cities)}")
logging.info(f"Training places: {len(train_dataset)}")
logging.info(f"Training images after filtering: {train_dataset.total_nb_images}")

# Multi-Similarity Loss and Miner
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)

        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss, miner_outputs

# loading training datasets
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=len(ds)*3, gamma=0.5)
if resume_checkpoint is not None:
    scheduler_restore_source = util.restore_scheduler_state(
        scheduler, optimizer, resume_checkpoint,
        completed_steps=start_epoch_num * len(ds))
    if scheduler_restore_source == "reconstructed":
        # Compatibility fallback for early v2 checkpoints. Do not pass
        # last_epoch to the constructor: it performs an implicit step and can
        # double-decay an already-restored optimizer at a decay boundary.
        logging.warning(
            "Resume checkpoint has no scheduler state; reconstructed StepLR "
            "at completed step %d without changing optimizer LR",
            start_epoch_num * len(ds))

# mixed precision training
from torch.cuda.amp import GradScaler,autocast
scaler = GradScaler()
if resume_checkpoint is not None:
    scaler_state = resume_checkpoint.get("scaler_state_dict")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    else:
        logging.warning(
            "Resume checkpoint has no AMP scaler state; using a fresh scaler")

# 【相对 Baseline 新增：训练期 EMA Teacher】完整复制 Student 但关闭梯度；完整 checkpoint 可保存 Teacher 供续训，但部署时只加载 Student，不增加推理分支。
teacher_model = None
if args.multiview_distill:
    from model.multiview_distillation import (
        build_place_set_teacher,
        conflict_safe_distillation,
        get_distillation_scale,
        privileged_multiview_distillation,
        update_ema_teacher,
    )

    teacher_model = copy.deepcopy(model)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    if args.resume:
        teacher_state_dict = resume_checkpoint.get("teacher_state_dict")
        if teacher_state_dict is not None:
            teacher_model.load_state_dict(teacher_state_dict)
            logging.info("Restored EMA multi-view teacher from checkpoint")
        else:
            logging.info(
                "Resume checkpoint has no teacher state; initialized teacher "
                "from the resumed student")
    logging.info(
        "Enabled privileged multi-view distillation: views/place=%d, "
        "ema=%.5f, temperature=%.3f, relation_topk=%d, reliability="
        "(intra_threshold=%.3f, margin_threshold=%.3f, power=%.2f), "
        "schedule=(warmup=%d, ramp=%d), loss weights=(total=%.3f, "
        "prototype=%.3f, relational=%.3f), max_loss_ratio=%.3f, "
        "max_grad_ratio=%.3f, max_active_fraction=%.3f",
        img_per_place,
        args.teacher_ema_decay,
        args.distill_temperature,
        args.distill_relation_topk,
        args.teacher_reliability_threshold,
        args.teacher_margin_threshold,
        args.teacher_reliability_power,
        args.distill_warmup_epochs,
        args.distill_ramp_epochs,
        args.distill_weight,
        args.distill_prototype_weight,
        args.distill_relational_weight,
        args.distill_max_loss_ratio,
        args.distill_max_grad_ratio,
        args.teacher_max_active_fraction)

if resume_checkpoint is not None:
    if resume_rng_state is not None:
        util.restore_rng_state(resume_rng_state)
        logging.info("Restored Python, NumPy, CPU and CUDA RNG states")
    del resume_checkpoint

#### Training loop
epoch_num = start_epoch_num - 1
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    distill_scale = 0.0
    if teacher_model is not None:
        # 【相对 Baseline 新增】先纯主任务预热，再同步 Teacher 并线性增大蒸馏权重，避免早期不可靠目标污染 Student。
        # At the warm-up boundary, discard the EMA mixture accumulated while
        # BoQ/adapters were still near initialization. Distillation therefore
        # starts from the fully warmed-up student, while later resumes retain
        # their checkpointed EMA teacher as usual.
        if epoch_num == args.distill_warmup_epochs:
            teacher_model.load_state_dict(model.state_dict())
            teacher_model.requires_grad_(False)
            teacher_model.eval()
            logging.info(
                "Hard-synchronized EMA teacher from warmed-up student before "
                "enabling privileged distillation")
        distill_scale = get_distillation_scale(
            epoch_num,
            args.distill_warmup_epochs,
            args.distill_ramp_epochs)
        logging.info(
            "Epoch %02d privileged distillation scale: %.4f",
            epoch_num, distill_scale)
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    model = model.train()
    epoch_losses=[]
    epoch_metric_losses=[]
    epoch_prototype_losses=[]
    epoch_relational_losses=[]
    epoch_teacher_reliability=[]
    epoch_positive_place_fractions=[]
    epoch_active_place_fractions=[]
    epoch_effective_place_fractions=[]
    epoch_teacher_margins=[]
    epoch_place_reliabilities=[]
    epoch_distillation_cap_scales=[]
    epoch_effective_distillation_losses=[]
    epoch_distillation_gradient_cosines=[]
    epoch_distillation_gradient_scales=[]
    epoch_distillation_gradient_conflicts=[]
    for batch in tqdm(ds):
        if teacher_model is not None:
            images, place_id, teacher_images = batch
        else:
            images, place_id = batch
            teacher_images = None
        
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)
        if teacher_images is not None:
            teacher_images = teacher_images.view(BS*N, ch, h, w)

        optimizer.zero_grad()
        with autocast():
            teacher_prototypes = leave_one_out_prototypes = None
            place_reliability = teacher_diagnostics = None
            if teacher_model is not None and distill_scale > 0.0:
                # 【相对 Baseline 新增】Teacher 将同地点 N 个弱增强描述子聚合为加权原型、留一原型和可靠性权重。
                # Run the no-grad teacher first so its intermediate activations
                # are released before the student builds its backward graph.
                with torch.no_grad():
                    teacher_descriptors = teacher_model(
                        teacher_images.to(args.device))
                    (teacher_prototypes, leave_one_out_prototypes,
                     place_reliability, _, teacher_diagnostics) = build_place_set_teacher(
                        teacher_descriptors,
                        num_places=BS,
                        views_per_place=N,
                        view_temperature=args.teacher_view_temperature,
                        reliability_threshold=args.teacher_reliability_threshold,
                        reliability_margin_threshold=args.teacher_margin_threshold,
                        max_active_fraction=args.teacher_max_active_fraction,
                        reliability_power=args.teacher_reliability_power)
                    del teacher_descriptors

            float_descriptors = model(images.to(args.device))
            metric_loss, miner_outputs = loss_function(float_descriptors, labels)
            loss = metric_loss

            distillation_terms = None
            if teacher_model is not None and distill_scale > 0.0:
                # 【相对 Baseline 新增】原型损失传递地点集合知识；关系损失传递困难负地点之间的相似度结构。
                distillation_terms = privileged_multiview_distillation(
                    float_descriptors,
                    teacher_prototypes,
                    leave_one_out_prototypes,
                    place_reliability,
                    views_per_place=N,
                    temperature=args.distill_temperature,
                    relation_topk=args.distill_relation_topk)
                distillation_loss = (
                    args.distill_prototype_weight
                    * distillation_terms["prototype_loss"]
                    + args.distill_relational_weight
                    * distillation_terms["relational_loss"])
                weighted_distillation_loss = (
                    args.distill_weight * distillation_loss)
                max_distillation_loss = (
                    args.distill_max_loss_ratio * metric_loss.detach())
                # 【第一层保护】蒸馏损失不得超过主度量损失的指定比例，防止异常 Teacher 目标主导训练。
                distillation_cap_scale = (
                    max_distillation_loss
                    / weighted_distillation_loss.detach().clamp_min(
                        torch.finfo(weighted_distillation_loss.dtype).eps)
                ).clamp(max=1.0)
                effective_distillation_loss = (
                    weighted_distillation_loss * distillation_cap_scale)
                # 【第二层保护】逐视图投影掉与主任务冲突的梯度分量，再限制蒸馏梯度相对主梯度的范数。
                safe_distillation_loss, gradient_diagnostics = (
                    conflict_safe_distillation(
                        metric_loss,
                        effective_distillation_loss,
                        float_descriptors,
                        max_gradient_ratio=args.distill_max_grad_ratio))
                loss = (
                    loss
                    + distill_scale * safe_distillation_loss)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss detected at epoch {epoch_num:02d}; "
                    f"loss={loss.detach().float().item()}")
            del float_descriptors
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if teacher_model is not None and distill_scale > 0.0:
            # 【相对 Baseline 新增】Student 更新后再用 EMA 更新 Teacher；Teacher 不反向传播，是 Student 历史状态的平滑集成。
            update_ema_teacher(
                teacher_model, model, decay=args.teacher_ema_decay)
        scheduler.step()   
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        epoch_metric_losses.append(metric_loss.detach().float().item())
        if distillation_terms is not None:
            epoch_prototype_losses.append(
                distillation_terms["prototype_loss"].detach().float().item())
            epoch_relational_losses.append(
                distillation_terms["relational_loss"].detach().float().item())
            epoch_teacher_reliability.append(
                distillation_terms["mean_reliability"].detach().float().item())
            epoch_positive_place_fractions.append(
                distillation_terms["positive_place_fraction"].detach().float().item())
            epoch_active_place_fractions.append(
                distillation_terms["active_place_fraction"].detach().float().item())
            epoch_effective_place_fractions.append(
                distillation_terms["effective_place_fraction"].detach().float().item())
            epoch_teacher_margins.append(
                teacher_diagnostics["discriminative_margin"].mean().item())
            epoch_place_reliabilities.append(
                place_reliability.detach().float().cpu().numpy())
            epoch_distillation_cap_scales.append(
                distillation_cap_scale.detach().float().item())
            epoch_effective_distillation_losses.append(
                effective_distillation_loss.detach().float().item())
            epoch_distillation_gradient_cosines.append(
                gradient_diagnostics["gradient_cosine"].float().item())
            epoch_distillation_gradient_scales.append(
                gradient_diagnostics["gradient_cap_scale"].float().item())
            epoch_distillation_gradient_conflicts.append(
                gradient_diagnostics["gradient_conflict"].float().item())
        del loss
        
    print(f"lr:{optimizer.param_groups[0]['lr']}")
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average total training loss = {epoch_losses.mean():.4f}")
    if teacher_model is not None and epoch_prototype_losses:
        # 【相对 Baseline 新增】汇总门控覆盖率、置信度、损失截断和梯度冲突，用于诊断蒸馏是否过弱或干扰主任务。
        reliability_quantiles = np.quantile(
            np.concatenate(epoch_place_reliabilities),
            [0.1, 0.25, 0.5, 0.75, 0.9])
        logging.info(
            "Distillation epoch %02d: metric=%.4f, prototype=%.4f, "
            "relational=%.4f, teacher_reliability=%.4f, positive_places=%.4f, "
            "active_places_r>=0.05=%.4f, effective_places=%.4f, "
            "teacher_margin=%.4f, reliability_q10/q25/q50/q75/q90="
            "%.4f/%.4f/%.4f/%.4f/%.4f, cap_scale=%.4f, "
            "effective_distill=%.4f, grad_cosine=%.4f, grad_cap_scale=%.4f, "
            "grad_conflict_fraction=%.4f, scale=%.4f",
            epoch_num,
            np.mean(epoch_metric_losses),
            np.mean(epoch_prototype_losses),
            np.mean(epoch_relational_losses),
            np.mean(epoch_teacher_reliability),
            np.mean(epoch_positive_place_fractions),
            np.mean(epoch_active_place_fractions),
            np.mean(epoch_effective_place_fractions),
            np.mean(epoch_teacher_margins),
            *reliability_quantiles,
            np.mean(epoch_distillation_cap_scales),
            np.mean(epoch_effective_distillation_losses),
            np.mean(epoch_distillation_gradient_cosines),
            np.mean(epoch_distillation_gradient_scales),
            np.mean(epoch_distillation_gradient_conflicts),
            distill_scale)
    elif teacher_model is not None:
        logging.info(
            "Distillation epoch %02d is in metric-only warm-up (scale=0)",
            epoch_num)
    
    # Compute recalls on validation set
    recalls0, recalls_str0 = test.test(args, val_ds0, model)
    logging.info(f"Recalls on val set0 {val_ds0}: {recalls_str0}")
    recalls1, recalls_str1 = test.test(args, val_ds1, model)
    logging.info(f"Recalls on val set1 {val_ds1}: {recalls_str1}")
    
    current_score = recalls1[0] + recalls1[1]
    # Match the original float baseline: best_model.pth is selected only by
    # the highest MSLS validation R@1 + R@5 observed in this training run.
    is_best = current_score > best_r1r5
    epoch_model_path = util.save_epoch_model_checkpoint(
        args, model, epoch_num, recalls0, recalls1, current_score, is_best)
    logging.info("Saved epoch %02d Student weights to %s", epoch_num,
                 epoch_model_path)
    
    # Save checkpoint, which contains all training parameters
    updated_best = current_score if is_best else best_r1r5
    updated_not_improved = 0 if is_best else not_improved_num + 1
    # 【相对 Baseline 新增：完整续训状态】除 Student/Optimizer 外，还保存 Scheduler、AMP、随机状态、目标版本、训练配置和 EMA Teacher。
    checkpoint_state = {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls1, "best_r5": updated_best,
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "not_improved_num": updated_not_improved,
        "multiview_distillation": bool(args.multiview_distill),
        "multiview_distillation_version": (
            MULTIVIEW_DISTILLATION_VERSION if args.multiview_distill else 0),
        "training_config": training_config,
        "rng_state": util.capture_rng_state(),
        "best_selection_policy": "run_msls_val_r1_plus_r5",
    }
    if teacher_model is not None:
        checkpoint_state["teacher_state_dict"] = teacher_model.state_dict()
    util.save_checkpoint(
        args, checkpoint_state, is_best, filename="last_model.pth")
    
    # If recall@1+recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@1+R@5 = {best_r1r5:.1f}, current R@1+R@5 = {(recalls1[0]+recalls1[1]):.1f}")
        best_r1r5 = current_score
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1+R@5 = {best_r1r5:.1f}, current R@1+R@5 = {(recalls1[0]+recalls1[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best R@1+R@5: {best_r1r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")

recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")
