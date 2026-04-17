import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark = True

import util
import test
import parser
import commons
import datasets_ws
import network_sage
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities
from geo_visual_sampler import GeoVisualGraphSampler

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_arguments_sage():
    parser = argparse.ArgumentParser(description="EDTformer with SAGE improvements")
    parser.add_argument("--train_batch_size", type=int, default=72)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=15)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument('--pca_dim', type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2)
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", 
                                 "five_crops", "nearest_crop", "maj_voting"])
    parser.add_argument("--majority_weight", type=float, default=0.01)
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25)
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10)
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+")
    parser.add_argument("--eval_datasets_folder", type=str, default=None)
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k")
    parser.add_argument("--pca_dataset_folder", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="default")
    
    parser.add_argument("--use_soft_probing", action='store_true', default=True,
                        help="Enable Soft Probing module from SAGE")
    parser.add_argument("--use_interact_head", action='store_true', default=True,
                        help="Enable InteractHead module from SAGE")
    parser.add_argument("--use_geo_sampling", action='store_true', default=False,
                        help="Enable Geo-Visual Graph sampling from SAGE")
    parser.add_argument("--soft_probing_alpha", type=float, default=0.5,
                        help="Alpha parameter for Soft Probing")
    parser.add_argument("--interact_num_segments", type=int, default=4,
                        help="Number of segments for InteractHead")
    parser.add_argument("--interact_num_heads", type=int, default=16,
                        help="Number of attention heads for InteractHead")
    parser.add_argument("--clique_size", type=int, default=4,
                        help="Clique size for Geo-Visual sampling")
    
    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("Set --eval_datasets_folder or export DATASETS_FOLDER")
    
    return args


def extract_r1_from_recalls(recalls_dict):
    """Extract scalar R@1 from test.py-style recall dict."""
    if not isinstance(recalls_dict, dict) or len(recalls_dict) == 0:
        raise ValueError(f"Unexpected recalls format: {type(recalls_dict)}")

    # Prefer the canonical "queries" split, otherwise fallback to first entry.
    split_key = "queries" if "queries" in recalls_dict else next(iter(recalls_dict))
    split_metrics = recalls_dict[split_key]
    if split_metrics is None or len(split_metrics) == 0:
        raise ValueError(f"Empty metrics for split: {split_key}")

    r_at_k = split_metrics[0]
    if len(r_at_k) == 0:
        raise ValueError(f"Empty R@K array for split: {split_key}")

    return float(r_at_k[0])


def train_sage(args):
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"EDTformer + SAGE Training")
    logging.info(f"SAGE features: SoftProbing={args.use_soft_probing}, "
                 f"InteractHead={args.use_interact_head}, "
                 f"GeoSampling={args.use_geo_sampling}")
    
    val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 
                                     args.eval_dataset_name, "val")
    logging.info(f"Val set: {val_ds}")
    
    test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 
                                      args.eval_dataset_name, "test")
    logging.info(f"Test set: {test_ds}")
    
    model = network_sage.VPRNetSAGE(
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path,
        use_soft_probing=args.use_soft_probing,
        use_interact_head=args.use_interact_head,
        soft_probing_alpha=args.soft_probing_alpha,
        interact_num_segments=args.interact_num_segments,
        interact_num_heads=args.interact_num_heads
    )
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)
    
    args.features_dim = 4096
    
    for name, param in model.module.backbone.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    momentum=0.9, weight_decay=0.001)
    
    if args.resume:
        model, optimizer, best_r1, start_epoch_num, not_improved_num = \
            util.resume_train(args, model, optimizer)
        logging.info(f"Resuming from epoch {start_epoch_num}")
    else:
        best_r1 = start_epoch_num = not_improved_num = 0
    
    logging.info(f"Output dimension: {args.features_dim}")
    
    train_dataset = get_GSVCities()
    
    train_loader_config = {
        'batch_size': args.train_batch_size,
        'num_workers': args.num_workers,
        'drop_last': False,
        'pin_memory': True,
        'shuffle': False
    }
    
    ds = DataLoader(dataset=train_dataset, **train_loader_config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(ds)*3, gamma=0.7, last_epoch=-1)
    
    graph_sampler = None
    if args.use_geo_sampling:
        graph_sampler = GeoVisualGraphSampler(clique_size=args.clique_size)
        logging.info(f"Geo-Visual Graph Sampler initialized with clique_size={args.clique_size}")
    
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")
        
        epoch_start_time = datetime.now()
        epoch_losses = []
        
        model = model.train()
        
        if graph_sampler is not None and epoch_num % 1 == 0:
            logging.info("Rebuilding geo-visual graph...")
            all_descriptors = []
            all_labels = []
            
            with torch.no_grad():
                for images, place_id in tqdm(ds, desc="Collecting descriptors"):
                    BS, N, ch, h, w = images.shape
                    images_flat = images.view(BS*N, ch, h, w)
                    desc = model(images_flat.to(args.device))
                    all_descriptors.append(desc.cpu())
                    all_labels.append(place_id)
            
            all_descriptors = torch.cat(all_descriptors, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            W = graph_sampler.compute_affinity_matrix(all_descriptors)
            seed_scores = graph_sampler.compute_seed_scores(W)
            best_seeds = torch.topk(seed_scores, min(10, len(seed_scores)))[1]
            logging.info(f"Top seed indices: {best_seeds.tolist()}")
        
        for images, place_id in tqdm(ds):
            BS, N, ch, h, w = images.shape
            images = images.view(BS*N, ch, h, w)
            labels = place_id.view(-1)
            
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cuda()
            loss = loss_function(descriptors, labels)
            
            del descriptors
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            del loss
        
        logging.info(f"Finished epoch {epoch_num:02d} in "
                     f"{str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"avg loss = {np.mean(epoch_losses):.4f}")
        
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
        
        current_r1 = extract_r1_from_recalls(recalls)
        is_best = current_r1 > best_r1
        
        util.save_checkpoint(args, {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r1": best_r1,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")
        
        if is_best:
            logging.info(f"Improved: R@1 = {best_r1:.3f} -> {current_r1:.3f}")
            best_r1 = current_r1
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num}/{args.patience}, "
                f"best R@1 = {best_r1:.3f}, current R@1 = {current_r1:.3f}"
            )
            if not_improved_num >= args.patience:
                logging.info("Early stopping.")
                break
    
    logging.info(f"Best R@1: {best_r1:.1f}")
    logging.info(f"Trained for {epoch_num+1} epochs")
    
    logging.info("Test best model on test set")
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
    model.load_state_dict(best_model_state_dict)
    recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
    
    return best_r1


if __name__ == "__main__":
    import argparse
    args = parse_arguments_sage()
    train_sage(args)
