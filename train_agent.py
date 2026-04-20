# -*- coding: UTF-8 -*-
"""
Training entrypoint for SAGE (triplet loss + periodic cache refresh).
Mining mode 'sage' approximates the paper's geo-visual graph + clique-style expansion on a candidate pool.
"""
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datetime import datetime
from os.path import join
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import logging
import numpy as np

import parser
import commons
import datasets_ws
import network
import test
import util

torch.backends.cudnn.benchmark = True

args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
torch.backends.cudnn.benchmark = True

args.features_dim = 8448
args.datasets_folder = args.train_datasets_folder

logging.info(f"Arguments: {args}")
logging.info(f"Outputs: {args.save_dir}")

if torch.cuda.device_count() >= 2 and args.criterion in ("sare_joint", "sare_ind"):
    raise NotImplementedError("SARE losses are not verified under DataParallel with multiple GPUs.")

triplets_ds = datasets_ws.TripletsDataset(
    args, args.train_datasets_folder, args.train_dataset_name, "train", args.negs_num_per_query
)
logging.info(f"Train triplets base: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.train_datasets_folder, args.train_dataset_name, "val")
logging.info(f"Val: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.train_datasets_folder, args.train_dataset_name, "test")
logging.info(f"Test: {test_ds}")

model = network.SAGE(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
else:
    raise ValueError(args.optim)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
else:
    raise NotImplementedError("Only --criterion triplet is wired in this repo; extend for sare_* if needed.")

if args.resume:
    model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resume epoch {start_epoch_num}, best R@5={best_r5:.1f}")
else:
    best_r5 = 0.0
    start_epoch_num = 0
    not_improved_num = 0

loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Epoch {epoch_num:02d}")
    epoch_start_time = datetime.now()
    epoch_losses = []

    for loop_num in range(loops_num):
        logging.debug(f"triplet cache {loop_num + 1}/{loops_num}")
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False

        triplets_dl = DataLoader(
            dataset=triplets_ds,
            num_workers=args.num_workers,
            batch_size=args.train_batch_size,
            collate_fn=datasets_ws.collate_fn,
            pin_memory=(args.device == "cuda"),
            drop_last=True,
        )

        model.train()
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100, leave=False):
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)

            features = model(images.to(args.device))
            loss_triplet = 0.0
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0
            )
            for triplets in triplets_local_indexes:
                q_idx, p_idx, n_idx = triplets.T
                loss_triplet += criterion_triplet(
                    features[q_idx], features[p_idx], features[n_idx]
                )
            del features
            loss_triplet /= args.train_batch_size * args.negs_num_per_query

            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            epoch_losses.append(loss_triplet.item())
            del loss_triplet

    if len(epoch_losses) == 0:
        logging.warning(
            "本 epoch 未产生任何训练 batch（例如 cache_refresh_rate 过小或 train_batch_size 过大）。"
        )
    else:
        logging.info(
            f"Epoch {epoch_num:02d} done in {str(datetime.now() - epoch_start_time)[:-7]}, "
            f"mean triplet loss={np.mean(epoch_losses):.4f}"
        )

    model.eval()
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Val {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5
    if is_best:
        logging.info(f"New best R@5: {recalls[1]:.1f} (was {best_r5:.1f})")
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"No improvement {not_improved_num}/{args.patience}; best R@5={best_r5:.1f}")

    util.save_training_checkpoint(
        args,
        {
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r5": best_r5,
            "not_improved_num": not_improved_num,
        },
        is_best,
    )

    if not_improved_num >= args.patience:
        logging.info("Early stopping.")
        break

best_path = join(args.save_dir, "best_model.pth")
if os.path.isfile(best_path):
    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"Test {test_ds}: {recalls_str}")
else:
    logging.warning("best_model.pth missing; skip final test.")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}, best R@5={best_r5:.1f}")
