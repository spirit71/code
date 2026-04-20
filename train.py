# -*- coding: UTF-8 -*-
"""Repo-adapted training entrypoint for SAGE.

This script is designed to live alongside the public SAGE repository without
modifying the original `parser.py` or `datasets_ws.py`. It uses:

- parser_train.py
- datasets_ws_train.py
- the public repo's network.py / test.py / config.py / commons.py / util.py

Expected usage:
    python train.py --train_datasets_folder /path/to/datasets --train_dataset_name pitts30k
"""

import copy
import math
import os
import shutil
import logging
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import parser_train as parser
import commons
import config
import datasets_ws_train as datasets_ws
import network
import test
import util


def save_training_checkpoint(args, state, is_best):
    """Save last checkpoint and optionally update best checkpoint."""
    os.makedirs(args.save_dir, exist_ok=True)
    last_path = join(args.save_dir, "last_model.pth")
    torch.save(state, last_path)
    if is_best:
        shutil.copy2(last_path, join(args.save_dir, "best_model.pth"))



def build_optimizer(args, model):
    if args.optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    raise ValueError(args.optim)



def metric_to_index(args, metric_name):
    mapping = {
        "r1": 1,
        "r5": 5,
        "r10": 10,
        "r20": 20,
        "r100": 100,
    }
    target = mapping[metric_name.lower()]
    if target not in args.recall_values:
        raise ValueError(f"selection metric {metric_name} not present in args.recall_values={args.recall_values}")
    return args.recall_values.index(target)



def make_eval_args(args, dataset_name):
    eval_args = copy.deepcopy(args)
    config.apply_config(eval_args, dataset_name)
    return eval_args



def main():
    torch.backends.cudnn.benchmark = True

    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True

    logging.info(f"Arguments: {args}")
    logging.info(f"Outputs: {args.save_dir}")

    if torch.cuda.device_count() >= 2 and args.criterion in ("sare_joint", "sare_ind"):
        raise NotImplementedError("SARE losses are not verified under DataParallel with multiple GPUs.")

    # Training datasets use train resize; val/test use repo config (typically 322x322).
    train_args = copy.deepcopy(args)
    val_args = make_eval_args(args, args.train_dataset_name)
    test_args = make_eval_args(args, args.train_dataset_name)

    triplets_ds = datasets_ws.TripletsDataset(
        train_args, train_args.train_datasets_folder, train_args.train_dataset_name, "train", train_args.negs_num_per_query
    )
    logging.info(f"Train triplets base: {triplets_ds}")

    val_ds = datasets_ws.BaseDataset(val_args, val_args.eval_datasets_folder, train_args.train_dataset_name, "val")
    logging.info(f"Val: {val_ds}")

    test_ds = datasets_ws.BaseDataset(test_args, test_args.eval_datasets_folder, train_args.train_dataset_name, "test")
    logging.info(f"Test: {test_ds}")

    model = network.SAGE(args)
    model = model.to(args.device)
    if args.device == "cuda":
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer(args, model)

    if args.criterion == "triplet":
        criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
    else:
        raise NotImplementedError("Only --criterion triplet is wired in this training script.")

    if args.resume:
        model, optimizer, best_metric, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
        logging.info(f"Resume epoch {start_epoch_num}, best tracked metric={best_metric:.4f}")
    else:
        best_metric = float("-inf")
        start_epoch_num = 0
        not_improved_num = 0

    if train_args.cache_refresh_rate > triplets_ds.queries_num:
        raise ValueError(
        f"cache_refresh_rate={train_args.cache_refresh_rate} is larger than queries_num={triplets_ds.queries_num}"
    )

    if train_args.neg_samples_num > triplets_ds.database_num:
        raise ValueError(
        f"neg_samples_num={train_args.neg_samples_num} is larger than database_num={triplets_ds.database_num}"
    )

    metric_idx = metric_to_index(val_args, args.selection_metric)
    loops_num = math.ceil(train_args.queries_per_epoch / train_args.cache_refresh_rate)

    for epoch_num in range(start_epoch_num, train_args.epochs_num):
        logging.info(f"Epoch {epoch_num:02d}")
        epoch_start_time = datetime.now()
        epoch_losses = []

        for loop_num in range(loops_num):
            logging.debug(f"triplet cache {loop_num + 1}/{loops_num}")
            triplets_ds.is_inference = True
            triplets_ds.compute_triplets(train_args, model)
            triplets_ds.is_inference = False

            triplets_dl = DataLoader(
                dataset=triplets_ds,
                num_workers=train_args.num_workers,
                batch_size=train_args.train_batch_size,
                collate_fn=datasets_ws.collate_fn,
                pin_memory=(train_args.device == "cuda"),
                drop_last=True,
            )

            model.train()
            for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100, leave=False):
                if train_args.horizontal_flip:
                    images = transforms.RandomHorizontalFlip()(images)

                features = model(images.to(train_args.device))
                loss_triplet = 0.0
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(train_args.train_batch_size, train_args.negs_num_per_query, 3),
                    1,
                    0,
                )
                for triplets in triplets_local_indexes:
                    q_idx, p_idx, n_idx = triplets.T
                    loss_triplet += criterion_triplet(features[q_idx], features[p_idx], features[n_idx])
                loss_triplet /= train_args.train_batch_size * train_args.negs_num_per_query

                optimizer.zero_grad()
                loss_triplet.backward()
                optimizer.step()

                epoch_losses.append(float(loss_triplet.item()))
                del loss_triplet, features

        if len(epoch_losses) == 0:
            logging.warning("This epoch produced no training batch. Check cache_refresh_rate/train_batch_size.")
        else:
            logging.info(
                f"Epoch {epoch_num:02d} done in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"mean triplet loss={np.mean(epoch_losses):.4f}"
            )

        model.eval()
        recalls, recalls_str = test.test(val_args, val_ds, model)
        logging.info(f"Val {val_ds}: {recalls_str}")

        current_metric = float(recalls[metric_idx])
        is_best = current_metric > best_metric
        if is_best:
            logging.info(
                f"New best {args.selection_metric.upper()}: {current_metric:.4f} "
                f"(was {best_metric if best_metric != float('-inf') else 'N/A'})"
            )
            best_metric = current_metric
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"No improvement {not_improved_num}/{args.patience}; "
                f"best {args.selection_metric.upper()}={best_metric:.4f}"
            )

        save_training_checkpoint(
            args,
            {
                "epoch_num": epoch_num + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "recalls": recalls,
                "best_r5": best_metric,
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
        recalls, recalls_str = test.test(test_args, test_ds, model)
        logging.info(f"Test {test_ds}: {recalls_str}")
    else:
        logging.warning("best_model.pth missing; skip final test.")

    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}, best {args.selection_metric.upper()}={best_metric:.4f}")


if __name__ == "__main__":
    main()
