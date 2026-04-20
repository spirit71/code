# -*- coding: UTF-8 -*-
"""Training parser for repo-adapted SAGE experiments.

This file mirrors the public SAGE `parser.py` interface as much as possible,
while restoring the training arguments that are currently absent/commented out
in the public repository.
"""

import os
import argparse


def _default_datasets_folder() -> str | None:
    return os.environ.get("DATASETS_FOLDER")



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmarking Visual Geolocalization - training entrypoint for SAGE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=60,
        help="Number of triplets in a batch. Each triplet contains 1 query + 1 positive + N negatives.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "sgd", "adamw"])
    parser.add_argument("--epochs_num", type=int, default=50, help="Number of epochs to train for.")
    parser.add_argument(
        "--criterion",
        type=str,
        default="triplet",
        choices=["triplet", "sare_ind", "sare_joint"],
        help="Loss to use during training. This training script only implements triplet.",
    )
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for triplet loss.")
    parser.add_argument(
        "--cache_refresh_rate",
        type=int,
        default=1000,
        help="How often to refresh mining cache, in number of queries.",
    )
    parser.add_argument(
        "--queries_per_epoch",
        type=int,
        default=5000,
        help="How many queries to consider for one epoch. Ideally a multiple of cache_refresh_rate.",
    )
    parser.add_argument(
        "--negs_num_per_query",
        type=int,
        default=10,
        help="How many negatives to include per query in the loss.",
    )
    parser.add_argument(
        "--neg_samples_num",
        type=int,
        default=1000,
        help="How many database images to sample when mining hard negatives.",
    )
    parser.add_argument(
        "--mining",
        type=str,
        default="sage",
        choices=["partial", "full", "random", "msls_weighted", "sage"],
        help="Negative mining policy.",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="r5",
        choices=["r1", "r5", "r10", "r20", "r100"],
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--features_dim",
        type=int,
        default=8448,
        help="Output descriptor dimensionality expected from network.SAGE.",
    )

    # ------------------------------------------------------------------
    # Inference / evaluation parameters
    # ------------------------------------------------------------------
    parser.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="PCA dimension. If None, PCA is not used.",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=None,
        help="Output dimension of fully connected layer. If None, do not use an extra FC layer.",
    )

    # ------------------------------------------------------------------
    # Initialization / runtime parameters
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=4)

    # ------------------------------------------------------------------
    # Image / testing parameters
    # ------------------------------------------------------------------
    parser.add_argument("--resize", type=int, default=[224, 224], nargs=2, help="Training resize H W.")
    parser.add_argument(
        "--test_method",
        type=str,
        default="hard_resize",
        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
        help="Query pre/post-processing used during evaluation.",
    )
    parser.add_argument(
        "--majority_weight",
        type=float,
        default=0.01,
        help="Only for majority voting refinement.",
    )
    parser.add_argument("--efficient_ram_testing", action="store_true")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25)
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10)
    parser.add_argument(
        "--recall_values",
        type=int,
        default=[1, 5, 10, 20],
        nargs="+",
        help="Recall levels to compute.",
    )

    # ------------------------------------------------------------------
    # Data augmentation parameters
    # ------------------------------------------------------------------
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--contrast", type=float, default=0.0)
    parser.add_argument("--saturation", type=float, default=0.0)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--rand_perspective", type=float, default=0.0)
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--random_resized_crop", type=float, default=0.0)
    parser.add_argument("--random_rotation", type=float, default=0.0)

    # ------------------------------------------------------------------
    # SAGE-style mining parameters
    # ------------------------------------------------------------------
    parser.add_argument("--sage_vis_weight", type=float, default=0.7)
    parser.add_argument("--sage_geo_weight", type=float, default=0.3)
    parser.add_argument(
        "--sage_pool_size",
        type=int,
        default=64,
        help="Candidate pool size before greedy expansion in approximate SAGE mining.",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    parser.add_argument(
        "--train_datasets_folder",
        type=str,
        default=_default_datasets_folder(),
        help="Path containing training datasets laid out as datasets_vg/datasets/<dataset>/images/...",
    )
    parser.add_argument("--train_dataset_name", type=str, default="pitts30k")
    parser.add_argument(
        "--eval_datasets_folder",
        type=str,
        default=None,
        help="Path containing evaluation datasets. Defaults to train_datasets_folder when omitted.",
    )
    parser.add_argument("--eval_dataset_name", type=str, default="sped")
    parser.add_argument("--pca_dataset_folder", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="default", help="Run folder name, stored under ./logs/")
    parser.add_argument(
        "--eval_dataset_names",
        type=str,
        nargs="+",
        # default=["sped", "amstertime", "Msls_740", "pitts30k", "tokyo247", "pitts250k", "nordland", "eynsham"],
        default=["sped", "amstertime",  "pitts30k", "Tokyo247",  "Nordland", "eynsham"],
        help="List of datasets for multi-dataset evaluation (kept for compatibility).",
    )
    parser.add_argument("--crossimage_encoder", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--pca_dir",
        type=str,
        default="./pca_path",
        help="Directory or full filepath used to save/load PCA models.",
    )

    args = parser.parse_args()

    if args.train_datasets_folder is None:
        raise Exception(
            "Please provide --train_datasets_folder or export DATASETS_FOLDER, for example:\n"
            "export DATASETS_FOLDER=../datasets_vg/datasets"
        )

    if args.eval_datasets_folder is None:
        args.eval_datasets_folder = args.train_datasets_folder

    if args.cache_refresh_rate <= 0:
        raise ValueError("--cache_refresh_rate must be > 0")
    if args.queries_per_epoch <= 0:
        raise ValueError("--queries_per_epoch must be > 0")
    if args.negs_num_per_query <= 0:
        raise ValueError("--negs_num_per_query must be > 0")
    if args.neg_samples_num < args.negs_num_per_query:
        raise ValueError("--neg_samples_num must be >= --negs_num_per_query")

    args.train_batch_size = int(args.train_batch_size)
    return args
