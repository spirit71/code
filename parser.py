
import os
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="SelaVPR++",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=120,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference (caching and testing)")
    # parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
    #                     choices=["triplet", "sare_ind", "sare_joint"])
    # parser.add_argument("--margin", type=float, default=0.1,
    #                     help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=25,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=25) #12
    parser.add_argument("--lr", type=float, default=0.0004, help="_")
    # parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd", "adamw"])
    # parser.add_argument("--cache_refresh_rate", type=int, default=1000,
    #                     help="How often to refresh cache, in number of queries")
    # parser.add_argument("--queries_per_epoch", type=int, default=5000,
    #                     help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    # parser.add_argument("--negs_num_per_query", type=int, default=10,
    #                     help="How many negatives to consider per each query in the loss")
    # parser.add_argument("--neg_samples_num", type=int, default=1000,
    #                     help="How many negatives to use to compute the hardest ones")
    # parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2-large",
                        choices=["dinov2-large","dinov2-base"], help="_")
    parser.add_argument("--aggregation", type=str, default="gem", choices=["gem", "boq", "salad"])
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    # parser.add_argument('--fc_output_dim', type=int, default=None,
    #                     help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--hashing', action='store_true', help="_")
    parser.add_argument("--rerank", action="store_true", help="_")
    # 【相对 Baseline 新增：训练期专用】以下参数控制 EMA Teacher、可靠性门控、蒸馏损失和安全保护；不改变部署 Student。
    # Training-only privileged multi-view teacher. The deployed model remains
    # the same single-branch floating-point VPR network.
    parser.add_argument("--multiview_distill", action="store_true",
                        help="Enable EMA place-set teacher distillation")
    parser.add_argument("--teacher_ema_decay", type=float, default=0.999,
                        help="EMA decay used to update the multi-view teacher")
    parser.add_argument("--teacher_view_temperature", type=float, default=0.1,
                        help="Temperature for reliability-weighted teacher view pooling")
    parser.add_argument("--teacher_reliability_threshold", type=float, default=0.2,
                        help="Minimum mean cross-view cosine for teacher supervision")
    parser.add_argument("--teacher_margin_threshold", type=float, default=0.05,
                        help="Minimum intra-place minus nearest-negative similarity margin")
    parser.add_argument("--teacher_max_active_fraction", type=float, default=0.75,
                        help="Maximum fraction of places retained by the reliability gate")
    parser.add_argument("--teacher_reliability_power", type=float, default=2.0,
                        help="Power used to sharpen calibrated place reliability")
    parser.add_argument("--distill_temperature", type=float, default=0.2,
                        help="Temperature for cross-place relational distillation")
    parser.add_argument("--distill_relation_topk", type=int, default=16,
                        help="Number of hardest in-batch negative places used by relational distillation")
    parser.add_argument("--distill_warmup_epochs", type=int, default=3,
                        help="Metric-only epochs before enabling teacher supervision")
    parser.add_argument("--distill_ramp_epochs", type=int, default=5,
                        help="Epochs used to linearly ramp distillation to full weight")
    parser.add_argument("--distill_weight", type=float, default=1.0,
                        help="Weight of the complete privileged distillation objective")
    parser.add_argument("--distill_prototype_weight", type=float, default=0.1,
                        help="Weight of single-view to place-prototype consistency")
    parser.add_argument("--distill_relational_weight", type=float, default=1.0,
                        help="Weight of cross-place relational KL distillation")
    parser.add_argument("--distill_max_loss_ratio", type=float, default=0.05,
                        help="Maximum unscaled distillation loss as a fraction of metric loss")
    parser.add_argument("--distill_max_grad_ratio", type=float, default=0.10,
                        help="Maximum safe distillation descriptor-gradient norm relative to metric loss")
    parser.add_argument("--best_min_delta", type=float, default=1e-4,
                        help="Minimum MSLS R@1+R@5 increase required to replace best_model")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--training_dataset", type=str, default="gsv_cities", choices=["gsv_cities", "unified_dataset"],
                        help="Dataset for model training")
    # 【相对 Baseline 新增】断点恢复时控制最佳模型轨迹；safe-v4 默认保留历史最佳。
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--reset_best_tracking_on_resume", action="store_true",
                        help="Restart best-model tracking from the resumed checkpoint metrics")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--rerank_num", type=int, default=100, help="_")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    # 【相对 Baseline 新增】启动前校验蒸馏配置，拒绝不兼容分支和无效超参数。

    if args.multiview_distill and (args.hashing or args.rerank):
        raise ValueError(
            "Multi-view privileged distillation supports only the single-branch "
            "floating-point model (do not set --hashing or --rerank).")
    if not 0.0 <= args.teacher_ema_decay < 1.0:
        raise ValueError("--teacher_ema_decay must be in [0, 1)")
    for argument_name in (
            "teacher_view_temperature", "teacher_reliability_power",
            "distill_temperature"):
        if getattr(args, argument_name) <= 0:
            raise ValueError(f"--{argument_name} must be positive")
    if not -1.0 <= args.teacher_reliability_threshold < 1.0:
        raise ValueError("--teacher_reliability_threshold must be in [-1, 1)")
    if not -1.0 <= args.teacher_margin_threshold < 1.0:
        raise ValueError("--teacher_margin_threshold must be in [-1, 1)")
    if not 0.0 < args.teacher_max_active_fraction <= 1.0:
        raise ValueError("--teacher_max_active_fraction must be in (0, 1]")
    if args.distill_relation_topk <= 0:
        raise ValueError("--distill_relation_topk must be positive")
    if args.distill_warmup_epochs < 0:
        raise ValueError("--distill_warmup_epochs must be non-negative")
    if args.distill_ramp_epochs <= 0:
        raise ValueError("--distill_ramp_epochs must be positive")
    for argument_name in (
            "distill_weight", "distill_prototype_weight",
            "distill_relational_weight"):
        if getattr(args, argument_name) < 0:
            raise ValueError(f"--{argument_name} must be non-negative")
    if args.distill_max_loss_ratio <= 0:
        raise ValueError("--distill_max_loss_ratio must be positive")
    if args.distill_max_grad_ratio <= 0:
        raise ValueError("--distill_max_grad_ratio must be positive")
    if args.best_min_delta < 0:
        raise ValueError("--best_min_delta must be non-negative")
    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    # if args.queries_per_epoch % args.cache_refresh_rate != 0:
    #     raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
    #                      f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    criterion = getattr(args, "criterion", None)
    if torch.cuda.device_count() >= 2 and criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {criterion} loss.")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    return args
