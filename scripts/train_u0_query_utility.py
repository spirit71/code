#!/usr/bin/env python3
"""在冻结 GSV O2/contribution cache 上训练最小 U0 query-only utility head。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class UtilityHead(nn.Module):
    def __init__(
        self, architecture: str = "u0", static_prior: torch.Tensor | None = None
    ):
        super().__init__()
        self.architecture = architecture
        if architecture == "u0":
            input_dim, hidden_dim = 512, 128
        elif architecture == "u1":
            input_dim, hidden_dim = 4 * 512, 256
        elif architecture in ("c0", "c1"):
            input_dim, hidden_dim = 512 + 6, 128
        else:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if architecture == "c1":
            if static_prior is None or static_prior.shape != (64,):
                raise ValueError("C1 requires a [64] static prior")
            self.register_buffer("static_prior", static_prior.float().clone())
            self.gamma = nn.Parameter(torch.ones(()))
            # 初始 residual 严格为 0，因此 optimizer 第一步前等价于 static。
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        expected_dim = 518 if self.architecture in ("c0", "c1") else 512
        if query.ndim != 3 or query.shape[1:] != (64, expected_dim):
            raise ValueError(
                f"expected [B,64,{expected_dim}], got {tuple(query.shape)}"
            )
        query = query.float()
        if self.architecture == "u1":
            # 图像内 64 个 query 的集合均值：[B,64,512] -> [B,1,512]。
            context = query.mean(dim=1, keepdim=True)
            # expand 仅创建广播视图；concat 后得到每个 query 的 2048 维上下文输入。
            context = context.expand_as(query)
            query = torch.cat(
                [query, context, query * context, query - context], dim=-1
            )
            if query.shape[1:] != (64, 2048):
                raise RuntimeError(f"bad U1 context shape: {tuple(query.shape)}")
        output = self.mlp(query).squeeze(-1)
        if self.architecture == "c1":
            output = output + self.gamma * self.static_prior[None]
        return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--train-cache", type=Path,
        help="可选的扁平训练缓存，需含 query_o2/contribution/image_ids",
    )
    parser.add_argument(
        "--val-cache", type=Path,
        help="可选的扁平验证缓存，需含 query_o2/contribution/image_ids",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("u0", "u1", "c0", "c1"), default="u0",
        help="u0=单query；u1=图像上下文；c0=候选统计；c1=static+候选残差",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--delta-scales", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_flat_cache(path: Path, split: str, architecture: str):
    item = torch.load(path, map_location="cpu", weights_only=True)
    query = item["query_o2"]
    if architecture in ("c0", "c1"):
        candidate_stats = item.get("candidate_stats")
        if (
            candidate_stats is None
            or candidate_stats.shape != (len(query), 64, 6)
            or item.get("top_k") != 20
            or item.get("gt_used_for_candidate_selection") is not False
        ):
            raise RuntimeError(f"{split}: invalid C0 candidate statistics")
        query = torch.cat([query, candidate_stats], dim=-1)
    target = item["contribution"]
    image_ids = item["image_ids"]
    expected_dim = 518 if architecture in ("c0", "c1") else 512
    if query.ndim != 3 or query.shape[1:] != (64, expected_dim):
        raise RuntimeError(f"{split}: bad flat query shape {tuple(query.shape)}")
    if target.shape != query.shape[:2] or len(image_ids) != len(query):
        raise RuntimeError(f"{split}: inconsistent flat cache")
    if not torch.isfinite(query).all() or not torch.isfinite(target).all():
        raise RuntimeError(f"{split}: non-finite flat cache")
    checkpoint_hash = item.get("checkpoint_sha256")
    if not checkpoint_hash:
        raise RuntimeError(f"{split}: missing checkpoint hash")
    return query, target, [split] * len(query), image_ids, checkpoint_hash


def load_split(root: Path, split: str):
    paths = sorted((root / "cities").glob(f"{split}_*.pt"))
    expected = 18 if split == "train" else 5
    if len(paths) != expected:
        raise RuntimeError(f"{split}: expected {expected} city caches, got {len(paths)}")
    queries, contribution, cities, image_ids = [], [], [], []
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=True)
        query = item["projection_inputs"][:, -1]
        target = item["contribution"]
        if query.shape != (512, 64, 512) or target.shape != (512, 64):
            raise RuntimeError(f"bad cache shape: {path}")
        if not torch.isfinite(query).all() or not torch.isfinite(target).all():
            raise RuntimeError(f"non-finite cache: {path}")
        queries.append(query)
        contribution.append(target)
        cities.extend([item["city"]] * 512)
        image_ids.extend(item["image_paths"])
    return torch.cat(queries), torch.cat(contribution), cities, image_ids


def rank_tensor(value: torch.Tensor, dim: int) -> torch.Tensor:
    return value.argsort(dim=dim).argsort(dim=dim).float()


def spearman_rows(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_rank, target_rank = rank_tensor(prediction, 1), rank_tensor(target, 1)
    pred_rank -= pred_rank.mean(1, keepdim=True)
    target_rank -= target_rank.mean(1, keepdim=True)
    return float(F.cosine_similarity(pred_rank, target_rank, dim=1).mean())


def spearman_pooled(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_rank = rank_tensor(prediction.flatten(), 0)
    target_rank = rank_tensor(target.flatten(), 0)
    pred_rank -= pred_rank.mean()
    target_rank -= target_rank.mean()
    return float(F.cosine_similarity(pred_rank[None], target_rank[None]))


def spearman_indices(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, list[float]]:
    pred_rank, target_rank = rank_tensor(prediction, 0), rank_tensor(target, 0)
    pred_rank -= pred_rank.mean(0, keepdim=True)
    target_rank -= target_rank.mean(0, keepdim=True)
    values = F.cosine_similarity(pred_rank, target_rank, dim=0)
    return float(values.mean()), [float(value) for value in values]


def pairwise_accuracy(prediction: torch.Tensor, target: torch.Tensor, delta: float) -> float:
    upper = torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1)
    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    valid = upper[None] & (target_difference.abs() > delta)
    if not valid.any():
        raise RuntimeError("no valid validation pairs")
    return float((
        torch.sign(target_difference[valid]) == torch.sign(prediction_difference[valid])
    ).float().mean())


def pairwise_loss(prediction: torch.Tensor, target: torch.Tensor, delta: float):
    upper = torch.triu(
        torch.ones(64, 64, dtype=torch.bool, device=prediction.device), diagonal=1
    )
    target_difference = target[:, :, None] - target[:, None, :]
    prediction_difference = prediction[:, :, None] - prediction[:, None, :]
    valid = upper[None] & (target_difference.abs() > delta)
    if not valid.any():
        raise RuntimeError("no valid training pairs")
    signs = torch.sign(target_difference[valid])
    return F.softplus(-signs * prediction_difference[valid]).mean(), signs.numel()


@torch.inference_mode()
def predict(model, query, device, batch_size):
    outputs = []
    for start in range(0, len(query), batch_size):
        outputs.append(model(query[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def static_metrics(train_target, val_target, delta):
    static_score = train_target.mean(0)[None].expand_as(val_target)
    return {
        "per_image_spearman": spearman_rows(static_score, val_target),
        "pooled_spearman": spearman_pooled(static_score, val_target),
        "pairwise_accuracy": pairwise_accuracy(static_score, val_target, delta),
    }


def build_static_rank_prior(train_target: torch.Tensor) -> torch.Tensor:
    """仅由训练 target 构造固定 slot prior；不读取 validation/test。"""
    prior = rank_tensor(train_target.mean(0), 0)
    prior = prior - prior.mean()
    prior = prior / prior.std(unbiased=False).clamp_min(1e-12)
    if prior.shape != (64,) or not torch.isfinite(prior).all():
        raise RuntimeError("invalid static rank prior")
    return prior


def full_metrics(model, train_query, train_target, val_query, val_target, delta, args):
    train_prediction = predict(model, train_query, args.device, args.batch_size)
    val_prediction = predict(model, val_query, args.device, args.batch_size)
    train_prediction_index_mean = train_prediction.mean(0)
    train_target_index_mean = train_target.mean(0)
    residual_prediction = val_prediction - train_prediction_index_mean[None]
    residual_target = val_target - train_target_index_mean[None]
    index_mean, index_values = spearman_indices(val_prediction, val_target)
    residual_index_mean, residual_index_values = spearman_indices(
        residual_prediction, residual_target
    )
    return {
        "per_image_spearman": spearman_rows(val_prediction, val_target),
        "pooled_spearman": spearman_pooled(val_prediction, val_target),
        "per_index_spearman_mean": index_mean,
        "per_index_spearman": index_values,
        "pairwise_accuracy": pairwise_accuracy(val_prediction, val_target, delta),
        "residual_pooled_spearman": spearman_pooled(
            residual_prediction, residual_target
        ),
        "residual_per_index_spearman_mean": residual_index_mean,
        "residual_per_index_spearman": residual_index_values,
        "train_prediction_index_mean": [
            float(value) for value in train_prediction_index_mean
        ],
    }


def train_one(
    train_query, train_target, val_query, val_target, val_image_ids,
    delta_scale, seed, args,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    delta = delta_scale * float(train_target.std(unbiased=False))
    output = args.output_root / f"delta_{delta_scale:g}" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    static_prior = (
        build_static_rank_prior(train_target)
        if args.architecture == "c1" else None
    )
    model = UtilityHead(args.architecture, static_prior).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # 冻结缓存约 0.6GB，H100 可一次容纳。一次上传后只做 GPU index，
    # 避免小 MLP 训练被逐 batch CPU collate/PCIe 搬运完全主导。
    train_query_device = train_query.to(args.device)
    train_target_device = train_target.to(args.device)
    generator = torch.Generator().manual_seed(seed)
    history, best_score, best_epoch = [], -math.inf, -1
    for epoch in range(args.epochs):
        model.train()
        loss_sum, pair_sum, image_sum = 0.0, 0, 0
        permutation = torch.randperm(len(train_query), generator=generator)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size].to(args.device)
            query = train_query_device[indices]
            target = train_target_device[indices]
            if query.requires_grad or target.requires_grad:
                raise RuntimeError("cache tensors unexpectedly require gradients")
            prediction = model(query)
            loss, pairs = pairwise_loss(prediction, target, delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Utility Head gradient missing or non-finite")
            optimizer.step()
            loss_sum += float(loss) * len(query)
            pair_sum += pairs
            image_sum += len(query)
        model.eval()
        val_prediction = predict(model, val_query, args.device, args.batch_size)
        val_score = spearman_rows(val_prediction, val_target)
        val_pair_accuracy = pairwise_accuracy(val_prediction, val_target, delta)
        row = {
            "epoch": epoch + 1, "train_loss": loss_sum / image_sum,
            "train_valid_pairs": pair_sum,
            "val_per_image_spearman": val_score,
            "val_pairwise_accuracy": val_pair_accuracy,
        }
        history.append(row)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch + 1
            torch.save({
                "state_dict": model.state_dict(), "seed": seed,
                "architecture": args.architecture,
                "delta_scale": delta_scale, "delta": delta,
                "epoch": best_epoch, "val_per_image_spearman": best_score,
            }, output / "best.pt")
    with (output / "train_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    metrics = full_metrics(
        model, train_query, train_target, val_query, val_target, delta, args
    )
    val_prediction = predict(model, val_query, args.device, args.batch_size)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_id", "query_index", "gt_contribution", "pred_utility",
            "gt_rank", "pred_rank",
        ])
        writer.writeheader()
        gt_rank = rank_tensor(val_target, 1).long()
        pred_rank = rank_tensor(val_prediction, 1).long()
        for image, image_id in enumerate(val_image_ids):
            for query_index in range(64):
                writer.writerow({
                    "image_id": image_id, "query_index": query_index,
                    "gt_contribution": float(val_target[image, query_index]),
                    "pred_utility": float(val_prediction[image, query_index]),
                    "gt_rank": int(gt_rank[image, query_index]),
                    "pred_rank": int(pred_rank[image, query_index]),
                })
    static = static_metrics(train_target, val_target, delta)
    result = {
        "seed": seed, "delta_scale": delta_scale, "delta": delta,
        "best_epoch": best_epoch, "metrics": metrics, "static": static,
        "architecture": (
            "Linear(512,128)-GELU-Linear(128,1)"
            if args.architecture == "u0"
            else (
                "ContextConcat(4x512)-Linear(2048,256)-GELU-Linear(256,1)"
                if args.architecture == "u1"
                else (
                    "CandidateConcat(512+6)-Linear(518,128)-GELU-Linear(128,1)"
                    if args.architecture == "c0"
                    else "StaticRankPrior+ZeroInitCandidateResidual(518,128,1)"
                )
            )
        ),
        "boq_participates": False,
        "only_utility_head_trainable": True,
    }
    if args.architecture == "c1":
        result["gamma"] = float(model.gamma.detach().cpu())
        result["static_prior"] = [
            float(value) for value in model.static_prior.detach().cpu()
        ]
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    using_flat = args.train_cache is not None or args.val_cache is not None
    if using_flat:
        if args.train_cache is None or args.val_cache is None:
            raise RuntimeError("--train-cache and --val-cache must be used together")
        train_query, train_target, train_cities, train_image_ids, train_hash = (
            load_flat_cache(args.train_cache, "train", args.architecture)
        )
        val_query, val_target, val_cities, val_image_ids, val_hash = (
            load_flat_cache(args.val_cache, "validation", args.architecture)
        )
        if train_hash != val_hash:
            raise RuntimeError("train/validation caches use different checkpoints")
    else:
        if args.cache_root is None:
            raise RuntimeError("--cache-root is required for legacy city caches")
        train_query, train_target, train_cities, train_image_ids = load_split(
            args.cache_root, "train"
        )
        val_query, val_target, val_cities, val_image_ids = load_split(
            args.cache_root, "val"
        )
        if set(train_cities) & set(val_cities):
            raise RuntimeError("train/validation cities overlap")
        train_hash = val_hash = None
    if args.smoke:
        train_query, train_target = train_query[:256], train_target[:256]
        val_query, val_target = val_query[:128], val_target[:128]
        val_image_ids = val_image_ids[:128]
        args.epochs = min(args.epochs, 2)
        args.delta_scales = args.delta_scales[:1]
        args.seeds = args.seeds[:1]
    results = []
    for delta_scale in args.delta_scales:
        for seed in args.seeds:
            print(f"[train] delta={delta_scale} seed={seed}", flush=True)
            results.append(train_one(
                train_query, train_target, val_query, val_target,
                val_image_ids, delta_scale, seed, args,
            ))
    grouped = []
    for delta_scale in args.delta_scales:
        selected = [result for result in results if result["delta_scale"] == delta_scale]
        for metric in (
            "per_image_spearman", "pooled_spearman", "per_index_spearman_mean",
            "pairwise_accuracy", "residual_pooled_spearman",
            "residual_per_index_spearman_mean",
        ):
            values = [result["metrics"][metric] for result in selected]
            static_values = [result["static"].get(metric, float("nan")) for result in selected]
            grouped.append({
                "delta_scale": delta_scale, "metric": metric,
                "predictor_mean": float(np.mean(values)),
                "predictor_std": float(np.std(values, ddof=0)),
                "static_mean": float(np.nanmean(static_values))
                if not all(math.isnan(value) for value in static_values) else "",
            })
    primary = {
        scale: np.mean([
            result["metrics"]["per_image_spearman"]
            for result in results if result["delta_scale"] == scale
        ])
        for scale in args.delta_scales
    }
    selected_delta = max(primary, key=primary.get)
    selected_results = [
        result for result in results if result["delta_scale"] == selected_delta
    ]
    u0_per_image = float(np.mean([
        result["metrics"]["per_image_spearman"] for result in selected_results
    ]))
    static_per_image = float(np.mean([
        result["static"]["per_image_spearman"] for result in selected_results
    ]))
    u0_pair = float(np.mean([
        result["metrics"]["pairwise_accuracy"] for result in selected_results
    ]))
    static_pair = float(np.mean([
        result["static"]["pairwise_accuracy"] for result in selected_results
    ]))
    residual_index = float(np.mean([
        result["metrics"]["residual_per_index_spearman_mean"]
        for result in selected_results
    ]))
    passes = (
        u0_per_image > static_per_image + 0.05
        and residual_index > 0.05
        and u0_pair > static_pair + 0.02
    )
    with (args.output_root / "delta_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    model_label = args.architecture.lower()
    summary = {
        "experiment": (
            "U0 query-only utility predictor"
            if args.architecture == "u0"
            else (
                "U1 query-set context utility predictor"
                if args.architecture == "u1"
                else (
                    "C0 candidate-aware utility predictor"
                    if args.architecture == "c0"
                    else "C1 static-prior candidate-residual utility predictor"
                )
            )
        ),
        "architecture_mode": args.architecture,
        "train_images": len(train_query), "val_images": len(val_query),
        "delta_scales": args.delta_scales, "seeds": args.seeds,
        "epochs": args.epochs, "selected_delta_scale": selected_delta,
        "selection_metric": "3-seed mean val per-image Spearman",
        "selected_metrics": {
            f"{model_label}_per_image_spearman_mean": u0_per_image,
            "static_per_image_spearman": static_per_image,
            f"{model_label}_pairwise_accuracy_mean": u0_pair,
            "static_pairwise_accuracy": static_pair,
            f"{model_label}_residual_per_index_spearman_mean": residual_index,
        },
        "masking_gate": {
            "required_u0_over_static_spearman": 0.05,
            "required_residual_per_index": 0.05,
            "required_u0_over_static_pairwise": 0.02,
            "passed": passes,
        },
        "test_data_used": False,
        "label_protocol": (
            "GSV global hard negatives -> MSLS-val full gallery"
            if using_flat else "legacy city caches"
        ),
        "checkpoint_sha256": train_hash,
        "boq_participates": False,
        "only_utility_head_trainable": True,
        "smoke": args.smoke,
        "runs": results,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        (
            "# U0 Query-only Utility Predictor"
            if args.architecture == "u0"
            else (
                "# U1 Query-set Context Utility Predictor"
                if args.architecture == "u1"
                else (
                    "# C0 Candidate-aware Utility Predictor"
                    if args.architecture == "c0"
                    else "# C1 Static Prior + Candidate Residual Utility Predictor"
                )
            )
        ), "",
        f"- selected delta scale: `{selected_delta}`",
        f"- masking gate passed: `{str(passes).lower()}`", "",
        f"| Metric | {model_label.upper()} | Static | Δ |",
        "|---|---:|---:|---:|",
        f"| per-image Spearman | {u0_per_image:.4f} | {static_per_image:.4f} | "
        f"{u0_per_image-static_per_image:+.4f} |",
        f"| pairwise accuracy | {u0_pair:.4f} | {static_pair:.4f} | "
        f"{u0_pair-static_pair:+.4f} |",
        f"| residual per-index Spearman | {residual_index:.4f} | — | — |",
    ]
    (args.output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["selected_metrics"], indent=2), flush=True)
    print(f"masking_gate_passed={passes}", flush=True)


if __name__ == "__main__":
    main()
