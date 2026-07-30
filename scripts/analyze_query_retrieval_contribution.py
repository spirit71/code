#!/usr/bin/env python3
"""固定 QRL-BoQ checkpoint 的 LOO query retrieval-contribution 离线诊断。

该脚本不训练模型，也不修改 reference gallery。它只在 query 侧逐个删除最后
一层 O2 query，复用原 BoQ FC 重建 descriptor，并测量 hardest-positive /
hardest-negative margin 的变化。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import utils
from src.analysis.reliability_diagnostics import (
    descriptor_from_query_outputs,
    mask_from_scores,
    random_mask,
)
from src.backbones import DinoV2
from src.boq import BoQ
from src.dataloaders.datamodule import TEST_DATASET_BUILDERS
from src.model import BoQModel
from src.query_reliability import attention_statistics


DATASET_ROOTS = {
    "pitts30k-test": "/root/data/Pittsburgh/pitts30k",
    "nordland": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/images",
    "sped": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/sped/images",
    "amstertime": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/amstertime/images",
    "tokyo247": "/root/data/Tokyo247/images",
    "svox-all": "/home/code_qy_7_28/VPR-datasets-downloader/datasets/svox/images",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-variant", choices=["e0", "e2"], default="e2")
    parser.add_argument("--gate-mode", choices=["on", "off"], default="on")
    parser.add_argument("--positive-mode", choices=["best", "hardest", "both"], default="both")
    parser.add_argument("--dataset", choices=DATASET_ROOTS, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--loo-query-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--random-repeats", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.batch_size < 1 or args.loo_query_batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch sizes must be positive and num_workers must be non-negative")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.mask_ratios):
        raise ValueError("mask ratios must be in [0,1]")
    if args.random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    if not args.device.startswith("cuda"):
        raise ValueError("the current implementation requires CUDA for tractable LOO evaluation")
    if args.model_variant == "e0" and args.gate_mode != "off":
        raise ValueError("E0 has no reliability gate; use --gate-mode off")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint_path: Path, device: str, model_variant: str, gate_mode: str) -> BoQModel:
    """按 E0/E2 真实结构建模并 strict 加载，禁止静默漏权重。"""
    backbone = DinoV2("dinov2_vitb14", unfreeze_n_blocks=2)
    use_qrl = model_variant == "e2"
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=512,
        num_queries=64,
        num_layers=2,
        row_dim=16,
        use_query_reliability=use_qrl,
        reliability_mode="learned",
        reliability_gate_mode="centered_residual" if gate_mode == "on" else "none",
    )
    model = BoQModel(backbone, aggregator)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a Lightning state_dict")
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    gc.collect()
    return model.eval().requires_grad_(False).to(device)


def assert_finite(name: str, tensor: torch.Tensor, shape_tail: tuple[int, ...] | None = None) -> None:
    if shape_tail is not None and tuple(tensor.shape[-len(shape_tail):]) != shape_tail:
        raise RuntimeError(f"{name} expected trailing shape {shape_tail}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or inf")


@torch.inference_mode()
def extract_features(model: BoQModel, dataset, loader: DataLoader, device: str):
    """单次前向提取冻结 descriptor、O1/O2、reliability 和 attention 统计。"""
    descriptors: dict[int, torch.Tensor] = {}
    raw_outputs: dict[int, torch.Tensor] = {}
    weighted_outputs: dict[int, torch.Tensor] = {}
    reliability: dict[int, torch.Tensor] = {}
    query_norm: dict[int, torch.Tensor] = {}
    entropy: dict[int, torch.Tensor] = {}
    max_attention: dict[int, torch.Tensor] = {}

    for images, indices in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            desc, aux = model(images.to(device, non_blocking=True), return_aux=True)
        raw = aux["query_outputs_raw"]
        weighted = aux["query_outputs_weighted"]
        scores = aux["reliability_scores"]
        attn_entropy, attn_max = attention_statistics(aux["last_attention"])
        norms = raw[:, -1].float().norm(dim=-1)

        assert_finite("descriptor", desc, (8192,))
        assert_finite("raw outputs", raw, (2, 64, 512))
        # E0 没有 QRL 分支，此时正式 projection input 就是 raw O1/O2。
        projection_outputs = weighted if weighted is not None else raw
        assert_finite("projection outputs", projection_outputs, (2, 64, 512))
        if scores is not None:
            assert_finite("reliability", scores, (64,))
        for row, index in enumerate(indices.tolist()):
            descriptors[index] = desc[row].float().cpu()
            raw_outputs[index] = raw[row].float().cpu()
            weighted_outputs[index] = projection_outputs[row].float().cpu()
            reliability[index] = (
                scores[row].float().cpu()
                if scores is not None else torch.full((raw.shape[2],), float("nan"))
            )
            query_norm[index] = norms[row].cpu()
            entropy[index] = attn_entropy[row].float().cpu()
            max_attention[index] = attn_max[row].float().cpu()

    expected = len(dataset)
    if len(descriptors) != expected:
        raise RuntimeError(f"extracted {len(descriptors)} of {expected} dataset items")
    stack = lambda mapping: torch.stack([mapping[index] for index in range(expected)])
    return tuple(stack(mapping) for mapping in (
        descriptors, raw_outputs, weighted_outputs, reliability, query_norm, entropy, max_attention
    ))


def positive_and_negative_stats(similarities: torch.Tensor, ground_truth, query_offset: int = 0):
    """按 benchmark GT 计算 hardest positive、best positive、hardest negative 和正样本 rank。"""
    batch, references = similarities.shape
    hard_pos = torch.empty(batch, dtype=similarities.dtype, device=similarities.device)
    best_pos = torch.empty_like(hard_pos)
    hard_neg = torch.empty_like(hard_pos)
    positive_rank = torch.empty(batch, dtype=torch.long, device=similarities.device)
    for row in range(batch):
        positives_np = np.asarray(ground_truth[query_offset + row], dtype=np.int64)
        if positives_np.size == 0:
            raise RuntimeError(f"query {query_offset + row} has no GT positives")
        positives = torch.as_tensor(positives_np, device=similarities.device)
        positive_scores = similarities[row, positives]
        hard_pos[row] = positive_scores.min()
        best_pos[row] = positive_scores.max()
        negative_mask = torch.ones(references, dtype=torch.bool, device=similarities.device)
        negative_mask[positives] = False
        if not negative_mask.any():
            raise RuntimeError("a query has no negatives")
        hard_neg[row] = similarities[row, negative_mask].max()
        positive_rank[row] = 1 + (similarities[row] > best_pos[row]).sum()
    return hard_pos, best_pos, hard_neg, positive_rank


def repeatability_against_gt(
    query_o2: torch.Tensor, reference_o2: torch.Tensor, ground_truth
) -> torch.Tensor:
    """用全部 GT reference 的 row-max query matching 均值近似训练期 repeatability target。"""
    query = torch.nn.functional.normalize(query_o2.float(), dim=-1)
    reference = torch.nn.functional.normalize(reference_o2.float(), dim=-1)
    result = torch.empty(query.shape[:2], dtype=torch.float32)
    for query_index, positives_np in enumerate(ground_truth):
        positives = torch.as_tensor(np.asarray(positives_np, dtype=np.int64))
        # [M,D] x [P,M,D] -> [P,M,M]，对目标 query 取 max，再对全部 GT view 平均。
        similarities = torch.einsum("md,pnd->pmn", query[query_index], reference[positives])
        result[query_index] = similarities.max(dim=-1).values.mean(dim=0)
    return result


@torch.inference_mode()
def loo_contribution(
    model: BoQModel,
    references: torch.Tensor,
    query_weighted: torch.Tensor,
    ground_truth,
    device: str,
    query_batch_size: int,
):
    """逐 O2 query 删除，返回 margin 及 contribution；所有相似度用 FP32。"""
    refs = references.to(device)
    queries, layers, num_queries, dim = query_weighted.shape
    if (layers, num_queries, dim) != (2, 64, 512):
        raise RuntimeError(f"unexpected query outputs shape {tuple(query_weighted.shape)}")

    baseline_desc = descriptor_from_query_outputs(query_weighted.to(device), model.aggregator.fc).float()
    baseline_sim = baseline_desc @ refs.T
    base_hard_pos, base_best_pos, base_hard_neg, base_rank = positive_and_negative_stats(
        baseline_sim, ground_truth
    )
    base_margin = base_hard_pos - base_hard_neg
    base_best_margin = base_best_pos - base_hard_neg

    masked_hard_margin = torch.empty(queries, num_queries, dtype=torch.float32)
    masked_best_margin = torch.empty_like(masked_hard_margin)
    masked_hard_pos = torch.empty_like(masked_hard_margin)
    masked_best_pos = torch.empty_like(masked_hard_margin)
    masked_hard_neg = torch.empty_like(masked_hard_margin)
    masked_rank = torch.empty(queries, num_queries, dtype=torch.long)
    for start in range(0, queries, query_batch_size):
        stop = min(start + query_batch_size, queries)
        source = query_weighted[start:stop].to(device)
        current = stop - start
        # [b,M,L,M,D]：每个副本只删除与副本索引相同的最后层 query。
        expanded = source[:, None].expand(current, num_queries, layers, num_queries, dim).clone()
        diagonal = torch.arange(num_queries, device=device)
        expanded[:, diagonal, -1, diagonal] = 0
        masked_desc = descriptor_from_query_outputs(
            expanded.reshape(current * num_queries, layers, num_queries, dim),
            model.aggregator.fc,
        ).float()
        similarities = (masked_desc @ refs.T).reshape(current, num_queries, -1)
        for local in range(current):
            hp, bp, hn, rank = positive_and_negative_stats(
                similarities[local], [ground_truth[start + local]] * num_queries
            )
            masked_hard_pos[start + local] = hp.cpu()
            masked_best_pos[start + local] = bp.cpu()
            masked_hard_neg[start + local] = hn.cpu()
            masked_rank[start + local] = rank.cpu()
            masked_hard_margin[start + local] = (hp - hn).cpu()
            masked_best_margin[start + local] = (bp - hn).cpu()
        del expanded, masked_desc, similarities

    contribution_hard = base_margin.cpu().unsqueeze(1) - masked_hard_margin
    contribution_best = base_best_margin.cpu().unsqueeze(1) - masked_best_margin
    return {
        "baseline_descriptor": baseline_desc.cpu(),
        "baseline_hard_positive": base_hard_pos.cpu(),
        "baseline_best_positive": base_best_pos.cpu(),
        "baseline_hard_negative": base_hard_neg.cpu(),
        "baseline_hard_margin": base_margin.cpu(),
        "baseline_best_margin": base_best_margin.cpu(),
        "baseline_rank": base_rank.cpu(),
        "masked_hard_positive": masked_hard_pos,
        "masked_best_positive": masked_best_pos,
        "masked_hard_negative": masked_hard_neg,
        "masked_hard_margin": masked_hard_margin,
        "masked_best_margin": masked_best_margin,
        "masked_rank": masked_rank,
        "contribution_hard": contribution_hard,
        "contribution_best": contribution_best,
    }


def per_image_spearman(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    values = []
    for row in range(left.shape[0]):
        if left[row].max() == left[row].min() or right[row].max() == right[row].min():
            continue
        correlation = spearmanr(left[row].numpy(), right[row].numpy()).statistic
        if math.isfinite(correlation):
            values.append(float(correlation))
    pooled = (
        spearmanr(left.flatten().numpy(), right.flatten().numpy()).statistic
        if left.max() != left.min() and right.max() != right.min() else float("nan")
    )
    return float(np.mean(values)), float(pooled)


@torch.inference_mode()
def masking_rows(model, dataset, refs, weighted, reliability, repeatability, stats, args):
    """实验 B：保持 reference gallery 不变，仅改变 query 侧 O2。"""
    rows = []
    strategies = {"repeatability_low": (repeatability, False)}
    if torch.isfinite(reliability).all():
        strategies["predicted_low"] = (reliability, False)
    modes = ("best", "hardest") if args.positive_mode == "both" else (args.positive_mode,)
    # 单正样本数据集上 best/hard 完全相同，只评测一套 oracle Recall，避免重复耗时。
    masking_modes = modes
    if modes == ("best", "hardest") and torch.equal(
        stats["contribution_best"], stats["contribution_hard"]
    ):
        masking_modes = ("best",)
    for mode in masking_modes:
        contribution = stats["contribution_best" if mode == "best" else "contribution_hard"]
        strategies[f"contribution_{mode}_low"] = (contribution, False)
        strategies[f"contribution_{mode}_high"] = (contribution, True)
    for ratio in [0.0, *args.mask_ratios]:
        for strategy, (strategy_scores, remove_high) in strategies.items():
            keep = mask_from_scores(strategy_scores, ratio, remove_high)
            masked = weighted.clone()
            masked[:, -1] *= keep.unsqueeze(-1)
            desc = descriptor_from_query_outputs(masked.to(args.device), model.aggregator.fc).cpu()
            recalls = utils.compute_recall_performance(
                torch.cat((refs, desc), dim=0),
                dataset.num_references,
                dataset.num_queries,
                dataset.ground_truth,
                k_values=[1, 5, 10, 20],
            )
            rows.append({"strategy": strategy, "mask_ratio": ratio, "repeat": "", **{
                f"R@{k}": float(value) for k, value in recalls.items()
            }})
        for repeat in range(args.random_repeats if ratio else 1):
            keep = random_mask(weighted.shape[0], weighted.shape[2], ratio, args.seed + repeat)
            masked = weighted.clone()
            masked[:, -1] *= keep.unsqueeze(-1)
            desc = descriptor_from_query_outputs(masked.to(args.device), model.aggregator.fc).cpu()
            recalls = utils.compute_recall_performance(
                torch.cat((refs, desc), dim=0), dataset.num_references, dataset.num_queries,
                dataset.ground_truth, k_values=[1, 5, 10, 20],
            )
            rows.append({"strategy": "random", "mask_ratio": ratio,
                         "repeat": repeat if ratio else "", **{
                f"R@{k}": float(value) for k, value in recalls.items()
            }})
    return rows


def write_outputs(args, dataset, reliability, repeatability, stats, query_norm, entropy, attn_max,
                  masking, elapsed_seconds, checkpoint_sha256, projection_fidelity):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    correlations = {}
    correlation_inputs = []
    if torch.isfinite(reliability).all():
        correlation_inputs.append(("pred_repeat", reliability, repeatability))
    modes = ("best", "hardest") if args.positive_mode == "both" else (args.positive_mode,)
    for mode in modes:
        contribution = stats["contribution_best" if mode == "best" else "contribution_hard"]
        correlation_inputs.append((f"repeat_contrib_{mode}", repeatability, contribution))
        if torch.isfinite(reliability).all():
            correlation_inputs.append((f"pred_contrib_{mode}", reliability, contribution))
    if args.positive_mode == "both":
        correlation_inputs.append(
            ("contrib_best_hard", stats["contribution_best"], stats["contribution_hard"])
        )
    for name, left, right in correlation_inputs:
        per_image, pooled = per_image_spearman(left, right)
        correlations[name] = {"per_image_mean": per_image, "pooled": pooled}

    query_csv = args.output_dir / "query_level.csv"
    with query_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "query_image_id", "place_id", "query_index",
            "predicted_reliability", "repeatability_score",
            "retrieval_contribution_best", "retrieval_contribution_hard",
            "query_norm", "attention_entropy", "max_attention",
            "baseline_best_margin", "baseline_hard_margin",
            "masked_best_margin", "masked_hard_margin",
            "masked_best_positive_similarity", "masked_hard_positive_similarity",
            "masked_hard_negative_similarity", "baseline_rank", "masked_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in range(dataset.num_queries):
            image_id = dataset.qImages[image] if hasattr(dataset, "qImages") else str(image)
            place_id = "|".join(map(str, np.asarray(dataset.ground_truth[image]).tolist()))
            for query_index in range(reliability.shape[1]):
                writer.writerow({
                    "dataset": args.dataset, "query_image_id": image_id, "place_id": place_id,
                    "query_index": query_index,
                    "predicted_reliability": float(reliability[image, query_index]),
                    "repeatability_score": float(repeatability[image, query_index]),
                    "retrieval_contribution_best": float(stats["contribution_best"][image, query_index]),
                    "retrieval_contribution_hard": float(stats["contribution_hard"][image, query_index]),
                    "query_norm": float(query_norm[image, query_index]),
                    "attention_entropy": float(entropy[image, query_index]),
                    "max_attention": float(attn_max[image, query_index]),
                    "baseline_best_margin": float(stats["baseline_best_margin"][image]),
                    "baseline_hard_margin": float(stats["baseline_hard_margin"][image]),
                    "masked_best_margin": float(stats["masked_best_margin"][image, query_index]),
                    "masked_hard_margin": float(stats["masked_hard_margin"][image, query_index]),
                    "masked_best_positive_similarity": float(stats["masked_best_positive"][image, query_index]),
                    "masked_hard_positive_similarity": float(stats["masked_hard_positive"][image, query_index]),
                    "masked_hard_negative_similarity": float(stats["masked_hard_negative"][image, query_index]),
                    "baseline_rank": int(stats["baseline_rank"][image]),
                    "masked_rank": int(stats["masked_rank"][image, query_index]),
                })

    with (args.output_dir / "image_level.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "query_image_id", "place_id", "num_gt_positives", "baseline_rank",
                  "baseline_best_positive_similarity", "baseline_hardest_positive_similarity",
                  "baseline_hard_negative_similarity", "baseline_best_margin", "baseline_hard_margin"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in range(dataset.num_queries):
            writer.writerow({
                "dataset": args.dataset,
                "query_image_id": dataset.qImages[image] if hasattr(dataset, "qImages") else image,
                "place_id": "|".join(map(str, np.asarray(dataset.ground_truth[image]).tolist())),
                "num_gt_positives": len(dataset.ground_truth[image]),
                "baseline_rank": int(stats["baseline_rank"][image]),
                "baseline_best_positive_similarity": float(stats["baseline_best_positive"][image]),
                "baseline_hardest_positive_similarity": float(stats["baseline_hard_positive"][image]),
                "baseline_hard_negative_similarity": float(stats["baseline_hard_negative"][image]),
                "baseline_best_margin": float(stats["baseline_best_margin"][image]),
                "baseline_hard_margin": float(stats["baseline_hard_margin"][image]),
            })

    with (args.output_dir / "masking_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(masking[0]))
        writer.writeheader()
        writer.writerows(masking)

    gt_counts = np.asarray([len(gt) for gt in dataset.ground_truth], dtype=np.int64)
    contribution_summary = {}
    for mode in modes:
        contribution = stats["contribution_best" if mode == "best" else "contribution_hard"]
        contribution_summary[mode] = {
            "mean": float(contribution.mean()),
            "std": float(contribution.std(unbiased=False)),
            "min": float(contribution.min()),
            "max": float(contribution.max()),
            "positive_fraction": float((contribution > 0).float().mean()),
            "negative_fraction": float((contribution < 0).float().mean()),
        }
    summary = {
        "experiment": "LOO retrieval contribution diagnostic",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "model_variant": args.model_variant,
        "gate_mode": args.gate_mode,
        "positive_mode": args.positive_mode,
        "dataset": args.dataset,
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "seed": args.seed,
        "protocol": "reference gallery fixed; query-side projection O2 leave-one-out; FP32 similarity",
        "oracle_uses_test_gt": True,
        "deployable": False,
        "positive_definition": "best=max and hardest=min cosine over benchmark GT references",
        "negative_definition": "maximum cosine over all non-GT references",
        "repeatability_definition": "mean row-max query cosine over all benchmark GT references",
        "gt_positive_count": {
            "min": int(gt_counts.min()), "mean": float(gt_counts.mean()),
            "median": float(np.median(gt_counts)), "max": int(gt_counts.max()),
        },
        "projection_fidelity": projection_fidelity,
        "correlations": correlations,
        "contribution": contribution_summary,
        "elapsed_seconds": elapsed_seconds,
        "masking_rows": masking,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_row = {
        "experiment": "loo_contribution",
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_variant": args.model_variant,
        "gate_mode": args.gate_mode,
        "positive_mode": args.positive_mode,
        "dataset": args.dataset,
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "gt_positive_min": int(gt_counts.min()),
        "gt_positive_mean": float(gt_counts.mean()),
        "gt_positive_median": float(np.median(gt_counts)),
        "gt_positive_max": int(gt_counts.max()),
        "negative_contribution_ratio_best": contribution_summary.get("best", {}).get("negative_fraction", ""),
        "negative_contribution_ratio_hardest": contribution_summary.get("hardest", {}).get("negative_fraction", ""),
        "spearman_pred_repeat": correlations.get("pred_repeat", {}).get("per_image_mean", ""),
        "spearman_pred_contrib_best": correlations.get("pred_contrib_best", {}).get("per_image_mean", ""),
        "spearman_pred_contrib_hardest": correlations.get("pred_contrib_hardest", {}).get("per_image_mean", ""),
        "spearman_repeat_contrib_best": correlations.get("repeat_contrib_best", {}).get("per_image_mean", ""),
        "spearman_repeat_contrib_hardest": correlations.get("repeat_contrib_hardest", {}).get("per_image_mean", ""),
        "spearman_contrib_best_hard": correlations.get("contrib_best_hard", {}).get("per_image_mean", ""),
        "elapsed_seconds": elapsed_seconds,
    }
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row))
        writer.writeheader()
        writer.writerow(summary_row)
    lines = [
        f"# {args.dataset} Query Retrieval Contribution 诊断", "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- checkpoint SHA256: `{checkpoint_sha256}`",
        f"- model/gate: `{args.model_variant}/{args.gate_mode}`",
        f"- protocol: `{summary['protocol']}`",
        "- oracle uses test GT: `true`; deployable: `false`",
        f"- GT positives/query: min={gt_counts.min()}, mean={gt_counts.mean():.2f}, "
        f"median={np.median(gt_counts):.2f}, max={gt_counts.max()}",
        f"- elapsed: `{elapsed_seconds / 60:.2f} min`", "",
        "| Correlation | Per-image mean Spearman | Pooled Spearman |",
        "|---|---:|---:|",
    ]
    for name, value in correlations.items():
        lines.append(f"| {name} | {value['per_image_mean']:.4f} | {value['pooled']:.4f} |")
    lines += ["", "| Strategy | Mask | Repeat | R@1 | R@5 | R@10 | R@20 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in masking:
        lines.append(
            f"| {row['strategy']} | {row['mask_ratio']:.0%} | {row['repeat']} | "
            f"{100*row['R@1']:.2f} | {100*row['R@5']:.2f} | "
            f"{100*row['R@10']:.2f} | {100*row['R@20']:.2f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    root = args.dataset_root or Path(DATASET_ROOTS[args.dataset])
    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TEST_DATASET_BUILDERS[args.dataset](root, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    print(
        f"[1/6] strict-loading {args.model_variant}/{args.gate_mode}: "
        f"{args.checkpoint} (sha256={checkpoint_sha256})",
        flush=True,
    )
    model = build_model(args.checkpoint, args.device, args.model_variant, args.gate_mode)
    print(f"[2/6] extracting {len(dataset)} images", flush=True)
    desc, raw, weighted, reliability, query_norm, entropy, attn_max = extract_features(
        model, dataset, loader, args.device
    )
    refs, query_desc = desc[:dataset.num_references], desc[dataset.num_references:]
    ref_raw, query_raw = raw[:dataset.num_references], raw[dataset.num_references:]
    query_weighted = weighted[dataset.num_references:]

    # 离线重投影必须和真实 forward 一致，否则后续 contribution 全部无效。
    reconstructed = descriptor_from_query_outputs(query_weighted.to(args.device), model.aggregator.fc).cpu()
    max_abs = float((reconstructed - query_desc).abs().max())
    min_cos = float(torch.nn.functional.cosine_similarity(reconstructed, query_desc).min())
    if max_abs > 5e-3 or min_cos < 0.9999:
        raise RuntimeError(f"offline projection mismatch: max_abs={max_abs}, min_cos={min_cos}")
    print(f"[3/6] projection fidelity passed: max_abs={max_abs:.3g}, min_cos={min_cos:.6f}", flush=True)

    repeatability = repeatability_against_gt(query_raw[:, -1], ref_raw[:, -1], dataset.ground_truth)
    assert_finite("repeatability", repeatability, (64,))
    print("[4/6] computing FP32 leave-one-query-out margins", flush=True)
    stats = loo_contribution(
        model, refs, query_weighted, dataset.ground_truth, args.device, args.loo_query_batch_size
    )
    assert_finite("contribution_best", stats["contribution_best"], (64,))
    assert_finite("contribution_hard", stats["contribution_hard"], (64,))
    if all(len(gt) == 1 for gt in dataset.ground_truth):
        if not torch.equal(stats["contribution_best"], stats["contribution_hard"]):
            max_diff = float((stats["contribution_best"] - stats["contribution_hard"]).abs().max())
            raise RuntimeError(f"single-positive dataset has best/hard mismatch: {max_diff}")
    print("[5/6] evaluating query-only oracle masking", flush=True)
    # repeatability-low 使用和相关性分析完全相同的分数。
    masking = masking_rows(
        model, dataset, refs, query_weighted, reliability[dataset.num_references:],
        repeatability, stats, args
    )

    elapsed = time.perf_counter() - started
    print("[6/6] writing CSV/JSON/Markdown", flush=True)
    write_outputs(
        args, dataset, reliability[dataset.num_references:], repeatability, stats,
        query_norm[dataset.num_references:], entropy[dataset.num_references:],
        attn_max[dataset.num_references:], masking, elapsed, checkpoint_sha256,
        {"max_absolute_difference": max_abs, "minimum_cosine": min_cos},
    )
    print(f"Saved outputs to {args.output_dir} in {elapsed / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
