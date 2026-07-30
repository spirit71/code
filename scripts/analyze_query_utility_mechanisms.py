#!/usr/bin/env python3
"""分析 redundancy、sign switching、difficulty 与 LOO contribution 的关系。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eps-scale", type=float, default=0.25)
    parser.add_argument("--redundancy-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def per_image_spearman(frame, left, right):
    values = []
    for _, group in frame.groupby("query_image_id", sort=False):
        value = safe_spearman(group[left], group[right])
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


@torch.inference_mode()
def compute_redundancy(o2, batch_size, device):
    if o2.ndim != 3 or tuple(o2.shape[1:]) != (64, 512):
        raise ValueError(f"expected O2 [Q,64,512], got {tuple(o2.shape)}")
    outputs = []
    for start in range(0, len(o2), batch_size):
        query = torch.nn.functional.normalize(
            o2[start:start + batch_size].to(device).float(), dim=-1
        )
        similarity = query @ query.transpose(1, 2)
        diagonal = similarity.diagonal(dim1=1, dim2=2)
        if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=2e-4):
            raise RuntimeError("normalized O2 cosine diagonal is not one")
        mask = torch.eye(64, dtype=torch.bool, device=device).unsqueeze(0)
        without_self = similarity.masked_fill(mask, float("-inf"))
        maximum = without_self.max(dim=-1).values
        top5 = without_self.topk(5, dim=-1).values.mean(dim=-1)
        mean = similarity.masked_fill(mask, 0).sum(dim=-1) / 63
        outputs.append(torch.stack((maximum, mean, top5), dim=-1).cpu())
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise RuntimeError("redundancy contains NaN or inf")
    return result


def main():
    args = parse_args()
    required = [
        args.experiment_dir / "query_level.csv",
        args.experiment_dir / "image_level.csv",
        args.experiment_dir / "summary.json",
        args.feature_cache,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    query = pd.read_csv(required[0])
    images = pd.read_csv(required[1])
    summary = json.loads(required[2].read_text(encoding="utf-8"))
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    if cache["dataset"] != summary["dataset"]:
        raise RuntimeError("dataset mismatch between cache and contribution results")
    if cache["model_variant"] != summary["model_variant"]:
        raise RuntimeError("model variant mismatch")
    if cache["gate_mode"] != summary["gate_mode"]:
        raise RuntimeError("gate mode mismatch")
    if cache["checkpoint_sha256"] != summary["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    expected_rows = summary["num_queries"] * 64
    if len(query) != expected_rows:
        raise RuntimeError(f"query CSV has {len(query)} rows, expected {expected_rows}")
    expected_ids = np.repeat(np.asarray(cache["query_image_ids"], dtype=object), 64)
    if not np.array_equal(query["query_image_id"].astype(str).to_numpy(), expected_ids.astype(str)):
        raise RuntimeError("query image order differs between CSV and cache")
    if not np.array_equal(
        query["query_index"].to_numpy(), np.tile(np.arange(64), summary["num_queries"])
    ):
        raise RuntimeError("query_index order is not 0..63 per image")

    redundancy = compute_redundancy(
        cache["query_raw_o2"], args.redundancy_batch_size, args.device
    ).reshape(-1, 3).numpy()
    query["redundancy_max"] = redundancy[:, 0]
    query["redundancy_mean"] = redundancy[:, 1]
    query["redundancy_top5"] = redundancy[:, 2]
    image_baseline = images.set_index("query_image_id")
    # Correct/Wrong 必须严格复用正式评测的 FAISS IndexFlatL2，而不能用 cosine
    # margin 的符号或 cosine rank 代替；reference norm 的微小差异会改变近邻顺序。
    reference_descriptors = cache["reference_descriptors"].float().numpy()
    query_descriptors = cache["query_descriptors"].float().numpy()
    faiss_index = faiss.IndexFlatL2(reference_descriptors.shape[1])
    faiss_index.add(reference_descriptors)
    _, top1 = faiss_index.search(query_descriptors, 1)
    ground_truth = [
        {int(value) for value in str(place_id).split("|")}
        for place_id in images["place_id"].tolist()
    ]
    official_top1_correct = np.asarray([
        int(top1[index, 0]) in ground_truth[index] for index in range(len(top1))
    ])
    baseline_rows = [
        row["R@1"] for row in summary["masking_rows"] if float(row["mask_ratio"]) == 0
    ]
    expected_r1 = float(baseline_rows[0])
    measured_r1 = float(official_top1_correct.mean())
    if abs(measured_r1 - expected_r1) > 1e-12:
        raise RuntimeError(
            f"FAISS Top-1 mismatch: cache={measured_r1}, contribution={expected_r1}"
        )
    official_correct_by_image = pd.Series(
        official_top1_correct, index=images["query_image_id"].astype(str)
    )
    query["baseline_best_positive_similarity"] = query["query_image_id"].map(
        image_baseline["baseline_best_positive_similarity"]
    )
    query["baseline_hard_negative_similarity"] = query["query_image_id"].map(
        image_baseline["baseline_hard_negative_similarity"]
    )
    query["delta_positive_similarity"] = (
        query["baseline_best_positive_similarity"]
        - query["masked_best_positive_similarity"]
    )
    query["delta_negative_similarity"] = (
        query["baseline_hard_negative_similarity"]
        - query["masked_hard_negative_similarity"]
    )
    query["contribution_identity_error"] = (
        query["retrieval_contribution_best"]
        - (query["delta_positive_similarity"] - query["delta_negative_similarity"])
    ).abs()
    if query["contribution_identity_error"].max() > 2e-6:
        raise RuntimeError("contribution decomposition identity failed")

    dataset_std = float(query["retrieval_contribution_best"].std(ddof=0))
    eps = args.eps_scale * dataset_std
    contribution = query["retrieval_contribution_best"]
    query["utility_class"] = np.where(
        contribution > eps, "helpful",
        np.where(contribution < -eps, "harmful", "neutral"),
    )
    query["baseline_top1_correct"] = query["query_image_id"].map(
        official_correct_by_image
    ).astype(bool)
    args.output_dir.mkdir(parents=True)
    query.to_csv(args.output_dir / "redundancy_contribution.csv", index=False)

    index_rows = []
    eps_scales = (0.0, 0.1, 0.25, 0.5)
    for query_index, group in query.groupby("query_index"):
        row = {
            "dataset": summary["dataset"],
            "model_variant": summary["model_variant"],
            "gate_mode": summary["gate_mode"],
            "query_index": int(query_index),
            "mean_contribution": group["retrieval_contribution_best"].mean(),
            "std_contribution": group["retrieval_contribution_best"].std(ddof=0),
            "mean_abs_contribution": group["retrieval_contribution_best"].abs().mean(),
            "mean_redundancy": group["redundancy_top5"].mean(),
        }
        for scale in eps_scales:
            threshold = scale * dataset_std
            row[f"helpful_ratio_eps{scale}"] = (
                group["retrieval_contribution_best"] > threshold
            ).mean()
            row[f"harmful_ratio_eps{scale}"] = (
                group["retrieval_contribution_best"] < -threshold
            ).mean()
            row[f"neutral_ratio_eps{scale}"] = 1 - (
                row[f"helpful_ratio_eps{scale}"] + row[f"harmful_ratio_eps{scale}"]
            )
        index_rows.append(row)
    query_index_stats = pd.DataFrame(index_rows)
    query_index_stats.to_csv(args.output_dir / "query_index_statistics.csv", index=False)

    image_rows = []
    for image_id, group in query.groupby("query_image_id", sort=False):
        harmful = group[group["utility_class"] == "harmful"]
        helpful = group[group["utility_class"] == "helpful"]
        baseline = image_baseline.loc[image_id]
        image_rows.append({
            "dataset": summary["dataset"],
            "model_variant": summary["model_variant"],
            "gate_mode": summary["gate_mode"],
            "query_image_id": image_id,
            "baseline_rank_cosine": int(baseline["baseline_rank"]),
            "baseline_top1_correct": bool(official_correct_by_image.loc[str(image_id)]),
            "baseline_best_margin": baseline["baseline_best_margin"],
            "negative_query_ratio": len(harmful) / 64,
            "positive_query_ratio": len(helpful) / 64,
            "neutral_query_ratio": 1 - (len(harmful) + len(helpful)) / 64,
            "mean_negative_contribution": (
                harmful["retrieval_contribution_best"].mean() if len(harmful) else 0.0
            ),
            "mean_contribution": group["retrieval_contribution_best"].mean(),
            "std_contribution": group["retrieval_contribution_best"].std(ddof=0),
            "mean_redundancy": group["redundancy_top5"].mean(),
        })
    image_stats = pd.DataFrame(image_rows)
    image_stats.to_csv(args.output_dir / "image_difficulty_statistics.csv", index=False)

    harmful_decomposition = query[query["utility_class"] == "harmful"][[
        "dataset", "query_image_id", "query_index", "retrieval_contribution_best",
        "delta_positive_similarity", "delta_negative_similarity",
        "redundancy_max", "redundancy_mean", "redundancy_top5",
    ]].copy()
    harmful_decomposition["dominant_mechanism"] = np.where(
        harmful_decomposition["delta_negative_similarity"].abs()
        > harmful_decomposition["delta_positive_similarity"].abs(),
        "negative_similarity", "positive_similarity",
    )
    harmful_decomposition.to_csv(
        args.output_dir / "harmful_query_decomposition.csv", index=False
    )

    correlations = {}
    for name, left, right in (
        ("redundancy_max_vs_contribution", "redundancy_max", "retrieval_contribution_best"),
        ("redundancy_mean_vs_contribution", "redundancy_mean", "retrieval_contribution_best"),
        ("redundancy_top5_vs_contribution", "redundancy_top5", "retrieval_contribution_best"),
        ("redundancy_top5_vs_delta_negative", "redundancy_top5", "delta_negative_similarity"),
        ("contribution_vs_delta_negative", "retrieval_contribution_best", "delta_negative_similarity"),
        ("contribution_vs_delta_positive", "retrieval_contribution_best", "delta_positive_similarity"),
    ):
        correlations[name] = {
            "per_image_mean": per_image_spearman(query, left, right),
            "pooled": safe_spearman(query[left], query[right]),
        }
    correlations["margin_vs_negative_query_ratio"] = {
        "pooled": safe_spearman(
            image_stats["baseline_best_margin"], image_stats["negative_query_ratio"]
        )
    }
    correlations["margin_vs_mean_negative_contribution"] = {
        "pooled": safe_spearman(
            image_stats["baseline_best_margin"],
            image_stats["mean_negative_contribution"],
        )
    }

    grouped = image_stats.groupby("baseline_top1_correct").agg({
        "negative_query_ratio": "mean",
        "mean_negative_contribution": "mean",
        "mean_contribution": "mean",
        "std_contribution": "mean",
        "mean_redundancy": "mean",
        "query_image_id": "count",
    }).rename(columns={"query_image_id": "num_images"}).reset_index()
    grouped.to_csv(args.output_dir / "correct_wrong_statistics.csv", index=False)

    switching = (
        (query_index_stats["helpful_ratio_eps0.25"] >= 0.1)
        & (query_index_stats["harmful_ratio_eps0.25"] >= 0.1)
    )
    mechanism_summary = {
        "dataset": summary["dataset"],
        "model_variant": summary["model_variant"],
        "gate_mode": summary["gate_mode"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "num_queries": summary["num_queries"],
        "eps_scale": args.eps_scale,
        "eps": eps,
        "correlations": correlations,
        "query_indices_with_sign_switching_ratio": float(switching.mean()),
        "harmful_dominant_mechanism_ratio": (
            harmful_decomposition["dominant_mechanism"].value_counts(normalize=True).to_dict()
        ),
        "correct_wrong": grouped.to_dict(orient="records"),
        "oracle_uses_test_gt": True,
        "deployable": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(mechanism_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.json_normalize(mechanism_summary, sep=".").to_csv(
        args.output_dir / "mechanism_summary.csv", index=False
    )

    lines = [
        f"# {summary['dataset']} {summary['model_variant']}/{summary['gate_mode']} Utility机制分析",
        "",
        "> 使用测试 GT contribution，仅为机制诊断，不可部署。",
        "",
        f"- epsilon: `{eps:.6g}` (`{args.eps_scale} × dataset std`)",
        f"- sign-switching query-index ratio: `{switching.mean():.2%}`",
        "",
        "| Correlation | Per-image Spearman | Pooled Spearman |",
        "|---|---:|---:|",
    ]
    for name, values in correlations.items():
        lines.append(
            f"| {name} | {values.get('per_image_mean', float('nan')):.4f} | "
            f"{values['pooled']:.4f} |"
        )
    lines += [
        "", "## Correct vs Wrong", "",
        "| Baseline correct | Images | Negative ratio | Mean negative C | "
        "Mean C | C std | Mean redundancy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"| {bool(row['baseline_top1_correct'])} | {int(row['num_images'])} | "
            f"{row['negative_query_ratio']:.4f} | "
            f"{row['mean_negative_contribution']:.6f} | "
            f"{row['mean_contribution']:.6f} | "
            f"{row['std_contribution']:.6f} | {row['mean_redundancy']:.4f} |"
        )
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved mechanism analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
