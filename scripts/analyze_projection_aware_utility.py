#!/usr/bin/env python3
"""用正确的 O2 FC 槽位分析 projection-space redundancy 与 retrieval contribution。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--contribution-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_spearman(left, right) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def per_image_spearman(left: np.ndarray, right: np.ndarray) -> float:
    values = [finite_spearman(a, b) for a, b in zip(left, right)]
    return float(np.nanmean(values))


def load_contribution(path: Path, image_ids: list[str], queries: int):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != queries * 64:
        raise RuntimeError(f"contribution rows {len(rows)} != {queries * 64}")
    contribution = np.empty((queries, 64))
    for row_number, row in enumerate(rows):
        image, query = divmod(row_number, 64)
        if int(row["query_index"]) != query or row["query_image_id"] != image_ids[image]:
            raise RuntimeError(f"CSV/cache order mismatch at row {row_number}")
        contribution[image, query] = float(row["retrieval_contribution_best"])
    return contribution


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    checkpoint = Path(feature["checkpoint"])
    if sha256_file(checkpoint) != feature["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash differs from feature-cache provenance")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    weight = state["aggregator.fc.weight"].float()
    if weight.shape != (16, 128):
        raise RuntimeError(f"unexpected FC shape {tuple(weight.shape)}")
    # concat 顺序是 O1(0:64), O2(64:128)，这里只分析最后一层 O2。
    o2_weight = weight[:, 64:128].T.contiguous()
    if o2_weight.shape != (64, 16):
        raise RuntimeError("O2 FC slot slicing failed")

    outputs = feature["query_projection_outputs"][:, -1].float()
    queries = outputs.shape[0]
    if outputs.shape != (queries, 64, 512):
        raise RuntimeError(f"unexpected O2 projection input {tuple(outputs.shape)}")
    contribution = load_contribution(
        args.contribution_csv, feature["query_image_ids"], queries
    )

    query_direction = torch.nn.functional.normalize(outputs, dim=-1)
    weight_direction = torch.nn.functional.normalize(o2_weight, dim=-1)
    raw_similarity = torch.einsum("qmd,qnd->qmn", query_direction, query_direction)
    weight_similarity = weight_direction @ weight_direction.T
    # vec(w_m outer o_m) 的内积等于 <w_m,w_n>*<o_m,o_n>。
    projection_similarity = raw_similarity * weight_similarity[None]
    diagonal = torch.eye(64, dtype=torch.bool)[None]
    without_diagonal = projection_similarity.masked_fill(diagonal, -torch.inf)
    redundancy_max = without_diagonal.max(dim=-1).values.numpy()
    redundancy_top5 = without_diagonal.topk(5, dim=-1).values.mean(dim=-1).numpy()
    redundancy_mean = (
        projection_similarity.masked_fill(diagonal, 0).sum(dim=-1) / 63
    ).numpy()

    eps = 0.25 * float(contribution.std())
    mean_contribution = contribution.mean(axis=0)
    harmful_ratio = (contribution < -eps).mean(axis=0)
    helpful_ratio = (contribution > eps).mean(axis=0)
    slot_norm = o2_weight.norm(dim=-1).numpy()

    summary = {
        "dataset": feature["dataset"],
        "model_variant": feature["model_variant"],
        "gate_mode": feature["gate_mode"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": feature["checkpoint_sha256"],
        "fc_shape": list(weight.shape),
        "o2_fc_slot_range": [64, 128],
        "query_projection_input_shape": list(outputs.shape),
        "projection_vector_definition": "vec(W[:,64+m] outer O2[m])",
        "correlations": {
            "projection_max_vs_contribution_per_image": per_image_spearman(
                redundancy_max, contribution
            ),
            "projection_mean_vs_contribution_per_image": per_image_spearman(
                redundancy_mean, contribution
            ),
            "projection_top5_vs_contribution_per_image": per_image_spearman(
                redundancy_top5, contribution
            ),
            "projection_top5_vs_contribution_pooled": finite_spearman(
                redundancy_top5.ravel(), contribution.ravel()
            ),
            "fc_slot_norm_vs_mean_contribution": finite_spearman(
                slot_norm, mean_contribution
            ),
            "fc_slot_norm_vs_harmful_ratio": finite_spearman(slot_norm, harmful_ratio),
        },
        "eps": eps,
        "oracle_uses_test_gt": True,
        "deployable": False,
    }
    with (args.output_dir / "query_index_projection_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "query_index", "fc_slot_index", "fc_slot_norm", "mean_contribution",
            "harmful_ratio", "helpful_ratio", "mean_projection_top5_redundancy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(64):
            writer.writerow({
                "query_index": index, "fc_slot_index": 64 + index,
                "fc_slot_norm": slot_norm[index],
                "mean_contribution": mean_contribution[index],
                "harmful_ratio": harmful_ratio[index],
                "helpful_ratio": helpful_ratio[index],
                "mean_projection_top5_redundancy": redundancy_top5[:, index].mean(),
            })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# {feature['dataset']} {feature['model_variant']}/{feature['gate_mode']} "
        "Projection-aware Utility", "",
        "- O2 FC slots: `[64,128)`",
        "- oracle uses test GT: `true`; deployable: `false`", "",
        "| Metric | Spearman |", "|---|---:|",
    ]
    for name, value in summary["correlations"].items():
        lines.append(f"| {name} | {value:.4f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["correlations"], indent=2))


if __name__ == "__main__":
    main()
