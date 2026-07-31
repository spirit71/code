#!/usr/bin/env python3
"""用 train-only query ranking 在冻结 benchmark 上评估 O2 static masks。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs
from src.dataloaders.datamodule import TEST_DATASET_BUILDERS


DATA_ROOTS = {
    "amstertime": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/amstertime/images"),
    "sped": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/sped/images"),
    "pitts30k-test": Path("/root/data/Pittsburgh/pitts30k"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-chunk", type=int, default=128)
    return parser.parse_args()


def load_cache(dataset: str):
    if dataset == "amstertime":
        item = torch.load(
            "/root/qrl_utility_mechanisms/features/amstertime_e0.pt",
            map_location="cpu", weights_only=True,
        )
        return item["reference_descriptors"].float(), item["query_projection_outputs"]
    if dataset == "sped":
        item = torch.load(
            "/root/qrl_utility_mechanisms/features/sped_e0.pt",
            map_location="cpu", weights_only=True,
        )
        return item["reference_descriptors"].float(), item["query_projection_outputs"]
    shards = []
    for path in sorted(Path("/root/qrl_pitts30k_e0_loo/feature_shards").glob("*.pt")):
        shards.append(torch.load(path, map_location="cpu", weights_only=True))
    references = torch.cat([item["descriptors"] for item in shards])[:10000].float()
    query_outputs = torch.cat([
        item["query_outputs"] for item in shards if item["query_outputs"].numel()
    ])
    if query_outputs.shape != (6816, 2, 64, 512):
        raise RuntimeError(f"incomplete Pitts query cache: {tuple(query_outputs.shape)}")
    return references, query_outputs


@torch.inference_mode()
def reconstruct(model, outputs, remove_indices, device, chunk):
    result = []
    for start in range(0, len(outputs), chunk):
        stop = min(start + chunk, len(outputs))
        current = outputs[start:stop].float().to(device)
        if remove_indices:
            current[:, -1, remove_indices] = 0
        result.append(descriptor_from_query_outputs(current, model.aggregator.fc).float().cpu())
    return torch.cat(result)


@torch.inference_mode()
def l2_recall(references, queries, ground_truth, device, chunk):
    references = references.to(device)
    reference_norm = (references * references).sum(1)
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        query = queries[start:stop].to(device)
        distance = (
            (query * query).sum(1, keepdim=True) + reference_norm[None]
            - 2 * query @ references.T
        )
        prediction = distance.topk(20, largest=False).indices.cpu().numpy()
        for local, candidates in enumerate(prediction):
            positives = set(np.asarray(ground_truth[start + local], dtype=np.int64).tolist())
            for k in hits:
                hits[k] += int(any(int(value) in positives for value in candidates[:k]))
    return {f"R@{k}": hits[k] / len(queries) for k in hits}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static = json.loads(args.static_summary.read_text())
    train_order = static["train_static_order_low_to_high"]
    fc_order = static["fc_norm_order_low_to_high"]
    selected = int(static["selected_removed_by_val_r1"])
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("model is not frozen")
    rows = []
    expected_baseline = {
        "amstertime": 0.6344435418359058,
        "sped": 0.9110378912685338,
        "pitts30k-test": 0.9295774647887324,
    }
    generator = torch.Generator().manual_seed(42)
    random_orders = [torch.randperm(64, generator=generator).tolist() for _ in range(3)]
    for dataset_name, root in DATA_ROOTS.items():
        dataset = TEST_DATASET_BUILDERS[dataset_name](root, None)
        references, outputs = load_cache(dataset_name)
        if len(references) != dataset.num_references or len(outputs) != dataset.num_queries:
            raise RuntimeError(f"{dataset_name} cache/dataset size mismatch")
        configurations = [("baseline", 0, [], "formal")]
        for method, order in (
            ("train_mean_contribution", train_order), ("fc_slot_norm", fc_order)
        ):
            for count in (4, 8, 16, 24, 32):
                configurations.append((
                    method, count, order[:count],
                    "frozen_selected" if method == "train_mean_contribution" and count == selected
                    else "diagnostic_curve",
                ))
        for repeat, order in enumerate(random_orders):
            configurations.append((f"random_seed_{42+repeat}", selected, order[:selected], "control"))
        for method, count, remove, role in configurations:
            queries = reconstruct(model, outputs, remove, args.device, args.query_chunk)
            recall = l2_recall(
                references, queries, dataset.ground_truth, args.device, args.query_chunk
            )
            row = {
                "dataset": dataset_name, "method": method, "removed": count,
                "role": role, **recall,
            }
            rows.append(row)
            print(row, flush=True)
        baseline = next(
            row for row in rows
            if row["dataset"] == dataset_name and row["method"] == "baseline"
        )
        if abs(baseline["R@1"] - expected_baseline[dataset_name]) > 1e-12:
            raise RuntimeError(
                f"{dataset_name} baseline mismatch: {baseline['R@1']} "
                f"!= {expected_baseline[dataset_name]}"
            )
    with (args.output_dir / "test_static_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "train-only static O2 mask frozen test evaluation",
        "selected_removed_by_city_val": selected,
        "selection_source": str(args.static_summary),
        "checkpoint": str(args.checkpoint.resolve()),
        "all_parameters_frozen": True,
        "test_gt_used_for_training_or_selection": False,
        "selected_rows": [
            row for row in rows
            if row["method"] == "train_mean_contribution" and row["removed"] == selected
        ],
        "all_rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Train-only Static O2 Mask 测试结果", "",
        f"- frozen removed count: `{selected}`",
        "- test GT used for training/selection: `false`", "",
        "| Dataset | Method | Removed | Role | R@1 | R@5 | R@10 | R@20 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['removed']} | {row['role']} | "
            f"{100*row['R@1']:.2f} | {100*row['R@5']:.2f} | "
            f"{100*row['R@10']:.2f} | {100*row['R@20']:.2f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
