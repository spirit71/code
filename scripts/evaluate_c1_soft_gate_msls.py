#!/usr/bin/env python3
"""在 MSLS-val 上评估冻结 C1 score 的 query-side O2 soft gate。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from scripts.train_u0_query_utility import (
    UtilityHead,
    build_static_rank_prior,
    predict,
)
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--betas", type=float, nargs="+",
        default=[0, 0.02, 0.05, 0.1, 0.2, 0.3],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def load_msls_features(root: Path):
    shards = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in sorted((root / "msls_feature_shards").glob("*.pt"))
    ]
    if not shards:
        raise RuntimeError("missing MSLS feature shards")
    expected_start = 0
    for item in shards:
        if item["start"] != expected_start:
            raise RuntimeError("non-contiguous MSLS feature shards")
        expected_start = item["stop"]
    descriptors = torch.cat([item["descriptors"] for item in shards])
    outputs = torch.cat([
        item["query_outputs"] for item in shards if item["query_outputs"].numel()
    ])
    if descriptors.shape != (19611, 8192):
        raise RuntimeError(f"bad descriptors {tuple(descriptors.shape)}")
    if outputs.shape != (740, 2, 64, 512):
        raise RuntimeError(f"bad query outputs {tuple(outputs.shape)}")
    return descriptors[:18871], descriptors[18871:], outputs


@torch.inference_mode()
def recall_counts(references, queries, ground_truth, device):
    references = references.float().to(device)
    queries = queries.float().to(device)
    predictions = []
    for start in range(0, len(queries), 128):
        query = queries[start:start + 128]
        distance = (
            (query * query).sum(1, keepdim=True)
            + (references * references).sum(1)[None]
            - 2 * query @ references.T
        )
        predictions.append(distance.topk(20, dim=1, largest=False).indices.cpu())
    predictions = torch.cat(predictions)
    counts = {}
    for k in (1, 5, 10, 20):
        hits = 0
        for index, row in enumerate(predictions[:, :k]):
            gt = set(ground_truth[index].tolist())
            hits += int(any(int(value) in gt for value in row))
        counts[k] = hits
    return counts


def fractions(counts, total):
    return {f"r{k}": counts[k] / total for k in (1, 5, 10, 20)}


@torch.inference_mode()
def gated_descriptor(outputs, scores, beta, model, device, batch_size):
    result, minimum_weight, maximum_mean_error = [], float("inf"), 0.0
    for start in range(0, len(outputs), batch_size):
        stop = min(start + batch_size, len(outputs))
        score = scores[start:stop].float().to(device)
        z = (score - score.mean(1, keepdim=True)) / (
            score.std(1, keepdim=True, unbiased=False) + 1e-6
        )
        residual = torch.tanh(z)
        residual = residual - residual.mean(1, keepdim=True)
        maximum_mean_error = max(
            maximum_mean_error, float(residual.mean(1).abs().max())
        )
        weight = 1 + beta * residual
        minimum_weight = min(minimum_weight, float(weight.min()))
        if float(weight.min()) <= 0:
            raise RuntimeError(f"non-positive gate weight at beta={beta}")
        current = outputs[start:stop].float().to(device).clone()
        current[:, -1] = current[:, -1] * weight[..., None]
        descriptor = descriptor_from_query_outputs(
            current, model.aggregator.fc
        ).float()
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("non-finite gated descriptor")
        result.append(descriptor.cpu())
    return torch.cat(result), minimum_weight, maximum_mean_error


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train = torch.load(
        args.candidate_cache_root / "gsv_c0.pt",
        map_location="cpu", weights_only=True,
    )
    val = torch.load(
        args.candidate_cache_root / "msls_c0.pt",
        map_location="cpu", weights_only=True,
    )
    utility_val = torch.load(
        args.utility_cache_root / "msls_val.pt",
        map_location="cpu", weights_only=True,
    )
    checkpoint_hash = train["checkpoint_sha256"]
    if (
        val["checkpoint_sha256"] != checkpoint_hash
        or utility_val["checkpoint_sha256"] != checkpoint_hash
    ):
        raise RuntimeError("cache checkpoint mismatch")
    if train["top_k"] != 20 or val["top_k"] != 20:
        raise RuntimeError("C2 requires frozen Top-20 candidates")
    if (
        train["gt_used_for_candidate_selection"] is not False
        or val["gt_used_for_candidate_selection"] is not False
    ):
        raise RuntimeError("candidate feature GT leakage")

    train_target = train["contribution"]
    prior = build_static_rank_prior(train_target)
    val_input = torch.cat([val["query_o2"], val["candidate_stats"]], dim=-1)
    scores = {}
    for seed in args.seeds:
        path = args.c1_root / "delta_0.5" / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != "c1" or checkpoint["delta_scale"] != 0.5:
            raise RuntimeError(f"invalid C1 checkpoint {path}")
        model = UtilityHead("c1", prior)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(args.device).eval()
        scores[f"seed_{seed}"] = predict(
            model, val_input, args.device, args.batch_size
        )
    scores["ensemble"] = torch.stack(list(scores.values())).mean(0)

    references, official_queries, outputs = load_msls_features(
        args.utility_cache_root
    )
    e0 = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in e0.parameters()):
        raise RuntimeError("E0 must be frozen")
    reconstructed = descriptor_from_query_outputs(
        outputs.float().to(args.device), e0.aggregator.fc
    ).float().cpu()
    minimum_cosine = float(F.cosine_similarity(
        reconstructed, official_queries.float(), dim=-1
    ).min())
    if minimum_cosine < 0.9999:
        raise RuntimeError(f"projection fidelity failed {minimum_cosine}")
    ground_truth = utility_val["ground_truth"]
    baseline_counts = recall_counts(
        references, official_queries, ground_truth, args.device
    )
    expected = {1: 685, 5: 712, 10: 717, 20: 719}
    if baseline_counts != expected:
        raise RuntimeError(f"baseline Recall mismatch {baseline_counts}")

    rows = []
    beta_zero_descriptor = None
    for score_name, score in scores.items():
        for beta in args.betas:
            descriptor, minimum_weight, mean_error = gated_descriptor(
                outputs, score, beta, e0, args.device, args.batch_size
            )
            if beta == 0 and score_name == "ensemble":
                beta_zero_descriptor = descriptor
            counts = recall_counts(
                references, descriptor, ground_truth, args.device
            )
            row = {
                "score": score_name,
                "beta": beta,
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
                **fractions(counts, len(outputs)),
                "minimum_weight": minimum_weight,
                "max_gate_mean_error": mean_error,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    beta_zero_cosine = float(F.cosine_similarity(
        beta_zero_descriptor, reconstructed, dim=-1
    ).min())
    if beta_zero_cosine < 0.999999:
        raise RuntimeError(f"beta=0 parity failed {beta_zero_cosine}")

    ensemble_rows = [row for row in rows if row["score"] == "ensemble"]
    selected = sorted(
        ensemble_rows, key=lambda row: (-row["hits@1"], -row["hits@5"], row["beta"])
    )[0]
    individual_selected = [
        row for row in rows
        if row["score"].startswith("seed_") and row["beta"] == selected["beta"]
    ]
    passed = (
        selected["hits@1"] >= baseline_counts[1] + 1
        and selected["hits@5"] >= baseline_counts[5]
        and sum(
            row["hits@1"] >= baseline_counts[1] for row in individual_selected
        ) >= 2
    )
    with (args.output_root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "MSLS-val query-only O2 centered bounded soft gate",
        "checkpoint_sha256": checkpoint_hash,
        "delta_scale": 0.5,
        "top_k": 20,
        "seeds": args.seeds,
        "betas": args.betas,
        "selection": "ensemble max R1, then R5, then smaller beta",
        "baseline_counts": baseline_counts,
        "baseline_recall": fractions(baseline_counts, len(outputs)),
        "projection_minimum_cosine": minimum_cosine,
        "beta_zero_minimum_cosine": beta_zero_cosine,
        "selected": selected,
        "individual_at_selected_beta": individual_selected,
        "validation_gate_passed": passed,
        "test_data_used": False,
        "boq_and_c1_frozen": True,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
