#!/usr/bin/env python3
"""MSLS-val：比较 C1 predicted/static/random O2 hard removal。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from scripts.evaluate_c1_soft_gate_msls import load_msls_features, recall_counts
from scripts.train_u0_query_utility import UtilityHead, build_static_rank_prior, predict
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
    parser.add_argument("--remove-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--random-seeds", type=int, default=20)
    return parser.parse_args()


def make_keep_mask(scores, count, remove_high):
    keep = torch.ones_like(scores, dtype=torch.bool)
    order = scores.argsort(dim=1, descending=remove_high)[:, :count]
    keep.scatter_(1, order, False)
    if not torch.equal(
        (~keep).sum(1), torch.full((len(keep),), count, dtype=torch.long)
    ):
        raise RuntimeError("mask removal count mismatch")
    return keep


def random_keep_mask(samples, count, seed):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(samples, 64, generator=generator)
    return make_keep_mask(noise, count, remove_high=False)


@torch.inference_mode()
def masked_descriptor(outputs, keep, model, device, batch_size):
    result = []
    for start in range(0, len(outputs), batch_size):
        stop = min(start + batch_size, len(outputs))
        current = outputs[start:stop].float().to(device).clone()
        mask = keep[start:stop].to(device)
        current[:, -1] = current[:, -1] * mask[..., None]
        descriptor = descriptor_from_query_outputs(
            current, model.aggregator.fc
        ).float()
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("non-finite masked descriptor")
        result.append(descriptor.cpu())
    return torch.cat(result)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if sorted(set(args.remove_counts)) != args.remove_counts:
        raise RuntimeError("remove counts must be unique and sorted")
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
    prior = build_static_rank_prior(train["contribution"])
    val_input = torch.cat([val["query_o2"], val["candidate_stats"]], dim=-1)
    seed_scores = {}
    for seed in args.seeds:
        checkpoint = torch.load(
            args.c1_root / "delta_0.5" / f"seed_{seed}" / "best.pt",
            map_location="cpu", weights_only=True,
        )
        if checkpoint["architecture"] != "c1":
            raise RuntimeError("non-C1 checkpoint")
        model = UtilityHead("c1", prior)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(args.device).eval()
        seed_scores[seed] = predict(
            model, val_input, args.device, args.batch_size
        )
    ensemble = torch.stack(list(seed_scores.values())).mean(0)

    references, official_queries, outputs = load_msls_features(
        args.utility_cache_root
    )
    e0 = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in e0.parameters()):
        raise RuntimeError("E0 must be frozen")
    ground_truth = utility_val["ground_truth"]
    baseline = recall_counts(
        references, official_queries, ground_truth, args.device
    )
    if baseline != {1: 685, 5: 712, 10: 717, 20: 719}:
        raise RuntimeError(f"baseline mismatch {baseline}")

    rows = []

    def evaluate(label, count, keep, seed=""):
        descriptor = masked_descriptor(
            outputs, keep, e0, args.device, args.batch_size
        )
        counts = recall_counts(
            references, descriptor, ground_truth, args.device
        )
        row = {
            "method": label, "remove_count": count, "seed": seed,
            **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            **{f"r{k}": counts[k] / len(outputs) for k in (1, 5, 10, 20)},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        return row

    static_scores = prior[None].expand(len(outputs), -1)
    for count in args.remove_counts:
        evaluate(
            "pred_low_ensemble", count,
            make_keep_mask(ensemble, count, remove_high=False),
        )
        evaluate(
            "pred_high_ensemble", count,
            make_keep_mask(ensemble, count, remove_high=True),
        )
        evaluate(
            "static_low", count,
            make_keep_mask(static_scores, count, remove_high=False),
        )
        for random_seed in range(1000, 1000 + args.random_seeds):
            evaluate(
                "random", count,
                random_keep_mask(len(outputs), count, random_seed),
                random_seed,
            )
        for seed, score in seed_scores.items():
            evaluate(
                "pred_low_individual", count,
                make_keep_mask(score, count, remove_high=False),
                seed,
            )

    ensemble_rows = [
        row for row in rows if row["method"] == "pred_low_ensemble"
    ]
    selected = sorted(
        ensemble_rows,
        key=lambda row: (-row["hits@1"], -row["hits@5"], row["remove_count"]),
    )[0]
    count = selected["remove_count"]
    random_rows = [
        row for row in rows
        if row["method"] == "random" and row["remove_count"] == count
    ]
    random_mean_r1_hits = sum(row["hits@1"] for row in random_rows) / len(random_rows)
    static_row = next(
        row for row in rows
        if row["method"] == "static_low" and row["remove_count"] == count
    )
    high_row = next(
        row for row in rows
        if row["method"] == "pred_high_ensemble" and row["remove_count"] == count
    )
    individual_rows = [
        row for row in rows
        if row["method"] == "pred_low_individual" and row["remove_count"] == count
    ]
    passed = (
        selected["hits@1"] >= baseline[1] + 1
        and selected["hits@5"] >= baseline[5]
        and selected["hits@1"] >= random_mean_r1_hits + 1
        and selected["hits@1"] >= static_row["hits@1"]
        and high_row["hits@1"] <= selected["hits@1"]
        and sum(row["hits@1"] >= baseline[1] for row in individual_rows) >= 2
    )
    with (args.output_root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "MSLS-val query-only O2 hard removal",
        "remove_counts": args.remove_counts,
        "random_seeds": args.random_seeds,
        "baseline_counts": baseline,
        "selection": "ensemble pred-low max R1, then R5, then fewer removals",
        "selected": selected,
        "random_mean_r1_hits_at_selected": random_mean_r1_hits,
        "static_at_selected": static_row,
        "pred_high_at_selected": high_row,
        "individual_at_selected": individual_rows,
        "validation_gate_passed": passed,
        "test_data_used": False,
        "query_only_masking": True,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
