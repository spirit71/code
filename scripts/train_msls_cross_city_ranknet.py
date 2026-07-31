#!/usr/bin/env python3
"""MSLS CPH/SF 双向 cross-city Candidate RankNet 校准。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_c1_top20_reranking_msls import recall_from_candidates
from scripts.train_candidate_ranknet import (
    CandidateRankNet,
    pairwise_loss,
    predict_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def candidate_labels(indices, ground_truth):
    labels = torch.zeros_like(indices, dtype=torch.bool)
    for row in range(len(indices)):
        gt = set(ground_truth[row].tolist())
        labels[row] = torch.tensor(
            [int(value) in gt for value in indices[row]], dtype=torch.bool
        )
    return labels


def train_one_epoch(features, labels, train_indices, input_columns, seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CandidateRankNet(len(input_columns)).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    valid_base = train_indices[labels[train_indices].any(1)]
    generator = torch.Generator().manual_seed(seed)
    loss_sum, query_sum, pair_sum = 0.0, 0, 0
    for _ in range(args.epochs):
        valid = valid_base[
            torch.randperm(len(valid_base), generator=generator)
        ]
        model.train()
        for start in range(0, len(valid), args.batch_size):
            current = valid[start:start + args.batch_size]
            x = features[current][..., input_columns].float().to(args.device)
            y = labels[current].to(args.device)
            score = model(x)
            loss, pairs = pairwise_loss(score, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("cross-city RankNet gradient failure")
            optimizer.step()
            loss_sum += float(loss) * len(current)
            query_sum += len(current)
            pair_sum += pairs
    return model.eval(), loss_sum / query_sum, pair_sum, len(valid_base)


def counts_for_rows(scores, indices, gt, rows):
    order = scores[rows].argsort(dim=1, descending=True, stable=True)
    candidates = indices[rows].gather(1, order)
    subset_gt = [gt[int(index)] for index in rows]
    return recall_from_candidates(candidates, subset_gt)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    utility = torch.load(
        args.utility_cache_root / "msls_val.pt",
        map_location="cpu", weights_only=True,
    )
    candidate = torch.load(
        args.candidate_cache_root / "msls_c0.pt",
        map_location="cpu", weights_only=True,
    )
    feature_item = torch.load(
        args.feature_cache, map_location="cpu", weights_only=True
    )
    features = feature_item["val"].float()
    indices = candidate["candidate_indices"]
    gt = utility["ground_truth"]
    labels = candidate_labels(indices, gt)
    cities = [value.split("/")[0] for value in utility["image_ids"]]
    city_rows = {
        city: torch.tensor(
            [index for index, value in enumerate(cities) if value == city],
            dtype=torch.long,
        )
        for city in sorted(set(cities))
    }
    if {key: len(value) for key, value in city_rows.items()} != {
        "cph": 498, "sf": 242,
    }:
        raise RuntimeError("unexpected MSLS city split")
    baseline_scores = features[..., 0]
    baseline = counts_for_rows(
        baseline_scores, indices, gt, torch.arange(740)
    )
    baseline_city = {
        city: counts_for_rows(baseline_scores, indices, gt, rows)
        for city, rows in city_rows.items()
    }
    if baseline != {1: 685, 5: 712, 10: 717, 20: 719}:
        raise RuntimeError("cross-city baseline mismatch")

    variants = {"global_only": [0], "full7": list(range(7))}
    seed_results, heldout_scores, model_states = [], {}, {}
    for variant, columns in variants.items():
        for seed in args.seeds:
            combined = torch.empty(740, 20)
            fold_rows = []
            for train_city, heldout_city in (("cph", "sf"), ("sf", "cph")):
                model, loss, pairs, used = train_one_epoch(
                    features, labels, city_rows[train_city],
                    columns, seed, args,
                )
                heldout = city_rows[heldout_city]
                combined[heldout] = predict_scores(
                    model, features[heldout][..., columns].float(),
                    args.device, args.batch_size,
                )
                state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                model_states[(variant, seed, train_city)] = state
                fold_counts = counts_for_rows(
                    combined, indices, gt, heldout
                )
                fold_rows.append({
                    "train_city": train_city,
                    "heldout_city": heldout_city,
                    "train_queries": used,
                    "train_loss": loss,
                    "train_pairs": pairs,
                    **{f"hits@{k}": fold_counts[k] for k in (1, 5, 10, 20)},
                })
            counts = counts_for_rows(
                combined, indices, gt, torch.arange(740)
            )
            result = {
                "variant": variant, "seed": seed, "folds": fold_rows,
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            }
            seed_results.append(result)
            heldout_scores[(variant, seed)] = combined
            print(json.dumps(result), flush=True)

    ensembles = {}
    ensemble_scores = {}
    for variant in variants:
        score = torch.stack([
            heldout_scores[(variant, seed)] for seed in args.seeds
        ]).mean(0)
        ensemble_scores[variant] = score
        counts = counts_for_rows(
            score, indices, gt, torch.arange(740)
        )
        ensembles[variant] = {
            "overall": counts,
            "cities": {
                city: counts_for_rows(score, indices, gt, rows)
                for city, rows in city_rows.items()
            },
        }
    full = ensembles["full7"]
    global_only = ensembles["global_only"]
    full_seed = [row for row in seed_results if row["variant"] == "full7"]
    passed = (
        full["overall"][1] >= 686
        and full["overall"][5] >= 712
        and full["overall"][20] == 719
        and full["cities"]["cph"][1] >= baseline_city["cph"][1]
        and full["cities"]["sf"][1] >= baseline_city["sf"][1]
        and full["overall"][1] > global_only["overall"][1]
        and sum(row["hits@1"] >= 685 for row in full_seed) >= 2
    )

    final_models = []
    if passed:
        for seed in args.seeds:
            model, loss, pairs, used = train_one_epoch(
                features, labels, torch.arange(740),
                variants["full7"], seed, args,
            )
            output = args.output_root / "final_full_msls" / f"seed_{seed}"
            output.mkdir(parents=True, exist_ok=True)
            torch.save({
                "architecture": "CandidateRankNet(7,16,1)",
                "feature_names": feature_item["feature_names"],
                "seed": seed,
                "epochs": args.epochs,
                "state_dict": model.state_dict(),
                "train_queries": used,
                "train_loss": loss,
                "train_pairs": pairs,
            }, output / "model.pt")
            final_models.append(str(output / "model.pt"))

    torch.save({
        "heldout_scores": {
            f"{variant}_seed_{seed}": score
            for (variant, seed), score in heldout_scores.items()
        },
        "ensemble_scores": ensemble_scores,
        "candidate_indices": indices,
        "cities": cities,
        "epochs": args.epochs,
    }, args.output_root / "cross_city_scores.pt")

    summary = {
        "protocol": f"MSLS CPH<->SF fixed-{args.epochs}-epoch cross-city RankNet",
        "epochs": args.epochs,
        "city_sizes": {key: len(value) for key, value in city_rows.items()},
        "baseline": baseline,
        "baseline_cities": baseline_city,
        "seed_results": seed_results,
        "ensembles": ensembles,
        "validation_gate_passed": passed,
        "final_full_msls_models": final_models,
        "test_data_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
