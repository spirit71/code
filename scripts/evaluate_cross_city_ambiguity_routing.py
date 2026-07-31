#!/usr/bin/env python3
"""MSLS-val：仅对 global gap 最小的 queries 启用 cross-city RankNet。"""

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

from scripts.evaluate_c1_top20_reranking_msls import recall_from_candidates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cross-city-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--coverages", type=float, nargs="+",
        default=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def counts(scores, indices, ground_truth, rows=None):
    if rows is None:
        rows = torch.arange(len(scores))
    candidates = indices[rows].gather(
        1, scores[rows].argsort(dim=1, descending=True, stable=True)
    )
    gt = [ground_truth[int(index)] for index in rows]
    return recall_from_candidates(candidates, gt)


def routed_score(global_score, ranknet_score, coverage):
    samples = len(global_score)
    count = int(round(samples * coverage))
    output = global_score.clone()
    routed = torch.zeros(samples, dtype=torch.bool)
    if count:
        gap = global_score[:, 0] - global_score[:, 1]
        selected = gap.argsort(stable=True)[:count]
        output[selected] = ranknet_score[selected]
        routed[selected] = True
    if int(routed.sum()) != count:
        raise RuntimeError("routing count mismatch")
    return output, routed


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
    features = torch.load(
        args.feature_cache, map_location="cpu", weights_only=True
    )["val"].float()
    scores_item = torch.load(
        args.cross_city_scores, map_location="cpu", weights_only=True
    )
    global_score = features[..., 0]
    ranknet_ensemble = scores_item["ensemble_scores"]["full7"]
    seed_scores = {
        seed: scores_item["heldout_scores"][f"full7_seed_{seed}"]
        for seed in args.seeds
    }
    indices = candidate["candidate_indices"]
    gt = utility["ground_truth"]
    cities = [value.split("/")[0] for value in utility["image_ids"]]
    city_rows = {
        city: torch.tensor(
            [index for index, value in enumerate(cities) if value == city]
        )
        for city in sorted(set(cities))
    }
    baseline = counts(global_score, indices, gt)
    if baseline != {1: 685, 5: 712, 10: 717, 20: 719}:
        raise RuntimeError("routing baseline mismatch")
    baseline_city = {
        city: counts(global_score, indices, gt, rows)
        for city, rows in city_rows.items()
    }
    rows = []
    routed_masks = {}
    for coverage in args.coverages:
        score, routed = routed_score(
            global_score, ranknet_ensemble, coverage
        )
        result = counts(score, indices, gt)
        row = {
            "coverage": coverage,
            "routed_queries": int(routed.sum()),
            **{f"hits@{k}": result[k] for k in (1, 5, 10, 20)},
        }
        for city, city_index in city_rows.items():
            city_count = counts(score, indices, gt, city_index)
            row[f"{city}_hits@1"] = city_count[1]
        rows.append(row)
        routed_masks[coverage] = routed
        print(json.dumps(row), flush=True)
    selected = sorted(
        rows, key=lambda row: (
            -row["hits@1"], -row["hits@5"], row["coverage"]
        )
    )[0]
    individual = []
    for seed, seed_score in seed_scores.items():
        score, _ = routed_score(
            global_score, seed_score, selected["coverage"]
        )
        result = counts(score, indices, gt)
        individual.append({
            "seed": seed,
            **{f"hits@{k}": result[k] for k in (1, 5, 10, 20)},
        })
    passed = (
        selected["hits@1"] >= 686
        and selected["hits@5"] >= 712
        and selected["hits@20"] == 719
        and selected["cph_hits@1"] >= baseline_city["cph"][1]
        and selected["sf_hits@1"] >= baseline_city["sf"][1]
        and sum(row["hits@1"] >= 685 for row in individual) >= 2
    )
    with (args.output_root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "MSLS cross-city RankNet routed by global Top1-Top2 gap",
        "coverages": args.coverages,
        "baseline": baseline,
        "baseline_cities": baseline_city,
        "selected": selected,
        "individual_at_selected": individual,
        "validation_gate_passed": passed,
        "test_data_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
