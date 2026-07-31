#!/usr/bin/env python3
"""MSLS-val：C1-weighted symmetric Chamfer / MNN Top-20 reranking。"""

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

from scripts.evaluate_c1_soft_gate_msls import load_msls_features
from scripts.evaluate_c1_top20_reranking_msls import (
    global_candidate_scores,
    load_reference_o2,
    rank_weights,
    recall_from_candidates,
    zscore,
)
from scripts.train_u0_query_utility import UtilityHead, build_static_rank_prior, predict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


@torch.inference_mode()
def correspondence_cache(query_o2, reference_o2, indices, device, batch_size):
    qmax_values, rmax_values, mutual_masks = [], [], []
    for start in range(0, len(query_o2), batch_size):
        stop = min(start + batch_size, len(query_o2))
        query = F.normalize(query_o2[start:stop].float().to(device), dim=-1)
        reference = F.normalize(
            reference_o2[indices[start:stop]].float().to(device), dim=-1
        )
        # [B,K,M,N]
        similarity = torch.einsum("bmd,bknd->bkmn", query, reference)
        q_value, q_arg = similarity.max(dim=3)  # [B,K,M]
        r_value, r_arg = similarity.max(dim=2)  # [B,K,N]
        reverse_at_q_choice = r_arg.gather(2, q_arg)
        query_slot = torch.arange(64, device=device)[None, None]
        mutual = reverse_at_q_choice == query_slot
        if not mutual.any(dim=2).all():
            raise RuntimeError("candidate without any mutual match")
        qmax_values.append(q_value.cpu())
        rmax_values.append(r_value.cpu())
        mutual_masks.append(mutual.cpu())
    return (
        torch.cat(qmax_values),
        torch.cat(rmax_values),
        torch.cat(mutual_masks),
    )


def local_scores(qmax, rmax, mutual, weight):
    # qmax/mutual [B,K,M]，weight [B,M]。
    expanded_weight = weight[:, None]
    forward = (qmax * expanded_weight).sum(2) / expanded_weight.sum(2)
    reverse = rmax.mean(2)
    chamfer = 0.5 * (forward + reverse)
    mutual_weight = expanded_weight * mutual
    denominator = mutual_weight.sum(2)
    if float(denominator.min()) <= 0:
        raise RuntimeError("zero MNN denominator")
    mnn = (qmax * mutual_weight).sum(2) / denominator
    return {"chamfer": chamfer, "mnn": mnn}


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
    indices = val["candidate_indices"]
    prior = build_static_rank_prior(train["contribution"])
    val_input = torch.cat([val["query_o2"], val["candidate_stats"]], dim=-1)
    seed_scores = {}
    for seed in args.seeds:
        checkpoint = torch.load(
            args.c1_root / "delta_0.5" / f"seed_{seed}" / "best.pt",
            map_location="cpu", weights_only=True,
        )
        model = UtilityHead("c1", prior)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(args.device).eval()
        seed_scores[seed] = predict(model, val_input, args.device, 128)
    ensemble = torch.stack(list(seed_scores.values())).mean(0)

    references, queries, _ = load_msls_features(args.utility_cache_root)
    reference_o2 = load_reference_o2(args.candidate_cache_root)
    qmax, rmax, mutual = correspondence_cache(
        val["query_o2"], reference_o2, indices, args.device, args.batch_size
    )
    if qmax.shape != (740, 20, 64) or mutual.shape != qmax.shape:
        raise RuntimeError("bad correspondence cache shape")
    global_score = global_candidate_scores(
        queries, references, indices, args.device
    )
    if not torch.equal(
        global_score.argsort(dim=1, descending=True, stable=True),
        torch.arange(20)[None].expand(740, -1),
    ):
        raise RuntimeError("alpha=0 global parity failed")
    global_z = zscore(global_score)
    weights = {
        "uniform": torch.ones(740, 64),
        "static": rank_weights(prior[None].expand(740, -1)),
        "c1": rank_weights(ensemble),
    }
    gt = utility_val["ground_truth"]
    baseline = recall_from_candidates(indices, gt)
    rows = []
    for weighting, weight in weights.items():
        locals_by_match = local_scores(qmax, rmax, mutual, weight)
        for matching, local in locals_by_match.items():
            local_z = zscore(local)
            for alpha in args.alphas:
                fused = global_z + alpha * local_z
                candidates = indices.gather(
                    1, fused.argsort(dim=1, descending=True, stable=True)
                )
                if not torch.equal(
                    candidates.sort(1).values, indices.sort(1).values
                ):
                    raise RuntimeError("candidate set changed")
                counts = recall_from_candidates(candidates, gt)
                row = {
                    "matching": matching, "weighting": weighting, "alpha": alpha,
                    **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    preference = {"c1": 0, "static": 1, "uniform": 2}
    selected = sorted(rows, key=lambda row: (
        -row["hits@1"], -row["hits@5"], row["alpha"],
        preference[row["weighting"]], row["matching"],
    ))[0]
    same_config = [
        row for row in rows
        if row["matching"] == selected["matching"]
        and row["alpha"] == selected["alpha"]
    ]
    individual = []
    if selected["weighting"] == "c1":
        for seed, score in seed_scores.items():
            local = local_scores(
                qmax, rmax, mutual, rank_weights(score)
            )[selected["matching"]]
            fused = global_z + selected["alpha"] * zscore(local)
            candidates = indices.gather(
                1, fused.argsort(dim=1, descending=True, stable=True)
            )
            counts = recall_from_candidates(candidates, gt)
            individual.append({
                "seed": seed,
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            })
    comparator = {row["weighting"]: row for row in same_config}
    passed = (
        selected["hits@1"] >= baseline[1] + 1
        and selected["hits@5"] >= baseline[5]
        and selected["hits@20"] == baseline[20]
        and selected["weighting"] == "c1"
        and selected["hits@1"] >= comparator["uniform"]["hits@1"]
        and selected["hits@1"] >= comparator["static"]["hits@1"]
        and sum(row["hits@1"] >= baseline[1] for row in individual) >= 2
    )
    with (args.output_root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "MSLS-val bidirectional O2 Top-20 reranking",
        "baseline_counts": baseline,
        "selected": selected,
        "comparators_at_selected_matching_alpha": comparator,
        "individual_c1": individual,
        "mutual_matches_mean": float(mutual.float().sum(2).mean()),
        "validation_gate_passed": passed,
        "candidate_set_unchanged": True,
        "test_data_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {"qmax": qmax.half(), "rmax": rmax.half(), "mutual": mutual},
        args.output_root / "correspondence_cache.pt",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
