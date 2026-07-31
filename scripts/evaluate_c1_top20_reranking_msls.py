#!/usr/bin/env python3
"""MSLS-val：使用 O2 local matching 和 C1 utility 重排 E0 Top-20。"""

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
from scripts.train_u0_query_utility import (
    UtilityHead,
    build_static_rank_prior,
    predict,
    rank_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--matching", choices=("max", "same"), default="max",
        help="max=跨 candidate slots 最大值；same=相同 query index cosine",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


def load_reference_o2(root: Path):
    result, expected_start = [], 0
    for path in sorted((root / "msls_reference_o2_shards").glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] != expected_start:
            raise RuntimeError("non-contiguous reference O2")
        expected_start = item["stop"]
        result.append(item["o2"])
    output = torch.cat(result)
    if output.shape != (18871, 64, 512) or expected_start != 18871:
        raise RuntimeError("bad reference O2 cache")
    return output


@torch.inference_mode()
def local_slot_candidate_scores(
    query_o2, reference_o2, indices, device, batch_size, matching="max"
):
    result = []
    for start in range(0, len(query_o2), batch_size):
        stop = min(start + batch_size, len(query_o2))
        query = F.normalize(query_o2[start:stop].float().to(device), dim=-1)
        candidate = F.normalize(
            reference_o2[indices[start:stop]].float().to(device), dim=-1
        )
        if matching == "max":
            # [B,64,512] × [B,20,64,512] -> [B,64,20,64] -> [B,64,20]
            similarity = torch.einsum("bmd,bknd->bmkn", query, candidate)
            similarity = similarity.max(dim=-1).values
        elif matching == "same":
            # query/reference 两侧使用相同 slot m，不引入跨 slot 最大化。
            similarity = torch.einsum("bmd,bkmd->bmk", query, candidate)
        else:
            raise ValueError(f"unsupported matching {matching}")
        if similarity.shape != (stop - start, 64, 20):
            raise RuntimeError("bad local score shape")
        result.append(similarity.cpu())
    output = torch.cat(result)
    if not torch.isfinite(output).all():
        raise RuntimeError("non-finite local scores")
    return output


@torch.inference_mode()
def global_candidate_scores(query_desc, reference_desc, indices, device):
    query_desc = query_desc.float().to(device)
    reference_desc = reference_desc.float().to(device)
    result = []
    for start in range(0, len(query_desc), 128):
        stop = min(start + 128, len(query_desc))
        query = query_desc[start:stop]
        candidate = reference_desc[indices[start:stop].to(device)]
        # 与候选挖掘完全相同的展开式，避免近似并列项因运算顺序交换。
        distance = (
            (query * query).sum(1, keepdim=True)
            + (candidate * candidate).sum(2)
            - 2 * torch.einsum("bd,bkd->bk", query, candidate)
        )
        result.append((-distance).cpu())
    return torch.cat(result)


def rank_weights(score):
    rank = rank_tensor(score, 1)
    weight = 0.5 + rank / 63
    if not torch.allclose(weight.mean(1), torch.ones(len(weight)), atol=1e-6):
        raise RuntimeError("rank weights do not have unit mean")
    return weight


def zscore(value):
    return (value - value.mean(1, keepdim=True)) / (
        value.std(1, keepdim=True, unbiased=False) + 1e-8
    )


def recall_from_candidates(candidates, ground_truth):
    counts = {}
    for k in (1, 5, 10, 20):
        hits = 0
        for index, row in enumerate(candidates[:, :k]):
            gt = set(ground_truth[index].tolist())
            hits += int(any(int(value) in gt for value in row))
        counts[k] = hits
    return counts


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
    if indices.shape != (740, 20):
        raise RuntimeError("bad Top-20 candidate indices")
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
        seed_scores[seed] = predict(
            model, val_input, args.device, 128
        )
    ensemble = torch.stack(list(seed_scores.values())).mean(0)

    references, queries, _ = load_msls_features(args.utility_cache_root)
    reference_o2 = load_reference_o2(args.candidate_cache_root)
    local = local_slot_candidate_scores(
        val["query_o2"], reference_o2, indices, args.device, args.batch_size,
        args.matching,
    )
    global_score = global_candidate_scores(
        queries, references, indices, args.device
    )
    # Alpha=0 的 global score 必须保持原 Top-20 顺序。
    # 完全并列时保持已缓存 Top-K 的原顺序，与 baseline topk 结果一致。
    global_order = global_score.argsort(dim=1, descending=True, stable=True)
    identity = torch.arange(20)[None].expand(740, -1)
    if not torch.equal(global_order, identity):
        mismatches = int((global_order != identity).any(1).sum())
        raise RuntimeError(f"global Top-20 order mismatch for {mismatches} queries")

    weightings = {
        "uniform": torch.ones(740, 64),
        "static": rank_weights(prior[None].expand(740, -1)),
        "c1": rank_weights(ensemble),
    }
    ground_truth = utility_val["ground_truth"]
    baseline = recall_from_candidates(indices, ground_truth)
    if baseline != {1: 685, 5: 712, 10: 717, 20: 719}:
        raise RuntimeError(f"baseline mismatch {baseline}")
    global_z = zscore(global_score)
    rows = []
    reranked_sets = {}
    for name, weight in weightings.items():
        local_score = (local * weight[..., None]).mean(1)
        local_z = zscore(local_score)
        for alpha in args.alphas:
            fused = global_z + alpha * local_z
            order = fused.argsort(dim=1, descending=True, stable=True)
            candidates = indices.gather(1, order)
            if not torch.equal(
                candidates.sort(1).values, indices.sort(1).values
            ):
                raise RuntimeError("reranking changed candidate set")
            counts = recall_from_candidates(candidates, ground_truth)
            row = {
                "weighting": name, "alpha": alpha,
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
                **{f"r{k}": counts[k] / 740 for k in (1, 5, 10, 20)},
            }
            rows.append(row)
            reranked_sets[(name, alpha)] = candidates
            print(json.dumps(row), flush=True)

    preference = {"c1": 0, "static": 1, "uniform": 2}
    selected = sorted(
        rows,
        key=lambda row: (
            -row["hits@1"], -row["hits@5"], row["alpha"],
            preference[row["weighting"]],
        ),
    )[0]
    individual = []
    if selected["weighting"] == "c1":
        for seed, score in seed_scores.items():
            local_score = (local * rank_weights(score)[..., None]).mean(1)
            fused = global_z + selected["alpha"] * zscore(local_score)
            candidates = indices.gather(
                1, fused.argsort(dim=1, descending=True, stable=True)
            )
            counts = recall_from_candidates(candidates, ground_truth)
            individual.append({
                "seed": seed,
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            })
    c1_same_alpha = next(
        row for row in rows
        if row["weighting"] == "c1" and row["alpha"] == selected["alpha"]
    )
    uniform_same_alpha = next(
        row for row in rows
        if row["weighting"] == "uniform" and row["alpha"] == selected["alpha"]
    )
    static_same_alpha = next(
        row for row in rows
        if row["weighting"] == "static" and row["alpha"] == selected["alpha"]
    )
    passed = (
        selected["hits@1"] >= baseline[1] + 1
        and selected["hits@5"] >= baseline[5]
        and selected["hits@20"] == baseline[20]
        and selected["weighting"] == "c1"
        and c1_same_alpha["hits@1"] >= uniform_same_alpha["hits@1"]
        and c1_same_alpha["hits@1"] >= static_same_alpha["hits@1"]
        and sum(row["hits@1"] >= baseline[1] for row in individual) >= 2
    )
    with (args.output_root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save({
        "local_slot_candidate_scores": local.half(),
        "candidate_indices": indices,
    }, args.output_root / "local_scores.pt")
    summary = {
        "protocol": f"MSLS-val E0 Top-20 O2 {args.matching}-slot local reranking",
        "matching": args.matching,
        "alphas": args.alphas,
        "weightings": list(weightings),
        "baseline_counts": baseline,
        "selected": selected,
        "c1_at_selected_alpha": c1_same_alpha,
        "uniform_at_selected_alpha": uniform_same_alpha,
        "static_at_selected_alpha": static_same_alpha,
        "individual_c1_at_selected_alpha": individual,
        "validation_gate_passed": passed,
        "candidate_set_unchanged": True,
        "test_data_used": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
