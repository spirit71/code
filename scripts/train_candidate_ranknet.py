#!/usr/bin/env python3
"""GSV-train 监督 Candidate RankNet，MSLS-val 选择与评估。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_c1_soft_gate_msls import load_msls_features
from scripts.evaluate_c1_top20_reranking_msls import (
    global_candidate_scores,
    rank_weights,
    recall_from_candidates,
    zscore,
)
from scripts.train_u0_query_utility import UtilityHead, build_static_rank_prior, predict

FEATURE_NAMES = (
    "global", "qmax_mean", "same_mean", "chamfer", "mnn",
    "static_qmax", "c1_qmax",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsv-city-cache-root", type=Path, required=True)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--candidate-cache-root", type=Path, required=True)
    parser.add_argument("--c1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args()


class CandidateRankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] != 20:
            raise ValueError(f"bad candidate features {tuple(features.shape)}")
        # 第 0 维固定为 per-query z-scored global similarity。
        return features[..., 0] + self.mlp(features).squeeze(-1)


def load_gsv(city_root):
    descriptors, labels, image_ids = [], [], []
    offset = 0
    for path in sorted((city_root / "cities").glob("train_*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        descriptors.append(item["descriptors"])
        labels.append(item["labels"] + offset)
        image_ids.extend(item["image_paths"])
        offset += int(item["labels"].max()) + 1
    descriptors, labels = torch.cat(descriptors), torch.cat(labels)
    if descriptors.shape != (9216, 8192) or labels.shape != (9216,):
        raise RuntimeError("bad GSV cache")
    return descriptors, labels, image_ids


def load_c1_scores(train, val, c1_root, device):
    prior = build_static_rank_prior(train["contribution"])
    inputs = {
        "train": torch.cat([train["query_o2"], train["candidate_stats"]], -1),
        "val": torch.cat([val["query_o2"], val["candidate_stats"]], -1),
    }
    outputs = {"train": [], "val": []}
    for seed in (42, 43, 44):
        checkpoint = torch.load(
            c1_root / "delta_0.5" / f"seed_{seed}" / "best.pt",
            map_location="cpu", weights_only=True,
        )
        model = UtilityHead("c1", prior)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        for split in outputs:
            outputs[split].append(predict(model, inputs[split], device, 128))
    return (
        torch.stack(outputs["train"]).mean(0),
        torch.stack(outputs["val"]).mean(0),
        prior,
    )


@torch.inference_mode()
def pair_features(
    query_o2, reference_o2, indices, global_score, static_weight, c1_weight,
    device, batch_size,
):
    rows = []
    for start in range(0, len(query_o2), batch_size):
        stop = min(start + batch_size, len(query_o2))
        query = F.normalize(query_o2[start:stop].float().to(device), dim=-1)
        reference = F.normalize(
            reference_o2[indices[start:stop]].float().to(device), dim=-1
        )
        similarity = torch.einsum("bmd,bknd->bkmn", query, reference)
        qmax, qarg = similarity.max(3)
        rmax, rarg = similarity.max(2)
        same = similarity.diagonal(dim1=2, dim2=3)
        reverse_at_q = rarg.gather(2, qarg)
        mutual = reverse_at_q == torch.arange(64, device=device)[None, None]
        if not mutual.any(2).all():
            raise RuntimeError("candidate without mutual pair")
        sw = static_weight[start:stop].to(device)[:, None]
        cw = c1_weight[start:stop].to(device)[:, None]
        qmean = qmax.mean(2)
        rmean = rmax.mean(2)
        mnn = (qmax * mutual).sum(2) / mutual.sum(2)
        values = torch.stack([
            global_score[start:stop].to(device),
            qmean,
            same.mean(2),
            0.5 * (qmean + rmean),
            mnn,
            (qmax * sw).sum(2) / sw.sum(2),
            (qmax * cw).sum(2) / cw.sum(2),
        ], dim=-1)
        rows.append(values.cpu())
    features = torch.cat(rows)
    # 每个 query 的每一 feature 只在其 Top-20 内标准化。
    features = (features - features.mean(1, keepdim=True)) / (
        features.std(1, keepdim=True, unbiased=False) + 1e-8
    )
    if not torch.isfinite(features).all():
        raise RuntimeError("non-finite RankNet features")
    return features


def pairwise_loss(scores, labels):
    positive = labels[:, :, None]
    negative = ~labels[:, None, :]
    valid = positive & negative
    if not valid.any():
        raise RuntimeError("batch without positive-negative pairs")
    difference = scores[:, :, None] - scores[:, None, :]
    return F.softplus(-difference[valid]).mean(), int(valid.sum())


@torch.inference_mode()
def predict_scores(model, features, device, batch_size=256):
    result = []
    for start in range(0, len(features), batch_size):
        result.append(model(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(result)


def validation_counts(scores, indices, ground_truth):
    candidates = indices.gather(
        1, scores.argsort(dim=1, descending=True, stable=True)
    )
    if not torch.equal(candidates.sort(1).values, indices.sort(1).values):
        raise RuntimeError("RankNet changed candidate set")
    return recall_from_candidates(candidates, ground_truth)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_root / "features.pt"
    train_cache = torch.load(
        args.candidate_cache_root / "gsv_c0.pt",
        map_location="cpu", weights_only=True,
    )
    val_cache = torch.load(
        args.candidate_cache_root / "msls_c0.pt",
        map_location="cpu", weights_only=True,
    )
    utility_val = torch.load(
        args.utility_cache_root / "msls_val.pt",
        map_location="cpu", weights_only=True,
    )
    gsv_desc, gsv_labels, gsv_ids = load_gsv(args.gsv_city_cache_root)
    if gsv_ids != train_cache["image_ids"]:
        raise RuntimeError("GSV order mismatch")
    train_indices = train_cache["candidate_indices"]
    val_indices = val_cache["candidate_indices"]
    train_labels = gsv_labels[train_indices] == gsv_labels[:, None]
    coverage = float(train_labels.any(1).float().mean())
    if coverage < 0.998:
        raise RuntimeError("unexpectedly low GSV positive coverage")

    train_c1, val_c1, prior = load_c1_scores(
        train_cache, val_cache, args.c1_root, args.device
    )
    if feature_path.is_file():
        item = torch.load(feature_path, map_location="cpu", weights_only=True)
        train_features, val_features = item["train"], item["val"]
    else:
        msls_refs, msls_queries, _ = load_msls_features(args.utility_cache_root)
        train_global = global_candidate_scores(
            gsv_desc, gsv_desc, train_indices, args.device
        )
        val_global = global_candidate_scores(
            msls_queries, msls_refs, val_indices, args.device
        )
        train_features = pair_features(
            train_cache["query_o2"], train_cache["query_o2"], train_indices,
            train_global,
            rank_weights(prior[None].expand(9216, -1)),
            rank_weights(train_c1),
            args.device, args.feature_batch_size,
        )
        reference_o2 = []
        for path in sorted(
            (args.candidate_cache_root / "msls_reference_o2_shards").glob("*.pt")
        ):
            reference_o2.append(torch.load(
                path, map_location="cpu", weights_only=True
            )["o2"])
        reference_o2 = torch.cat(reference_o2)
        val_features = pair_features(
            val_cache["query_o2"], reference_o2, val_indices, val_global,
            rank_weights(prior[None].expand(740, -1)),
            rank_weights(val_c1),
            args.device, args.feature_batch_size,
        )
        torch.save({
            "feature_names": FEATURE_NAMES,
            "train": train_features.half(),
            "val": val_features.half(),
            "train_positive_labels": train_labels,
        }, feature_path)
    if train_features.shape != (9216, 20, 7):
        raise RuntimeError("bad train RankNet features")
    if val_features.shape != (740, 20, 7):
        raise RuntimeError("bad val RankNet features")
    baseline = validation_counts(
        val_features[..., 0], val_indices, utility_val["ground_truth"]
    )
    if baseline != {1: 685, 5: 712, 10: 717, 20: 719}:
        raise RuntimeError(f"epoch-0 parity failed {baseline}")

    variants = {"global_only": [0], "full7": list(range(7))}
    all_results, best_models = [], {}
    valid_train = train_labels.any(1)
    valid_indices = valid_train.nonzero(as_tuple=False).flatten()
    for variant, columns in variants.items():
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = CandidateRankNet(len(columns)).to(args.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            # CandidateRankNet 要求第 0 输入始终是 global。
            train_x = train_features[..., columns].float()
            val_x = val_features[..., columns].float()
            generator = torch.Generator().manual_seed(seed)
            history, best_key, best_state = [], None, None
            for epoch in range(1, args.epochs + 1):
                model.train()
                permutation = valid_indices[
                    torch.randperm(len(valid_indices), generator=generator)
                ]
                loss_sum, pairs_sum, query_sum = 0.0, 0, 0
                for start in range(0, len(permutation), args.train_batch_size):
                    index = permutation[start:start + args.train_batch_size]
                    x = train_x[index].to(args.device)
                    y = train_labels[index].to(args.device)
                    scores = model(x)
                    loss, pairs = pairwise_loss(scores, y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if any(
                        parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    ):
                        raise RuntimeError("RankNet gradient failure")
                    optimizer.step()
                    loss_sum += float(loss) * len(index)
                    pairs_sum += pairs
                    query_sum += len(index)
                model.eval()
                val_score = predict_scores(model, val_x, args.device)
                counts = validation_counts(
                    val_score, val_indices, utility_val["ground_truth"]
                )
                row = {
                    "variant": variant, "seed": seed, "epoch": epoch,
                    "train_loss": loss_sum / query_sum,
                    "train_pairs": pairs_sum,
                    **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
                }
                history.append(row)
                key = (counts[1], counts[5], -epoch)
                if best_key is None or key > best_key:
                    best_key = key
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            output = args.output_root / variant / f"seed_{seed}"
            output.mkdir(parents=True, exist_ok=True)
            torch.save({
                "variant": variant, "columns": columns, "seed": seed,
                "state_dict": best_state, "best_key": best_key,
                "feature_names": FEATURE_NAMES,
            }, output / "best.pt")
            with (output / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            model.load_state_dict(best_state, strict=True)
            model.eval()
            score = predict_scores(model, val_x, args.device)
            counts = validation_counts(
                score, val_indices, utility_val["ground_truth"]
            )
            result = {
                "variant": variant, "seed": seed,
                "best_epoch": -best_key[2],
                **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            }
            all_results.append(result)
            best_models[(variant, seed)] = score
            print(json.dumps(result), flush=True)

    ensembles = {}
    for variant in variants:
        score = torch.stack([
            best_models[(variant, seed)] for seed in args.seeds
        ]).mean(0)
        counts = validation_counts(
            score, val_indices, utility_val["ground_truth"]
        )
        ensembles[variant] = {
            "variant": variant,
            **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
        }
    full = ensembles["full7"]
    global_only = ensembles["global_only"]
    full_seeds = [row for row in all_results if row["variant"] == "full7"]
    passed = (
        full["hits@1"] >= 686
        and full["hits@5"] >= 712
        and full["hits@20"] == 719
        and full["hits@1"] > global_only["hits@1"]
        and sum(row["hits@1"] >= 685 for row in full_seeds) >= 2
    )
    summary = {
        "protocol": "GSV-train supervised candidate RankNet -> MSLS-val",
        "feature_names": FEATURE_NAMES,
        "gsv_top20_positive_coverage": coverage,
        "baseline_counts": baseline,
        "seed_results": all_results,
        "ensembles": ensembles,
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
