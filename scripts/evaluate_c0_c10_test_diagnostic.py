#!/usr/bin/env python3
"""冻结 MSLS-val 配置，在六个 held-out benchmark 上诊断 C0-C10。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from scripts.evaluate_c1_bidirectional_reranking_msls import local_scores
from scripts.evaluate_c1_hard_removal_msls import make_keep_mask, masked_descriptor
from scripts.evaluate_c1_top20_reranking_msls import (
    global_candidate_scores,
    local_slot_candidate_scores,
    rank_weights,
    zscore,
)
from scripts.train_candidate_ranknet import (
    CandidateRankNet,
    FEATURE_NAMES,
    pair_features,
    predict_scores,
)
from scripts.train_u0_query_utility import (
    UtilityHead,
    build_static_rank_prior,
    predict,
)
from src.dataloaders.datamodule import TEST_DATASET_BUILDERS


DATASET_ROOTS = {
    "pitts30k-test": Path("/root/data/Pittsburgh/pitts30k"),
    "nordland": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/Nordland/images"),
    "sped": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/sped/images"),
    "amstertime": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/amstertime/images"),
    "tokyo247": Path("/root/data/Tokyo247/images"),
    "svox-all": Path("/home/code_qy_7_28/VPR-datasets-downloader/datasets/svox/images"),
}
EXPECTED_SHA = "ec3431ae1728ae708b449c06dc016fcf6ce55ae852083911e88ac3a4dfb37d2e"
TOP_K = 20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=[*DATASET_ROOTS, "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def extract_cache(model, dataset, cache_root, checkpoint_sha, args):
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_size = args.batch_size * 20
    transform_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False, persistent_workers=False,
    )
    shard_desc, shard_o2, shard_indices = [], [], []
    shard_start = 0
    for images, indices in transform_loader:
        with torch.autocast("cuda", dtype=torch.float16):
            descriptor, aux = model(images.to(args.device, non_blocking=True), return_aux=True)
        outputs = aux["query_outputs_raw"]
        if descriptor.shape[1:] != (8192,) or outputs.shape[1:] != (2, 64, 512):
            raise RuntimeError(f"bad model outputs {descriptor.shape}, {outputs.shape}")
        if not torch.isfinite(descriptor).all() or not torch.isfinite(outputs).all():
            raise RuntimeError("non-finite model outputs")
        shard_desc.append(descriptor.half().cpu())
        shard_o2.append(outputs.half().cpu())
        shard_indices.extend(indices.tolist())
        if len(shard_indices) >= shard_size or int(indices[-1]) == len(dataset) - 1:
            stop = shard_start + len(shard_indices)
            if shard_indices != list(range(shard_start, stop)):
                raise RuntimeError("non-contiguous extraction order")
            path = cache_root / f"{shard_start:06d}_{stop:06d}.pt"
            atomic_save({
                "checkpoint_sha256": checkpoint_sha, "start": shard_start, "stop": stop,
                "descriptors": torch.cat(shard_desc), "outputs": torch.cat(shard_o2),
            }, path)
            print(f"[extract] {dataset.dataset_name} {stop}/{len(dataset)}", flush=True)
            shard_start = stop
            shard_desc, shard_o2, shard_indices = [], [], []


def load_or_extract(model, dataset, cache_root, checkpoint_sha, args):
    shards = sorted(cache_root.glob("*.pt"))
    complete = False
    if shards:
        expected = 0
        complete = True
        for path in shards:
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (
                item["checkpoint_sha256"] != checkpoint_sha
                or item["start"] != expected
                or item["descriptors"].shape != (item["stop"] - item["start"], 8192)
                or item["outputs"].shape != (item["stop"] - item["start"], 2, 64, 512)
            ):
                complete = False
                break
            expected = item["stop"]
        complete = complete and expected == len(dataset)
    if not complete:
        if shards:
            raise RuntimeError(f"incomplete/invalid cache exists: {cache_root}")
        extract_cache(model, dataset, cache_root, checkpoint_sha, args)
        shards = sorted(cache_root.glob("*.pt"))
    descriptors, o2, expected = [], [], 0
    for path in shards:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] != expected:
            raise RuntimeError("cache gap")
        expected = item["stop"]
        descriptors.append(item["descriptors"])
        o2.append(item["outputs"])
    descriptor, o2 = torch.cat(descriptors), torch.cat(o2)
    if descriptor.shape != (len(dataset), 8192) or o2.shape != (len(dataset), 2, 64, 512):
        raise RuntimeError("assembled cache shape mismatch")
    return descriptor, o2


@torch.inference_mode()
def retrieve_top20(queries, references, device):
    references = references.float().to(device)
    ref_norm = (references * references).sum(1)[None]
    rows, scores = [], []
    for start in range(0, len(queries), 64):
        query = queries[start:start + 64].float().to(device)
        distance = (query * query).sum(1, keepdim=True) + ref_norm - 2 * query @ references.T
        values, indices = distance.topk(TOP_K, dim=1, largest=False)
        rows.append(indices.cpu())
        scores.append((-values).cpu())
    return torch.cat(rows), torch.cat(scores)


def recall(indices, ground_truth):
    counts = {}
    for k in (1, 5, 10, 20):
        hits = 0
        for row, candidate in enumerate(indices[:, :k]):
            gt = set(int(value) for value in ground_truth[row])
            hits += int(any(int(value) in gt for value in candidate))
        counts[k] = hits
    return counts


def ordered_candidates(scores, indices):
    output = indices.gather(
        1, scores.argsort(dim=1, descending=True, stable=True)
    )
    if not torch.equal(output.sort(1).values, indices.sort(1).values):
        raise RuntimeError("candidate set changed")
    return output


def load_c1(train_cache, query_o2, reference_o2, indices, args):
    prior = build_static_rank_prior(train_cache["contribution"])
    local = local_slot_candidate_scores(
        query_o2, reference_o2, indices, args.device, args.feature_batch_size, "max"
    )
    top2 = local.topk(2, dim=2).values
    stats = torch.stack([
        top2[..., 0], top2[..., 1], top2[..., 0] - top2[..., 1],
        local.mean(2), local.std(2, unbiased=False), top2[..., 0] - local.mean(2),
    ], -1).half()
    c1_input = torch.cat([query_o2, stats], -1)
    seed_scores = []
    for seed in args.seeds:
        checkpoint = torch.load(
            Path("/root/qrl_c1_static_candidate_residual_msls") /
            "delta_0.5" / f"seed_{seed}" / "best.pt",
            map_location="cpu", weights_only=True,
        )
        head = UtilityHead("c1", prior)
        head.load_state_dict(checkpoint["state_dict"], strict=True)
        head.eval().requires_grad_(False).to(args.device)
        seed_scores.append(predict(head, c1_input, args.device, 128))
    return torch.stack(seed_scores).mean(0), prior, local


def load_ranknet_ensemble(root: Path, variant: str, features, seeds, device):
    scores = []
    columns = list(range(features.shape[-1]))
    for seed in seeds:
        checkpoint = torch.load(
            root / variant / f"seed_{seed}" / "best.pt",
            map_location="cpu", weights_only=True,
        )
        columns = checkpoint["columns"]
        model = CandidateRankNet(len(columns))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().requires_grad_(False).to(device)
        scores.append(predict_scores(model, features[..., columns].float(), device))
    return torch.stack(scores).mean(0)


def load_fold_ensemble(root: Path, features, seeds, device):
    scores = []
    for seed in seeds:
        for train_city in ("cph", "sf"):
            path = root / f"seed_{seed}_train_{train_city}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = CandidateRankNet(7)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval().requires_grad_(False).to(device)
            scores.append(predict_scores(model, features.float(), device))
    return torch.stack(scores).mean(0)


def evaluate_dataset(name, model, checkpoint_sha, train_cache, args):
    started = time.perf_counter()
    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TEST_DATASET_BUILDERS[name](DATASET_ROOTS[name], transform)
    if args.smoke:
        raise RuntimeError("smoke uses the real dataset protocol and is not implemented as a subset")
    descriptor, all_o2 = load_or_extract(
        model, dataset, args.output_root / "cache" / name / "shards",
        checkpoint_sha, args,
    )
    reference_desc = descriptor[:dataset.num_references]
    query_desc = descriptor[dataset.num_references:]
    reference_o2 = all_o2[:dataset.num_references, -1]
    query_outputs = all_o2[dataset.num_references:]
    query_o2 = query_outputs[:, -1]
    indices, global_score = retrieve_top20(query_desc, reference_desc, args.device)
    baseline = recall(indices, dataset.ground_truth)
    c1, prior, cross_local = load_c1(
        train_cache, query_o2, reference_o2, indices, args
    )
    global_order = global_score.argsort(dim=1, descending=True, stable=True)
    if not torch.equal(global_order, torch.arange(20)[None].expand(len(indices), -1)):
        raise RuntimeError("global Top-20 order parity failed")

    # C3: validation-frozen removal count=2, query side only.
    keep = make_keep_mask(c1, 2, remove_high=False)
    c3_desc = masked_descriptor(query_outputs, keep, model, args.device, 128)
    c3_indices, _ = retrieve_top20(c3_desc, reference_desc, args.device)

    # C4: validation-frozen C1 weighting + cross-slot max + alpha=0.2.
    c4_local = (cross_local * rank_weights(c1)[..., None]).mean(1)
    c4_score = zscore(global_score) + 0.2 * zscore(c4_local)
    c4_indices = ordered_candidates(c4_score, indices)

    static_weight = rank_weights(prior[None].expand(len(query_o2), -1))
    features = pair_features(
        query_o2, reference_o2, indices, global_score,
        static_weight, rank_weights(c1), args.device, args.feature_batch_size,
    )
    if features.shape != (dataset.num_queries, 20, 7):
        raise RuntimeError(f"bad candidate features {features.shape}")

    c7_score = load_ranknet_ensemble(
        Path("/root/qrl_c7_candidate_ranknet_msls"), "full7",
        features, args.seeds, args.device,
    )
    c7_indices = ordered_candidates(c7_score, indices)
    c8_score = load_fold_ensemble(
        args.output_root / "fold_models" / "c8_e1",
        features, args.seeds, args.device,
    )
    c9_score = load_fold_ensemble(
        args.output_root / "fold_models" / "c9_e20",
        features, args.seeds, args.device,
    )
    c8_indices = ordered_candidates(c8_score, indices)
    c9_indices = ordered_candidates(c9_score, indices)

    methods = {
        "E0": indices,
        "C2_beta0": indices.clone(),
        "C3_remove2": c3_indices,
        "C4_crossmax_a0.2": c4_indices,
        "C5_same_a0": indices.clone(),
        "C6_chamfer_a0": indices.clone(),
        "C7_gsv_ranknet": c7_indices,
        "C8_crosscity_e1": c8_indices,
        "C9_crosscity_e20": c9_indices,
        "C10_route0": indices.clone(),
    }
    for identity in ("C2_beta0", "C5_same_a0", "C6_chamfer_a0", "C10_route0"):
        if not torch.equal(methods[identity], indices):
            raise RuntimeError(f"{identity} is not identical to E0")
    rows = []
    for method, candidates in methods.items():
        counts = recall(candidates, dataset.ground_truth)
        row = {
            "dataset": name, "method": method,
            "queries": dataset.num_queries, "references": dataset.num_references,
            **{f"hits@{k}": counts[k] for k in (1, 5, 10, 20)},
            **{f"r@{k}": counts[k] / dataset.num_queries for k in (1, 5, 10, 20)},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = args.output_root / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "frozen MSLS-val configuration; diagnostic held-out evaluation",
        "dataset": name,
        "checkpoint_sha256": checkpoint_sha,
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "fixed_configs": {
            "C2": "beta=0", "C3": "remove_low=2",
            "C4": "cross-slot max, C1, alpha=0.2",
            "C5": "alpha=0", "C6": "alpha=0",
            "C7": "GSV full7 seed ensemble",
            "C8": "MSLS cross-city 1-epoch 2-fold x 3-seed ensemble",
            "C9": "MSLS cross-city 20-epoch 2-fold x 3-seed ensemble",
            "C10": "coverage=0",
        },
        "c0_c1_retrieval_status": "predictor-only; no independent retrieval output",
        "test_gt_used_for_selection": False,
        "test_gt_used_only_for_final_recall": True,
        "all_models_frozen": True,
        "results": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_SHA:
        raise RuntimeError(f"unexpected checkpoint SHA256 {checkpoint_sha}")
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("E0 is not frozen")
    train_cache = torch.load(
        "/root/qrl_candidate_aware_features_k20/gsv_c0.pt",
        map_location="cpu", weights_only=True,
    )
    datasets = list(DATASET_ROOTS) if args.dataset == "all" else [args.dataset]
    all_rows = []
    for name in datasets:
        all_rows.extend(evaluate_dataset(name, model, checkpoint_sha, train_cache, args))
    with (args.output_root / "all_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_root / "summary.json").write_text(json.dumps({
        "checkpoint_sha256": checkpoint_sha,
        "datasets": datasets,
        "test_gt_used_for_selection": False,
        "results": all_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
