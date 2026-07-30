#!/usr/bin/env python3
"""Pitts30k-test 原生 E0 的可恢复、多正样本 O2 leave-one-query-out 诊断。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs
from src.dataloaders.datamodule import TEST_DATASET_BUILDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-shard-size", type=int, default=512)
    parser.add_argument("--loo-image-chunk", type=int, default=16)
    parser.add_argument("--search-query-chunk", type=int, default=128)
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    for name in ("batch_size", "feature_shard_size", "loo_image_chunk", "search_query_chunk"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or args.random_repeats < 1:
        raise ValueError("num_workers must be non-negative and random_repeats positive")
    if any(not 0 < ratio < 1 for ratio in args.mask_ratios):
        raise ValueError("mask ratios must be strictly between zero and one")
    if not args.device.startswith("cuda"):
        raise ValueError("Pitts full LOO requires CUDA")


def validate_dataset(dataset) -> dict:
    if dataset.num_references != 10000 or dataset.num_queries != 6816:
        raise RuntimeError(
            f"unexpected Pitts split: refs={dataset.num_references}, queries={dataset.num_queries}"
        )
    counts = np.asarray([len(gt) for gt in dataset.ground_truth], dtype=np.int64)
    if (counts == 0).any():
        raise RuntimeError("Pitts query without GT positive")
    all_gt = np.concatenate([np.asarray(gt, dtype=np.int64) for gt in dataset.ground_truth])
    if all_gt.min() < 0 or all_gt.max() >= dataset.num_references:
        raise RuntimeError("GT reference index out of range")
    return {
        "min": int(counts.min()), "mean": float(counts.mean()),
        "median": float(np.median(counts)), "max": int(counts.max()),
    }


@torch.inference_mode()
def extract_feature_shards(model, dataset, args, checkpoint_hash: str) -> tuple[torch.Tensor, ...]:
    feature_dir = args.output_dir / "feature_shards"
    feature_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    if args.smoke:
        total = min(total, dataset.num_references + 4)
    for start in range(0, total, args.feature_shard_size):
        stop = min(start + args.feature_shard_size, total)
        path = feature_dir / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if (
                cached["checkpoint_sha256"] != checkpoint_hash
                or cached["start"] != start or cached["stop"] != stop
                or cached["descriptors"].shape != (stop - start, 8192)
            ):
                raise RuntimeError(f"invalid cached feature shard: {path}")
            print(f"[feature cache] {path.name}", flush=True)
            continue
        loader = DataLoader(
            Subset(dataset, range(start, stop)), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=False,
        )
        descriptors, query_outputs = [], []
        for images, indices in loader:
            expected = torch.arange(start + len(descriptors) * args.batch_size,
                                    start + len(descriptors) * args.batch_size + len(indices))
            if not torch.equal(indices, expected):
                raise RuntimeError("feature loader index order changed")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                desc, aux = model(images.to(args.device, non_blocking=True), return_aux=True)
            projection_input = aux["query_outputs_raw"]
            if projection_input.shape[1:] != (2, 64, 512):
                raise RuntimeError(f"unexpected query outputs {tuple(projection_input.shape)}")
            # 正式 L2 排名保留 descriptor FP32；仅大体积 O1/O2 缓存使用 FP16。
            descriptors.append(desc.float().cpu())
            if start >= dataset.num_references:
                query_outputs.append(projection_input.half().cpu())
            elif stop > dataset.num_references:
                first_query = max(dataset.num_references - int(indices[0]), 0)
                if first_query < len(indices):
                    query_outputs.append(projection_input[first_query:].half().cpu())
        payload = {
            "checkpoint_sha256": checkpoint_hash, "start": start, "stop": stop,
            "descriptors": torch.cat(descriptors),
            "query_outputs": torch.cat(query_outputs) if query_outputs else torch.empty(0, 2, 64, 512),
        }
        atomic_torch_save(payload, path)
        print(f"[feature saved] {path.name}", flush=True)

    shards = []
    for path in sorted(feature_dir.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] >= total:
            continue
        shards.append(item)
    descriptors = torch.cat([item["descriptors"] for item in shards])
    if len(descriptors) != total:
        raise RuntimeError(f"feature cache incomplete: {len(descriptors)} != {total}")
    query_outputs = torch.cat([item["query_outputs"] for item in shards if item["query_outputs"].numel()])
    expected_queries = max(total - dataset.num_references, 0)
    if query_outputs.shape != (expected_queries, 2, 64, 512):
        raise RuntimeError(f"query cache shape mismatch: {tuple(query_outputs.shape)}")
    return descriptors[:dataset.num_references], descriptors[dataset.num_references:], query_outputs


def score_stats(similarity: torch.Tensor, ground_truth, query_offset: int = 0):
    """支持 [B,R] 或 [B,M,R]；一次按图排除全部 GT positives。"""
    squeeze = similarity.ndim == 2
    if squeeze:
        similarity = similarity[:, None, :]
    if similarity.ndim != 3:
        raise ValueError(f"unexpected similarity shape {tuple(similarity.shape)}")
    batch, variants, references = similarity.shape
    best = torch.empty(batch, variants, device=similarity.device)
    hardest = torch.empty_like(best)
    hard_negative = torch.empty_like(best)
    rank = torch.empty(batch, variants, dtype=torch.long, device=similarity.device)
    work = similarity.clone()
    for row in range(batch):
        positives = torch.as_tensor(
            np.asarray(ground_truth[query_offset + row], dtype=np.int64),
            device=similarity.device,
        )
        positive_scores = similarity[row, :, positives]
        best[row] = positive_scores.max(dim=-1).values
        hardest[row] = positive_scores.min(dim=-1).values
        rank[row] = 1 + (similarity[row] > best[row, :, None]).sum(dim=-1)
        work[row, :, positives] = -torch.inf
    hard_negative.copy_(work.max(dim=-1).values)
    if squeeze:
        return best[:, 0], hardest[:, 0], hard_negative[:, 0], rank[:, 0]
    return best, hardest, hard_negative, rank


@torch.inference_mode()
def compute_loo(model, refs_half, query_outputs_half, ground_truth, args, checkpoint_hash):
    loo_dir = args.output_dir / "loo_chunks"
    loo_dir.mkdir(parents=True, exist_ok=True)
    refs = torch.nn.functional.normalize(refs_half.float().to(args.device), dim=-1)
    queries = len(query_outputs_half)
    for start in range(0, queries, args.loo_image_chunk):
        stop = min(start + args.loo_image_chunk, queries)
        path = loo_dir / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if (
                cached["checkpoint_sha256"] != checkpoint_hash
                or cached["start"] != start or cached["stop"] != stop
                or cached["contribution_best"].shape != (stop - start, 64)
            ):
                raise RuntimeError(f"invalid LOO chunk: {path}")
            print(f"[LOO cache] {path.name}", flush=True)
            continue
        source = query_outputs_half[start:stop].to(args.device).float()
        baseline_desc = descriptor_from_query_outputs(source, model.aggregator.fc).float()
        baseline_desc = torch.nn.functional.normalize(baseline_desc, dim=-1)
        baseline_similarity = baseline_desc @ refs.T
        base_best, base_hard, base_neg, base_rank = score_stats(
            baseline_similarity, ground_truth, start
        )
        current = stop - start
        expanded = source[:, None].expand(current, 64, 2, 64, 512).clone()
        diagonal = torch.arange(64, device=args.device)
        expanded[:, diagonal, -1, diagonal] = 0
        masked_desc = descriptor_from_query_outputs(
            expanded.reshape(current * 64, 2, 64, 512), model.aggregator.fc
        ).float()
        masked_desc = torch.nn.functional.normalize(masked_desc, dim=-1)
        masked_similarity = (masked_desc @ refs.T).reshape(current, 64, -1)
        masked_best, masked_hard, masked_neg, masked_rank = score_stats(
            masked_similarity, ground_truth, start
        )
        base_best_margin = base_best - base_neg
        base_hard_margin = base_hard - base_neg
        masked_best_margin = masked_best - masked_neg
        masked_hard_margin = masked_hard - masked_neg
        contribution_best = base_best_margin[:, None] - masked_best_margin
        contribution_hard = base_hard_margin[:, None] - masked_hard_margin
        identity_best = (
            contribution_best
            - ((base_best[:, None] - masked_best) - (base_neg[:, None] - masked_neg))
        ).abs().max()
        identity_hard = (
            contribution_hard
            - ((base_hard[:, None] - masked_hard) - (base_neg[:, None] - masked_neg))
        ).abs().max()
        if max(float(identity_best), float(identity_hard)) > 2e-6:
            raise RuntimeError("contribution decomposition identity failed")
        payload = {
            "checkpoint_sha256": checkpoint_hash, "start": start, "stop": stop,
            "baseline_best_positive": base_best.cpu(),
            "baseline_hard_positive": base_hard.cpu(),
            "baseline_hard_negative": base_neg.cpu(),
            "baseline_rank": base_rank.cpu(),
            "baseline_best_margin": base_best_margin.cpu(),
            "baseline_hard_margin": base_hard_margin.cpu(),
            "masked_best_positive": masked_best.cpu(),
            "masked_hard_positive": masked_hard.cpu(),
            "masked_hard_negative": masked_neg.cpu(),
            "masked_rank": masked_rank.cpu(),
            "masked_best_margin": masked_best_margin.cpu(),
            "masked_hard_margin": masked_hard_margin.cpu(),
            "contribution_best": contribution_best.cpu(),
            "contribution_hard": contribution_hard.cpu(),
        }
        atomic_torch_save(payload, path)
        print(f"[LOO saved] {path.name}", flush=True)
        del source, expanded, baseline_desc, masked_desc, baseline_similarity, masked_similarity

    chunks = []
    for path in sorted(loo_dir.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] < queries:
            chunks.append(item)
    result = {}
    tensor_keys = [key for key in chunks[0] if isinstance(chunks[0][key], torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.cat([item[key] for item in chunks])
        if len(result[key]) != queries:
            raise RuntimeError(f"incomplete LOO result {key}: {len(result[key])} != {queries}")
    return result


@torch.inference_mode()
def descriptors_with_mask(model, query_outputs, keep, args):
    result = []
    for start in range(0, len(query_outputs), args.search_query_chunk):
        stop = min(start + args.search_query_chunk, len(query_outputs))
        current = query_outputs[start:stop].to(args.device).float()
        current[:, -1] *= keep[start:stop].to(args.device).unsqueeze(-1)
        result.append(descriptor_from_query_outputs(current, model.aggregator.fc).float().cpu())
    return torch.cat(result)


@torch.inference_mode()
def formal_l2_recall(refs_half, queries_half, ground_truth, args) -> dict[int, float]:
    refs = refs_half.float().to(args.device)
    ref_norm = (refs * refs).sum(dim=-1)
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    for start in range(0, len(queries_half), args.search_query_chunk):
        stop = min(start + args.search_query_chunk, len(queries_half))
        query = queries_half[start:stop].float().to(args.device)
        distance = (
            (query * query).sum(dim=-1, keepdim=True)
            + ref_norm[None]
            - 2 * (query @ refs.T)
        )
        prediction = distance.topk(20, largest=False).indices.cpu().numpy()
        for local, row in enumerate(prediction):
            positives = set(np.asarray(ground_truth[start + local], dtype=np.int64).tolist())
            for k in hits:
                hits[k] += int(any(int(index) in positives for index in row[:k]))
    return {k: hits[k] / len(queries_half) for k in hits}


def keep_mask(scores: torch.Tensor, ratio: float, remove_high: bool) -> torch.Tensor:
    remove = int(round(scores.shape[1] * ratio))
    keep = torch.ones_like(scores)
    if remove:
        indices = torch.argsort(scores, dim=1, descending=remove_high)[:, :remove]
        keep.scatter_(1, indices, 0)
    return keep


@torch.inference_mode()
def evaluate_masking(model, refs, query_outputs, ground_truth, stats, args):
    rows = []
    baseline_keep = torch.ones(len(query_outputs), 64)
    baseline_desc = descriptors_with_mask(model, query_outputs, baseline_keep, args)
    rows.append({"strategy": "baseline", "mask_ratio": 0.0, "repeat": "", **formal_l2_recall(
        refs, baseline_desc, ground_truth, args
    )})
    generator = torch.Generator().manual_seed(args.seed)
    for ratio in args.mask_ratios:
        for mode in ("best", "hard"):
            scores = stats[f"contribution_{mode}"]
            for direction, remove_high in (("low", False), ("high", True)):
                keep = keep_mask(scores, ratio, remove_high)
                desc = descriptors_with_mask(model, query_outputs, keep, args)
                rows.append({
                    "strategy": f"contribution_{mode}_{direction}",
                    "mask_ratio": ratio, "repeat": "",
                    **formal_l2_recall(refs, desc, ground_truth, args),
                })
        for repeat in range(args.random_repeats):
            random_scores = torch.rand(len(query_outputs), 64, generator=generator)
            keep = keep_mask(random_scores, ratio, False)
            desc = descriptors_with_mask(model, query_outputs, keep, args)
            rows.append({
                "strategy": "random", "mask_ratio": ratio, "repeat": repeat,
                **formal_l2_recall(refs, desc, ground_truth, args),
            })
    return rows


def write_outputs(args, dataset, stats, masking, metadata):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_path = args.output_dir / "query_level.csv"
    with query_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "query_image_id", "query_index", "num_gt_positives",
            "contribution_best", "contribution_hard",
            "baseline_best_margin", "baseline_hard_margin",
            "masked_best_margin", "masked_hard_margin",
            "delta_best_positive", "delta_hard_positive", "delta_hard_negative",
            "baseline_rank", "masked_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in range(dataset.num_queries):
            for query_index in range(64):
                writer.writerow({
                    "query_image_id": dataset.qImages[image],
                    "query_index": query_index,
                    "num_gt_positives": len(dataset.ground_truth[image]),
                    "contribution_best": float(stats["contribution_best"][image, query_index]),
                    "contribution_hard": float(stats["contribution_hard"][image, query_index]),
                    "baseline_best_margin": float(stats["baseline_best_margin"][image]),
                    "baseline_hard_margin": float(stats["baseline_hard_margin"][image]),
                    "masked_best_margin": float(stats["masked_best_margin"][image, query_index]),
                    "masked_hard_margin": float(stats["masked_hard_margin"][image, query_index]),
                    "delta_best_positive": float(
                        stats["baseline_best_positive"][image]
                        - stats["masked_best_positive"][image, query_index]
                    ),
                    "delta_hard_positive": float(
                        stats["baseline_hard_positive"][image]
                        - stats["masked_hard_positive"][image, query_index]
                    ),
                    "delta_hard_negative": float(
                        stats["baseline_hard_negative"][image]
                        - stats["masked_hard_negative"][image, query_index]
                    ),
                    "baseline_rank": int(stats["baseline_rank"][image]),
                    "masked_rank": int(stats["masked_rank"][image, query_index]),
                })
    with (args.output_dir / "masking_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["strategy", "mask_ratio", "repeat", 1, 5, 10, 20]
        )
        writer.writeheader()
        writer.writerows(masking)
    summary = dict(metadata)
    summary["contribution"] = {}
    for mode in ("best", "hard"):
        value = stats[f"contribution_{mode}"]
        summary["contribution"][mode] = {
            "mean": float(value.mean()), "std": float(value.std(unbiased=False)),
            "min": float(value.min()), "max": float(value.max()),
            "negative_fraction": float((value < 0).float().mean()),
            "positive_fraction": float((value > 0).float().mean()),
        }
    left, right = stats["contribution_best"], stats["contribution_hard"]
    rank_left = left.argsort(dim=1).argsort(dim=1).float()
    rank_right = right.argsort(dim=1).argsort(dim=1).float()
    summary["best_hard_per_image_spearman_mean"] = float(
        torch.nn.functional.cosine_similarity(
            rank_left - rank_left.mean(1, keepdim=True),
            rank_right - rank_right.mean(1, keepdim=True),
        ).mean()
    )
    summary["masking_rows"] = [
        {
            **{k: v for k, v in row.items() if not isinstance(k, int)},
            **{f"R@{k}": row[k] for k in (1, 5, 10, 20)},
        }
        for row in masking
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Pitts30k-test E0 多正样本 O2 LOO", "",
        f"- checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- refs/queries: `{summary['num_references']}/{summary['num_queries']}`",
        f"- GT positives: `{summary['gt_positive_count']}`",
        "- oracle uses test GT: `true`; deployable: `false`",
        f"- projection minimum cosine: `{summary['projection_minimum_cosine']:.8f}`",
        f"- best/hard per-image Spearman: `{summary['best_hard_per_image_spearman_mean']:.4f}`",
        f"- elapsed: `{summary['elapsed_seconds']/60:.2f} min`", "",
        "| Strategy | Mask | Repeat | R@1 | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in masking:
        lines.append(
            f"| {row['strategy']} | {row['mask_ratio']:.0%} | {row['repeat']} | "
            f"{100*row[1]:.2f} | {100*row[5]:.2f} | {100*row[10]:.2f} | {100*row[20]:.2f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TEST_DATASET_BUILDERS["pitts30k-test"](args.dataset_root, transform)
    gt_stats = validate_dataset(dataset)
    if args.smoke:
        dataset.num_queries = 4
        dataset.qImages = dataset.qImages[:4]
        dataset.ground_truth = dataset.ground_truth[:4]
        dataset.image_paths = dataset.image_paths[:dataset.num_references + 4]
    checkpoint_hash = sha256_file(args.checkpoint)
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("model parameters are not fully frozen")
    refs, query_desc, query_outputs = extract_feature_shards(
        model, dataset, args, checkpoint_hash
    )
    reconstructed = descriptors_with_mask(
        model, query_outputs, torch.ones(len(query_outputs), 64), args
    )
    projection_minimum_cosine = float(torch.nn.functional.cosine_similarity(
        reconstructed.float(), query_desc.float()
    ).min())
    if projection_minimum_cosine < 0.9999:
        raise RuntimeError(f"projection fidelity failed: {projection_minimum_cosine}")
    stats = compute_loo(
        model, refs, query_outputs, dataset.ground_truth, args, checkpoint_hash
    )
    masking = evaluate_masking(
        model, refs, query_outputs, dataset.ground_truth, stats, args
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "experiment": "Pitts30k E0 multi-positive O2 LOO",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "dataset": "pitts30k-test",
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "gt_positive_count": gt_stats,
        "model_variant": "e0", "gate_mode": "off",
        "protocol": "fixed gallery; query-side O2 LOO; FP32 cosine contribution; FP32 L2 Recall",
        "projection_minimum_cosine": projection_minimum_cosine,
        "all_parameters_frozen": True,
        "oracle_uses_test_gt": True, "deployable": False,
        "smoke": args.smoke, "elapsed_seconds": elapsed,
    }
    write_outputs(args, dataset, stats, masking, metadata)
    print(f"complete: {args.output_dir} ({elapsed/60:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
