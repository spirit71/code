#!/usr/bin/env python3
"""生成 GSV 全局 memory-bank 与 MSLS-val full-gallery O2 LOO utility labels。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs
from src.dataloaders import MapillarySLSDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gsv-cache-root", type=Path, required=True)
    parser.add_argument("--msls-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-shard-size", type=int, default=2048)
    parser.add_argument("--loo-chunk", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_gsv_train(root: Path):
    paths = sorted((root / "cities").glob("train_*.pt"))
    if len(paths) != 18:
        raise RuntimeError(f"expected 18 train city caches, got {len(paths)}")
    descriptors, outputs, image_ids, composite_labels = [], [], [], []
    next_label = 0
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["descriptors"].shape != (512, 8192):
            raise RuntimeError(f"bad city cache {path}")
        descriptors.append(item["descriptors"])
        outputs.append(item["projection_inputs"])
        image_ids.extend(item["image_paths"])
        local = item["labels"]
        composite_labels.append(local + next_label)
        next_label += int(local.max()) + 1
    descriptor = torch.cat(descriptors)
    output = torch.cat(outputs)
    labels = torch.cat(composite_labels)
    if descriptor.shape != (9216, 8192) or output.shape != (9216, 2, 64, 512):
        raise RuntimeError("unexpected global GSV cache shapes")
    counts = torch.bincount(labels)
    if len(counts) != 2304 or not torch.equal(counts, torch.full_like(counts, 4)):
        raise RuntimeError("GSV composite place labels are not exactly four-view")
    return descriptor, output, labels, image_ids


def positive_indices_from_labels(labels: torch.Tensor):
    groups = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(label, []).append(index)
    result, exclusions = [], []
    for index, label in enumerate(labels.tolist()):
        exclusions.append(groups[label])
        positives = [value for value in groups[label] if value != index]
        if len(positives) != 3:
            raise RuntimeError("expected exactly three GSV positives")
        result.append(positives)
    return (
        torch.tensor(result, dtype=torch.long),
        torch.tensor(exclusions, dtype=torch.long),
    )


def gt_positive_indices(ground_truth):
    return [
        torch.tensor(np.asarray(value, dtype=np.int64), dtype=torch.long)
        for value in ground_truth
    ]


def similarity_stats(
    similarity, positives, query_offset, negative_exclusions=None
):
    """返回最佳正样本和最难负样本相似度。

    GSV 的 reference bank 含 query 自身，因此正样本集合只含同地点另外
    三张图，而负样本排除集合必须含同地点四张图（包括 self）。MSLS 的
    query/reference 天然分离，正样本集合也就是负样本排除集合。
    """
    squeeze = similarity.ndim == 2
    if squeeze:
        similarity = similarity[:, None]
    batch, variants, references = similarity.shape
    best, negative = [], []
    work = similarity.clone()
    for row in range(batch):
        positive = positives[query_offset + row].to(similarity.device)
        excluded = (
            negative_exclusions[query_offset + row]
            if negative_exclusions is not None else positive
        ).to(similarity.device)
        if positive.numel() == 0:
            raise RuntimeError("query without positive")
        best.append(similarity[row, :, positive].max(dim=-1).values)
        work[row, :, excluded] = -torch.inf
    best = torch.stack(best)
    negative = work.max(dim=-1).values
    if squeeze:
        return best[:, 0], negative[:, 0]
    return best, negative


@torch.inference_mode()
def loo_chunks(
    model, references, outputs, positives, output_dir, checkpoint_hash, args,
    negative_exclusions=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    references = torch.nn.functional.normalize(references.float().to(args.device), dim=-1)
    samples = len(outputs)
    for start in range(0, samples, args.loo_chunk):
        stop = min(start + args.loo_chunk, samples)
        path = output_dir / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (
                item["checkpoint_sha256"] != checkpoint_hash
                or item["start"] != start or item["stop"] != stop
                or item["contribution"].shape != (stop - start, 64)
            ):
                raise RuntimeError(f"invalid LOO chunk {path}")
            continue
        source = outputs[start:stop].float().to(args.device)
        baseline_desc = descriptor_from_query_outputs(source, model.aggregator.fc).float()
        baseline_desc = torch.nn.functional.normalize(baseline_desc, dim=-1)
        baseline_similarity = baseline_desc @ references.T
        baseline_positive, baseline_negative = similarity_stats(
            baseline_similarity, positives, start, negative_exclusions
        )
        current = stop - start
        expanded = source[:, None].expand(current, 64, 2, 64, 512).clone()
        diagonal = torch.arange(64, device=args.device)
        expanded[:, diagonal, -1, diagonal] = 0
        masked_desc = descriptor_from_query_outputs(
            expanded.reshape(current * 64, 2, 64, 512), model.aggregator.fc
        ).float()
        masked_desc = torch.nn.functional.normalize(masked_desc, dim=-1)
        masked_similarity = (masked_desc @ references.T).reshape(current, 64, -1)
        masked_positive, masked_negative = similarity_stats(
            masked_similarity, positives, start, negative_exclusions
        )
        contribution = (
            (baseline_positive - baseline_negative)[:, None]
            - (masked_positive - masked_negative)
        )
        identity = (
            contribution
            - (
                (baseline_positive[:, None] - masked_positive)
                - (baseline_negative[:, None] - masked_negative)
            )
        ).abs().max()
        if float(identity) > 2e-6:
            raise RuntimeError(f"contribution identity failed: {float(identity)}")
        atomic_save({
            "checkpoint_sha256": checkpoint_hash, "start": start, "stop": stop,
            "baseline_positive": baseline_positive.cpu(),
            "baseline_negative": baseline_negative.cpu(),
            "masked_positive": masked_positive.cpu(),
            "masked_negative": masked_negative.cpu(),
            "contribution": contribution.cpu(),
        }, path)
        if start % (args.loo_chunk * 20) == 0:
            print(f"[LOO] {output_dir.name} {start}/{samples}", flush=True)

    chunks = []
    for path in sorted(output_dir.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] < samples:
            chunks.append(item)
    result = {}
    for key in (
        "baseline_positive", "baseline_negative", "masked_positive",
        "masked_negative", "contribution",
    ):
        result[key] = torch.cat([item[key] for item in chunks])
        if len(result[key]) != samples:
            raise RuntimeError(f"incomplete {output_dir.name} {key}")
    return result


@torch.inference_mode()
def extract_msls_shards(model, dataset, checkpoint_hash, args):
    shard_dir = args.output_root / "msls_feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    for start in range(0, total, args.feature_shard_size):
        stop = min(start + args.feature_shard_size, total)
        path = shard_dir / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (
                item["checkpoint_sha256"] != checkpoint_hash
                or item["start"] != start or item["stop"] != stop
                or item["descriptors"].shape != (stop - start, 8192)
            ):
                raise RuntimeError(f"invalid MSLS feature shard {path}")
            continue
        loader = DataLoader(
            Subset(dataset, range(start, stop)), batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, shuffle=False,
        )
        descriptors, query_outputs = [], []
        for images, indices in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                desc, aux = model(images.to(args.device, non_blocking=True), return_aux=True)
            descriptors.append(desc.float().cpu())
            outputs = aux["query_outputs_raw"]
            query_start = max(dataset.num_references - int(indices[0]), 0)
            if query_start < len(indices):
                query_outputs.append(outputs[query_start:].half().cpu())
        atomic_save({
            "checkpoint_sha256": checkpoint_hash, "start": start, "stop": stop,
            "descriptors": torch.cat(descriptors),
            "query_outputs": torch.cat(query_outputs)
            if query_outputs else torch.empty(0, 2, 64, 512),
        }, path)
        print(f"[MSLS feature] {start}/{total}", flush=True)
    shards = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in sorted(shard_dir.glob("*.pt"))
    ]
    expected_start = 0
    for item in shards:
        if item["start"] != expected_start:
            raise RuntimeError("non-contiguous MSLS feature shards")
        expected_start = item["stop"]
    if expected_start != total:
        raise RuntimeError("incomplete MSLS feature shard coverage")
    descriptors = torch.cat([item["descriptors"] for item in shards])
    query_outputs = torch.cat([
        item["query_outputs"] for item in shards if item["query_outputs"].numel()
    ])
    if descriptors.shape != (19611, 8192) or query_outputs.shape != (740, 2, 64, 512):
        raise RuntimeError("incomplete MSLS feature cache")
    return descriptors[:18871], descriptors[18871:], query_outputs


@torch.inference_mode()
def formal_msls_r1(references, queries, ground_truth, device):
    references = references.float().to(device)
    hits = 0
    for start in range(0, len(queries), 128):
        query = queries[start:start + 128].float().to(device)
        distance = (
            (query * query).sum(1, keepdim=True)
            + (references * references).sum(1)[None]
            - 2 * query @ references.T
        )
        prediction = distance.argmin(dim=1).cpu().tolist()
        for local, value in enumerate(prediction):
            hits += int(value in set(
                np.asarray(ground_truth[start + local], dtype=np.int64).tolist()
            ))
    return hits / len(queries)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(args.checkpoint)
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("E0 not fully frozen")
    started = time.perf_counter()

    gsv_desc, gsv_outputs, gsv_labels, gsv_ids = load_gsv_train(args.gsv_cache_root)
    if args.smoke:
        # 8 places，保持每 place 四视图。
        gsv_desc, gsv_outputs, gsv_labels, gsv_ids = (
            gsv_desc[:32], gsv_outputs[:32], gsv_labels[:32], gsv_ids[:32]
        )
    gsv_positive, gsv_negative_exclusions = positive_indices_from_labels(gsv_labels)
    gsv_stats = loo_chunks(
        model, gsv_desc, gsv_outputs, gsv_positive,
        args.output_root / ("gsv_loo_smoke" if args.smoke else "gsv_loo_chunks"),
        checkpoint_hash, args, gsv_negative_exclusions,
    )
    gsv_output = args.output_root / ("gsv_global_smoke.pt" if args.smoke else "gsv_global.pt")
    atomic_save({
        "format_version": 1, "split": "train",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "image_ids": gsv_ids, "query_o2": gsv_outputs[:, -1],
        "contribution": gsv_stats["contribution"],
        "baseline_positive": gsv_stats["baseline_positive"],
        "baseline_negative": gsv_stats["baseline_negative"],
        "negative_pool_size": len(gsv_desc) - 4,
        "all_parameters_frozen": True,
    }, gsv_output)
    if args.smoke:
        print("smoke complete", flush=True)
        return

    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    msls = MapillarySLSDataset(args.msls_root, transform)
    msls_refs, msls_query_desc, msls_outputs = extract_msls_shards(
        model, msls, checkpoint_hash, args
    )
    reconstructed = descriptor_from_query_outputs(
        msls_outputs.float().to(args.device), model.aggregator.fc
    ).cpu()
    minimum_cosine = float(F.cosine_similarity(
        reconstructed, msls_query_desc.float()
    ).min())
    if minimum_cosine < 0.9999:
        raise RuntimeError(f"MSLS projection fidelity failed {minimum_cosine}")
    baseline_r1 = formal_msls_r1(
        msls_refs, msls_query_desc, msls.ground_truth, args.device
    )
    if abs(baseline_r1 - 0.9256756756756757) > 1e-12:
        raise RuntimeError(f"MSLS baseline mismatch {baseline_r1}")
    reconstructed_r1 = formal_msls_r1(
        msls_refs, reconstructed, msls.ground_truth, args.device
    )
    msls_stats = loo_chunks(
        model, msls_refs, msls_outputs, gt_positive_indices(msls.ground_truth),
        args.output_root / "msls_loo_chunks", checkpoint_hash, args,
    )
    atomic_save({
        "format_version": 1, "split": "validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "image_ids": [str(value) for value in msls.qImages],
        "query_o2": msls_outputs[:, -1],
        "contribution": msls_stats["contribution"],
        "baseline_positive": msls_stats["baseline_positive"],
        "baseline_negative": msls_stats["baseline_negative"],
        # 保存为 Tensor，确保下游可使用 torch.load(weights_only=True) 安全读取。
        "ground_truth": gt_positive_indices(msls.ground_truth),
        "num_references": msls.num_references,
        "baseline_r1": baseline_r1,
        "reconstructed_r1": reconstructed_r1,
        "projection_minimum_cosine": minimum_cosine,
        "all_parameters_frozen": True,
    }, args.output_root / "msls_val.pt")
    elapsed = time.perf_counter() - started
    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "gsv": {
            "images": len(gsv_outputs), "negative_pool_size": len(gsv_desc) - 4,
            "contribution_mean": float(gsv_stats["contribution"].mean()),
            "contribution_std": float(gsv_stats["contribution"].std(unbiased=False)),
            "negative_fraction": float((gsv_stats["contribution"] < 0).float().mean()),
            "baseline_negative_mean": float(gsv_stats["baseline_negative"].mean()),
        },
        "msls": {
            "references": msls.num_references, "queries": msls.num_queries,
            "baseline_r1": baseline_r1,
            "reconstructed_r1": reconstructed_r1,
            "projection_minimum_cosine": minimum_cosine,
            "contribution_mean": float(msls_stats["contribution"].mean()),
            "contribution_std": float(msls_stats["contribution"].std(unbiased=False)),
            "negative_fraction": float((msls_stats["contribution"] < 0).float().mean()),
            "baseline_negative_mean": float(msls_stats["baseline_negative"].mean()),
        },
        "all_parameters_frozen": True,
        "test_data_used": False,
        "elapsed_seconds": elapsed,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
