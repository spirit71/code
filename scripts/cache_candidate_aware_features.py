#!/usr/bin/env python3
"""缓存无 GT 的 Top-K candidate-aware query statistics。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import build_model
from src.dataloaders import MapillarySLSDataset

STAT_NAMES = ("top1", "top2", "top1_minus_top2", "mean", "std", "top1_minus_mean")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gsv-city-cache-root", type=Path, required=True)
    parser.add_argument("--utility-cache-root", type=Path, required=True)
    parser.add_argument("--msls-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--reference-shard-size", type=int, default=2048)
    parser.add_argument("--stats-chunk-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_gsv_descriptors(city_root: Path, expected_image_ids):
    paths = sorted((city_root / "cities").glob("train_*.pt"))
    if len(paths) != 18:
        raise RuntimeError(f"expected 18 GSV city caches, got {len(paths)}")
    descriptors, image_ids = [], []
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=True)
        descriptors.append(item["descriptors"])
        image_ids.extend(item["image_paths"])
    descriptors = torch.cat(descriptors)
    if descriptors.shape != (9216, 8192):
        raise RuntimeError(f"bad GSV descriptors {tuple(descriptors.shape)}")
    if list(expected_image_ids) != image_ids:
        raise RuntimeError("GSV descriptor/O2 image ordering mismatch")
    return descriptors


def load_msls_descriptors(shard_root: Path, checkpoint_hash: str):
    paths = sorted(shard_root.glob("*.pt"))
    if not paths:
        raise RuntimeError("missing MSLS descriptor shards")
    descriptors, expected_start = [], 0
    for path in paths:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError(f"checkpoint mismatch in {path}")
        if item["start"] != expected_start:
            raise RuntimeError(f"non-contiguous descriptor shard {path}")
        expected_start = item["stop"]
        descriptors.append(item["descriptors"])
    descriptors = torch.cat(descriptors)
    if descriptors.shape != (19611, 8192) or expected_start != 19611:
        raise RuntimeError("incomplete MSLS descriptor cache")
    return descriptors[:18871], descriptors[18871:]


@torch.inference_mode()
def extract_msls_reference_o2(model, dataset, checkpoint_hash, args):
    shard_root = args.output_root / "msls_reference_o2_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    total = dataset.num_references
    for start in range(0, total, args.reference_shard_size):
        stop = min(start + args.reference_shard_size, total)
        path = shard_root / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (
                item["checkpoint_sha256"] != checkpoint_hash
                or item["start"] != start or item["stop"] != stop
                or item["o2"].shape != (stop - start, 64, 512)
            ):
                raise RuntimeError(f"invalid reference O2 shard {path}")
            continue
        loader = DataLoader(
            Subset(dataset, range(start, stop)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        outputs = []
        for images, indices in loader:
            if int(indices.min()) < start or int(indices.max()) >= stop:
                raise RuntimeError("MSLS subset ordering error")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, aux = model(
                    images.to(args.device, non_blocking=True), return_aux=True
                )
            outputs.append(aux["query_outputs_raw"][:, -1].half().cpu())
        output = torch.cat(outputs)
        if output.shape != (stop - start, 64, 512):
            raise RuntimeError("bad extracted reference O2 shape")
        atomic_save({
            "checkpoint_sha256": checkpoint_hash,
            "start": start,
            "stop": stop,
            "o2": output,
        }, path)
        print(f"[reference O2] {stop}/{total}", flush=True)
    shards, expected_start = [], 0
    for path in sorted(shard_root.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] != expected_start:
            raise RuntimeError("non-contiguous reference O2 shards")
        expected_start = item["stop"]
        shards.append(item["o2"])
    result = torch.cat(shards)
    if result.shape != (18871, 64, 512) or expected_start != 18871:
        raise RuntimeError("incomplete reference O2 cache")
    return result


@torch.inference_mode()
def retrieve_topk(query_desc, reference_desc, k, device, self_indices=None):
    """只使用 descriptor；函数不接收 GT，避免候选选择泄漏。"""
    query_desc = query_desc.float().to(device)
    reference_desc = reference_desc.float().to(device)
    result = []
    for start in range(0, len(query_desc), 128):
        query = query_desc[start:start + 128]
        # 精确 L2，与正式 FAISS IndexFlatL2 的排序口径一致。
        distance = (
            (query * query).sum(1, keepdim=True)
            + (reference_desc * reference_desc).sum(1)[None]
            - 2 * query @ reference_desc.T
        )
        if self_indices is not None:
            rows = torch.arange(len(query), device=device)
            excluded = self_indices[start:start + len(query)].to(device)
            distance[rows, excluded] = torch.inf
        result.append(distance.topk(k, dim=1, largest=False).indices.cpu())
    return torch.cat(result)


@torch.inference_mode()
def candidate_stats_chunks(
    query_o2, reference_o2, candidate_indices, output_root, checkpoint_hash, args
):
    output_root.mkdir(parents=True, exist_ok=True)
    samples = len(query_o2)
    for start in range(0, samples, args.stats_chunk_size):
        stop = min(start + args.stats_chunk_size, samples)
        path = output_root / f"{start:05d}_{stop:05d}.pt"
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=True)
            if (
                item["checkpoint_sha256"] != checkpoint_hash
                or item["top_k"] != args.top_k
                or item["start"] != start or item["stop"] != stop
                or item["stats"].shape != (stop - start, 64, len(STAT_NAMES))
            ):
                raise RuntimeError(f"invalid candidate stats chunk {path}")
            continue
        query = F.normalize(
            query_o2[start:stop].float().to(args.device), dim=-1
        )
        indices = candidate_indices[start:stop]
        candidate = F.normalize(
            reference_o2[indices].float().to(args.device), dim=-1
        )
        # [b,64,512] x [b,K,64,512] -> [b,64,K,64] -> [b,64,K]
        all_slot_similarity = torch.einsum("bmd,bknd->bmkn", query, candidate)
        similarity = all_slot_similarity.max(dim=-1).values
        top2 = similarity.topk(2, dim=-1).values
        mean = similarity.mean(dim=-1)
        std = similarity.std(dim=-1, unbiased=False)
        stats = torch.stack([
            top2[..., 0],
            top2[..., 1],
            top2[..., 0] - top2[..., 1],
            mean,
            std,
            top2[..., 0] - mean,
        ], dim=-1)
        if stats.shape != (stop - start, 64, len(STAT_NAMES)):
            raise RuntimeError("candidate stats shape error")
        if not torch.isfinite(stats).all():
            raise RuntimeError("non-finite candidate stats")
        atomic_save({
            "checkpoint_sha256": checkpoint_hash,
            "top_k": args.top_k,
            "stat_names": STAT_NAMES,
            "start": start,
            "stop": stop,
            "candidate_indices": indices,
            "stats": stats.half().cpu(),
        }, path)
        if start % (args.stats_chunk_size * 20) == 0:
            print(f"[candidate stats] {output_root.name} {stop}/{samples}", flush=True)
    chunks, expected_start = [], 0
    for path in sorted(output_root.glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["start"] != expected_start:
            raise RuntimeError(f"non-contiguous stats chunks in {output_root}")
        expected_start = item["stop"]
        chunks.append(item)
    stats = torch.cat([item["stats"] for item in chunks])
    indices = torch.cat([item["candidate_indices"] for item in chunks])
    if expected_start != samples:
        raise RuntimeError("incomplete candidate stats")
    return stats, indices


def main():
    args = parse_args()
    if args.top_k < 2:
        raise ValueError("top-k must be at least 2")
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(args.checkpoint)
    started = time.perf_counter()

    train = torch.load(
        args.utility_cache_root / "gsv_global.pt",
        map_location="cpu", weights_only=True,
    )
    val = torch.load(
        args.utility_cache_root / "msls_val.pt",
        map_location="cpu", weights_only=True,
    )
    if train["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("GSV utility checkpoint mismatch")
    if val["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("MSLS utility checkpoint mismatch")
    gsv_desc = load_gsv_descriptors(
        args.gsv_city_cache_root, train["image_ids"]
    )
    msls_refs, msls_queries = load_msls_descriptors(
        args.utility_cache_root / "msls_feature_shards", checkpoint_hash
    )
    if args.smoke:
        # 32 GSV query/reference；self exclusion 后仍有足够 Top-20 candidates。
        smoke_count = 32
        gsv_desc = gsv_desc[:smoke_count]
        train = {
            key: value[:smoke_count] if isinstance(value, torch.Tensor)
            and len(value) == 9216 else value
            for key, value in train.items()
        }
        train["image_ids"] = train["image_ids"][:smoke_count]
    gsv_topk = retrieve_topk(
        gsv_desc, gsv_desc, args.top_k, args.device,
        self_indices=torch.arange(len(gsv_desc)),
    )
    rows = torch.arange(len(gsv_topk))[:, None]
    if (gsv_topk == rows).any():
        raise RuntimeError("GSV Top-K contains self")
    gsv_stats, gsv_indices = candidate_stats_chunks(
        train["query_o2"], train["query_o2"], gsv_topk,
        args.output_root / ("gsv_stats_smoke" if args.smoke else "gsv_stats_chunks"),
        checkpoint_hash, args,
    )
    atomic_save({
        "format_version": 1,
        "split": "train",
        "checkpoint_sha256": checkpoint_hash,
        "top_k": args.top_k,
        "candidate_protocol": "descriptor L2 Top-K; self excluded; no GT filtering",
        "stat_names": STAT_NAMES,
        "query_o2": train["query_o2"],
        "candidate_stats": gsv_stats,
        "candidate_indices": gsv_indices,
        "contribution": train["contribution"],
        "image_ids": train["image_ids"],
        "gt_used_for_candidate_selection": False,
    }, args.output_root / ("gsv_c0_smoke.pt" if args.smoke else "gsv_c0.pt"))
    if args.smoke:
        print("candidate-aware smoke complete", flush=True)
        return

    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    msls = MapillarySLSDataset(args.msls_root, transform)
    if msls.num_references != 18871 or msls.num_queries != 740:
        raise RuntimeError("unexpected MSLS cardinality")
    model = build_model(args.checkpoint, args.device, "e0", "off")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("E0 must be frozen")
    reference_o2 = extract_msls_reference_o2(
        model, msls, checkpoint_hash, args
    )
    msls_topk = retrieve_topk(
        msls_queries, msls_refs, args.top_k, args.device
    )
    msls_stats, msls_indices = candidate_stats_chunks(
        val["query_o2"], reference_o2, msls_topk,
        args.output_root / "msls_stats_chunks", checkpoint_hash, args,
    )
    atomic_save({
        "format_version": 1,
        "split": "validation",
        "checkpoint_sha256": checkpoint_hash,
        "top_k": args.top_k,
        "candidate_protocol": "descriptor L2 Top-K; query/reference disjoint; no GT filtering",
        "stat_names": STAT_NAMES,
        "query_o2": val["query_o2"],
        "candidate_stats": msls_stats,
        "candidate_indices": msls_indices,
        "contribution": val["contribution"],
        "image_ids": val["image_ids"],
        "gt_used_for_candidate_selection": False,
    }, args.output_root / "msls_c0.pt")
    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "top_k": args.top_k,
        "stat_names": STAT_NAMES,
        "gsv_shape": list(gsv_stats.shape),
        "msls_shape": list(msls_stats.shape),
        "gt_used_for_candidate_selection": False,
        "test_data_used": False,
        "all_boq_parameters_frozen": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
