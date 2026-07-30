#!/usr/bin/env python3
"""为已有 LOO 实验缓存冻结模型的 O2 和标准 descriptors。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_query_retrieval_contribution import (
    DATASET_ROOTS,
    build_model,
    extract_features,
    sha256_file,
)
from src.analysis.reliability_diagnostics import descriptor_from_query_outputs
from src.dataloaders.datamodule import TEST_DATASET_BUILDERS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-variant", choices=["e0", "e2"], required=True)
    parser.add_argument("--gate-mode", choices=["on", "off"], required=True)
    parser.add_argument("--dataset", choices=DATASET_ROOTS, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.model_variant == "e0" and args.gate_mode != "off":
        raise ValueError("E0 requires gate-mode=off")
    root = args.dataset_root or Path(DATASET_ROOTS[args.dataset])
    transform = T.Compose([
        T.Resize((322, 322), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = TEST_DATASET_BUILDERS[args.dataset](root, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, shuffle=False,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    print(f"[1/3] strict-loading {args.model_variant}/{args.gate_mode}", flush=True)
    model = build_model(
        args.checkpoint, args.device, args.model_variant, args.gate_mode
    )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("feature cache model must be fully frozen")
    print(f"[2/3] extracting {len(dataset)} images", flush=True)
    descriptors, raw, projection, reliability, _, _, _ = extract_features(
        model, dataset, loader, args.device
    )
    refs = descriptors[:dataset.num_references]
    queries = descriptors[dataset.num_references:]
    query_raw = raw[dataset.num_references:]
    query_projection = projection[dataset.num_references:]

    reconstructed = descriptor_from_query_outputs(
        query_projection.to(args.device), model.aggregator.fc
    ).cpu()
    minimum_cosine = float(
        torch.nn.functional.cosine_similarity(reconstructed, queries).min()
    )
    if minimum_cosine < 0.9999:
        raise RuntimeError(f"projection fidelity failed: {minimum_cosine}")
    payload = {
        "format_version": 1,
        "dataset": args.dataset,
        "model_variant": args.model_variant,
        "gate_mode": args.gate_mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "num_references": dataset.num_references,
        "num_queries": dataset.num_queries,
        "query_image_ids": list(dataset.qImages),
        "reference_descriptors": refs.half(),
        "query_descriptors": queries.half(),
        "query_raw_o2": query_raw[:, -1].half(),
        "query_projection_outputs": query_projection.half(),
        "reliability": reliability[dataset.num_references:].half(),
        "projection_minimum_cosine": minimum_cosine,
        "all_parameters_frozen": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata = {key: value for key, value in payload.items() if not torch.is_tensor(value)}
    metadata["tensor_shapes"] = {
        key: list(value.shape) for key, value in payload.items() if torch.is_tensor(value)
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[3/3] saved {args.output} "
        f"(O2={tuple(payload['query_raw_o2'].shape)}, cosine={minimum_cosine:.6f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
