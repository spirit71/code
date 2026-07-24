#!/usr/bin/env python
"""Training-free X_L reranking for BoQ checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_checkpoints_in_dir import build_model_and_datamodule
from src import utils
from src.xl_reranking import RerankConfig, rerank_topk
from train import HyperParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training-free X_L reranking on BoQ test datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="BoQ Lightning checkpoint to evaluate.")
    parser.add_argument("--datasets", default="", help="Comma-separated test dataset names. Empty means all configured test sets.")
    parser.add_argument("--backbone", default="dinov2_vitb14", help="Backbone name used by the checkpoint.")
    parser.add_argument("--unfreeze-n", type=int, default=2, help="Number of unfrozen backbone blocks used by the checkpoint.")
    parser.add_argument("--output-dim", type=int, default=8192, help="BoQ output dimensionality.")
    parser.add_argument("--num-queries", type=int, default=64, help="Number of BoQ queries.")
    parser.add_argument("--eval-bs", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--nw", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=int, default=0, help="GPU device index.")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU.")

    parser.add_argument("--top-k", type=int, default=50, help="Global retrieval candidate count for reranking.")
    parser.add_argument("--global-weight", type=float, default=0.7, help="Weight of min-max normalized global score.")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Minimum token cosine similarity for MNN matches.")
    parser.add_argument("--spatial-sigma", type=float, default=0.15, help="Sigma for displacement residual spatial score.")
    parser.add_argument("--spatial-weight", type=float, default=0.3, help="Weight of spatial consistency inside local pair score.")
    parser.add_argument("--min-matches", type=int, default=4, help="Minimum MNN matches required for spatial consistency.")
    parser.add_argument("--debug-num-queries", type=int, default=None, help="Only rerank the first N queries per dataset.")
    parser.add_argument("--rerank-device", default=None, help="Device for X_L reranking. Defaults to the inference device.")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def select_test_sets(hparams: HyperParams, dataset_names: str) -> None:
    if not dataset_names:
        return
    requested = [name.strip() for name in dataset_names.split(",") if name.strip()]
    unknown = [name for name in requested if name not in hparams.test_sets]
    if unknown:
        raise ValueError(f"Unknown test datasets: {unknown}. Available: {list(hparams.test_sets)}")
    hparams.test_sets = {name: hparams.test_sets[name] for name in requested}


def extract_dataset_features(model, dataloader, device: torch.device) -> tuple[dict, float]:
    model.eval()
    global_chunks = []
    local_chunks = []
    index_chunks = []
    spatial_shape = None
    attention_shape = None

    start = time.perf_counter()
    with torch.inference_mode():
        for images, indices in dataloader:
            images = images.to(device, non_blocking=True)
            features = model.aggregator(model.backbone(images), return_local=True)
            if spatial_shape is None:
                spatial_shape = features["spatial_shape"]
                attention = features.get("attention")
                attention_shape = tuple(attention.shape) if attention is not None else None
            global_chunks.append(features["global"].detach().cpu())
            local_chunks.append(features["local"].detach().cpu())
            index_chunks.append(indices.detach().cpu())

    indices = torch.cat(index_chunks, dim=0)
    order = indices.argsort()
    output = {
        "global": torch.cat(global_chunks, dim=0)[order].float(),
        "local": torch.cat(local_chunks, dim=0)[order].float(),
        "indices": indices[order],
        "spatial_shape": spatial_shape,
        "attention_shape": attention_shape,
    }
    return output, time.perf_counter() - start


def format_recalls(recalls: dict[int, float]) -> str:
    return "  ".join(f"R@{k}: {100.0 * recalls[k]:.2f}" for k in [1, 5, 10, 20])


def assert_baseline_matches_original(result_recalls: dict[int, float], original_recalls: dict[int, float]) -> None:
    mismatches = []
    for k in [1, 5, 10, 20]:
        if abs(result_recalls[k] - original_recalls[k]) > 1e-12:
            mismatches.append((k, result_recalls[k], original_recalls[k]))
    if mismatches:
        details = ", ".join(
            f"R@{k}: rerank_baseline={rerank_value:.12f}, original={original_value:.12f}"
            for k, rerank_value, original_value in mismatches
        )
        raise RuntimeError(f"Baseline Recall mismatch against original project FAISS L2 metric: {details}")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.device}")

    hparams = HyperParams()
    hparams.backbone_name = args.backbone
    hparams.unfreeze_n_blocks = args.unfreeze_n
    hparams.output_dim = args.output_dim
    hparams.num_queries = args.num_queries
    hparams.eval_batch_size = args.eval_bs
    hparams.num_workers = args.nw
    hparams.silent = True
    select_test_sets(hparams, args.datasets)

    model, datamodule = build_model_and_datamodule(hparams)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()
    datamodule.setup(stage="test")

    config = RerankConfig(
        top_k=args.top_k,
        global_weight=args.global_weight,
        similarity_threshold=args.similarity_threshold,
        spatial_sigma=args.spatial_sigma,
        spatial_weight=args.spatial_weight,
        min_matches=args.min_matches,
        debug_num_queries=args.debug_num_queries,
        rerank_device=args.rerank_device if args.rerank_device is not None else str(device),
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(
        "Rerank params: "
        f"top_k={config.top_k}, global_weight={config.global_weight}, "
        f"similarity_threshold={config.similarity_threshold}, spatial_sigma={config.spatial_sigma}, "
        f"spatial_weight={config.spatial_weight}, min_matches={config.min_matches}"
    )

    for dataloader in datamodule.test_dataloader():
        dataset = dataloader.dataset
        print(f"\n[{dataset.dataset_name}] extracting global descriptors and X_L ...")
        features, extraction_seconds = extract_dataset_features(model, dataloader, device)
        num_references = dataset.num_references
        ref_global = features["global"][:num_references]
        query_global = features["global"][num_references:]
        ref_local = features["local"][:num_references]
        query_local = features["local"][num_references:]
        assert ref_global.shape[0] == dataset.num_references
        assert query_global.shape[0] == dataset.num_queries
        assert ref_local.shape[0] == dataset.num_references
        assert query_local.shape[0] == dataset.num_queries

        original_baseline_recalls = None
        if config.debug_num_queries is None:
            original_baseline_recalls = utils.compute_recall_performance(
                features["global"],
                dataset.num_references,
                dataset.num_queries,
                dataset.ground_truth,
                k_values=[1, 5, 10, 20],
            )

        result = rerank_topk(
            ref_global=ref_global,
            query_global=query_global,
            ref_local=ref_local,
            query_local=query_local,
            spatial_shape=features["spatial_shape"],
            ground_truth=dataset.ground_truth,
            config=config,
        )
        result.latencies["feature_extraction_seconds"] = extraction_seconds
        if original_baseline_recalls is not None:
            assert_baseline_matches_original(result.baseline_recalls, original_baseline_recalls)
            print("Baseline verification: matches original project FAISS L2 Recall@K")
        else:
            print("Baseline verification: skipped because --debug-num-queries changes the evaluated query set")

        print(f"Spatial shape: {features['spatial_shape']}  attention shape: {features['attention_shape']}")
        print(f"Baseline: {format_recalls(result.baseline_recalls)}")
        print(f"Reranked: {format_recalls(result.reranked_recalls)}")
        print(
            "Top-1 transitions: "
            f"Fixed={result.transitions['fixed']}  "
            f"New Error={result.transitions['new_error']}  "
            f"Still Wrong={result.transitions['still_wrong']}  "
            f"Both Correct={result.transitions['both_correct']}  "
            f"Net Gain={result.transitions['net_gain']}"
        )
        print(
            "Latency: "
            f"feature_extraction={result.latencies['feature_extraction_seconds']:.3f}s  "
            f"global_retrieval={result.latencies['global_retrieval_seconds']:.3f}s  "
            f"reranking={result.latencies['reranking_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
