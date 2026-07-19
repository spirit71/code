from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import faiss
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RerankConfig:
    top_k: int = 50
    global_weight: float = 0.7
    similarity_threshold: float = 0.5
    spatial_sigma: float = 0.15
    spatial_weight: float = 0.3
    min_matches: int = 4
    debug_num_queries: int | None = None
    rerank_device: str | None = None


@dataclass
class PairMatchResult:
    pair_score: float
    local_score: float
    quality: float
    coverage: float
    spatial_score: float
    num_matches: int


@dataclass
class RerankResult:
    baseline_indices: torch.Tensor
    reranked_indices: torch.Tensor
    baseline_scores: torch.Tensor
    pair_scores: torch.Tensor
    final_scores: torch.Tensor
    baseline_recalls: dict[int, float]
    reranked_recalls: dict[int, float]
    transitions: dict[str, int]
    latencies: dict[str, float]


def normalized_xy_grid(spatial_shape: tuple[int, int], device=None, dtype=torch.float32) -> torch.Tensor:
    """Return [H*W, 2] xy coordinates normalized to [0, 1]."""
    height, width = spatial_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid spatial shape: {spatial_shape}")

    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype) if height > 1 else torch.zeros(1, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype) if width > 1 else torch.zeros(1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def mutual_nearest_neighbor_matches(
    query_local: torch.Tensor,
    ref_local: torch.Tensor,
    similarity_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find thresholded mutual nearest-neighbor token matches."""
    if query_local.ndim != 2 or ref_local.ndim != 2:
        raise ValueError("query_local and ref_local must be [N, C] tensors")
    if query_local.shape[1] != ref_local.shape[1]:
        raise ValueError("query_local and ref_local channel dimensions must match")

    query_tokens = F.normalize(query_local, p=2, dim=-1)
    ref_tokens = F.normalize(ref_local, p=2, dim=-1)
    sim = query_tokens @ ref_tokens.transpose(0, 1)

    q_to_r = sim.argmax(dim=1)
    r_to_q = sim.argmax(dim=0)
    q_indices = torch.arange(sim.shape[0], device=sim.device)
    is_mutual = r_to_q[q_to_r] == q_indices
    matched_q = q_indices[is_mutual]
    matched_r = q_to_r[is_mutual]
    matched_sim = sim[matched_q, matched_r]

    keep = matched_sim >= similarity_threshold
    return matched_q[keep], matched_r[keep], matched_sim[keep]


def score_local_pair(
    query_local: torch.Tensor,
    ref_local: torch.Tensor,
    spatial_shape: tuple[int, int],
    similarity_threshold: float = 0.5,
    spatial_sigma: float = 0.15,
    spatial_weight: float = 0.3,
    min_matches: int = 4,
) -> PairMatchResult:
    """Score one query/reference X_L pair using MNN token matching and spatial consistency."""
    matched_q, matched_r, matched_sim = mutual_nearest_neighbor_matches(
        query_local,
        ref_local,
        similarity_threshold=similarity_threshold,
    )
    num_matches = int(matched_sim.numel())
    if num_matches == 0:
        return PairMatchResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    quality = float(matched_sim.mean().item())
    coverage = float(num_matches / min(query_local.shape[0], ref_local.shape[0]))
    local_score = quality * coverage

    if num_matches < min_matches:
        spatial_score = 0.0
    else:
        sigma = max(float(spatial_sigma), 1e-12)
        coords = normalized_xy_grid(spatial_shape, device=query_local.device, dtype=query_local.dtype)
        query_xy = coords[matched_q]
        ref_xy = coords[matched_r]
        displacement = ref_xy - query_xy
        main_displacement = displacement.median(dim=0).values
        residual = torch.linalg.norm(displacement - main_displacement, dim=1)
        median_residual = float(residual.median().item())
        spatial_score = math.exp(-((median_residual ** 2) / (sigma ** 2)))

    weighted_spatial = (1.0 - spatial_weight) + spatial_weight * spatial_score
    pair_score = local_score * weighted_spatial
    return PairMatchResult(
        pair_score=float(pair_score),
        local_score=float(local_score),
        quality=float(quality),
        coverage=float(coverage),
        spatial_score=float(spatial_score),
        num_matches=num_matches,
    )


def score_local_pairs_for_query(
    query_local: torch.Tensor,
    candidate_locals: torch.Tensor,
    spatial_shape: tuple[int, int],
    similarity_threshold: float = 0.5,
    spatial_sigma: float = 0.15,
    spatial_weight: float = 0.3,
    min_matches: int = 4,
) -> torch.Tensor:
    """Vectorized pair scores for one query against its Top-K candidates.

    This keeps the exact MNN/local/spatial formula but computes the expensive
    token-similarity matrices as one batched matmul on the tensor device.
    """
    if query_local.ndim != 2 or candidate_locals.ndim != 3:
        raise ValueError("query_local must be [N, C] and candidate_locals must be [K, N, C]")
    if query_local.shape[1] != candidate_locals.shape[2]:
        raise ValueError("query/candidate channel dimensions must match")

    query_tokens = F.normalize(query_local, p=2, dim=-1)
    candidate_tokens = F.normalize(candidate_locals, p=2, dim=-1)
    sim = torch.einsum("qc,krc->kqr", query_tokens, candidate_tokens)

    q_to_r = sim.argmax(dim=2)
    r_to_q = sim.argmax(dim=1)
    q_indices = torch.arange(sim.shape[1], device=sim.device).unsqueeze(0).expand_as(q_to_r)
    is_mutual = r_to_q.gather(1, q_to_r) == q_indices
    matched_sim = sim.gather(2, q_to_r.unsqueeze(-1)).squeeze(-1)
    keep = is_mutual & (matched_sim >= similarity_threshold)

    num_matches = keep.sum(dim=1)
    score_sums = (matched_sim * keep.to(matched_sim.dtype)).sum(dim=1)
    quality = torch.where(num_matches > 0, score_sums / num_matches.clamp_min(1), torch.zeros_like(score_sums))
    coverage = num_matches.to(matched_sim.dtype) / min(query_local.shape[0], candidate_locals.shape[1])
    local_score = quality * coverage

    spatial_score = torch.zeros_like(local_score)
    valid_spatial = num_matches >= min_matches
    if valid_spatial.any():
        sigma = max(float(spatial_sigma), 1e-12)
        coords = normalized_xy_grid(spatial_shape, device=sim.device, dtype=sim.dtype)
        query_xy_all = coords[q_indices[0]]
        for cand_idx in valid_spatial.nonzero(as_tuple=False).flatten().tolist():
            mask = keep[cand_idx]
            matched_q = q_indices[cand_idx, mask]
            matched_r = q_to_r[cand_idx, mask]
            query_xy = query_xy_all[matched_q]
            ref_xy = coords[matched_r]
            displacement = ref_xy - query_xy
            main_displacement = displacement.median(dim=0).values
            residual = torch.linalg.norm(displacement - main_displacement, dim=1)
            median_residual = residual.median()
            spatial_score[cand_idx] = torch.exp(-((median_residual ** 2) / (sigma ** 2)))

    weighted_spatial = (1.0 - spatial_weight) + spatial_weight * spatial_score
    return local_score * weighted_spatial


def minmax_normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    min_value = values.min(dim=-1, keepdim=True).values
    max_value = values.max(dim=-1, keepdim=True).values
    denom = max_value - min_value
    return torch.where(denom > eps, (values - min_value) / denom.clamp_min(eps), torch.zeros_like(values))


def global_topk_search(query_global: torch.Tensor, ref_global: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the original project baseline by using FAISS IndexFlatL2."""
    if query_global.ndim != 2 or ref_global.ndim != 2:
        raise ValueError("query_global and ref_global must be [N, C] tensors")
    if query_global.shape[1] != ref_global.shape[1]:
        raise ValueError("query_global and ref_global dimensions must match")

    k = min(int(top_k), ref_global.shape[0])
    if k <= 0:
        raise ValueError("top_k must be positive and reference set must be non-empty")

    ref_np = np.ascontiguousarray(ref_global.detach().cpu().numpy().astype("float32", copy=False))
    query_np = np.ascontiguousarray(query_global.detach().cpu().numpy().astype("float32", copy=False))
    index = faiss.IndexFlatL2(ref_np.shape[1])
    index.add(ref_np)
    distances, indices = index.search(query_np, k)

    # Higher is better for later min-max fusion; FAISS returns lower-is-better L2 distances.
    scores = torch.from_numpy(-distances)
    predictions = torch.from_numpy(indices).long()
    return scores, predictions


def recall_at_k(predictions: torch.Tensor, ground_truth: Iterable[np.ndarray], k_values=(1, 5, 10, 20)) -> dict[int, float]:
    if predictions.ndim != 2:
        raise ValueError("predictions must be [num_queries, top_k]")
    recalls = {}
    num_queries = predictions.shape[0]
    if num_queries == 0:
        return {int(k): 0.0 for k in k_values}

    pred_np = predictions.detach().cpu().numpy()
    for k in k_values:
        capped_k = min(int(k), predictions.shape[1])
        correct = 0
        for query_idx, positives in enumerate(ground_truth):
            if np.any(np.isin(pred_np[query_idx, :capped_k], positives)):
                correct += 1
        recalls[int(k)] = correct / num_queries
    return recalls


def top1_transition_stats(
    baseline_predictions: torch.Tensor,
    reranked_predictions: torch.Tensor,
    ground_truth: Iterable[np.ndarray],
) -> dict[str, int]:
    baseline_top1 = baseline_predictions[:, 0].detach().cpu().numpy()
    reranked_top1 = reranked_predictions[:, 0].detach().cpu().numpy()
    stats = {"fixed": 0, "new_error": 0, "still_wrong": 0, "both_correct": 0, "net_gain": 0}
    for q_idx, positives in enumerate(ground_truth):
        base_ok = bool(np.isin(baseline_top1[q_idx], positives))
        rerank_ok = bool(np.isin(reranked_top1[q_idx], positives))
        if not base_ok and rerank_ok:
            stats["fixed"] += 1
        elif base_ok and not rerank_ok:
            stats["new_error"] += 1
        elif not base_ok and not rerank_ok:
            stats["still_wrong"] += 1
        else:
            stats["both_correct"] += 1
    stats["net_gain"] = stats["fixed"] - stats["new_error"]
    return stats


def rerank_topk(
    ref_global: torch.Tensor,
    query_global: torch.Tensor,
    ref_local: torch.Tensor,
    query_local: torch.Tensor,
    spatial_shape: tuple[int, int],
    ground_truth: Iterable[np.ndarray],
    config: RerankConfig | None = None,
) -> RerankResult:
    config = config or RerankConfig()
    if config.top_k < 20:
        raise ValueError("--top-k must be at least 20 to report baseline/reranked R@20 consistently with the original project.")
    latencies: dict[str, float] = {}

    retrieval_start = time.perf_counter()
    baseline_scores, baseline_indices = global_topk_search(query_global, ref_global, config.top_k)
    latencies["global_retrieval_seconds"] = time.perf_counter() - retrieval_start

    if config.debug_num_queries is not None:
        limit = min(int(config.debug_num_queries), baseline_indices.shape[0])
        baseline_scores = baseline_scores[:limit]
        baseline_indices = baseline_indices[:limit]
        query_local = query_local[:limit]
        query_global = query_global[:limit]
        ground_truth = list(ground_truth)[:limit]
    else:
        ground_truth = list(ground_truth)

    rerank_start = time.perf_counter()
    num_queries, top_k = baseline_indices.shape
    pair_scores = torch.zeros((num_queries, top_k), dtype=torch.float32)
    rerank_device = torch.device(config.rerank_device) if config.rerank_device else query_local.device
    if rerank_device.type == "cuda" and not torch.cuda.is_available():
        rerank_device = torch.device("cpu")

    with torch.inference_mode():
        for q_idx in range(num_queries):
            candidate_indices = baseline_indices[q_idx]
            q_local = query_local[q_idx].to(rerank_device, dtype=torch.float32, non_blocking=True)
            candidate_locals = ref_local[candidate_indices].to(rerank_device, dtype=torch.float32, non_blocking=True)
            scores = score_local_pairs_for_query(
                q_local,
                candidate_locals,
                spatial_shape=spatial_shape,
                similarity_threshold=config.similarity_threshold,
                spatial_sigma=config.spatial_sigma,
                spatial_weight=config.spatial_weight,
                min_matches=config.min_matches,
            )
            pair_scores[q_idx] = scores.detach().cpu()

    global_norm = minmax_normalize(baseline_scores)
    pair_norm = minmax_normalize(pair_scores)
    final_scores = config.global_weight * global_norm + (1.0 - config.global_weight) * pair_norm
    rerank_order = final_scores.argsort(dim=1, descending=True)
    reranked_indices = baseline_indices.gather(1, rerank_order)
    final_scores = final_scores.gather(1, rerank_order)
    pair_scores = pair_scores.gather(1, rerank_order)
    latencies["reranking_seconds"] = time.perf_counter() - rerank_start

    baseline_recalls = recall_at_k(baseline_indices, ground_truth)
    reranked_recalls = recall_at_k(reranked_indices, ground_truth)
    transitions = top1_transition_stats(baseline_indices, reranked_indices, ground_truth)

    return RerankResult(
        baseline_indices=baseline_indices,
        reranked_indices=reranked_indices,
        baseline_scores=baseline_scores,
        pair_scores=pair_scores,
        final_scores=final_scores,
        baseline_recalls=baseline_recalls,
        reranked_recalls=reranked_recalls,
        transitions=transitions,
        latencies=latencies,
    )
