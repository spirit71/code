import numpy as np
import torch

from src import utils
from src.boq import BoQ
from src.xl_reranking import (
    RerankConfig,
    global_topk_search,
    mutual_nearest_neighbor_matches,
    normalized_xy_grid,
    recall_at_k,
    rerank_topk,
    score_local_pair,
    score_local_pairs_for_query,
)


def test_global_forward_unchanged():
    torch.manual_seed(0)
    aggregator = BoQ(in_channels=8, proj_channels=64, num_queries=4, num_layers=2, row_dim=8).eval()
    x = torch.randn(2, 8, 4, 4)
    baseline_global, baseline_attns = aggregator(x)
    local_output = aggregator(x, return_local=True)

    torch.testing.assert_close(local_output["global"], baseline_global)
    assert isinstance(baseline_attns, list)


def test_return_local_shape():
    torch.manual_seed(1)
    aggregator = BoQ(in_channels=8, proj_channels=64, num_queries=4, num_layers=2, row_dim=8).eval()
    x = torch.randn(3, 8, 5, 6)
    output = aggregator(x, return_local=True)

    assert output["global"].shape == (3, 64 * 8)
    assert output["local"].shape == (3, 5 * 6, 64)
    assert output["spatial_shape"] == (5, 6)
    assert output["attention"] is not None


def test_normalized_grid():
    grid = normalized_xy_grid((2, 3))
    expected = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [1.0, 1.0],
    ])
    torch.testing.assert_close(grid, expected)


def test_identity_mutual_matching():
    tokens = torch.eye(4)
    matched_q, matched_r, sims = mutual_nearest_neighbor_matches(tokens, tokens, similarity_threshold=0.9)

    torch.testing.assert_close(matched_q, torch.arange(4))
    torch.testing.assert_close(matched_r, torch.arange(4))
    torch.testing.assert_close(sims, torch.ones(4))


def test_no_match_case():
    query = torch.eye(4)
    ref = -torch.eye(4)
    result = score_local_pair(query, ref, spatial_shape=(2, 2), similarity_threshold=0.5, min_matches=1)

    assert result.num_matches == 0
    assert result.local_score == 0.0
    assert result.spatial_score == 0.0
    assert result.pair_score == 0.0


def test_perfect_translation_spatial_consistency():
    query = torch.eye(4)
    ref = torch.zeros_like(query)
    ref[1] = query[0]
    ref[2] = query[1]
    ref[3] = query[2]
    ref[0] = -query[3]

    result = score_local_pair(
        query,
        ref,
        spatial_shape=(1, 4),
        similarity_threshold=0.9,
        spatial_sigma=0.15,
        min_matches=3,
    )

    assert result.num_matches == 3
    assert abs(result.spatial_score - 1.0) < 1e-10
    assert result.pair_score > 0.0


def test_reranking_preserves_candidate_set():
    torch.manual_seed(2)
    ref_global = torch.eye(20, 20)
    query_global = torch.eye(2, 20)
    ref_local = torch.randn(20, 4, 8)
    query_local = torch.randn(2, 4, 8)
    gt = [np.array([0]), np.array([1])]

    result = rerank_topk(
        ref_global=ref_global,
        query_global=query_global,
        ref_local=ref_local,
        query_local=query_local,
        spatial_shape=(2, 2),
        ground_truth=gt,
        config=RerankConfig(top_k=20, similarity_threshold=2.0),
    )

    for q_idx in range(result.baseline_indices.shape[0]):
        assert set(result.baseline_indices[q_idx].tolist()) == set(result.reranked_indices[q_idx].tolist())


def test_recall_calculation():
    predictions = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    gt = [np.array([3]), np.array([4])]
    recalls = recall_at_k(predictions, gt, k_values=(1, 2, 3))

    assert recalls[1] == 0.5
    assert recalls[2] == 0.5
    assert recalls[3] == 1.0


def test_global_topk_shapes():
    ref_global = torch.eye(4)
    query_global = torch.eye(2, 4)
    scores, indices = global_topk_search(query_global, ref_global, top_k=3)

    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)


def test_faiss_l2_baseline_matches_original_recall():
    descriptors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    ref_global = descriptors[:3]
    query_global = descriptors[3:]
    gt = [np.array([0]), np.array([1])]

    _, predictions = global_topk_search(query_global, ref_global, top_k=3)
    rerank_recalls = recall_at_k(predictions, gt, k_values=(1, 2, 3))
    original_recalls = utils.compute_recall_performance(
        descriptors,
        num_references=3,
        num_queries=2,
        ground_truth=gt,
        k_values=[1, 2, 3],
    )

    assert rerank_recalls == original_recalls


def test_vectorized_pair_scores_match_scalar_scores():
    torch.manual_seed(3)
    query = torch.randn(6, 8)
    candidates = torch.randn(5, 6, 8)
    vector_scores = score_local_pairs_for_query(
        query,
        candidates,
        spatial_shape=(2, 3),
        similarity_threshold=0.1,
        spatial_sigma=0.15,
        spatial_weight=0.3,
        min_matches=2,
    )
    scalar_scores = torch.tensor([
        score_local_pair(
            query,
            candidates[idx],
            spatial_shape=(2, 3),
            similarity_threshold=0.1,
            spatial_sigma=0.15,
            spatial_weight=0.3,
            min_matches=2,
        ).pair_score
        for idx in range(candidates.shape[0])
    ])
    torch.testing.assert_close(vector_scores.cpu(), scalar_scores, rtol=1e-6, atol=1e-6)
