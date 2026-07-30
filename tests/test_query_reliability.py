import torch

from src.boq import BoQ
from src.losses.query_reliability_loss import (
    compute_cross_view_repeatability_targets,
    compute_query_reliability_loss,
    spearman_correlation,
)
from src.query_reliability import QueryReliabilityHead, apply_reliability_gate


def test_disabled_module_matches_original_forward_path():
    torch.manual_seed(3)
    model = BoQ(in_channels=64, proj_channels=64, num_queries=4, num_layers=2, row_dim=3)
    model.eval()
    image_features = torch.randn(2, 64, 5, 5)
    descriptor, attentions = model(image_features)
    descriptor_aux, aux = model(image_features, return_aux=True)
    torch.testing.assert_close(descriptor, descriptor_aux)
    assert len(attentions) == 2
    assert aux["query_outputs_raw"].shape == (2, 2, 4, 64)
    assert aux["query_outputs_weighted"] is None
    assert aux["reliability_scores"] is None


def test_centered_gate_initialization_is_identity():
    head = QueryReliabilityHead(8)
    queries = torch.randn(2, 5, 8)
    attention = torch.softmax(torch.randn(2, 5, 11), dim=-1)
    reliability = head(queries, attention)
    weighted, weights = apply_reliability_gate(queries, reliability)
    torch.testing.assert_close(reliability, torch.full_like(reliability, 0.5))
    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(weighted, queries)


def test_identical_and_permuted_views_have_max_continuous_targets():
    base = torch.eye(4).unsqueeze(0)
    permuted = base[:, [2, 0, 3, 1]]
    query_outputs = torch.cat((base, permuted), dim=0)
    targets, valid = compute_cross_view_repeatability_targets(
        query_outputs, torch.tensor([7, 7]), target_mode="continuous"
    )
    assert valid.all()
    torch.testing.assert_close(targets, torch.ones_like(targets))


def test_random_views_are_lower_than_identical_views():
    torch.manual_seed(9)
    first = torch.randn(1, 32, 64)
    identical = torch.cat((first, first), dim=0)
    random_pair = torch.randn(2, 32, 64)
    same_targets, _ = compute_cross_view_repeatability_targets(identical, torch.tensor([1, 1]), "continuous")
    random_targets, _ = compute_cross_view_repeatability_targets(random_pair, torch.tensor([1, 1]), "continuous")
    assert same_targets.mean() > random_targets.mean() + 0.2


def test_rank_targets_cover_zero_to_one_per_image():
    queries = torch.randn(4, 7, 12)
    targets, valid = compute_cross_view_repeatability_targets(queries, torch.tensor([1, 1, 2, 2]), "rank")
    assert valid.all()
    torch.testing.assert_close(targets.min(dim=-1).values, torch.zeros(4))
    torch.testing.assert_close(targets.max(dim=-1).values, torch.ones(4))


def test_single_view_is_invalid_and_finite():
    targets, valid = compute_cross_view_repeatability_targets(
        torch.randn(1, 5, 8), torch.tensor([1]), "rank"
    )
    assert not valid.any()
    assert torch.isfinite(targets).all()


def test_head_without_attention_features():
    head = QueryReliabilityHead(8, use_entropy=False, use_max_attention=False)
    scores = head(torch.randn(3, 5, 8), None)
    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()


def test_top_bottom_mask_and_all_loss_modes_are_finite():
    queries = torch.randn(4, 8, 12)
    targets, valid = compute_cross_view_repeatability_targets(
        queries, torch.tensor([1, 1, 2, 2]), "top_bottom"
    )
    assert (valid.sum(dim=-1) == 4).all()
    reliability = torch.sigmoid(torch.randn(4, 8, requires_grad=True))
    for loss_type in ("smooth_l1", "bce", "ranking"):
        loss = compute_query_reliability_loss(reliability, targets, valid, loss_type)
        assert loss.ndim == 0 and torch.isfinite(loss)


def test_constant_prediction_spearman_is_zero():
    prediction = torch.full((2, 4), 0.5)
    targets = torch.tensor([[0.0, 0.3, 0.7, 1.0]]).expand(2, -1)
    assert spearman_correlation(prediction, targets, torch.ones_like(targets, dtype=torch.bool)) == 0
