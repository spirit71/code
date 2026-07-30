"""Cross-view pseudo targets and auxiliary losses for QRL-BoQ."""

import torch
from torch.nn import functional as F


def _rank_rows(values: torch.Tensor) -> torch.Tensor:
    """Map each row's ascending order to evenly spaced values in [0,1]."""
    rows, queries = values.shape
    if queries == 1:
        return torch.ones_like(values)
    order = values.argsort(dim=-1)
    ranks = torch.zeros_like(values)
    rank_values = torch.linspace(0.0, 1.0, queries, device=values.device, dtype=values.dtype)
    return ranks.scatter(-1, order, rank_values.expand(rows, -1))


def compute_cross_view_repeatability_targets(
    query_outputs: torch.Tensor,
    place_ids: torch.Tensor,
    target_mode: str = "rank",
    stop_gradient: bool = True,
    min_views: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_outputs.ndim != 3:
        raise ValueError("query_outputs must have shape [B,M,D]")
    place_ids = place_ids.reshape(-1).to(query_outputs.device)
    if place_ids.numel() != query_outputs.shape[0]:
        raise ValueError("place_ids must contain one id per image")
    if min_views < 2:
        raise ValueError("min_views must be at least 2")
    if target_mode not in {"continuous", "rank", "top_bottom"}:
        raise ValueError(f"unsupported target_mode: {target_mode}")

    q = F.normalize(query_outputs.float(), dim=-1)
    batch, queries, _ = q.shape
    targets = torch.zeros(batch, queries, device=q.device, dtype=torch.float32)
    valid_mask = torch.zeros(batch, queries, device=q.device, dtype=torch.bool)

    for place_id in torch.unique(place_ids):
        indices = torch.nonzero(place_ids == place_id, as_tuple=False).flatten()
        if indices.numel() < min_views:
            continue
        group = q[indices]  # [V,M,D]
        views = group.shape[0]
        # [V,V,M,M], then row-max query matching. Exclude self-view pairs.
        similarities = torch.einsum("vmd,wnd->vwmn", group, group)
        row_max = similarities.max(dim=-1).values
        other_view = ~torch.eye(views, device=q.device, dtype=torch.bool)
        stability = (row_max * other_view[:, :, None]).sum(dim=1) / (views - 1)

        if target_mode == "continuous":
            group_targets = ((stability + 1.0) / 2.0).clamp(0.0, 1.0)
            group_valid = torch.ones_like(group_targets, dtype=torch.bool)
        elif target_mode == "rank":
            group_targets = _rank_rows(stability)
            group_valid = torch.ones_like(group_targets, dtype=torch.bool)
        else:
            k = max(1, queries // 4)
            order = stability.argsort(dim=-1)
            low_idx, high_idx = order[:, :k], order[:, -k:]
            group_targets = torch.zeros_like(stability)
            group_targets.scatter_(-1, high_idx, 1.0)
            group_valid = torch.zeros_like(stability, dtype=torch.bool)
            group_valid.scatter_(-1, low_idx, True)
            group_valid.scatter_(-1, high_idx, True)

        targets[indices] = group_targets
        valid_mask[indices] = group_valid
    return (targets.detach() if stop_gradient else targets), valid_mask


def compute_query_reliability_loss(
    reliability: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    if reliability.shape != targets.shape or valid_mask.shape != targets.shape:
        raise ValueError("reliability, targets and valid_mask must have the same shape")
    if not valid_mask.any():
        return reliability.float().sum() * 0.0
    prediction, target = reliability.float()[valid_mask], targets.float()[valid_mask]
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if loss_type == "bce":
        return F.binary_cross_entropy(prediction.clamp(1e-7, 1 - 1e-7), target)
    if loss_type == "ranking":
        losses = []
        for row in range(reliability.shape[0]):
            mask = valid_mask[row]
            r, t = reliability[row].float()[mask], targets[row].float()[mask]
            if r.numel() < 2:
                continue
            target_diff = t[:, None] - t[None, :]
            pair_mask = target_diff != 0
            score_diff = r[:, None] - r[None, :]
            losses.append(F.softplus(-score_diff[pair_mask] * target_diff[pair_mask].sign()).mean())
        return torch.stack(losses).mean() if losses else reliability.float().sum() * 0.0
    raise ValueError(f"unsupported reliability loss type: {loss_type}")


@torch.no_grad()
def spearman_correlation(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Average per-image Spearman correlation; constant/invalid rows are skipped."""
    correlations = []
    for row in range(prediction.shape[0]):
        mask = valid_mask[row]
        x, y = prediction[row].float()[mask], target[row].float()[mask]
        if x.numel() < 2:
            continue
        # A constant predictor has no defined ordering and must not receive an
        # artificial correlation from argsort's arbitrary tie ordering.
        if (x.max() - x.min()) <= 1e-12 or (y.max() - y.min()) <= 1e-12:
            continue
        x_rank = x.argsort().argsort().float()
        y_rank = y.argsort().argsort().float()
        x_rank, y_rank = x_rank - x_rank.mean(), y_rank - y_rank.mean()
        denominator = x_rank.norm() * y_rank.norm()
        if denominator > 0:
            correlations.append((x_rank * y_rank).sum() / denominator)
    return torch.stack(correlations).mean() if correlations else prediction.float().new_zeros(())
