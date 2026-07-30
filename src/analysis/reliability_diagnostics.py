"""Small, GT-free building blocks used by offline QRL-BoQ diagnostics."""

import torch
from torch.nn import functional as F


def mask_from_scores(scores: torch.Tensor, masked_fraction: float, remove_high: bool) -> torch.Tensor:
    """由逐 query 分数生成 mask；True 表示保留，False 表示删除。"""
    if scores.ndim != 2 or not 0.0 <= masked_fraction <= 1.0:
        raise ValueError("scores must be [B,M] and masked_fraction must be in [0,1]")
    count = int(round(scores.shape[1] * masked_fraction))
    keep = torch.ones_like(scores, dtype=torch.bool)
    if count:
        order = scores.argsort(dim=-1, descending=remove_high)[:, :count]
        keep.scatter_(1, order, False)
    return keep


def random_mask(batch: int, queries: int, masked_fraction: float, seed: int, device=None) -> torch.Tensor:
    """每张图随机删除相同数量 query，并用固定 seed 保证复现。"""
    if batch < 1 or queries < 1:
        raise ValueError("batch and queries must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(batch, queries, generator=generator)
    return mask_from_scores(noise.to(device), masked_fraction, remove_high=False)


def reliability_mask(
    query_outputs: torch.Tensor, reliability: torch.Tensor, lowest_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the predicted lowest-reliability queries (10/25/50% diagnostics)."""
    if not 0.0 <= lowest_fraction <= 1.0:
        raise ValueError("lowest_fraction must be in [0,1]")
    if query_outputs.shape[:2] != reliability.shape:
        raise ValueError("expected query_outputs [B,M,D] and reliability [B,M]")
    count = int(query_outputs.shape[1] * lowest_fraction)
    keep = torch.ones_like(reliability, dtype=torch.bool)
    if count:
        low = reliability.argsort(dim=-1)[:, :count]
        keep.scatter_(-1, low, False)
    return query_outputs * keep.unsqueeze(-1), keep


def descriptor_from_query_outputs(query_outputs_by_layer: torch.Tensor, projection) -> torch.Tensor:
    """Re-run the unchanged BoQ projection for baseline/masking comparisons."""
    if query_outputs_by_layer.ndim != 4:
        raise ValueError("query_outputs_by_layer must have shape [B,L,M,D]")
    batch, layers, queries, dim = query_outputs_by_layer.shape
    projection_input = query_outputs_by_layer.reshape(batch, layers * queries, dim).permute(0, 2, 1)
    # 离线诊断会把大量中间 query 缓存在 CPU；直接复用冻结 FC 权重，
    # 既保持与 BoQ.forward 相同的线性变换，又允许调用方自主选择计算设备。
    weight = projection.weight.detach().to(projection_input.device)
    bias = projection.bias.detach().to(projection_input.device) if projection.bias is not None else None
    projected = F.linear(projection_input, weight, bias)
    return F.normalize(projected.flatten(1), p=2, dim=-1)
