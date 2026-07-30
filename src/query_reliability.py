"""Single-image query reliability prediction and soft gating for BoQ."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def attention_statistics(attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized entropy and maximum mass for [B,M,N]/[B,H,M,N]."""
    # attention 表示每个 BoQ query 对图像 token 的注意力分布：
    #   B：batch 中的图像数量
    #   M：每层 BoQ query 数量（当前配置为 64）
    #   N：图像 token 数量（224 输入时为 16*16=256）
    # 如果保留了多头维度，则输入可能是 [B,H,M,N]；这里先对 H 个头取平均，
    # 得到每个 query 的一条总体注意力分布 [B,M,N]。
    if attention.ndim == 4:
        attention = attention.float().mean(dim=1)
    elif attention.ndim == 3:
        attention = attention.float()
    else:
        raise ValueError("attention must have shape [B,M,N] or [B,H,M,N]")

    # MultiheadAttention 已经输出概率，但混合精度和多头平均可能产生很小的数值误差。
    # 因此先截断负数，再沿 token 维重新归一化，确保每个 query 的权重和为 1。
    attention = attention.clamp_min(0.0)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # 归一化熵 entropy ∈ [0,1]：
    #   接近 0：注意力集中在少数 token；
    #   接近 1：注意力均匀分散在大量 token。
    # 注意：注意力集中不等于“地点检索可靠”，这里只把它作为预测头的输入线索。
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
    if attention.shape[-1] > 1:
        entropy = entropy / math.log(attention.shape[-1])
    else:
        entropy = torch.zeros_like(entropy)

    # max_attention 是每个 query 对某一个图像 token 的最大注意力质量。
    # entropy 和 max_attention 的输出形状均为 [B,M]。
    return entropy, attention.max(dim=-1).values


class QueryReliabilityHead(nn.Module):
    """根据单张图像中的 query 表征和注意力统计，预测每个 query 的可靠性。

    输入：
        query_outputs: [B,M,D]，当前为最后一层 BoQ 输出 O2=[B,64,512]
        attention:     [B,M,N] 或 [B,H,M,N]，O2 对图像 token 的注意力

    输出：
        reliability:  [B,M]，每个值位于 [0,1]

    reliability 是一个“单图可预测分数”，不是概率真值，也不是 oracle。
    跨视角稳定性只在训练时用于构造监督 target；部署推理时不需要正样本或 GT。
    """

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int = 128,
        use_entropy: bool = True,
        use_max_attention: bool = True,
        use_query_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_entropy = use_entropy
        self.use_max_attention = use_max_attention
        self.use_query_norm = use_query_norm

        # 每个 query 的基础输入是归一化后的 D 维向量。
        # 根据开关，最多再拼接 3 个标量：entropy、max-attention、原始 query 范数。
        # 当前默认配置：input_dim = 512 + 3 = 515。
        input_dim = query_dim + sum((use_entropy, use_max_attention, use_query_norm))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # 最后一层零初始化，使初始 logits 全为 0，sigmoid(0)=0.5。
        # 后面的 centered-residual gate 会减去每张图的 reliability 均值，
        # 所以初始所有 gate 权重严格等于 1，不会一开始就破坏原始 BoQ descriptor。
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, query_outputs: torch.Tensor, attention: torch.Tensor | None) -> torch.Tensor:
        if query_outputs.ndim != 3:
            raise ValueError("query_outputs must have shape [B,M,D]")

        # 可靠性统计统一用 FP32，避免混合精度下 entropy、norm 和 sigmoid 数值不稳。
        query_fp32 = query_outputs.float()

        # 方向特征 [B,M,D]：
        # L2 归一化后去掉向量整体尺度，保留“这个 query 表达了什么内容”。
        features = [F.normalize(query_fp32, dim=-1)]
        if self.use_query_norm:
            # 幅值特征 [B,M,1]：
            # 保留归一化前 query 的强度，作为预测可靠性的一个候选线索。
            features.append(query_fp32.norm(dim=-1, keepdim=True))
        if self.use_entropy or self.use_max_attention:
            if attention is None:
                raise ValueError("attention is required by the selected reliability features")
            entropy, max_attention = attention_statistics(attention)
            if entropy.shape != query_outputs.shape[:2]:
                raise ValueError("attention query dimensions do not match query_outputs")
            if self.use_entropy:
                # [B,M] -> [B,M,1]，和 query 特征在最后一维拼接。
                features.append(entropy.unsqueeze(-1))
            if self.use_max_attention:
                features.append(max_attention.unsqueeze(-1))

        # 当前默认形状变化：
        #   cat(features): [B,64,515]
        #   MLP:           [B,64,515] -> [B,64,128] -> [B,64,1]
        #   squeeze:       [B,64]
        # 同一个小 MLP 共享地应用于所有图像、所有 query，而不是为 64 个位置各建一个头。
        logits = self.mlp(torch.cat(features, dim=-1)).squeeze(-1)

        # 把无界 logits 映射到 [0,1]，得到可排序的 reliability。
        # 当前训练主要使用它的相对顺序；0/1 不应解释为严格校准后的概率。
        return torch.sigmoid(logits)


def entropy_reliability(attention: torch.Tensor) -> torch.Tensor:
    """Parameter-free E1 score: r=1-normalized_attention_entropy."""
    entropy, _ = attention_statistics(attention)
    return (1.0 - entropy).clamp(0.0, 1.0)


def apply_reliability_gate(
    query_outputs: torch.Tensor,
    reliability: torch.Tensor,
    mode: str = "centered_residual",
    alpha: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把可靠性分数转换为 query 权重，并逐 query 缩放 O2。

    输入：
        query_outputs: [B,M,D]，当前为 O2=[B,64,512]
        reliability:   [B,M]，QRL Head 输出

    输出：
        weighted:      [B,M,D]，加权后的 O2
        weights:       [B,M]，实际乘到每个 query 上的 gate 权重
    """
    if query_outputs.ndim != 3 or reliability.shape != query_outputs.shape[:2]:
        raise ValueError("expected query_outputs [B,M,D] and reliability [B,M]")
    if mode == "none":
        # 消融模式：保留 QRL 监督，但推理表示完全不使用 gate。
        weights = torch.ones_like(reliability)
    elif mode == "direct":
        # 直接权重：q'_m = r_m*q_m。
        # 缺点是初始 r=0.5 会把整层输出缩小一半，因此不是当前主方案。
        weights = reliability
    elif mode == "centered_residual":
        # 当前主方案，对每张图分别中心化：
        #   centered_m = r_m - mean(r_1,...,r_M)
        #   w_m = clip(1 + alpha*centered_m, min_weight, max_weight)
        #
        # “centered”：只关心同一张图内 64 个 query 的相对可靠性，
        # reliability 整体偏高或偏低不会让整张 descriptor 一起放大或缩小。
        #
        # “residual”：权重围绕 1 做温和调整，而不是从 0 开始重建表示。
        # 高于本图平均可靠性的 query 权重大于 1，低于平均值的则小于 1。
        centered = reliability - reliability.mean(dim=-1, keepdim=True)
        weights = (1.0 + alpha * centered).clamp(min=min_weight, max=max_weight)
    else:
        raise ValueError(f"unsupported reliability gate mode: {mode}")

    # [B,M] -> [B,M,1]，将每个标量权重广播到该 query 的全部 D 个通道：
    #   weighted[b,m,d] = query_outputs[b,m,d] * weights[b,m]
    # 这不会混合不同 query，也不会改变形状，只改变每个 query 整体贡献的幅度。
    return query_outputs * weights.to(query_outputs.dtype).unsqueeze(-1), weights
