from __future__ import annotations

import warnings
import torch

from src.xl_reranking import normalized_xy_grid


def select_tokens(features, attention, spatial_shape, method="all", count=None, seed=0):
    """从每张图的空间 token 中选出后续参与局部匹配的 token。

    初学者可以把一张图理解为一个 ``H x W`` 的小网格。网格中的每个格子
    对应一个 C 维特征向量，也就是一个 token。例如当前 DINOv2+BoQ 配置中，
    ``T = H*W = 23*23 = 529``，``C = 512``。

    参数形状：
        features:  [B, T, C]，B 张图、每张图 T 个 token、每个 token C 维。
        attention: [B, Q, T]，Q 个 BoQ query 对 T 个空间 token 的注意力。
        spatial_shape: (H, W)，用于把 token 编号还原为空间坐标。

    返回字典：
        features:    [B, K, C]，被选中的 K 个局部特征。
        roles:       [B, K, Q]，每个 token 被各个 BoQ query 关注的程度。
        coordinates: [B, K, 2]，归一化到 [0,1] 的 (x,y) 坐标。
        indices:     [B, K]，token 在原始 T 个位置中的编号。
        scores:      [B, K]，用于选择 token 的重要性分数。

    此函数只做选择，不会训练模型或修改特征值。
    """
    if features.ndim != 3:
        raise ValueError("features must be [B,T,C]")
    batch, tokens, _ = features.shape
    # K 不能超过原始 token 数；all 表示一个也不丢，K=T。
    count = tokens if method == "all" or count is None else min(int(count), tokens)
    if attention is not None and attention.shape[-1] != tokens:
        raise ValueError("attention token dimension does not match features")
    if method == "all":
        # all 不需要真正排序，统一给每个位置一个 1 分即可。
        scores = torch.ones(batch, tokens, device=features.device)
    elif method == "random":
        # 固定 seed 后，同一张图每次都会得到相同的随机选择，便于复现实验。
        gen = torch.Generator(device=features.device).manual_seed(seed)
        scores = torch.rand(batch, tokens, generator=gen, device=features.device)
    elif method == "feature_norm":
        # 对每个 C 维向量计算 L2 范数：sqrt(v1^2 + ... + vC^2)。
        # 注意：x2 经过 LayerNorm，范数可能非常接近，此方法未必有区分力。
        scores = features.float().norm(dim=-1)
    elif method in {"attention_mean", "attention_max", "entropy_aware"}:
        if attention is None:
            raise ValueError(f"{method} requires attention")
        attn = attention.float().clamp_min(1e-12)
        if method == "attention_mean":
            # 对 Q 个 BoQ query 求平均：大家平均有多关注这个空间位置。
            scores = attn.mean(dim=1)
            if (scores.amax(dim=1) - scores.amin(dim=1)).mean() < 1e-6:
                warnings.warn("attention_mean is nearly constant; selection may be uninformative", RuntimeWarning)
        elif method == "attention_max":
            # 对 Q 维取最大值：只要至少一个 query 强烈关注，该 token 就会得高分。
            # [B,Q,T] 沿 Q 维取最大值后变为 [B,T]。
            scores = attn.max(dim=1).values
        else:
            # 熵越小，说明少数 query 对该位置的角色分工越明确。
            # 用最大注意力除以熵，同时奖励“关注强”和“角色明确”。
            role_p = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(role_p * role_p.log()).sum(dim=1)
            scores = attn.max(dim=1).values / (entropy + 1e-6)
    else:
        raise ValueError(f"Unknown token selection: {method}")
    # grid[t] 保存第 t 个 token 在原特征图中的二维位置，后面空间验证要用。
    grid = normalized_xy_grid(spatial_shape, device=features.device, dtype=torch.float32)
    if method == "all":
        # 保持原 token 顺序且不复制整套特征；Pitts30k 的 529-token X_L 约 9 GiB。
        indices = torch.arange(tokens, device=features.device).expand(batch, -1)
        roles = None if attention is None else attention.transpose(1, 2)
        coords = grid.unsqueeze(0).expand(batch, -1, -1)
        return {"features": features, "roles": roles, "coordinates": coords, "indices": indices, "scores": scores}
    # 每张图独立选择分数最高的 K 个位置，indices 形状为 [B,K]。
    indices = scores.topk(count, dim=1).indices
    # gather 要求索引与 features 维数一致，所以把 [B,K] 扩展成 [B,K,C]。
    gather_feat = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    selected = features.gather(1, gather_feat)
    # attention 的 token 维在最后一维，先取相同 K 个位置，再转成 [B,K,Q]。
    roles = None if attention is None else attention.gather(2, indices.unsqueeze(1).expand(-1, attention.shape[1], -1)).transpose(1, 2)
    # 使用同一批 indices 取坐标，保证特征、角色、坐标一一对应。
    coords = grid[indices]
    return {"features": selected, "roles": roles, "coordinates": coords, "indices": indices, "scores": scores.gather(1, indices)}
