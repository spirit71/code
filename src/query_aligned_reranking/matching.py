from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def correlation_matrix(q_feat, r_feat, q_role=None, r_role=None, mode="feature_only", role_weight=0.5):
    """计算 query 图与候选图之间所有局部 token 的两两相似度。

    q_feat/r_feat 是 [P,T,C]：P 个图像对、每图 T 个 token、每 token C 维。
    返回 [P,Tq,Tr] 相关矩阵；计算全程留在 GPU，避免反复搬运数据。
    """
    if q_feat.ndim == 2:
        q_feat = q_feat.unsqueeze(0).expand(r_feat.shape[0], -1, -1)
    # 特征单位化后，批量矩阵乘法得到的点积就是余弦相似度。
    feat = torch.bmm(F.normalize(q_feat.float(), dim=-1), F.normalize(r_feat.float(), dim=-1).transpose(1, 2))
    if mode == "feature_only":
        return feat, None
    if q_role is None or r_role is None:
        raise ValueError(f"{mode} requires query-role vectors")
    if q_role.ndim == 2:
        q_role = q_role.unsqueeze(0).expand(r_role.shape[0], -1, -1)
    # role 向量描述一个 token 分别受各个 BoQ query 关注的模式。
    role = torch.bmm(F.normalize(q_role.float(), dim=-1), F.normalize(r_role.float(), dim=-1).transpose(1, 2))
    if mode == "role_only":
        combined = role
    elif mode == "product_relu":
        combined = feat * role.relu()
    elif mode == "product_sigmoid":
        combined = feat * torch.sigmoid(role)
    elif mode == "additive":
        combined = (1.0 - role_weight) * feat + role_weight * role
    elif mode == "residual_role":
        # V2：把 role 当成“有界的小修正”，而不是 product_relu 那样的硬乘法门。
        # role 的余弦值位于 [-1, 1]。下面的调制因子位于
        # [1-role_weight, 1+role_weight]，因此 role_weight=0.25 时，原始
        # feature similarity 最多只被压低 25%，同时相同 role 可获得适度增强。
        # 这保留了局部外观相似度这一主证据，符合 R2Former 中联合使用
        # feature/attention 的思想，但不引入新的可训练参数。
        if not 0.0 <= role_weight <= 1.0:
            raise ValueError("residual_role requires role_weight in [0, 1]")
        combined = feat * (1.0 + role_weight * role.clamp(-1.0, 1.0))
    else:
        raise ValueError(f"Unknown role matching mode: {mode}")
    return combined, role


def _entropy(values, mask):
    """计算有效匹配分数的熵；熵低通常表示匹配更集中、更明确。"""
    weights = torch.where(mask, values.clamp_min(0), torch.zeros_like(values))
    probs = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(torch.where(probs > 0, probs * probs.clamp_min(1e-12).log(), torch.zeros_like(probs))).sum(dim=1)


def batched_mutual_matching(q_feat, r_feat, q_role=None, r_role=None, mode="feature_only", threshold=0.5, role_weight=0.5):
    """批量执行双向最近邻匹配，并汇总每个图像对的局部分数。"""
    corr, role_corr = correlation_matrix(q_feat, r_feat, q_role, r_role, mode, role_weight)
    # q_to_r[p,i]：图像对 p 中，与 query token i 最相似的 reference token。
    q_to_r = corr.argmax(dim=2)
    # r_to_q 从反方向找每个 reference token 最喜欢哪个 query token。
    r_to_q = corr.argmax(dim=1)
    q_idx = torch.arange(corr.shape[1], device=corr.device).expand(corr.shape[0], -1)
    # 只有双方互相把对方选为第一名，才叫双向最近邻。
    mutual = r_to_q.gather(1, q_to_r).eq(q_idx)
    chosen = corr.gather(2, q_to_r.unsqueeze(-1)).squeeze(-1)
    # 还要去掉余弦相似度低于 threshold 的弱匹配。
    keep = mutual & chosen.ge(threshold)
    count = keep.sum(1)
    # quality 是有效匹配的平均相似度；clamp 防止零匹配时除以零。
    quality = (chosen * keep).sum(1) / count.clamp_min(1)
    quality = torch.where(count > 0, quality, torch.zeros_like(quality))
    # coverage 表示有多少比例的 token 成功形成可靠匹配。
    coverage = count.float() / min(corr.shape[1], corr.shape[2])
    std = torch.sqrt(((chosen - quality[:, None]).square() * keep).sum(1) / count.clamp_min(1))
    mean_role = torch.zeros_like(quality)
    if role_corr is not None:
        selected_role = role_corr.gather(2, q_to_r.unsqueeze(-1)).squeeze(-1)
        mean_role = (selected_role * keep).sum(1) / count.clamp_min(1)
        mean_role = torch.where(count > 0, mean_role, torch.zeros_like(mean_role))
    return {
        "matches": keep, "match_indices": q_to_r, "match_count": count, "quality": quality,
        # local_score 同时要求“匹配得准”(quality)和“匹配得多”(coverage)。
        "coverage": coverage, "local_score": quality * coverage, "score_std": std,
        "match_entropy": _entropy(chosen, keep), "mean_role_compatibility": mean_role,
        "correlation": corr,
    }


def spatial_verification(matches, q_to_r, q_coords, r_coords, mode="none", sigma=0.15):
    """检查匹配点的空间布局是否一致，返回 [0,1] 的空间可信度。"""
    batch = matches.shape[0]
    if mode == "none":
        # 关闭空间验证时返回 1，乘到 local_score 上不会改变原分数。
        return torch.ones(batch, device=matches.device), torch.zeros(batch, device=matches.device)
    selected_r = r_coords.gather(1, q_to_r.unsqueeze(-1).expand(-1, -1, 2))
    displacement = selected_r - q_coords if q_coords.ndim == 3 else selected_r - q_coords.unsqueeze(0)
    if mode == "translation_inlier":
        # V2 加速版：每一行仍独立执行“中位数主平移→内点比例”，但把所有
        # query-candidate pair 一次性放在 GPU 上计算。NaN 只负责让未通过
        # MNN/相似度阈值的 token 不参与中位数，不改变有效匹配的数学定义。
        valid_displacement=displacement.masked_fill(~matches.unsqueeze(-1),float("nan"))
        center=torch.nanmedian(valid_displacement,dim=1).values
        errors=torch.linalg.vector_norm(displacement-center.unsqueeze(1),dim=2)
        inliers=matches & errors.le(max(sigma,1e-12))
        count=matches.sum(1)
        scores=inliers.sum(1).float()/count.clamp_min(1)
        inlier_errors=errors.masked_fill(~inliers,float("nan"))
        residuals=torch.nanmedian(inlier_errors,dim=1).values
        # 零匹配 pair 与旧循环版保持一致：score=0，residual=+inf。
        scores=torch.where(count>0,scores,torch.zeros_like(scores))
        residuals=torch.where(count>0,residuals,torch.full_like(residuals,float("inf")))
        return scores,residuals
    residuals, scores = [], []
    for idx in range(batch):
        disp = displacement[idx][matches[idx]]
        if disp.numel() == 0:
            residual = torch.tensor(float("inf"), device=matches.device)
            score = torch.tensor(0.0, device=matches.device)
        elif mode == "translation_median":
            residual = torch.linalg.vector_norm(disp - disp.median(0).values, dim=1).median()
            score = torch.exp(-residual.square() / max(sigma, 1e-12) ** 2)
        elif mode == "pairwise_relative":
            q = (q_coords[idx] if q_coords.ndim == 3 else q_coords)[matches[idx]]
            r = selected_r[idx][matches[idx]]
            if len(q) < 2:
                residual, score = torch.tensor(float("inf"), device=q.device), torch.tensor(0.0, device=q.device)
            else:
                dq, dr = torch.pdist(q), torch.pdist(r)
                residual = (dq - dr).abs().median()
                score = torch.exp(-residual.square() / max(sigma, 1e-12) ** 2)
        else:
            raise ValueError(f"Unknown spatial mode: {mode}")
        residuals.append(residual); scores.append(score)
    return torch.stack(scores), torch.stack(residuals)
