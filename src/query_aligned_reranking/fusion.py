from __future__ import annotations
import torch


def normalize_scores(values, method="none", eps=1e-8, temperature=1.0):
    """逐个 query 归一化 Top-K 候选分数，使两类分数可以安全融合。"""
    # 一行最大值与最小值几乎相同，说明这些分数没有排序信息。
    degenerate = (values.amax(1, keepdim=True) - values.amin(1, keepdim=True)) < eps
    if method == "none": out = values
    elif method == "minmax": out = (values-values.amin(1,keepdim=True))/(values.amax(1,keepdim=True)-values.amin(1,keepdim=True)).clamp_min(eps)
    elif method == "zscore": out = (values-values.mean(1,keepdim=True))/values.std(1,keepdim=True,unbiased=False).clamp_min(eps)
    elif method == "softmax": out = torch.softmax(values/max(temperature,eps),dim=1)
    elif method == "rank":
        order=values.argsort(1,descending=True); ranks=torch.empty_like(order); ranks.scatter_(1,order,torch.arange(values.shape[1],device=values.device).expand_as(order)); out=1.0/(ranks.float()+1)
    else: raise ValueError(f"Unknown normalization: {method}")
    out = torch.where(degenerate, torch.zeros_like(out), out) if method != "none" else out
    return out, degenerate.squeeze(1)


def fuse_scores(global_scores, local_scores, normalization="minmax", fusion="weighted_sum", global_weight=0.7, eps=1e-8):
    """融合原 BoQ 全局分数和 QASSR 局部分数，产生最终排序分数。"""
    # 两种分数量纲不同，先分别归一化再组合。
    gn, gd = normalize_scores(global_scores, normalization, eps)
    ln, ld = normalize_scores(local_scores, normalization, eps)
    # 默认 global_weight=0.7，即 70% 全局分数 + 30% 局部分数。
    if fusion == "weighted_sum": final=global_weight*gn+(1-global_weight)*ln
    elif fusion == "additive_residual": final=global_scores+(1-global_weight)*ln
    elif fusion == "reciprocal_rank":
        gr,_=normalize_scores(global_scores,"rank",eps); lr,_=normalize_scores(local_scores,"rank",eps); final=global_weight*gr+(1-global_weight)*lr
    else: raise ValueError(f"Unknown fusion: {fusion}")
    return final, {"global_degenerate":gd,"local_degenerate":ld}
