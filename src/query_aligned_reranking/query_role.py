from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np


def query_readout(features, attention):
    """用每个 BoQ query 的 attention 对空间 token 加权求和。

    输入 features=[B,T,C]、attention=[B,Q,T]，输出 [B,Q,C]；可以理解为
    Q 个 query 分别从同一张图中读出自己负责的内容。
    """
    if attention.shape[-1] != features.shape[1]: raise ValueError("attention/features token mismatch")
    weights=attention/attention.sum(-1,keepdim=True).clamp_min(1e-12)
    return torch.bmm(weights.float(),features.float())


def aligned_readout_score(q, r, alignment="same-index", seed=42, fixed_permutation=None):
    """用多种对齐方式比较两张图的 Q 个 query readout，用于机制分析。"""
    q=F.normalize(q.float(),dim=-1); r=F.normalize(r.float(),dim=-1); sim=torch.bmm(q,r.transpose(1,2))
    m=q.shape[1]
    # same-index 检验“相同编号的 BoQ query 是否具有跨图片一致角色”。
    if alignment=="same-index": return sim.diagonal(dim1=1,dim2=2).mean(1)
    if alignment in {"random-per-pair","fixed-random"}:
        gen=torch.Generator(device=q.device).manual_seed(seed)
        if alignment=="fixed-random":
            perm=fixed_permutation if fixed_permutation is not None else torch.randperm(m,generator=gen,device=q.device)
            return sim[:,torch.arange(m,device=q.device),perm].mean(1)
        vals=[]
        for i in range(len(q)):
            perm=torch.randperm(m,generator=gen,device=q.device); vals.append(sim[i,torch.arange(m,device=q.device),perm].mean())
        return torch.stack(vals)
    if alignment=="hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc: raise RuntimeError("Hungarian alignment requires scipy") from exc
        vals=[]
        for matrix in sim.detach().cpu().numpy():
            rows,cols=linear_sum_assignment(-matrix); vals.append(matrix[rows,cols].mean())
        return torch.tensor(vals,device=q.device)
    if alignment=="mean-pool": return (q.mean(1)*r.mean(1)).sum(1)
    raise ValueError(f"Unknown alignment: {alignment}")


def binary_auc(labels, scores):
    """不用 sklearn 计算二分类 ROC-AUC/PR-AUC；越接近 1 区分能力越强。"""
    labels=np.asarray(labels,dtype=bool); scores=np.asarray(scores,float); pos=labels.sum(); neg=(~labels).sum()
    if pos==0 or neg==0: return {"roc_auc":float("nan"),"pr_auc":float("nan")}
    order=np.argsort(-scores); y=labels[order].astype(float); tp=np.cumsum(y); fp=np.cumsum(1-y)
    tpr=np.r_[0,tp/pos]; fpr=np.r_[0,fp/neg]; precision=tp/np.arange(1,len(y)+1); recall=tp/pos
    return {"roc_auc":float(np.trapezoid(tpr,fpr)),"pr_auc":float(np.sum((recall-np.r_[0,recall[:-1]])*precision))}


def bootstrap_auc(labels, scores, samples=2000, seed=42):
    """bootstrap 重采样 AUC，估计结果的不确定性区间。"""
    labels=np.asarray(labels); scores=np.asarray(scores); rng=np.random.default_rng(seed); values=[]
    for _ in range(samples):
        idx=rng.integers(0,len(labels),len(labels)); value=binary_auc(labels[idx],scores[idx])["roc_auc"]
        if np.isfinite(value): values.append(value)
    return {"roc_auc":binary_auc(labels,scores)["roc_auc"],"ci95_low":float(np.quantile(values,.025)),"ci95_high":float(np.quantile(values,.975)),"seed":seed}
