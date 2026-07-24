from __future__ import annotations
import math
import numpy as np


def transition_stats(base_ok, rerank_ok):
    """统计重排前后 Top-1 的四种转换：修复、新错、仍错、都正确。"""
    base_ok=np.asarray(base_ok,bool); rerank_ok=np.asarray(rerank_ok,bool)
    fixed=int((~base_ok & rerank_ok).sum()); new=int((base_ok & ~rerank_ok).sum())
    still=int((~base_ok & ~rerank_ok).sum()); both=int((base_ok & rerank_ok).sum())
    return {"fixed":fixed,"new_error":new,"still_wrong":still,"both_correct":both,"net_gain":fixed-new,
            "repair_rate":fixed/max(fixed+still,1),"corruption_rate":new/max(new+both,1)}


def paired_bootstrap(base_ok, rerank_ok, samples=10000, seed=42):
    """成对重采样同一批 query，估计 Recall@1 差值及 95% 置信区间。"""
    base=np.asarray(base_ok,float); rerank=np.asarray(rerank_ok,float); rng=np.random.default_rng(seed); n=len(base)
    idx=rng.integers(0,n,size=(samples,n)); delta=(rerank[idx]-base[idx]).mean(1)
    return {"delta":float((rerank-base).mean()),"ci95_low":float(np.quantile(delta,.025)),"ci95_high":float(np.quantile(delta,.975)),"samples":samples,"seed":seed}


def mcnemar_exact(base_ok, rerank_ok):
    """根据 Fixed 与 NewError 数量执行精确 McNemar 显著性检验。"""
    base=np.asarray(base_ok,bool); rerank=np.asarray(rerank_ok,bool); b=int((base & ~rerank).sum()); c=int((~base & rerank).sum()); n=b+c
    if n==0:return 1.0
    tail=sum(math.comb(n,k) for k in range(0,min(b,c)+1))/(2**n)
    return min(1.0,2*tail)
