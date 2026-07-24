from __future__ import annotations
import torch


def early_global_gate(mode, global_scores, thresholds):
    """在昂贵的局部匹配前，仅用全局分数筛选需要处理的 query。

    global_margin/combined 都包含全局 margin 条件，可以安全地提前计算；
    其他门控依赖局部指标，必须先匹配，所以返回全 True。
    """
    if mode not in {"global_margin", "combined"}:
        return torch.ones(global_scores.shape[0], dtype=torch.bool, device=global_scores.device)
    if global_scores.shape[1] < 2:
        raise ValueError("global-margin gate requires at least two candidates")
    global_margin = global_scores[:, 0] - global_scores[:, 1]
    return global_margin < thresholds["global_margin_max"]


def gate_queries(mode, global_scores, local_scores, match_entropy, match_count, thresholds,
                 reranked_order=None, early_decision=None, oracle=None):
    """决定每个 query 是否采用重排结果，返回 [N] 布尔向量。

    True 表示采用 QASSR 顺序，False 表示保留原 BoQ 顺序。

    V2 关键修正：entropy/count 必须读取“重排后第一名”所在的候选列；
    旧实现固定读取第 0 列，实际对应的是“原始全局第一名”。
    """
    if mode in {"none", "all-query"}: return torch.ones(global_scores.shape[0], dtype=torch.bool, device=global_scores.device)
    # 全局第一、二名越接近，原检索越不确定，越值得尝试重排。
    global_margin = global_scores[:,0] - global_scores[:,1]
    local_top = local_scores.topk(min(2, local_scores.shape[1]), 1).values
    # 局部第一名领先第二名越多，说明局部判断越明确。
    local_margin = local_top[:,0] - local_top[:,1] if local_top.shape[1] > 1 else local_top[:,0]
    if match_entropy.ndim == 2:
        if reranked_order is None:
            raise ValueError("reranked_order is required for candidate-aligned gate metrics")
        top_position = reranked_order[:, 0:1]
        entropy = match_entropy.gather(1, top_position).squeeze(1)
        counts = match_count.gather(1, top_position).squeeze(1)
    else:
        entropy, counts = match_entropy, match_count
    global_decision = (global_margin < thresholds["global_margin_max"])
    if early_decision is not None:
        # 防御式检查：确保前后两处 margin 判定完全一致，避免跳过了本应匹配的 query。
        if not torch.equal(global_decision, early_decision.to(global_decision.device)):
            raise RuntimeError("early and final global-margin decisions disagree")
        global_decision = early_decision.to(global_decision.device)
    if mode == "global_margin": decision=global_decision
    elif mode == "local_margin": decision=local_margin >= thresholds["local_margin_min"]
    elif mode in {"local_entropy", "entropy"}: decision=entropy <= thresholds["entropy_max"]
    elif mode == "combined": decision=global_decision & (local_margin >= thresholds["local_margin_min"]) & (entropy <= thresholds["entropy_max"]) & (counts >= thresholds["min_matches"])
    elif mode == "oracle":
        if oracle is None: raise ValueError("oracle labels are analysis-only and required")
        decision=oracle.bool()
    else: raise ValueError(f"Unknown gate: {mode}")
    return decision


def apply_gate(original_indices, reranked_indices, decision):
    """按 query 逐行选择原始顺序或重排后的候选顺序。"""
    return torch.where(decision[:,None], reranked_indices, original_indices)


def risk_coverage(correct, confidence, points=20):
    """按置信度从高到低扩大覆盖率，并统计当前覆盖部分的错误率。"""
    order=confidence.argsort(descending=True); correct=correct[order].float(); n=len(correct); rows=[]
    for end in torch.linspace(1,n,steps=min(points,n)).round().long().unique():
        rows.append({"coverage":float(end/n),"risk":float(1-correct[:end].mean())})
    return rows
