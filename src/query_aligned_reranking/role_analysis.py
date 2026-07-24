from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import torch
from src.xl_reranking import global_topk_search
from .query_role import aligned_readout_score, binary_auc, bootstrap_auc, query_readout


def run_query_role_analysis(bundle, ground_truth, output_dir, layer="x2", top_k=50, seed=42, bootstrap_samples=2000):
    """比较正样本、困难负样本和随机负样本的 query-role 分数。

    该函数只用于回答“相同编号的 BoQ query 是否学到稳定角色”，不会参与
    QASSR 主 Recall 计算。所有随机选择都固定 seed，保证可以复现。
    """
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); refs_g,queries_g=bundle.global_split
    _,neighbors=global_topk_search(queries_g,refs_g,top_k); refs_x,queries_x=bundle.layer(layer); refs_a,queries_a=bundle.split(bundle.attentions)
    ref_z=query_readout(refs_x,refs_a); query_z=query_readout(queries_x,queries_a); rng=np.random.default_rng(seed); fixed=torch.randperm(query_z.shape[1],generator=torch.Generator().manual_seed(seed))
    rows=[]; methods=["same-index","random-per-pair","fixed-random","hungarian","mean-pool"]
    for qi,positives in enumerate(ground_truth):
        # positive 来自数据集 GT；hard 是全局检索靠前但错误的图；random 是随机负例。
        positive=int(positives[0]); hard=next(int(x) for x in neighbors[qi] if not np.isin(int(x),positives)); random=int(rng.integers(0,len(ref_z)))
        while np.isin(random,positives): random=int(rng.integers(0,len(ref_z)))
        for pair_type,ri in [("positive",positive),("hard_negative",hard),("random_negative",random)]:
            row={"query_index":qi,"reference_index":ri,"pair_type":pair_type}
            for method in methods:
                row[method]=float(aligned_readout_score(query_z[qi:qi+1],ref_z[ri:ri+1],method,seed+qi,fixed).item())
            rows.append(row)
    with (out/"query_role_pair_scores.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary={"seed":seed,"pairs":len(rows),"methods":{}}
    for method in methods:
        values=np.array([r[method] for r in rows]); labels=np.array([r["pair_type"]=="positive" for r in rows]); groups={}
        for kind in ["positive","hard_negative","random_negative"]:
            group=values[[r["pair_type"]==kind for r in rows]]; groups[kind]={"mean":float(group.mean()),"std":float(group.std())}
        summary["methods"][method]={**groups,**binary_auc(labels,values),"bootstrap_roc_auc":bootstrap_auc(labels,values,bootstrap_samples,seed)}
    (out/"query_role_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary
