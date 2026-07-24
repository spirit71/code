from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
import torch.nn.functional as F

from src import utils
from src.xl_reranking import global_topk_search, recall_at_k
from .fusion import fuse_scores
from .gating import apply_gate, early_global_gate, gate_queries, risk_coverage
from .matching import batched_mutual_matching, spatial_verification
from .selection import select_tokens
from .statistics import mcnemar_exact, paired_bootstrap, transition_stats


def baseline_verification(model, images, bundle, dataset):
    """验证新增中间输出接口没有改变原始 descriptor 和 Recall。"""
    with torch.inference_mode():
        legacy, _ = model(images)
        modern = model(images, return_intermediates=True)["global"]
    # 同一批图走旧/新接口，数值必须一致，否则立即停止实验。
    torch.testing.assert_close(legacy, modern, rtol=1e-5, atol=1e-6)
    diff=(legacy-modern).abs(); cosine=F.cosine_similarity(legacy,modern).mean()
    original=utils.compute_recall_performance(bundle.global_descriptors,dataset.num_references,dataset.num_queries,dataset.ground_truth,[1,5,10,20])
    refs,queries=bundle.global_split; _,pred=global_topk_search(queries,refs,max(20,50)); new=recall_at_k(pred,dataset.ground_truth)
    if original != new: raise RuntimeError(f"Baseline Recall mismatch: original={original}, new={new}")
    return {"max_descriptor_difference":float(diff.max()),"mean_cosine":float(cosine),"original_recall":original,"new_recall":new}


def _sync(device):
    if torch.device(device).type == "cuda": torch.cuda.synchronize(device)


def run_reranking(bundle, dataset, config, device):
    """执行完整 QASSR：全局召回候选→局部匹配→融合→门控→统计。"""
    # 原 BoQ 描述符先召回 Top-K；QASSR 只调整这 K 个候选内部的顺序。
    refs_g, queries_g=bundle.global_split; global_scores,base_idx=global_topk_search(queries_g,refs_g,config["top_k"])
    local_refs,local_queries=bundle.layer(config["feature_layer"])
    attn_refs,attn_queries=(bundle.split(bundle.attentions) if bundle.attentions is not None else (None,None))
    spatial_shape=tuple(bundle.metadata["spatial_shape"])
    # reference 和 query 必须使用完全相同的 token 选择规则。
    ref_sel=select_tokens(local_refs,attn_refs,spatial_shape,config["token_selection"],config["token_count"],config["random_seed"])
    query_sel=select_tokens(local_queries,attn_queries,spatial_shape,config["token_selection"],config["token_count"],config["random_seed"])
    n,k=base_idx.shape
    # V2 early gate：只把全局第一、二名差距较小的“困难 query”送入局部匹配。
    # eligible_ids 保存原 query 行号，后续写回 metrics 时仍保持 [N,K] 对齐。
    early_gate=early_global_gate(config["gate"],global_scores,config["gate_thresholds"])
    eligible_ids=early_gate.nonzero(as_tuple=False).squeeze(1)
    metric_names=["local_score","match_count","quality","coverage","score_std","match_entropy","mean_role_compatibility","spatial_score","spatial_residual"]
    # 每个 [query,候选] 图像对都保存一组可解释指标。
    metrics={name:torch.zeros(n,k) for name in metric_names}; pair_ms=torch.zeros(n,k)
    _sync(device); start=time.perf_counter()
    with torch.inference_mode():
        for qs in range(0,len(eligible_ids),config["query_chunk"]):
            query_ids=eligible_ids[qs:qs+config["query_chunk"]]
            q_count=len(query_ids)
            for cs in range(0,k,config["candidate_chunk"]):
                ce=min(k,cs+config["candidate_chunk"]); c_count=ce-cs
                ids=base_idx[query_ids,cs:ce]
                # 将 Q 个 query 和各自 C 个候选展平成 Q*C 个 pair，一次性送入 GPU。
                qf=query_sel["features"][query_ids].to(device).unsqueeze(1).expand(-1,c_count,-1,-1).reshape(q_count*c_count,query_sel["features"].shape[1],-1)
                rf=ref_sel["features"][ids].to(device).reshape(q_count*c_count,ref_sel["features"].shape[1],-1)
                if query_sel["roles"] is None:
                    qr=rr=None
                else:
                    qr=query_sel["roles"][query_ids].to(device).unsqueeze(1).expand(-1,c_count,-1,-1).reshape(q_count*c_count,query_sel["roles"].shape[1],-1)
                    rr=ref_sel["roles"][ids].to(device).reshape(q_count*c_count,ref_sel["roles"].shape[1],-1)
                qcoords=query_sel["coordinates"][query_ids].to(device).unsqueeze(1).expand(-1,c_count,-1,-1).reshape(q_count*c_count,query_sel["coordinates"].shape[1],2)
                rcoords=ref_sel["coordinates"][ids].to(device).reshape(q_count*c_count,ref_sel["coordinates"].shape[1],2)
                _sync(device); pair_start=time.perf_counter()
                result=batched_mutual_matching(qf,rf,qr,rr,config["matching"],config["similarity_threshold"],config["role_weight"])
                spatial,residual=spatial_verification(result["matches"],result["match_indices"],qcoords,rcoords,config["spatial"],config["spatial_sigma"])
                _sync(device); elapsed=(time.perf_counter()-pair_start)*1000
                result["spatial_score"],result["spatial_residual"]=spatial,residual
                result["local_score"]=result["local_score"]*spatial
                for name in metric_names:
                    # 使用 eligible_ids 属于高级索引；PyTorch 不会像普通切片那样
                    # 自动把 Long(match_count) 转成 Float，因此这里显式对齐目标 dtype。
                    value=result[name].reshape(q_count,c_count).detach().cpu()
                    metrics[name][query_ids,cs:ce]=value.to(metrics[name].dtype)
                pair_ms[query_ids,cs:ce]=elapsed/max(q_count*c_count,1)
    total_matching=time.perf_counter()-start
    # 把全局检索证据和局部匹配证据合成最终分数，再按降序排列。
    final,degenerate=fuse_scores(global_scores,metrics["local_score"],config["normalization"],config["fusion"],config["global_weight"])
    order=final.argsort(1,descending=True); proposed=base_idx.gather(1,order)
    gate=gate_queries(config["gate"],global_scores,metrics["local_score"],metrics["match_entropy"],metrics["match_count"],config["gate_thresholds"],order,early_gate)
    reranked=apply_gate(base_idx,proposed,gate)
    # Top-1 只要属于该 query 的任意一个 GT positive，就算检索正确。
    gt=dataset.ground_truth; base_ok=np.array([np.isin(int(base_idx[i,0]),gt[i]) for i in range(n)]); rerank_ok=np.array([np.isin(int(reranked[i,0]),gt[i]) for i in range(n)])
    transitions=transition_stats(base_ok,rerank_ok)
    confidence=(metrics["local_score"].topk(2,1).values[:,0]-metrics["local_score"].topk(2,1).values[:,1])
    summary={"baseline_recall":recall_at_k(base_idx,gt),"reranked_recall":recall_at_k(reranked,gt),"delta_recall":{},"transitions":transitions,
             "rerank_coverage":float(gate.float().mean()),"bootstrap":paired_bootstrap(base_ok,rerank_ok,config["bootstrap_samples"],config["seed"]),
             "mcnemar_p":mcnemar_exact(base_ok,rerank_ok),"latency":{"local_matching_seconds":total_matching,"ms_query":1000*total_matching/n,"ms_pair":float(pair_ms[early_gate].mean()) if early_gate.any() else 0.0,
             "matched_queries":int(early_gate.sum()),"skipped_queries":int((~early_gate).sum()),"matched_pairs":int(early_gate.sum())*k,"pair_saving_ratio":float((~early_gate).float().mean())},
             "risk_coverage":risk_coverage(torch.as_tensor(rerank_ok),confidence),
             "pair_score_distribution":{"all_mean":float(metrics["local_score"].mean()),"all_std":float(metrics["local_score"].std())},
             "bytes_image":bundle.bytes_per_image(),"peak_gpu_memory":torch.cuda.max_memory_allocated(device) if torch.device(device).type=="cuda" else 0}
    summary["delta_recall"]={str(k):summary["reranked_recall"][k]-summary["baseline_recall"][k] for k in summary["baseline_recall"]}
    return {"summary":summary,"base_idx":base_idx,"reranked":reranked,"global_scores":global_scores,"metrics":metrics,"gate":gate,"degenerate":degenerate,"pair_ms":pair_ms,"base_ok":base_ok,"rerank_ok":rerank_ok}


def write_outputs(output_dir, config, dataset, verification, result, environment):
    """把配置、汇总指标、逐 query 和逐候选结果完整落盘，便于复查。"""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"resolved_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (out/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
    (out/"baseline_verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    summary=result["summary"]; (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["metric","value"]); [w.writerow([k,json.dumps(v)]) for k,v in summary.items()]
    rows=[]
    for i in range(dataset.num_queries):
        gt=dataset.ground_truth[i]; positive_rank=next((j+1 for j,x in enumerate(result["reranked"][i].tolist()) if np.isin(x,gt)),None)
        final_top1=int(result["reranked"][i,0])
        final_pos=int((result["base_idx"][i] == final_top1).nonzero(as_tuple=False)[0,0])
        rows.append({"query_index":i,"query_path":dataset.qImages[i],"baseline_top1":int(result["base_idx"][i,0]),"reranked_top1":int(result["reranked"][i,0]),"positive_rank":positive_rank,
          "global_margin":float(result["global_scores"][i,0]-result["global_scores"][i,1]),"local_margin":float(result["metrics"]["local_score"][i].topk(2).values.diff().abs()[0]),
          "entropy":float(result["metrics"]["match_entropy"][i,final_pos]),"gate_decision":bool(result["gate"][i]),"match_count":int(result["metrics"]["match_count"][i,final_pos]),
          "quality":float(result["metrics"]["quality"][i,final_pos]),"coverage":float(result["metrics"]["coverage"][i,final_pos]),"role_score":float(result["metrics"]["mean_role_compatibility"][i,final_pos]),
          "spatial_score":float(result["metrics"]["spatial_score"][i,final_pos]),"latency_ms":float(result["pair_ms"][i].sum()),"case":("BothCorrect" if result["base_ok"][i] and result["rerank_ok"][i] else "NewError" if result["base_ok"][i] else "Fixed" if result["rerank_ok"][i] else "StillWrong")})
    with (out/"per_query_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    cand=[]
    for qi in range(dataset.num_queries):
        for rank,ri in enumerate(result["base_idx"][qi].tolist()):
            cand.append({"query_index":qi,"candidate_rank":rank+1,"reference_index":ri,"global_score":float(result["global_scores"][qi,rank]),**{k:float(v[qi,rank]) for k,v in result["metrics"].items()}})
    try:
        import pandas as pd; pd.DataFrame(cand).to_parquet(out/"per_candidate_scores.parquet",index=False)
    except Exception:
        with gzip.open(out/"per_candidate_scores.csv.gz","wt",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=cand[0].keys()); w.writeheader(); w.writerows(cand)
    for case in ["Fixed","NewError","StillWrong","BothCorrect"]:
        (out/f"{case}.txt").write_text("\n".join(row["query_path"] for row in rows if row["case"]==case),encoding="utf-8")
    (out/"bootstrap.json").write_text(json.dumps(summary["bootstrap"],indent=2),encoding="utf-8")
    (out/"transition_stats.json").write_text(json.dumps(summary["transitions"],indent=2),encoding="utf-8")


def environment_info():
    return {"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu","platform":platform.platform(),"pid":os.getpid()}
