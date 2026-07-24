#!/usr/bin/env python
"""QASSR 推理入口：加载模型/数据，提特征，执行重排并保存完整结果。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.test_checkpoints_in_dir import build_model_and_datamodule
from src.query_aligned_reranking.features import FeatureCache, extract_feature_bundle, make_cache_metadata
from src.query_aligned_reranking.runner import baseline_verification, environment_info, run_reranking, write_outputs
from train import HyperParams

# 这些参数会直接影响最终 Recall；正式 test 阶段不应边看结果边修改它们。
CORE={"feature_layer","top_k","token_selection","token_count","matching","role_weight","similarity_threshold","spatial","spatial_sigma","normalization","fusion","global_weight","gate","gate_thresholds"}

def args_parser():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,default=ROOT/"configs/qassr/default.yaml")
    p.add_argument("--datasets",default=""); p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/qassr"); p.add_argument("--cache-dir",type=Path,default=ROOT/"feature_cache")
    p.add_argument("--backbone",default="dinov2_vitb14"); p.add_argument("--unfreeze-n",type=int,default=2); p.add_argument("--output-dim",type=int,default=8192); p.add_argument("--num-queries",type=int,default=64)
    p.add_argument("--eval-bs",type=int,default=32); p.add_argument("--nw",type=int,default=8); p.add_argument("--device",default="cuda:0"); p.add_argument("--set",action="append",default=[],help="Validation-only key=value override")
    p.add_argument("--allow-test-overrides",action="store_true",help="Diagnostic only: explicitly allow core overrides on test data; results are not protocol-valid model selection.")
    return p.parse_args()

def load_config(path, overrides, allow_test_overrides=False):
    """读取 YAML，并应用命令行 --set key=value；同时执行测试协议保护。"""
    config=yaml.safe_load(path.read_text())
    for item in overrides:
        key,value=item.split("=",1); parsed=yaml.safe_load(value); target=config
        parts=key.split(".")
        for part in parts[:-1]: target=target.setdefault(part,{})
        target[parts[-1]]=parsed
    if config.get("dataset_split")=="test" and overrides:
        changed={x.split("=",1)[0] for x in overrides}
        forbidden=changed&CORE
        if forbidden and not allow_test_overrides: raise RuntimeError(f"Test config forbids core overrides: {sorted(forbidden)}. Use a frozen YAML; --allow-test-overrides is diagnostic-only.")
        if forbidden:
            config["protocol_warning"]="DIAGNOSTIC ONLY: core parameters were overridden on test data"
    return config

def main():
    """串联一次实验的全部步骤；只做推理，不执行训练或反向传播。"""
    args=args_parser(); config=load_config(args.config,args.set,args.allow_test_overrides); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    hp=HyperParams(); hp.backbone_name=args.backbone; hp.unfreeze_n_blocks=args.unfreeze_n; hp.output_dim=args.output_dim; hp.num_queries=args.num_queries; hp.eval_batch_size=args.eval_bs; hp.num_workers=args.nw; hp.silent=True
    if args.datasets:
        names=[x.strip() for x in args.datasets.split(",")]; hp.test_sets={k:v for k,v in hp.test_sets.items() if k in names}; hp.val_sets={k:v for k,v in hp.val_sets.items() if k in names}
    # strict=True 保证 checkpoint 与当前模型结构完全匹配，避免悄悄漏加载参数。
    model,dm=build_model_and_datamodule(hp); checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True); model.load_state_dict(checkpoint.get("state_dict",checkpoint),strict=True); model.to(device).eval()
    split=config.get("dataset_split","validation")
    if split=="test": dm.setup("test"); loaders=dm.test_dataloader()
    else: dm.setup("fit"); loaders=dm.val_dataloader()
    if not loaders: raise RuntimeError(f"No datasets selected for split={split}")
    cache=FeatureCache(args.cache_dir)
    for loader in loaders:
        ds=loader.dataset; meta=make_cache_metadata(args.checkpoint,ds,dm.val_img_size,model,[config["feature_layer"],"attention"])
        bundle,timing=extract_feature_bundle(model,loader,device,meta,cache)
        images,_=next(iter(loader)); verification=baseline_verification(model,images.to(device),bundle,ds)
        result=run_reranking(bundle,ds,config,device); result["summary"]["latency"]["feature_extraction_seconds"]=timing["seconds"]; result["summary"]["cache_hit"]=timing["cache_hit"]
        target=args.output_dir/config["experiment_name"]/ds.dataset_name
        write_outputs(target,config,ds,verification,result,environment_info()); print(json.dumps({"dataset":ds.dataset_name,"output":str(target),**result["summary"]},indent=2))

if __name__=="__main__": main()
