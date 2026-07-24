import numpy as np
import pytest
import torch

from src.boq import BoQ
from src.query_aligned_reranking.features import BoQFeatureBundle, FeatureCache
from src.query_aligned_reranking.fusion import normalize_scores
from src.query_aligned_reranking.gating import apply_gate, early_global_gate, gate_queries
from src.query_aligned_reranking.matching import batched_mutual_matching, correlation_matrix, spatial_verification
from src.query_aligned_reranking.query_role import aligned_readout_score, query_readout
from src.query_aligned_reranking.selection import select_tokens
from src.query_aligned_reranking.statistics import paired_bootstrap, transition_stats


def test_global_descriptor_unchanged_and_feature_shapes():
    torch.manual_seed(0); model=BoQ(8,64,4,2,8).eval(); x=torch.randn(2,8,4,5)
    legacy,_=model(x); result=model(x,return_intermediates=True)
    torch.testing.assert_close(legacy,result["global"])
    assert result["x0"].shape==(2,20,64); assert len(result["encoder_outputs"])==2
    assert result["x_last"].shape==(2,20,64); assert len(result["block_outputs"])==2
    assert result["last_attention"].shape==(2,4,20); assert result["spatial_shape"]==(4,5)


def test_reference_query_split_and_index_order():
    bundle=BoQFeatureBundle(torch.randn(5,3),{"x1":torch.randn(5,4,2)},None,torch.arange(5),3,{})
    r,q=bundle.global_split; assert len(r)==3 and len(q)==2
    with pytest.raises(ValueError): BoQFeatureBundle(torch.randn(5,3),{},None,torch.tensor([1,0,2,3,4]),3,{})


def test_cache_key_changes_with_config(tmp_path):
    cache=FeatureCache(tmp_path); a={"dataset":"x","checkpoint_hash":"a","token_count":64}; b={"dataset":"x","checkpoint_hash":"b","token_count":64}
    assert cache.path(a)!=cache.path(b)


def test_attention_normalization_and_query_readout():
    x=torch.tensor([[[1.,0.],[0.,1.]]]); a=torch.tensor([[[2.,2.],[1.,3.]]])
    z=query_readout(x,a); torch.testing.assert_close(z,torch.tensor([[[.5,.5],[.25,.75]]]))


def test_random_selection_reproducible_and_coordinates():
    x=torch.randn(2,6,4); a=torch.softmax(torch.randn(2,3,6),-1)
    one=select_tokens(x,a,(2,3),"random",3,7); two=select_tokens(x,a,(2,3),"random",3,7)
    torch.testing.assert_close(one["indices"],two["indices"]); assert one["coordinates"].shape==(2,3,2); assert one["roles"].shape==(2,3,3)


def test_hungarian_is_upper_bound():
    torch.manual_seed(2); q=torch.randn(4,5,8); r=torch.randn(4,5,8)
    same=aligned_readout_score(q,r,"same-index"); hung=aligned_readout_score(q,r,"hungarian")
    assert torch.all(hung>=same-1e-6)


def test_feature_role_correlation_modes():
    q=torch.eye(3); r=torch.eye(3).unsqueeze(0); role=torch.eye(3)
    feat,_=correlation_matrix(q,r,mode="feature_only"); product,rc=correlation_matrix(q,r,role,role.unsqueeze(0),"product_relu")
    assert feat.shape==(1,3,3) and rc.shape==(1,3,3); torch.testing.assert_close(product,feat)


def test_residual_role_keeps_feature_as_main_evidence():
    q=torch.tensor([[1.,0.]]); r=torch.tensor([[[1.,0.],[-1.,0.]]])
    q_role=torch.tensor([[1.,0.]]); r_role=torch.tensor([[[1.,0.],[-1.,0.]]])
    feat,_=correlation_matrix(q,r,mode="feature_only")
    residual,_=correlation_matrix(q,r,q_role,r_role,"residual_role",.25)
    # 正 role 只增强 25%，负 role 也只衰减 25%，不会像 ReLU 乘法一样归零。
    torch.testing.assert_close(residual,feat*torch.tensor([[[1.25,.75]]]))


def test_mutual_identity_and_no_match():
    eye=torch.eye(4); yes=batched_mutual_matching(eye,eye.unsqueeze(0),threshold=.9); no=batched_mutual_matching(eye,(-eye).unsqueeze(0),threshold=.9)
    assert yes["match_count"].item()==4; assert no["match_count"].item()==0


def test_spatial_perfect_translation_and_pairwise():
    matches=torch.tensor([[1,1,1]],dtype=torch.bool); ids=torch.tensor([[0,1,2]]); q=torch.tensor([[0.,0.],[.5,0.],[1.,0.]]); r=(q+torch.tensor([.2,.3])).unsqueeze(0)
    for mode in ["translation_median","translation_inlier","pairwise_relative"]:
        score,res=spatial_verification(matches,ids,q,r,mode); assert score.item()==pytest.approx(1.); assert res.item()==pytest.approx(0.)


def test_spatial_translation_inlier_rejects_outlier():
    matches=torch.ones(1,4,dtype=torch.bool); ids=torch.arange(4).unsqueeze(0)
    q=torch.tensor([[0.,0.],[.2,0.],[.4,0.],[.6,0.]])
    r=(q+torch.tensor([.1,.1])).clone(); r[-1]=torch.tensor([1.,1.])
    score,_=spatial_verification(matches,ids,q,r.unsqueeze(0),"translation_inlier",.05)
    assert score.item()==pytest.approx(.75)


def test_constant_normalization_does_not_amplify():
    out,flag=normalize_scores(torch.ones(2,5),"minmax"); assert out.count_nonzero()==0; assert flag.all()


def test_gate_keep_original_and_candidate_set():
    original=torch.tensor([[1,2,3],[4,5,6]]); proposed=torch.tensor([[3,1,2],[6,5,4]]); decision=torch.tensor([False,True]); final=apply_gate(original,proposed,decision)
    torch.testing.assert_close(final[0],original[0]); assert set(final[1].tolist())==set(original[1].tolist())


def test_gate_metrics_follow_reranked_top1_and_early_margin():
    global_scores=torch.tensor([[.60,.59,.20],[.90,.40,.10]])
    local=torch.tensor([[.1,.8,.2],[.1,.8,.2]])
    entropy=torch.tensor([[9.,1.,9.],[9.,1.,9.]])
    counts=torch.tensor([[1,8,1],[1,8,1]])
    order=torch.tensor([[1,0,2],[1,0,2]])
    thresholds={"global_margin_max":.02,"local_margin_min":.1,"entropy_max":2.,"min_matches":4}
    early=early_global_gate("combined",global_scores,thresholds)
    decision=gate_queries("combined",global_scores,local,entropy,counts,thresholds,order,early)
    # 第 1 行应读取重排后第一名（原候选列 1）的 entropy=1/count=8；第 2 行被 early gate 跳过。
    torch.testing.assert_close(early,torch.tensor([True,False])); torch.testing.assert_close(decision,torch.tensor([True,False]))


def test_transition_sum():
    stats=transition_stats([1,1,0,0],[1,0,1,0]); assert sum(stats[k] for k in ["fixed","new_error","still_wrong","both_correct"])==4


def test_bootstrap_reproducibility():
    a=paired_bootstrap([1,0,1],[1,1,0],100,9); b=paired_bootstrap([1,0,1],[1,1,0],100,9); assert a==b
