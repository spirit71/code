import torch
import numpy as np
from soft_probing import SoftProbing, SoftProbingV2
from interact_head import InteractHead, InteractHeadLite
from geo_visual_sampler import GeoVisualGraphSampler, GeoVisualGraphSamplerV2


def test_soft_probing():
    print("\n" + "="*60)
    print("Testing SoftProbing Module")
    print("="*60)
    
    B, L, D = 4, 256, 768
    patch_tokens = torch.randn(B, L, D)
    
    sp = SoftProbing(embed_dim=D, alpha=0.5)
    
    modulated = sp(patch_tokens)
    assert modulated.shape == patch_tokens.shape, "Shape mismatch"
    print(f"✓ Input shape: {patch_tokens.shape}")
    print(f"✓ Output shape: {modulated.shape}")
    
    scale = (modulated / (patch_tokens + 1e-8)).mean().item()
    print(f"✓ Mean scale factor: {scale:.4f} (expected ~1.0)")
    
    loss = modulated.sum()
    loss.backward()
    assert sp.phi[0].weight.grad is not None, "No gradients"
    print("✓ Gradient flow verified")
    
    attn_weights = sp.get_attention_weights(patch_tokens)
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    
    print("\nTesting SoftProbingV2 (with channel attention)...")
    sp_v2 = SoftProbingV2(embed_dim=D, alpha=0.5, use_channel_attention=True)
    modulated_v2 = sp_v2(patch_tokens)
    assert modulated_v2.shape == patch_tokens.shape
    print(f"✓ SoftProbingV2 output shape: {modulated_v2.shape}")
    
    print("\n✅ SoftProbing validation PASSED")
    return True


def test_interact_head():
    print("\n" + "="*60)
    print("Testing InteractHead Module")
    print("="*60)
    
    B = 8
    D = 768
    descriptors = torch.randn(B, D)
    
    ih = InteractHead(embed_dim=D, num_heads=16, num_segments=4)
    
    enhanced = ih(descriptors)
    assert enhanced.shape == descriptors.shape, "Shape mismatch"
    print(f"✓ Input shape: {descriptors.shape}")
    print(f"✓ Output shape: {enhanced.shape}")
    
    original_var = descriptors.var().item()
    enhanced_var = enhanced.var().item()
    print(f"✓ Variance - Original: {original_var:.4f}, Enhanced: {enhanced_var:.4f}")
    
    enhanced_viz, attn = ih.forward_with_visualization(descriptors)
    print(f"✓ Visualization output shape: {attn.shape}")
    
    print("\nTesting InteractHeadLite (reduced parameters)...")
    ih_lite = InteractHeadLite(embed_dim=D, num_heads=8, num_segments=4)
    enhanced_lite = ih_lite(descriptors)
    assert enhanced_lite.shape == descriptors.shape
    print(f"✓ InteractHeadLite output shape: {enhanced_lite.shape}")
    
    num_params_original = sum(p.numel() for p in ih.parameters())
    num_params_lite = sum(p.numel() for p in ih_lite.parameters())
    print(f"✓ Original params: {num_params_original:,}, Lite params: {num_params_lite:,}")
    
    print("\n✅ InteractHead validation PASSED")
    return True


def test_geo_visual_sampler():
    print("\n" + "="*60)
    print("Testing GeoVisualGraph Sampler")
    print("="*60)
    
    N = 50
    D = 4096
    descriptors = torch.randn(N, D)
    geo_coords = torch.rand(N, 2) * 100
    
    sampler = GeoVisualGraphSampler(
        geo_threshold=25.0,
        affinity_threshold=-2.88e3,
        num_places=15,
        clique_size=4
    )
    
    W = sampler.compute_affinity_matrix(descriptors, geo_coords)
    assert W.shape == (N, N), "Affinity matrix shape mismatch"
    assert torch.allclose(W.diagonal(), torch.zeros(N)), "Diagonal should be 0"
    print(f"✓ Affinity matrix shape: {W.shape}")
    print(f"✓ Affinity range: [{W.min().item():.2f}, {W.max().item():.2f}]")
    
    seed_scores = sampler.compute_seed_scores(W)
    print(f"✓ Seed scores shape: {seed_scores.shape}")
    print(f"✓ Best seed index: {torch.argmax(seed_scores).item()}")
    
    clique = sampler.greedy_clique_expansion(W)
    assert len(clique) <= sampler.clique_size, "Clique too large"
    assert len(set(clique)) == len(clique), "Duplicate nodes"
    print(f"✓ Sampled clique size: {len(clique)}")
    print(f"✓ Clique indices: {clique}")
    
    edges = sampler.build_sparse_graph(W)
    print(f"✓ Sparse graph edges: {len(edges)}")
    
    print("\nTesting GeoVisualGraphSamplerV2...")
    sampler_v2 = GeoVisualGraphSamplerV2(
        geo_threshold=25.0,
        affinity_threshold=-2.88e3,
        num_places=15,
        clique_size=4,
        temperature=1.0
    )
    
    prob_samples = sampler_v2.probabilistic_sampling(W, num_samples=4)
    print(f"✓ Probabilistic samples: {prob_samples}")
    
    print("\n✅ GeoVisualGraph Sampler validation PASSED")
    return True


def test_module_integration():
    print("\n" + "="*60)
    print("Testing Module Integration")
    print("="*60)
    
    B, L, D = 4, 256, 768
    
    sp = SoftProbing(embed_dim=D, alpha=0.5)
    ih = InteractHead(embed_dim=D, num_heads=16, num_segments=4)
    
    patch_tokens = torch.randn(B, L, D)
    modulated = sp(patch_tokens)
    
    pooled = modulated.mean(dim=1)
    enhanced = ih(pooled)
    
    assert enhanced.shape == (B, D)
    print(f"✓ Integration test passed")
    print(f"  Input: {patch_tokens.shape} → SoftProbing → {modulated.shape}")
    print(f"  → Pool → {pooled.shape} → InteractHead → {enhanced.shape}")
    
    print("\n✅ Module Integration validation PASSED")
    return True


def test_sampling_strategies():
    print("\n" + "="*60)
    print("Testing Different Sampling Strategies")
    print("="*60)
    
    N = 100
    D = 4096
    np.random.seed(42)
    torch.manual_seed(42)
    
    descriptors = torch.randn(N, D)
    labels = torch.randint(0, 10, (N,))
    
    sampler = GeoVisualGraphSampler(clique_size=4)
    W = sampler.compute_affinity_matrix(descriptors)
    
    print("\n1. Greedy Clique Expansion:")
    greedy_clique = sampler.greedy_clique_expansion(W)
    print(f"   Sampled: {greedy_clique}")
    
    print("\n2. Probabilistic Sampling (different temperatures):")
    sampler_v2 = GeoVisualGraphSamplerV2(temperature=0.5)
    samples_t05 = sampler_v2.probabilistic_sampling(W, num_samples=4)
    print(f"   T=0.5: {samples_t05}")
    
    sampler_v2.temperature = 2.0
    samples_t2 = sampler_v2.probabilistic_sampling(W, num_samples=4)
    print(f"   T=2.0: {samples_t2}")
    
    print("\n✅ Sampling strategies validation PASSED")
    return True


def print_module_summary():
    print("\n" + "="*60)
    print("Module Parameter Count Summary")
    print("="*60)
    
    modules = {
        'SoftProbing': SoftProbing(embed_dim=768, alpha=0.5),
        'SoftProbingV2': SoftProbingV2(embed_dim=768, alpha=0.5),
        'InteractHead': InteractHead(embed_dim=768, num_heads=16, num_segments=4),
        'InteractHeadLite': InteractHeadLite(embed_dim=768, num_heads=8, num_segments=4),
    }
    
    print(f"\n{'Module':<20} {'Parameters':>15} {'Memory (MB)':>15}")
    print("-" * 55)
    
    for name, module in modules.items():
        num_params = sum(p.numel() for p in module.parameters())
        memory_mb = num_params * 4 / (1024 ** 2)
        print(f"{name:<20} {num_params:>15,} {memory_mb:>15.2f}")
    
    print("\nNote: GeoVisualGraphSampler has 0 parameters (sampling only)")
    
    baseline_dinov2_params = 86_000_000
    sage_additional = modules['InteractHead'].num_parameters()
    print(f"\nTotal SAGE additional params: ~{sage_additional:,}")
    print(f"DINOv2 backbone params: {baseline_dinov2_params:,}")
    print(f"Parameter increase: {sage_additional/baseline_dinov2_params*100:.2f}%")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  SAGE Module Validation Tests")
    print("#"*60)
    
    try:
        test_soft_probing()
        test_interact_head()
        test_geo_visual_sampler()
        test_module_integration()
        test_sampling_strategies()
        print_module_summary()
        
        print("\n" + "#"*60)
        print("#  ✅ ALL TESTS PASSED!")
        print("#"*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
