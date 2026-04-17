import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftProbing(nn.Module):
    """
    Soft Probing Module from SAGE paper
    Enhances discriminative local patches through residual weighting
    
    Paper: SAGE - Spatial-visual Adaptive Graph Exploration for Efficient VPR
    Section 3.2: Soft Probing
    
    原理：
    1. 计算每个descriptor的ℓ2响应: si = ||Xi||₂ + ε
    2. 通过MLP预测残差系数: βi = α · σ(φ(si))
    3. 残差调制: X̃i = Xi + βi·Xi = (1+βi)·Xi
    
    作用：自适应放大判别性局部区域，增强聚合对关键区域的敏感性
    """
    def __init__(self, embed_dim=768, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.embed_dim = embed_dim
        
        self.phi = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, L, D] - patch embeddings from backbone
        Returns:
            modulated_tokens: [B, L, D] - residual-weighted tokens
        """
        B, L, D = patch_tokens.shape
        
        si = torch.norm(patch_tokens, p=2, dim=-1, keepdim=True)
        si = si + 1e-6
        
        beta = self.phi(si)
        beta = torch.sigmoid(beta) * self.alpha
        
        modulated_tokens = patch_tokens * (1 + beta)
        
        return modulated_tokens
    
    def get_attention_weights(self, patch_tokens):
        """
        获取注意力权重用于可视化分析
        """
        si = torch.norm(patch_tokens, p=2, dim=-1, keepdim=True) + 1e-6
        beta = self.phi(si)
        beta = torch.sigmoid(beta) * self.alpha
        return (1 + beta).squeeze(-1)


class SoftProbingV2(nn.Module):
    """
    改进版SoftProbing，支持通道级和空间级双重加权
    """
    def __init__(self, embed_dim=768, alpha=0.5, use_channel_attention=True):
        super().__init__()
        self.alpha = alpha
        self.use_channel_attention = use_channel_attention
        
        self.spatial_phi = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim),
                nn.Sigmoid()
            )
    
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, L, D]
        Returns:
            modulated_tokens: [B, L, D]
        """
        B, L, D = patch_tokens.shape
        
        si = torch.norm(patch_tokens, p=2, dim=-1, keepdim=True) + 1e-6
        beta = self.spatial_phi(si)
        beta = torch.sigmoid(beta) * self.alpha
        
        modulated = patch_tokens * (1 + beta)
        
        if self.use_channel_attention:
            channel_weights = self.channel_attention(modulated)
            modulated = modulated * channel_weights
        
        return modulated
