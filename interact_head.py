import torch
import torch.nn as nn
from torch import Tensor


class InteractHead(nn.Module):
    """
    InteractHead Module from SAGE paper
    Cross-image attention through segment-wise processing
    
    Paper: SAGE - Spatial-visual Adaptive Graph Exploration for Efficient VPR
    Section 3.3: Online Graph Creation
    
    原理：
    1. 将descriptor分割为S个固定长度段
    2. 同位置段组成序列，通过Transformer encoder处理
    3. 捕获跨视图的一致性关联
    
    作用：增强特征鲁棒性，提高跨视图匹配能力
    """
    def __init__(self, embed_dim=768, num_heads=16, num_segments=4, 
                 ffn_dim=1024, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_segments = num_segments
        self.seg_dim = embed_dim // num_segments
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, D] - global descriptors
        Returns:
            enhanced_descriptors: [B, D] - cross-image enhanced descriptors
        """
        B = descriptors.shape[0]
        
        segments = descriptors.view(B, self.num_segments, self.seg_dim)
        
        enhanced = self.encoder(segments)
        enhanced = self.norm(enhanced)
        
        enhanced_descriptors = enhanced.reshape(B, -1)
        
        return enhanced_descriptors
    
    def forward_with_visualization(self, descriptors):
        """
        返回增强后的descriptors和注意力权重
        """
        B = descriptors.shape[0]
        
        segments = descriptors.view(B, self.num_segments, self.seg_dim)
        enhanced = self.encoder(segments)
        
        attention_weights = torch.softmax(enhanced, dim=1)
        
        enhanced_descriptors = enhanced.reshape(B, -1)
        enhanced_descriptors = self.norm(enhanced_descriptors)
        
        return enhanced_descriptors, attention_weights


class InteractHeadLite(nn.Module):
    """
    轻量版InteractHead，减少参数量的简化版本
    适用于资源受限场景
    """
    def __init__(self, embed_dim=768, num_heads=8, num_segments=4):
        super().__init__()
        self.num_segments = num_segments
        self.seg_dim = embed_dim // num_segments
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.seg_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.seg_dim, self.seg_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.seg_dim * 2, self.seg_dim)
        )
        
        self.norm1 = nn.LayerNorm(self.seg_dim)
        self.norm2 = nn.LayerNorm(self.seg_dim)
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, D]
        Returns:
            enhanced_descriptors: [B, D]
        """
        B = descriptors.shape[0]
        
        segments = descriptors.view(B, self.num_segments, self.seg_dim)
        
        attn_out, _ = self.self_attention(segments, segments, segments)
        segments = self.norm1(segments + attn_out)
        
        ffn_out = self.ffn(segments)
        segments = self.norm2(segments + ffn_out)
        
        enhanced_descriptors = segments.reshape(B, -1)
        
        return enhanced_descriptors
