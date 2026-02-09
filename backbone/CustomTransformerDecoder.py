"""
自定义TransformerDecoder，支持传递几何约束参数
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class CustomTransformerDecoder(nn.Module):
    """
    自定义TransformerDecoder，支持向decoder layer传递额外参数（如几何约束参数）
    """
    
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        # 几何约束相关参数
        patch_grid_h: Optional[int] = None,
        patch_grid_w: Optional[int] = None,
        patch_coords: Optional[Tensor] = None,
        return_stats: bool = False,
    ):
        """
        Args:
            tgt: [B, num_queries, embed_dim]
            memory: [B, num_patches, embed_dim]
            其他参数同标准TransformerDecoder
        """
        output = tgt
        all_stats = []
        
        for layer_idx, layer in enumerate(self.layers):
            result = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                patch_grid_h=patch_grid_h,
                patch_grid_w=patch_grid_w,
                patch_coords=patch_coords,
                return_stats=return_stats,
            )
            if return_stats:
                output, stats = result
                if stats is not None:
                    # 添加layer索引到统计信息
                    stats['layer_idx'] = layer_idx
                    all_stats.append(stats)
            else:
                output = result[0] if isinstance(result, tuple) else result
        
        if self.norm is not None:
            output = self.norm(output)
        
        if return_stats:
            return output, all_stats
        else:
            return output
