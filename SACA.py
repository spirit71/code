import copy
from typing import Optional, Any, Union, Callable, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module,MultiheadAttention,LayerNorm,Dropout

# 导入几何约束注意力
from backbone.GeometryConstrainedAttention import GeometryConstrainedAttention

class SA_CA(Module):
    #batch_first控制输入张量的维度顺序，决定 batch_size是否放在第0维。
    # 如果 batch_first=False，输入形状为 [seq_len, batch_size, dim]
    #norm_first （归一化顺序）控制 LayerNorm 是在注意力计算前（Pre-LN）还是后（Post-LN）执行。
    #如果 norm_first=True：先 LayerNorm，再计算注意力（Pre-LN）。
    __constants__ = ['batch_first', 'norm_first']           

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 use_geometry_constraint: bool = True,  # 是否使用几何约束
                 geometry_dim: int = 64,  # 几何嵌入维度
                 lambda_g: float = 0.1,   # 几何约束权重
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        
        # 根据use_geometry_constraint选择使用标准attention或几何约束attention
        self.use_geometry_constraint = use_geometry_constraint
        if use_geometry_constraint:
            self.multihead_attn = GeometryConstrainedAttention(
                embed_dim=d_model, 
                num_heads=nhead, 
                dropout=dropout, 
                batch_first=batch_first,
                geometry_dim=geometry_dim,
                lambda_g=lambda_g,
                **factory_kwargs
            )
        else:
            self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                     **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,        
        memory: Tensor,     #memory ([288, 257, 768])
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        # 几何约束相关参数
        patch_grid_h: Optional[int] = None,  # patch网格高度
        patch_grid_w: Optional[int] = None,  # patch网格宽度
        patch_coords: Optional[Tensor] = None,  # 归一化坐标 [num_patches, 2]
        return_stats: bool = False,  # 是否返回统计信息
    ) -> Tuple[Tensor, Optional[dict]]:

        x = tgt       #tgt torch.Size([288, 64, 768])
        stats = None
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            mha_output, stats = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal,
                                    patch_grid_h=patch_grid_h, patch_grid_w=patch_grid_w, patch_coords=patch_coords,
                                    return_stats=return_stats)
            x = x + mha_output
        else:
            #[batch_size, seq_len, dim]
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            mha_output, stats = self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal,
                                                patch_grid_h=patch_grid_h, patch_grid_w=patch_grid_w, patch_coords=patch_coords,
                                                return_stats=return_stats)
            x = self.norm2(x + mha_output)

        return x, stats

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,
                   patch_grid_h: Optional[int] = None,
                   patch_grid_w: Optional[int] = None,
                   patch_coords: Optional[Tensor] = None,
                   return_stats: bool = False) -> Tuple[Tensor, Optional[dict]]:
        if self.use_geometry_constraint:
            # 使用几何约束attention
            result = self.multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    patch_grid_h=patch_grid_h,
                                    patch_grid_w=patch_grid_w,
                                    patch_coords=patch_coords,
                                    need_weights=False,
                                    return_stats=return_stats)
            if return_stats:
                x, _, stats = result
            else:
                x = result[0]
                stats = None
        else:
            # 使用标准attention
            x = self.multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    is_causal=is_causal,
                                    need_weights=False)[0]
            stats = None
        return self.dropout2(x), stats

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
