"""
Geometry-Constrained Attention (借鉴 a²GC)
实现方案A：几何偏置直接加到 attention logits

核心思想：
1. 将patch坐标(x,y)归一化到[-1,1]，投影成几何嵌入g_xy
2. 每个query学习一个几何向量q_j^g
3. 计算几何兼容分数S_{j,i}^g = g_xy,i^T · q_j^g
4. 融合到attention logits: logit_{j,i} = (q_j^T · k_i) / sqrt(d) + λ_g · S_{j,i}^g
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class GeometryConstrainedAttention(nn.Module):
    """
    带几何约束的多头注意力机制
    
    将a²GC的几何兼容性思想移植到Transformer的Query↔Patch分配中：
    - 每个patch有归一化的坐标(x,y) ∈ [-1,1]²
    - 坐标通过MLP投影成几何嵌入g_xy
    - 每个query学习一个几何向量q_j^g
    - 几何兼容分数 = g_xy^T · q_j^g
    - 与内容相似度融合：logit = content_sim + λ_g · geometry_compat
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        batch_first: bool = False,
        geometry_dim: int = 64,  # 几何嵌入维度（可配置）
        lambda_g: float = 0.1,    # 几何约束权重（可配置）
        device=None,
        dtype=None,
    ):
        """
        Args:
            embed_dim: 特征维度
            num_heads: 注意力头数
            dropout: dropout概率
            batch_first: 是否batch维度在前
            geometry_dim: 几何嵌入维度（默认64，与a²GC类似）
            lambda_g: 几何约束权重λ_g（默认0.1）
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.lambda_g = lambda_g
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 标准注意力机制的QKV投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # ========== 几何约束相关组件 ==========
        # 1. 坐标归一化：输入是(x,y)坐标，输出是归一化到[-1,1]的坐标
        #    （坐标归一化在forward中动态计算，无需额外参数）
        
        # 2. 几何嵌入投影：将归一化坐标(x,y) ∈ [-1,1]² 投影到几何嵌入空间
        #    使用MLP: coord_xy -> geometry_dim
        self.geometry_embed = nn.Sequential(
            nn.Linear(2, geometry_dim, **factory_kwargs),  # (x,y) -> geometry_dim
            nn.LayerNorm(geometry_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(geometry_dim, geometry_dim, **factory_kwargs),  # 可选的第二层
        )
        # 初始化几何嵌入MLP，使用较小的初始化避免输出尺度过大
        for module in self.geometry_embed:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 3. Query的几何向量：每个query学习一个几何向量q_j^g
        #    注意：这里我们为每个query head分别学习几何向量，增强表达能力
        #    为了灵活性，我们使用一个可学习的投影层，将query特征映射到几何向量
        #    这样每个query可以根据其内容学习不同的几何偏好
        self.query_geometry_proj = nn.Linear(embed_dim, num_heads * geometry_dim, **factory_kwargs)
        # 初始化：使用Xavier初始化，确保几何向量的尺度合理
        nn.init.xavier_uniform_(self.query_geometry_proj.weight, gain=0.1)  # 使用较小的gain
        if self.query_geometry_proj.bias is not None:
            nn.init.zeros_(self.query_geometry_proj.bias)
        
        # 几何兼容分数的归一化因子，使其尺度与内容相似度匹配
        # 内容相似度经过 1/sqrt(head_dim) 缩放，几何兼容分数也应该有类似的缩放
        self.geometry_scale = (geometry_dim ** -0.5)  # 类似于attention的scale
        
    def _compute_patch_coordinates(
        self, 
        num_patches: int, 
        patch_grid_h: int, 
        patch_grid_w: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tensor:
        """
        计算patch的归一化坐标
        
        Args:
            num_patches: patch总数（通常等于patch_grid_h * patch_grid_w）
            patch_grid_h: patch网格高度
            patch_grid_w: patch网格宽度
            device: 设备
            dtype: 数据类型
            
        Returns:
            coords: [num_patches, 2] 归一化到[-1,1]的坐标
        """
        # 生成网格坐标
        # 使用0.5偏移，使坐标位于patch中心（与a²GC和RoPE一致）
        coords_h = torch.arange(0.5, patch_grid_h, device=device, dtype=dtype) / patch_grid_h  # [H]
        coords_w = torch.arange(0.5, patch_grid_w, device=device, dtype=dtype) / patch_grid_w  # [W]
        
        # 创建网格并展平
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), 
            dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        
        # 归一化到[-1, 1]范围（a²GC的做法）
        coords = 2.0 * coords - 1.0  # [HW, 2]
        
        return coords
    
    def forward(
        self,
        query: Tensor,           # [B, num_queries, embed_dim] 或 [num_queries, B, embed_dim]
        key: Tensor,            # [B, num_patches, embed_dim] 或 [num_patches, B, embed_dim]
        value: Tensor,          # [B, num_patches, embed_dim] 或 [num_patches, B, embed_dim]
        patch_grid_h: Optional[int] = None,  # patch网格高度（用于计算坐标）
        patch_grid_w: Optional[int] = None,  # patch网格宽度（用于计算坐标）
        patch_coords: Optional[Tensor] = None,  # 可选：直接提供归一化坐标 [num_patches, 2]
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        return_stats: bool = False,  # 是否返回统计信息用于监控
    ) -> Tuple[Tensor, Optional[Tensor], Optional[dict]]:
        """
        Forward pass with geometry-constrained attention
        
        Args:
            query: Query tokens
            key: Key tokens (patch tokens)
            value: Value tokens (patch tokens)
            patch_grid_h: Patch网格高度（如果提供patch_coords则忽略）
            patch_grid_w: Patch网格宽度（如果提供patch_coords则忽略）
            patch_coords: 归一化坐标 [num_patches, 2]，如果提供则直接使用
            attn_mask: Attention mask
            key_padding_mask: Key padding mask
            
        Returns:
            output: [B, num_queries, embed_dim] 或 [num_queries, B, embed_dim]
            attn_weights: Optional attention weights
        """
        if not self.batch_first:
            # 转换为batch_first格式便于处理
            query = query.transpose(0, 1)  # [B, num_queries, embed_dim]
            key = key.transpose(0, 1)      # [B, num_patches, embed_dim]
            value = value.transpose(0, 1)  # [B, num_patches, embed_dim]
        
        B, num_queries, _ = query.shape
        _, num_keys, _ = key.shape  # num_keys可能包含cls token + patch tokens
        
        # ========== 1. 标准注意力计算 ==========
        Q = self.q_proj(query)  # [B, num_queries, embed_dim]
        K = self.k_proj(key)    # [B, num_keys, embed_dim]
        V = self.v_proj(value)  # [B, num_keys, embed_dim]
        
        # 重塑为多头格式
        Q = Q.view(B, num_queries, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, num_queries, d]
        K = K.view(B, num_keys, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, num_keys, d]
        V = V.view(B, num_keys, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, num_keys, d]
        
        # 内容相似度：Q · K^T / sqrt(d)
        content_sim = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, num_queries, num_keys]
        attn_logits = content_sim
        
        # 统计信息（用于监控）
        stats = {}
        if return_stats:
            stats['content_sim_mean'] = content_sim.mean().item()
            stats['content_sim_std'] = content_sim.std().item()
            stats['content_sim_max'] = content_sim.max().item()
            stats['content_sim_min'] = content_sim.min().item()
        
        # ========== 2. 几何约束计算 ==========
        if patch_coords is None:
            # 需要从patch_grid计算坐标
            # 首先尝试从实际的key tokens数量推断patch tokens数量
            # 假设key中可能包含cls token，实际的patch tokens数量需要推断
            # 如果num_keys是平方数，假设没有cls token；否则假设有1个cls token
            sqrt_num_keys = int(num_keys ** 0.5)
            if sqrt_num_keys * sqrt_num_keys == num_keys:
                # num_keys是平方数，假设没有cls token
                num_patch_tokens = num_keys
                inferred_patch_grid_h = inferred_patch_grid_w = sqrt_num_keys
            else:
                # num_keys不是平方数，假设有1个cls token
                num_patch_tokens = num_keys - 1
                inferred_patch_grid_h = inferred_patch_grid_w = int(num_patch_tokens ** 0.5)
                if inferred_patch_grid_h * inferred_patch_grid_w != num_patch_tokens:
                    # 无法推断，跳过几何约束
                    geometry_bias = None
                    patch_coords = None
                else:
                    # 使用推断的网格大小
                    patch_coords = self._compute_patch_coordinates(
                        num_patch_tokens, inferred_patch_grid_h, inferred_patch_grid_w,
                        query.device, query.dtype
                    )
            
            # 如果提供了patch_grid_h和patch_grid_w，检查是否与推断的匹配
            if patch_coords is not None and patch_grid_h is not None and patch_grid_w is not None:
                grid_patch_count = patch_grid_h * patch_grid_w
                if grid_patch_count != num_patch_tokens:
                    # 提供的grid大小与实际的patch tokens数量不匹配
                    # 使用推断的网格大小重新计算（已经计算过了，无需重新计算）
                    pass
        
        if patch_coords is not None:
            # 2.1 计算patch的几何嵌入 g_xy
            # patch_coords: [num_patches, 2]
            num_patches = patch_coords.shape[0]
            g_xy = self.geometry_embed(patch_coords)  # [num_patches, geometry_dim]
            
            # 2.2 从query特征学习几何向量 q_j^g
            # 使用query的原始特征（在Q投影之前）来学习几何偏好
            # query: [B, num_queries, embed_dim]
            q_geometry_raw = self.query_geometry_proj(query)  # [B, num_queries, num_heads * geometry_dim]
            q_geometry = q_geometry_raw.view(B, num_queries, self.num_heads, -1)  # [B, num_queries, num_heads, geometry_dim]
            q_geometry = q_geometry.transpose(1, 2)  # [B, num_heads, num_queries, geometry_dim]
            
            # 2.3 计算几何兼容分数 S_{j,i}^g = g_xy,i^T · q_j^g
            # g_xy: [num_patches, geometry_dim]
            # q_geometry: [B, num_heads, num_queries, geometry_dim]
            # 需要: [B, num_heads, num_queries, num_patches]
            # 注意：添加归一化因子，使几何兼容分数的尺度与内容相似度匹配
            geometry_bias_patches = torch.matmul(
                q_geometry,  # [B, num_heads, num_queries, geometry_dim]
                g_xy.unsqueeze(0).unsqueeze(0).transpose(-2, -1)  # [1, 1, geometry_dim, num_patches]
            ) * self.geometry_scale  # [B, num_heads, num_queries, num_patches]
            # 应用归一化，使几何兼容分数的尺度与内容相似度 (Q·K^T / sqrt(d)) 匹配
            
            # 2.4 将几何偏置应用到对应的位置
            # 注意：如果key包含cls token（在开头），需要将几何偏置插入到正确位置
            # 假设patch_coords对应key中最后num_patches个位置（即patch tokens）
            if num_patches > num_keys:
                # patch_coords的数量大于key的tokens数量，说明patch_coords可能包含了额外的patches
                # 尝试推断key中实际的patch tokens数量
                # 如果num_keys是平方数，假设没有cls token；否则假设有1个cls token
                sqrt_num_keys = int(num_keys ** 0.5)
                if sqrt_num_keys * sqrt_num_keys == num_keys:
                    # key中没有cls token，使用前num_keys个patch的几何偏置
                    num_patch_tokens_in_key = num_keys
                    geometry_bias_patches_trimmed = geometry_bias_patches[:, :, :, :num_patch_tokens_in_key]
                    geometry_bias = geometry_bias_patches_trimmed
                elif (num_keys - 1) > 0:
                    # key中有1个cls token，使用前(num_keys-1)个patch的几何偏置
                    num_patch_tokens_in_key = num_keys - 1
                    if num_patch_tokens_in_key <= geometry_bias_patches.shape[-1]:
                        geometry_bias_patches_trimmed = geometry_bias_patches[:, :, :, :num_patch_tokens_in_key]
                        geometry_bias = torch.zeros(
                            B, self.num_heads, num_queries, num_keys,
                            device=geometry_bias_patches.device,
                            dtype=geometry_bias_patches.dtype
                        )
                        # 将几何偏置填充到最后的num_patch_tokens_in_key个位置（patch tokens）
                        geometry_bias[:, :, :, -num_patch_tokens_in_key:] = geometry_bias_patches_trimmed
                    else:
                        # 无法匹配，跳过几何约束
                        geometry_bias = None
                else:
                    # 无法匹配，跳过几何约束
                    geometry_bias = None
            elif num_keys > num_patches:
                # key包含cls token等，几何偏置只应用到patch tokens位置
                # 创建全零的几何偏置，然后填充到patch tokens位置
                geometry_bias = torch.zeros(
                    B, self.num_heads, num_queries, num_keys,
                    device=geometry_bias_patches.device,
                    dtype=geometry_bias_patches.dtype
                )
                # 将几何偏置填充到最后的num_patches个位置（patch tokens）
                geometry_bias[:, :, :, -num_patches:] = geometry_bias_patches
            else:
                # num_keys == num_patches，key只包含patch tokens
                geometry_bias = geometry_bias_patches
            
            # 2.5 融合到attention logits
            attn_logits = attn_logits + self.lambda_g * geometry_bias
            
            # 统计几何约束相关信息
            if return_stats:
                stats['geometry_bias_mean'] = geometry_bias.mean().item()
                stats['geometry_bias_std'] = geometry_bias.std().item()
                stats['geometry_bias_max'] = geometry_bias.max().item()
                stats['geometry_bias_min'] = geometry_bias.min().item()
                stats['geometry_scale'] = self.geometry_scale
                stats['lambda_g'] = self.lambda_g
                # 计算尺度比例
                if geometry_bias.abs().max() > 0:
                    stats['scale_ratio'] = (content_sim.abs().mean() / geometry_bias.abs().mean()).item()
                else:
                    stats['scale_ratio'] = float('inf')
        else:
            geometry_bias = None
            if return_stats:
                stats['geometry_bias_mean'] = 0.0
                stats['geometry_bias_std'] = 0.0
                stats['scale_ratio'] = float('inf')
        
        # ========== 3. 应用mask和softmax ==========
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        
        if key_padding_mask is not None:
            # key_padding_mask: [B, num_patches], True表示需要mask的位置
            attn_logits = attn_logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, num_queries, num_patches]
        attn_weights = self.dropout(attn_weights)
        
        # 统计attention权重信息
        if return_stats:
            stats['attn_weights_mean'] = attn_weights.mean().item()
            stats['attn_weights_std'] = attn_weights.std().item()
            stats['attn_weights_max'] = attn_weights.max().item()
            stats['attn_weights_min'] = attn_weights.min().item()
            # 计算attention权重的熵（衡量分布的均匀性）
            attn_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean()
            stats['attn_entropy'] = attn_entropy.item()
            # 保存attention权重用于可视化（只保存第一个batch，第一个head）
            stats['attn_weights_sample'] = attn_weights[0, 0].detach().cpu()  # [num_queries, num_keys]
        
        # ========== 4. 加权聚合value ==========
        output = torch.matmul(attn_weights, V)  # [B, H, num_queries, d]
        output = output.transpose(1, 2).contiguous()  # [B, num_queries, H, d]
        output = output.view(B, num_queries, self.embed_dim)  # [B, num_queries, embed_dim]
        output = self.out_proj(output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if need_weights:
            # 返回平均后的attention weights
            attn_weights_avg = attn_weights.mean(dim=1)  # [B, num_queries, num_patches]
            if return_stats:
                return output, attn_weights_avg, stats
            else:
                return output, attn_weights_avg
        else:
            if return_stats:
                return output, None, stats
            else:
                return output, None
