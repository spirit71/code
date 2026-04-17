import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from backbone.vision_transformer import vit_base
from SACA import SA_CA
from soft_probing import SoftProbing
from interact_head import InteractHead


class VPRNetSAGE(nn.Module):
    def __init__(self, pretrained_foundation=False, foundation_model_path=None,
                 use_soft_probing=True, use_interact_head=True,
                 soft_probing_alpha=0.5, interact_num_segments=4,
                 interact_num_heads=16, interact_ffn_dim=1024,
                 interact_num_layers=2):
        super().__init__()
        
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        
        self.use_soft_probing = use_soft_probing
        self.use_interact_head = use_interact_head
        
        if use_soft_probing:
            self.soft_probing = SoftProbing(embed_dim=768, alpha=soft_probing_alpha)
        
        if use_interact_head:
            self.interact_head = InteractHead(
                embed_dim=768,
                num_heads=interact_num_heads,
                num_segments=interact_num_segments,
                ffn_dim=interact_ffn_dim,
                num_layers=interact_num_layers
            )
        
        self.fc = nn.Linear(768, 768, bias=True)
        decoderlayer = SA_CA(d_model=768, nhead=16, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)
        
        self.queries = nn.Parameter(torch.zeros(1, 64, 768))
        nn.init.normal_(self.queries, std=1e-6)
        
        self.channel_proj = nn.Linear(768, 256)
        self.row_proj = nn.Linear(64, 16)
    
    def forward(self, x):
        x = self.backbone(x)
        
        B, P, D = x["x_norm"].shape
        
        x_c = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"]
        
        if self.use_soft_probing:
            x_p = self.soft_probing(x_p)
        
        x_cp = torch.cat([x_c, x_p], dim=1)
        
        x_cp = self.fc(x_cp)
        
        queries = self.queries.expand(B, -1, -1)
        x = self.decoder(queries, x_cp)
        
        x = self.channel_proj(x)
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x
    
    def forward_with_interact(self, x, batch_descriptors=None):
        """
        Forward pass with InteractHead for cross-image attention during training
        
        Args:
            x: input images
            batch_descriptors: other descriptors in batch for cross-attention
        """
        x = self.backbone(x)
        
        B, P, D = x["x_norm"].shape
        
        x_c = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"]
        
        if self.use_soft_probing:
            x_p = self.soft_probing(x_p)
        
        x_cp = torch.cat([x_c, x_p], dim=1)
        x_cp = self.fc(x_cp)
        
        if self.use_interact_head and batch_descriptors is not None:
            combined = torch.cat([x_cp, batch_descriptors], dim=1)
            x_cp = self.interact_head.encoder(combined)[:, :x_cp.size(1), :]
        
        queries = self.queries.expand(B, -1, -1)
        x = self.decoder(queries, x_cp)
        
        x = self.channel_proj(x)
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        return x
    
    def get_attention_maps(self, x):
        """
        获取注意力图用于可视化
        
        Returns:
            soft_probing_weights: 软探针注意力权重
        """
        x = self.backbone(x)
        x_p = x["x_norm_patchtokens"]
        
        if self.use_soft_probing:
            sp_weights = self.soft_probing.get_attention_weights(x_p)
        else:
            sp_weights = None
        
        return {
            'soft_probing_weights': sp_weights,
            'patch_tokens': x_p
        }


def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone


def create_sage_model(config):
    """
    工厂函数：创建SAGE增强版EDTformer
    
    Args:
        config: 配置字典
    """
    model = VPRNetSAGE(
        pretrained_foundation=config.get('pretrained_foundation', True),
        foundation_model_path=config.get('foundation_model_path'),
        use_soft_probing=config.get('use_soft_probing', True),
        use_interact_head=config.get('use_interact_head', True),
        soft_probing_alpha=config.get('soft_probing_alpha', 0.5),
        interact_num_segments=config.get('interact_num_segments', 4),
        interact_num_heads=config.get('interact_num_heads', 16),
        interact_ffn_dim=config.get('interact_ffn_dim', 1024),
        interact_num_layers=config.get('interact_num_layers', 2)
    )
    return model
