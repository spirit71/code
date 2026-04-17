import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from backbone.vision_transformer import vit_base
# from backbone.vision_transformer_dinov3 import vit_base
from SACA import SA_CA

class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation = False, foundation_model_path = None, 
                 use_decoder=True, num_queries=64):
        super().__init__()
        self.use_decoder = use_decoder
        self.num_queries = num_queries
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fc = nn.Linear(768, 768, bias=True)

        if use_decoder:
            decoderlayer = SA_CA(d_model=768, nhead=16, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)
            self.queries = nn.Parameter(torch.zeros(1, num_queries, 768))
            nn.init.normal_(self.queries, std=1e-6)
            # 使用 queries 数量作为 row_proj 输入
            self.row_proj = nn.Linear(num_queries, 16)
        else:
            self.decoder = None
            self.queries = None
            # 不使用 row_proj，改用自适应池化
            self.row_proj = None

        self.channel_proj = nn.Linear(768, 256)

    def forward(self, x):
        x = self.backbone(x)
        
        x_c = x["x_norm_clstoken"]           # [B, 1, 768]
        x_p = x["x_norm_patchtokens"]        # [B, N, 768], N 可能是 256 或 529
        x_cp = torch.cat([x_c, x_p], dim=1)  # [B, N+1, 768]
        x_cp = self.fc(x_cp)
        
        if self.use_decoder:
            # 使用 decoder 聚合（固定输出维度）
            B = x_cp.shape[0]
            queries = self.queries.expand(B, -1, -1)
            x = self.decoder(queries, x_cp)        # [B, num_queries, 768]
            x = self.channel_proj(x)               # [B, num_queries, 256]
            x = self.row_proj(x.permute(0, 2, 1))  # [B, 256, 16]
        else:
            # 不使用 decoder，仅使用 patch tokens
            x = x_p                                # [B, N, 768]
            x = self.channel_proj(x)               # [B, N, 256]
            
            # 使用自适应池化替代 row_proj，处理不同 N
            x = x.permute(0, 2, 1)                 # [B, 256, N]
            # 自适应平均池化到固定大小 16
            x = F.adaptive_avg_pool1d(x, 16)       # [B, 256, 16]
        
        x = x.flatten(1)                    # [B, 4096]
        x = F.normalize(x, p=2, dim=-1)
        
        return x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict, strict=False)
    return backbone

    



