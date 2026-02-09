import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from backbone.vision_transformer import vit_base
from SACA import SA_CA
from backbone.CustomTransformerDecoder import CustomTransformerDecoder

class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fc = nn.Linear(768,768,bias=True)
        # 使用几何约束的decoder layer（默认启用）
        decoderlayer = SA_CA(
            d_model=768, 
            nhead=16, 
            batch_first=True,
            use_geometry_constraint=True,  # 启用几何约束
            geometry_dim=64,               # 几何嵌入维度
            lambda_g=0.05,                 # 几何约束权重（降低到0.05，因为已添加归一化）
        )
        # 使用自定义TransformerDecoder以支持传递几何约束参数
        self.decoder = CustomTransformerDecoder(decoder_layer=decoderlayer, num_layers=2)

        # learnable queries
        self.queries = nn.Parameter(torch.zeros(1, 64, 768))
        nn.init.normal_(self.queries, std=1e-6)

        # linear projection for dimensionality adjustment
        self.channel_proj = nn.Linear(768, 256)
        self.row_proj = nn.Linear(64, 16)

    def forward(self, x):
        x = self.backbone(x)    #x.keys()
                                #dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'x_norm', 'masks'])
       
        B,P,D = x["x_norm"].shape       #x["x_norm"].shape=torch.Size([288, 257, 768])
                                        #B=288
       
        queries = self.queries.expand(B,-1,-1)      #self.queries.shape=([1, 64, 768])
                                                  #queries.shape=([288, 64, 768])
       
        x_c = x["x_norm_clstoken"]                 #x_c =x["x_norm_clstoken"].shape=([288, 1, 768])
        
        x_p = x["x_norm_patchtokens"]           #x_p =x["x_norm_patchtokens"].shape=([288, 256, 768])
       
        x_cp = torch.cat([x_c,x_p],dim=1)       #x_cp.shape=([288, 257, 768])
        
        x_cp = self.fc(x_cp)                    #x_cp.shape=([288, 257, 768])
       
        # ========== 计算patch坐标用于几何约束 ==========
        # 从backbone获取patch网格信息
        # 注意：x_p是patch tokens，数量为256，通常对应16x16的网格
        num_patches = x_p.shape[1]  # 256
        # 尝试从backbone获取patch网格大小，如果无法获取则推断
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'patches_resolution'):
            backbone_patch_grid_h, backbone_patch_grid_w = self.backbone.patch_embed.patches_resolution
            # 检查backbone的patch网格大小是否与实际patch tokens数量匹配
            if backbone_patch_grid_h * backbone_patch_grid_w == num_patches:
                # 匹配，使用backbone的网格大小
                patch_grid_h, patch_grid_w = backbone_patch_grid_h, backbone_patch_grid_w
            else:
                # 不匹配，使用实际的patch tokens数量推断网格大小
                patch_grid_h = patch_grid_w = int(num_patches ** 0.5)
                if patch_grid_h * patch_grid_w != num_patches:
                    # 无法推断，设为None（几何约束将跳过）
                    patch_grid_h = patch_grid_w = None
        else:
            # 推断：假设是正方形网格
            patch_grid_h = patch_grid_w = int(num_patches ** 0.5)
            if patch_grid_h * patch_grid_w != num_patches:
                # 无法推断，设为None（几何约束将跳过）
                patch_grid_h = patch_grid_w = None
        
        # 计算归一化坐标（仅对patch tokens，不包括cls token）
        patch_coords = None
        if patch_grid_h is not None and patch_grid_w is not None:
            # 确保patch_grid_h * patch_grid_w == num_patches
            assert patch_grid_h * patch_grid_w == num_patches, \
                f"patch_grid_h * patch_grid_w ({patch_grid_h * patch_grid_w}) != num_patches ({num_patches})"
            # 生成归一化坐标
            coords_h = torch.arange(0.5, patch_grid_h, device=x_p.device, dtype=x_p.dtype) / patch_grid_h
            coords_w = torch.arange(0.5, patch_grid_w, device=x_p.device, dtype=x_p.dtype) / patch_grid_w
            coords = torch.stack(
                torch.meshgrid(coords_h, coords_w, indexing="ij"), 
                dim=-1
            )  # [H, W, 2]
            patch_coords = coords.flatten(0, 1)  # [HW, 2]
            patch_coords = 2.0 * patch_coords - 1.0  # 归一化到[-1, 1]
        
        # 传递几何约束参数给decoder
        decoder_result = self.decoder(
            queries, x_cp,
            patch_grid_h=patch_grid_h,
            patch_grid_w=patch_grid_w,
            patch_coords=patch_coords,
            return_stats=getattr(self, '_return_stats', False),
        )
        if isinstance(decoder_result, tuple):
            x, decoder_stats = decoder_result  #([288, 64, 768])
            if hasattr(self, '_decoder_stats'):
                self._decoder_stats = decoder_stats
        else:
            x = decoder_result
       
        x = self.channel_proj(x)        #([288, 64, 256])
      
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)        #([288, 256, 64])->([288, 256, 16])->([288, 256*16])=([288, 4096])
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)   #对最后一维（dim=-1）进行 L2 归一化。
      
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



