import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F


from backbone.vision_transformer_dinov3 import vit_base
from SACA import SA_CA

class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fc = nn.Linear(768,768,bias=True)
        decoderlayer = SA_CA(d_model=768, nhead=16, batch_first=True)   # Simplified Decoder Block
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)

        # learnable queries
        self.queries = nn.Parameter(torch.zeros(1, 64, 768))
        # self.queries = nn.Parameter(torch.zeros(1, 768))
        nn.init.normal_(self.queries, std=1e-6)

        # linear projection for dimensionality adjustment
        self.channel_proj = nn.Linear(768, 256)
        self.row_proj = nn.Linear(256, 4096)

    def forward(self, x):
        x = self.backbone(x)        #[288, 768])
        B=x.shape[0]                
        
        queries = self.queries.expand(B,-1)     #([288, 768])
        
        
        x_cp = self.fc(x)       #([288, 768])
        x = self.decoder(queries,x_cp)
        x = self.channel_proj(x)
        
        x = self.row_proj(x).flatten(1)
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=16,img_size=592,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict, strict=False)
    return backbone


