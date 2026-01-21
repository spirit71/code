import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from backbone.vision_transformer import vit_large
from SACA import SA_CA

class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fc = nn.Linear(1024,1024,bias=True)
        decoderlayer = SA_CA(d_model=1024, nhead=16, batch_first=True)   # Simplified Decoder Block
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)

        # learnable queries
        self.queries = nn.Parameter(torch.zeros(1, 64, 1024))
        nn.init.normal_(self.queries, std=1e-6)

        # linear projection for dimensionality adjustment
        self.channel_proj = nn.Linear(1024, 256)
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
       
        x = self.decoder(queries,x_cp)  #([288, 64, 768])
       
        x = self.channel_proj(x)        #([288, 64, 256])
      
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)        #([288, 256, 64])->([288, 256, 16])->([288, 256*16])=([288, 4096])
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)   #对最后一维（dim=-1）进行 L2 归一化。
      
        return x

def get_backbone(pretrained_foundation, foundation_model_path):

    backbone = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict, strict=False)
    return backbone



