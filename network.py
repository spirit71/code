import logging
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from backbone.vision_transformer import vit_base
from SACA import SA_CA

class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)

        # feature dimensions are fixed by backbone (DINOv2-b / ViT-B/14)
        d_model = 768
        self.num_experts = 4  # K = 4 domain/routing experts for query bank

        self.fc = nn.Linear(768,768,bias=True)
        decoderlayer = SA_CA(d_model=d_model, nhead=16, batch_first=True)   # Simplified Decoder Block
        self.decoder = nn.TransformerDecoder(decoder_layer=decoderlayer, num_layers=2)

        # -----------------------------
        # Domain-Routed Query Bank
        # -----------------------------
        # multi-expert learnable queries: [K, Qn, C]
        self.query_bank = nn.Parameter(torch.zeros(self.num_experts, 64, d_model))
        nn.init.normal_(self.query_bank, std=1e-6)

        # lightweight gating network, takes global (cls) feature -> routing scores over experts
        self.gating = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_experts)
        )

        # linear projection for dimensionality adjustment
        self.channel_proj = nn.Linear(d_model, 256)
        self.row_proj = nn.Linear(64, 16)

    def forward(self, x, return_gates: bool = False):
        x = self.backbone(x)    #x.keys()
                                #dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'x_norm', 'masks'])
       
        B,P,D = x["x_norm"].shape       #x["x_norm"].shape=torch.Size([288, 257, 768])
                                        #B=288
       
        # -----------------------------
        # Domain routing over query experts
        # -----------------------------
        # use cls token as global descriptor for gating
        x_c = x["x_norm_clstoken"]                 # [B, 1, D]
        gating_input = x_c.squeeze(1)              # [B, D]
        gating_logits = self.gating(gating_input)  # [B, K]
        gating_scores = torch.softmax(gating_logits, dim=-1)  # [B, K]

        # query_bank: [K, Qn, D] -> [1, K, Qn, D]
        Q = self.query_bank.unsqueeze(0)
        # gating_scores: [B, K] -> [B, K, 1, 1]
        gates = gating_scores.unsqueeze(-1).unsqueeze(-1)
        # weighted sum over experts -> [B, Qn, D]
        queries = (gates * Q).sum(dim=1)
       
        x_p = x["x_norm_patchtokens"]           #x_p =x["x_norm_patchtokens"].shape=([288, 256, 768])
       
        x_cp = torch.cat([x_c,x_p],dim=1)       #x_cp.shape=([288, 257, 768])
        
        x_cp = self.fc(x_cp)                    #x_cp.shape=([288, 257, 768])
       
        x = self.decoder(queries,x_cp)  #([288, 64, 768])
       
        x = self.channel_proj(x)        #([288, 64, 256])
      
        x = self.row_proj(x.permute(0, 2, 1)).flatten(1)        #([288, 256, 64])->([288, 256, 16])->([288, 256*16])=([288, 4096])
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)   #对最后一维（dim=-1）进行 L2 归一化。
      
        if return_gates:
            return x, gating_scores
        return x

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path, weights_only=False)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict, strict=False)
    return backbone

