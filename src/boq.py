# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.) # in_dim=512, nheads=8, dim_feedforward=4*512=2048 batch_first=True, dropout=0.
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim)) # 1, 64, 512 num_queries=64, in_dim=512
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True) # in_dim=512, nheads=8 batch_first=True
        self.norm_q = torch.nn.LayerNorm(in_dim) # 512
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True) # in_dim=512, nheads=8 batch_first=True
        self.norm_out = torch.nn.LayerNorm(in_dim) # 512
        

    def forward(self, x):
        print(f"x.shape: {x.shape}") #[512, 256, 512]
        B = x.size(0)
        x = self.encoder(x)
        q = self.queries.repeat(B, 1, 1)  #[512, 64, 512]
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######  
        out, attn = self.cross_attn(q, x, x)        #attn [512, 64, 256]
        out = self.norm_out(out) #[512, 64, 512]
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1) # 12288, 512 in_channels=768, proj_channels=512
        self.norm_input = torch.nn.LayerNorm(proj_channels) # 512
        
        in_dim = proj_channels # 512
        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)]) # 512//64 = 8 nheads=8 num_layers=2
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim) # 2*64 = 128, row_dim=16
        
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        print(f"x.shape: {x.shape}") #[512, 768, 16, 16]
        x = self.proj_c(x)  #[512, 512, 16, 16]
        x = x.flatten(2).permute(0, 2, 1) #[512, 256, 512]
        x = self.norm_input(x)  #[512, 256, 512]
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)
        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns