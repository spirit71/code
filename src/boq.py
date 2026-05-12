# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
class BoQBlock(torch.nn.Module):
    # Shared Query + Routed Delta Query Bank.

    # 原始 BoQ:
    #     q = shared_q

    # 本模块:
    #     q = shared_q + alpha * routed_delta_q

    # routed_delta_q = sum_m gate_m(x) * delta_bank_m

    # 这样设计的好处：
    # 1. shared_q 保留原始 BoQ 的通用 query；
    # 2. routed_delta_q 学不同域/场景下的 query 修正；
    # 3. alpha 初始化为 0，训练初期等价于原始 BoQ，更稳。
    def __init__(self, in_dim, num_queries, nheads=8,num_banks = 4,
        router_hidden_ratio=4,
        use_balance_loss=True,delta_init_scale: float = 0.02,router_temperature=0.7, # 0.7 0.5
        max_delta_scale=0.2,):
        super(BoQBlock, self).__init__()
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.num_banks = num_banks
        self.router_temperature = router_temperature
        self.max_delta_scale = max_delta_scale
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        # shared query: 对应原始 BoQ 的固定 learnable queries
        self.shared_queries = torch.nn.Parameter(
            torch.randn(1, num_queries, in_dim) * 0.02
        )
        # raw 参数，不直接使用；通过 tanh 映射到有限范围
        self.raw_delta_scale = torch.nn.Parameter(torch.zeros(1))
        # delta query banks: domain-specific / context-specific query correction
        # shape: [M, K, C]
        self.delta_query_banks = nn.Parameter(
            torch.randn(num_banks, num_queries, in_dim) * delta_init_scale
        )
        # alpha 控制 routed delta 的强度
        # 初始化为 0：开始时 q = shared_q，等价于原始 BoQ
        # self.delta_scale = nn.Parameter(torch.zeros(1))
        hidden_dim = max(in_dim // router_hidden_ratio, 64)

        # Image-level router: [B, C] -> [B, M]
        self.router = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, num_banks),
        )
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x) ##[512, 256, 512]  ->
        # 2. image-level context
        z = x.mean(dim=1)                    # [B, C]
        # 3. router 预测每张图像的 bank 权重
        gate_logits = self.router(z)             # [B, M]
        # temperature < 1 会让 softmax 更尖锐
        gate = torch.softmax(
            gate_logits / self.router_temperature,
            dim=-1,
        )

        shared_q = self.shared_queries.repeat(B, 1, 1)
        # 4. soft routing 得到当前图像专属 queries
        routed_delta_q = torch.einsum(
            "bm,mkc->bkc",
            gate,
            self.delta_query_banks,
        )  # [B, K, C]
        #添加残差保持稳定
        # 限制 delta 强度
        delta_scale = self.max_delta_scale * torch.tanh(self.raw_delta_scale)
        q = shared_q + delta_scale * routed_delta_q  # [B, K, C]

        # 5. query self-attention，保持原逻辑
        
        # q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
         # 7. balance loss，防止所有样本都走同一个 bank
        aux = {
            "gate": gate,
            "gate_logits": gate_logits.detach(),
            "delta_scale": delta_scale.detach(),
            "raw_delta_scale": self.raw_delta_scale.detach(),
        }

        return x, out, attn.detach(), aux
    


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32,use_domain_routing=True,num_query_banks=4,routing_type="delta",router_temperature=0.7,
        max_delta_scale=0.2,
        routing_layers="last",  # "all" or "last"
        ):
        super().__init__()
        self.use_domain_routing = use_domain_routing
        self.num_query_banks = num_query_banks
        self.routing_type = routing_type
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        # self.boqs = torch.nn.ModuleList([
        #     BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        nheads = in_dim // 64
        blocks = []

        for layer_idx in range(num_layers):
            use_routing_this_layer = False

            if use_domain_routing:
                if routing_layers == "all":
                    use_routing_this_layer = True
                elif routing_layers == "last":
                    use_routing_this_layer = layer_idx == num_layers - 1
                else:
                    raise ValueError(f"Unknown routing_layers: {routing_layers}")

            if use_routing_this_layer:
                blocks.append(
                    BoQBlock(
                        in_dim=in_dim,
                        num_queries=num_queries,
                        nheads=nheads,
                        num_banks=num_query_banks,
                        router_temperature=router_temperature,
                        max_delta_scale=max_delta_scale,
                    )
                )
            else:
                blocks.append(
                    BoQBlock(
                        in_dim=in_dim,
                        num_queries=num_queries,
                        nheads=nheads,
                    )
                )

        self.boqs = torch.nn.ModuleList(blocks)

        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        routing_aux = []
        # for i in range(len(self.boqs)):
        #     x, out, attn = self.boqs[i](x)
        #     outs.append(out)
        #     attns.append(attn)
        for block in self.boqs:
            if self.use_domain_routing:
                x, out, attn, aux = block(x)
                routing_aux.append(aux)
            else:
                x, out, attn = block(x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)# [B, L*K, C]
        out = self.fc(out.permute(0, 2, 1))# [B, C, row_dim]
        out = out.flatten(1) # [B, C*row_dim]
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if self.use_domain_routing:
            return out, attns, routing_aux
        return out, attns