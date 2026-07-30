# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch

from src.query_reliability import QueryReliabilityHead, apply_reliability_gate, entropy_reliability

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(
        self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32,
        use_query_reliability=False, reliability_mode="learned", reliability_apply_layer="last",
        reliability_hidden_dim=128, reliability_dropout=0.0, reliability_use_entropy=True,
        reliability_use_max_attention=True, reliability_use_query_norm=True,
        reliability_gate_mode="centered_residual", reliability_residual_scale=0.5,
        reliability_min_weight=0.5, reliability_max_weight=1.5,
    ):
        super().__init__()
        if reliability_apply_layer != "last":
            raise ValueError("QRL-BoQ v1 only supports reliability_apply_layer='last'")
        if reliability_mode not in {"learned", "entropy"}:
            raise ValueError("reliability_mode must be 'learned' or 'entropy'")
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        self.use_query_reliability = use_query_reliability
        self.reliability_mode = reliability_mode
        self.reliability_gate_mode = reliability_gate_mode
        self.reliability_residual_scale = reliability_residual_scale
        self.reliability_min_weight = reliability_min_weight
        self.reliability_max_weight = reliability_max_weight
        self.reliability_head = None
        if use_query_reliability and reliability_mode == "learned":
            self.reliability_head = QueryReliabilityHead(
                proj_channels, reliability_hidden_dim, reliability_use_entropy,
                reliability_use_max_attention, reliability_use_query_norm, reliability_dropout,
            )
        
    def forward(self, x, return_aux=False):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        raw_outputs = torch.stack(outs, dim=1)
        weighted_outputs = None
        reliability = None
        gate_weights = None
        projection_outputs = outs
        if self.use_query_reliability:
            # ---------------- QRL-BoQ 相对原始 BoQ 新增的路径 ----------------
            # 原始 BoQ 到这里已经得到两层 query 输出：
            #   outs[0] = O1: [B,64,512]
            #   outs[1] = O2: [B,64,512]
            #
            # QRL-BoQ v1 只分析最后一层 O2。原因：
            # 1. O2 经过更深的图像 token 编码，更接近最终 descriptor；
            # 2. 只改最后一层属于低扰动扩展，O1 保留为原始 BoQ 安全支路；
            # 3. 两层 query 是独立参数，不能假设同编号 query 具有相同语义。
            reliability = (
                # learned 模式输入：
                #   O2 [B,64,512] + 最后一层 attention [B,64,N]
                # 输出：
                #   reliability [B,64]
                self.reliability_head(outs[-1], attns[-1])
                if self.reliability_mode == "learned"
                # entropy 模式是不训练预测头的消融：
                # reliability = 1 - normalized_attention_entropy。
                else entropy_reliability(attns[-1])
            )

            # 把 reliability [B,64] 转成 gate_weights [B,64]，
            # 再广播乘到 O2 的 512 个通道，得到 weighted_last [B,64,512]。
            # centered-residual 初始权重为 1，因此初始输出与原始 BoQ 一致。
            weighted_last, gate_weights = apply_reliability_gate(
                outs[-1], reliability, self.reliability_gate_mode,
                self.reliability_residual_scale, self.reliability_min_weight,
                self.reliability_max_weight,
            )

            # O1 不变，只用加权后的 O2 替换最后一层输出：
            #   原始 BoQ 投影输入：[O1, O2]
            #   QRL-BoQ 投影输入：[O1, weighted_O2]
            projection_outputs = [*outs[:-1], weighted_last]

            # 额外保留按层堆叠后的加权输出，供训练辅助损失和离线诊断使用。
            # 它不引入新的 descriptor 计算，只是 return_aux=True 时返回中间量。
            weighted_outputs = raw_outputs.clone()
            weighted_outputs[:, -1] = weighted_last

        # 从这里开始重新回到原始 BoQ 聚合路径。
        # 两层拼接：[B,64,512] + [B,64,512] -> [B,128,512]
        out = torch.cat(projection_outputs, dim=1)

        # [B,128,512] -> [B,512,128]，然后对 query/layer 这一维做线性压缩：
        # fc: 128 -> row_dim=16，输出 [B,512,16]。
        out = self.fc(out.permute(0, 2, 1))

        # [B,512,16] -> [B,8192]，最后 L2 归一化得到全局地点 descriptor。
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if return_aux:
            return out, {
                "query_outputs_raw": raw_outputs,
                "query_outputs_weighted": weighted_outputs,
                "reliability_scores": reliability,
                "gate_weights": gate_weights,
                "last_attention": attns[-1] if attns else None,
            }
        return out, attns
