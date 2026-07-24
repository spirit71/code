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
        
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x, return_intermediates=False):
        """运行一个 BoQ 块。

        x 的形状为 [B,T,C]。默认仍返回原项目的三元组；只有 QASSR 显式传入
        return_intermediates=True 时才返回带名字的字典，因而不影响训练旧路径。
        """
        B = x.size(0)
        # 空间 token 先互相交换上下文信息，输出仍为 [B,T,C]。
        x = self.encoder(x)
        
        # self.queries 是模型学习的 Q 个“信息收集器”，复制到 batch 中每张图。
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        # Q 个 query 从 T 个空间 token 中读取信息：
        # out=[B,Q,C]；attn=[B,Q,T]（PyTorch 默认已对多头求平均）。
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        if return_intermediates:
            return {
                "encoder_output": x,
                "query_output": out,
                "attention": attn.detach(),
            }
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x, return_local=False, return_intermediates=False):
        """把 backbone 特征图聚合为最终全局描述符。

        return_intermediates 是纯推理分析开关：关闭时返回值与原 BoQ 完全一致；
        打开时额外暴露 x0/x1/x2、attention 等中间量供 QASSR 使用。
        """
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        Hf, Wf = x.shape[-2:]
        # [B,C,H,W] -> [B,C,H*W] -> [B,H*W,C]，每个网格格子成为一个 token。
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        # x0 是进入第一个 Transformer Encoder 之前的空间 token。
        x0 = x
        
        outs = []
        attns = []
        encoder_outputs = []
        for i in range(len(self.boqs)):
            block = self.boqs[i](x, return_intermediates=return_intermediates)
            if return_intermediates:
                x, out, attn = block["encoder_output"], block["query_output"], block["attention"]
            else:
                x, out, attn = block
            # encoder_outputs[0] 命名为 x1，encoder_outputs[1] 命名为 x2。
            encoder_outputs.append(x)
            outs.append(out)
            attns.append(attn)

        # 当前模型有两层 BoQBlock，所以 x_last 与 x2 是同一个张量。
        x_last = x
        # 拼接每层的 query 输出，再压缩、展平并做 L2 归一化，得到全局描述符。
        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if return_intermediates:
            return {
                "global": out,
                "x0": x0,
                "encoder_outputs": encoder_outputs,
                "x_last": x_last,
                "block_outputs": outs,
                "attentions": attns,
                "last_attention": attns[-1] if attns else None,
                "spatial_shape": (Hf, Wf),
            }
        if return_local:
            return {
                "global": out,
                "local": x_last,
                "spatial_shape": (Hf, Wf),
                "attention": attns[-1] if attns else None,
            }
        return out, attns
