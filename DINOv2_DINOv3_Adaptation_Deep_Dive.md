# DINOv2 与 DINOv3 适配详解（从小白到进阶）

## 0. 这份文档是给谁看的
这份文档面向第一次接触 ViT / DINO / BoQ 的同学。目标是回答三个问题：

1. `DINOv2` 和 `DINOv3` 在这个项目里到底差在哪。
2. 为什么我们不能“直接把名字改成 dinov3 就完事”。
3. 我们改了哪些代码，每一处改动的意义是什么。

---

## 1. 先建立最小直觉：Backbone 在 BoQ 里做什么
在这个项目中，`backbone`（例如 `DINOv2` / `DINOv3`）负责把图片变成特征图，`BoQ` 聚合器再把特征图变成全局描述子。

完整链路可以理解为：

1. 输入图像：`[B, 3, H, W]`
2. Backbone 输出特征图：`[B, C, H', W']`
3. BoQ 输出描述子：`[B, D]`

只要我们能保证第 2 步的接口契约稳定（维度与语义一致），BoQ 主体训练代码几乎不需要大改。

---

## 2. 从 patch 到 token：到底发生了什么

## 2.1 patch 的含义
Vision Transformer 会把图像切成 patch。每个 patch 被映射成一个 token。

- 若 patch size = 14，输入 `224x224`，网格约为 `16x16`，patch token 数约 `256`
- 若 patch size = 16，输入 `224x224`，网格为 `14x14`，patch token 数 `196`

这就是你会看到 `dinov2` 与 `dinov3` 输出空间大小不同的根本原因之一。

## 2.2 token 组成
在 DINO 系列中，进入 transformer block 的 token 序列通常不是“纯 patch token”，还可能包含前缀 token：

1. `CLS token`
2. `storage tokens`（某些模型中存在）
3. `patch tokens`

因此，总 token 序列是：

`[CLS (+ storage...)] + [patch tokens]`

在我们的 wrapper 里，最终做空间 reshape 前，必须把前缀 token 去掉，只保留 patch tokens。

代码位置：`src/backbones.py:167-169`

---

## 3. DINOv2 与 DINOv3 在本项目语境下的关键区别

## 3.1 相同点（所以能共用骨架）

1. 都是 ViT 风格主干，核心有 `blocks`。
2. 都能输出 patch-level token 表示。
3. 都能通过“冻结前层 + 解冻后 N 层”微调。
4. 都能提供 `embed_dim` 作为 `out_channels`。

这就是我们抽象 `_DinoBackbone` 的基础。

## 3.2 关键差异（决定了必须改 forward）

1. **`prepare_tokens_with_masks` 返回格式差异**
   - 在我们遇到的 `dinov3` 实现中，返回 `(tokens, (Hpatch, Wpatch))`
   - 不能像旧版 `dinov2` wrapper 那样直接当成单个 Tensor 用

2. **RoPE（旋转位置编码）参与 block 前向**
   - `dinov3` block 前向常见签名是 `blk(x, rope_sincos)`
   - 如果不传 rope 或传错类型，会在 block 内部断言失败（你遇到的 `AssertionError`）

3. **prefix tokens 处理更敏感**
   - `dinov3` 里 `n_storage_tokens` 可能非 0
   - 不能再硬编码只去掉 1 个 `CLS`，要去掉 `1 + n_storage_tokens`

4. **patch size 默认不同**
   - 常见 `dinov2` 是 p14
   - 你当前 `dinov3_vitb16` 是 p16
   - 这会直接改变输出特征图分辨率，进而影响 BoQ 输入 token 数

---

## 4. 这次代码修改清单（由小到大）

## 4.1 小修改：训练入口识别 dinov3
文件：`train.py`

1. 导入 `DinoV3`：`train.py:16`
2. 增加 `dinov3` 分支：`train.py:98-109`
3. 增加 `--dino_weights` 参数：`train.py:288`

意义：训练脚本能够像 `dinov2` 一样自然选择 `dinov3`。

## 4.2 中修改：hubconf 支持 dinov3
文件：`hubconf.py`

1. 可选 backbone 增加 `dinov3`：`hubconf.py:31-37`
2. URL 解析逻辑增加 `model_url` / 环境变量：`hubconf.py:48-60`
3. `get_trained_boq` 增加 `dinov3` 分支：`hubconf.py:84-94`

意义：兼容 torch.hub 场景，允许在线/离线路径都可配置。

## 4.3 大修改：Backbone 抽象与前向逻辑升级
文件：`src/backbones.py`

1. 新增 `_DinoBackbone` 统一逻辑：`src/backbones.py:15`
2. 新增 `DinoV3` 类：`src/backbones.py:208`
3. 新增本地默认权重：`src/backbones.py:220`
4. 新增 rate-limit 容错加载：`src/backbones.py:79-130`
5. 修正 token/rope 处理（核心）：`src/backbones.py:139-169`

意义：不仅“能跑”，还解决了真实工程中的三类问题：

1. GitHub 403 限流。
2. 本地离线权重加载。
3. `dinov3` block 前向协议差异导致的断言错误。

---

## 5. 重点深挖：`src/backbones.py` 里的 forward 到底在做什么

下面按执行顺序解释 `forward`（`src/backbones.py:139-176`）。

## 5.1 取输入形状
`B, _, H, W = x.shape`

后面 reshape 回特征图需要 `H/W`。

## 5.2 准备 token，并识别是否带 `(Hpatch, Wpatch)`

`tokens_out = self.dino.prepare_tokens_with_masks(x)`

然后分两种情况：

1. 若 `tokens_out` 是 tuple：
   - `x, hw_tuple = tokens_out`
   - 若模型有 `rope_embed`，计算 `rope_sincos = rope_embed(H, W)`
2. 否则：
   - 直接 `x = tokens_out`

这一步就是为同时兼容 `dinov2/dinov3`。

## 5.3 冻结块与可训练块分段前向

1. 前半段（冻结块）放在 `torch.no_grad()` 内
2. 后半段（解冻块）正常前向，允许反向传播

并且每次调用 block 时：

1. 若有 `rope_sincos`，调用 `blk(x, rope_sincos)`
2. 否则调用 `blk(x)`

这一步修复了你遇到的 `AssertionError`。

## 5.4 去掉前缀 token（CLS + storage）

`num_prefix_tokens = 1 + n_storage_tokens`

`x = x[:, num_prefix_tokens:]`

如果只删 1 个 token，在有 storage token 的模型上，patch token 对齐会错位。

## 5.5 从 token 序列还原为二维特征图

已知：

1. `x` 现在是 `[B, Npatch, C]`
2. `patch_size = p`
3. `Npatch = (H/p) * (W/p)`

执行：

`x -> permute -> view(B, C, H/p, W/p)`

得到 BoQ 所需的卷积式特征图输入。

---

## 6. 训练参数冻结逻辑：为什么你看到 trainable 只有后两层 block
你日志里显示 `backbone.dino.blocks.10/11` 可训练，这和 `unfreeze_n_blocks=2` 完全一致。

逻辑在：`src/backbones.py:60-74`

1. 先把 backbone 所有参数 `requires_grad=False`
2. 再把最后 2 个 block 参数设为 `True`

这是迁移学习常见策略：

1. 稳定训练
2. 减少显存和训练成本
3. 避免小数据集把底层视觉特征破坏掉

---

## 7. 架构层面对 BoQ 的影响（从浅到深）

## 7.1 浅层影响：尺寸变化
`dinov3_vitb16` 的 patch size=16，`224x224` 输入时空间分辨率是 `14x14`。

## 7.2 中层影响：token 数变化
patch token 数变少（相对 p14 的 16x16），注意力计算规模下降，速度/显存行为会不同。

## 7.3 深层影响：位置编码与 block 协议
`dinov3` 强依赖 rope 信息在 block 中正确传递。wrapper 若不处理 rope，模型语义就不完整，甚至直接报错。

---

## 8. 新手最容易踩的坑

1. 只改模型名，不改 forward 协议。
2. 把 tuple token 输出当 Tensor 继续传。
3. 忽略 `storage tokens`，只删 CLS。
4. 网络受限时只依赖在线下载，不配本地权重回退。
5. 在非 `BoQ` 环境（例如 py39）测试 `dinov3`，遇到 `|` 类型语法兼容问题。

---

## 9. 推荐验证顺序（快速排错）

1. 语法检查

```bash
python -m py_compile src/backbones.py train.py hubconf.py
```

2. 单独前向检查（在 BoQ 环境）

```bash
/root/miniconda3/envs/BoQ/bin/python - <<'PY'
import torch
from src.backbones import DinoV3
m = DinoV3('dinov3_vitb16', unfreeze_n_blocks=2)
y = m(torch.randn(2,3,224,224))
print(y.shape, m.out_channels, m.patch_size)
PY
```

3. 训练 smoke run

```bash
python train.py --backbone dinov3_vitb16 --dev
```

---

## 10. 一句话总结
这次适配的本质不是“新增一个类名”，而是把 `dinov3` 的 **token 组织方式、rope 前向协议、prefix token 规则、离线加载需求** 全部接到 BoQ 的训练与推理链路中，确保它在真实环境可训练、可复现、可维护。
