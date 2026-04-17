# EDTformer 项目完整分析报告

## 1. 项目任务概述

### 1.1 核心任务：视觉位置识别（Visual Place Recognition, VPR）

- **目标**：给定一张查询图像，从图像数据库中检索出拍摄于同一位置的图像
- **应用场景**：机器人导航、自动驾驶、AR/VR等需要视觉定位的场景
- **评估指标**：Recall@K（R@1, R@5, R@10, R@20），即正确位置是否出现在检索结果的前K个中

### 1.2 数据集

**训练数据**：GSV-Cities 数据集
- 来源：Google Street View
- 包含多个城市的街景图像（London, Boston等）
- 每个地点（place）包含至少4张图像
- 数据路径：`/root/data/gsv_cities/`

**测试数据**：Pitts30k 数据集
- 分为 `database`（数据库图像）和 `queries`（查询图像）
- 通过UTM坐标计算软正样本（25米半径内的数据库图像）
- 数据路径：`/root/data/Pittsburgh/pitts30k/`

---

## 2. 整体模型架构

```
输入: [B, 3, 224, 224]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Patch Embedding (patch_embed.py)                          │
│     将图像划分为 patches 并线性投影                             │
│     [B, 3, 224, 224] → [B, 256, 768]                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 添加 [CLS Token] + [Positional Encoding]                  │
│     [B, 257, 768]                                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Transformer Blocks (12层) + LadderAdapter (12层)      │
│     for i in range(12):                                  │
│         x = Block_i(x)           # 标准 ViT Block         │
│         y = Adapter_i(y + x)     # 并行适配器          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 输出处理 (VPRNet)                                    │
│     ├─ 提取 CLS Token + Patch Tokens                       │
│     ├─ channel_proj: 768 → 256                          │
│     ├─ adaptive_avg_pool1d: [B,256,N]→[B,256,16]     │
│     └─ flatten + L2_norm → [B, 4096]                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 详细模块说明

### 3.1 PatchEmbed - 图像分块与嵌入

**文件**: `backbone/dinov2/patch_embed.py`

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)        # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x
```

**作用**： 
- 将输入图像划分为多个不重叠的 patch
- 通过卷积层将每个 patch 投影到 embed_dim 维空间
- 对于 224×224 图像，使用 16×16 的 patch，得到 14×14=256 个 patches
- 输出: `[B, 256, 768]`

---

### 3.2 Attention - 多头自注意力

**文件**: `backbone/dinov2/attention.py`

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

**作用**： 
- 计算 token 之间的自注意力，允许每个位置关注其他所有位置
- 多头机制允许模型学习不同类型的关系

---

### 3.3 Mlp - 前馈网络

**文件**: `backbone/dinov2/mlp.py`

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```

**作用**： 
- FFN 将维度扩展 4 倍（hidden_features = dim * mlp_ratio = 768 * 4 = 3072）
- 提供更强的特征变换能力

---

### 3.4 Block - Transformer 编码器块

**文件**: `backbone/dinov2/block.py`

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=None):
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.drop_path1 = DropPath(drop_path)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.ls2 = LayerScale(dim, init_values)
        self.drop_path2 = DropPath(drop_path)
        
    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
```

**架构图**:
```
输入 x
  │
  ├─> Norm1 → Attention → LayerScale → DropPath ─┐
  │                                          │
  └───────────────────────────────────────────＋─> + → 输出
  │
  ├─> Norm2 → MLP → LayerScale → DropPath ─┐
  │                                        │
  └──────────────────────────────────────────┘
```

**组件作用**:
- `LayerScale`: 对残差进行逐通道的可学习缩放，stable训练
- `DropPath`: 随机丢弃整个残差分支（Stochastic Depth）

---

### 3.5 LayerScale - 层缩放

**文件**: `backbone/dinov2/layer_scale.py`

```python
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
    def forward(self, x):
        return x * self.gamma
```

**作用**: 对输入特征按通道维做可学习的逐通道缩放，用于稳定深层Transformer的训练

---

### 3.6 DropPath - 随机深度

**文件**: `backbone/dinov2/drop_path.py`

```python
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output
```

**作用**: 随机丢弃整个残差分支，类似Dropout，但应用于残差连接，训练深层网络时有效减轻梯度消失

---

### 3.7 LadderAdapter / SideAdapter - 并行适配器

**文件**: `backbone/LadderAdapter.py`

```python
class SideAdapter(nn.Module):
    def __init__(self, D_features, D_hidden_features=4, act_layer=nn.GELU, skip_connect=True, alpha=0.5):
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)  # 768 → 4
        self.act = act_layer()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)  # 4 → 768
        self.alpha = alpha
        
    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        
        if self.skip_connect:
            x = x + self.alpha * xs  # 残差连接
        else:
            x = xs
        return x
```

**作用**:
- 轻量级适配器模块，只包含 2 个线性层+激活函数
- 隐藏维度只有 4，参数量极小
- `alpha=0.5` 控制残差连接的权重

---

### 3.8 DinoVisionTransformer - 完整骨干网络

**文件**: `backbone/vision_transformer.py`

```python
class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12):
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        
        self.adapters = nn.ModuleList([
            SideAdapter(D_features=embed_dim, D_hidden_features=4) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1)
        x = x + self.pos_embed
        
        y = x.clone()
        
        for blk, adapter in zip(self.blocks, self.adapters):
            x = blk(x)
            y = adapter(y + x)
        
        x = y
        x = self.norm(x)
        
        return {
            "x_norm_clstoken": x[:, 0:1, :],
            "x_norm_patchtokens": x[:, 1:, :]
        }
```

**架构图**:
```
输入图像
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Patch Embed + CLS Token + Pos Embed                      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Block 0    │  ...  │  Block 11                      │
��  ──────────┼───────┼─────────                         │
│  Adapter 0 │  ...  │  Adapter 11  (并行)            │
│  y = x + α*adapter(y + x)                           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  输出: CLS Token + Patch Tokens                        │
└──────────────────────────────────────────────────────────┘
```

---

### 3.9 VPRNet - 完整的 VPR 网络

**文件**: `network_1.py`

```python
class VPRNet(nn.Module):
    def __init__(self, pretrained_foundation=True, foundation_model_path=None):
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.fc = nn.Linear(768, 768)
        self.channel_proj = nn.Linear(768, 256)
        
    def forward(self, x):
        x = self.backbone(x)
        x_c = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"]
        
        x_cp = torch.cat([x_c, x_p], dim=1)
        x_cp = self.fc(x_cp)
        
        x = x_p
        x = self.channel_proj(x)
        
        x = x.permute(0, 2, 1)
        x = F.adaptive_avg_pool1d(x, 16)
        
        x = x.flatten(1)
        x = F.normalize(x, p=2, dim=-1)
        
        return x
```

**维度变化流程**:

```
输入:              [B, 3, 224, 224]
                     │
                     ▼
backbone(x):      {"x_norm_clstoken": [B,1,768],
                  "x_norm_patchtokens": [B,256, 768]}
                     │
                     ▼
torch.cat:        [B, 257, 768]
                     │
                     ▼
fc:             [B, 257, 768]
                     │
                     ▼
去掉CLS:         [B, 256, 768]
                     │
                     ▼
channel_proj:     [B, 256, 256]
                     │
                     ▼
permute:         [B, 256, 256]
                     │
                     ▼
adaptive_pool:   [B, 256, 16]
                     │
                     ▼
flatten:        [B, 4096]
                     │
                     ▼
L2_norm:       [B, 4096]
```

---

### 3.10 SA_CA - 交叉注意力模块（可选decoder）

**文件**: `SACA.py`

```python
class SA_CA(Module):
    def __init__(self, d_model=768, nhead=16, dropout=0.1):
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, tgt, memory):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
        return x
```

**作用**: 可选的Transformer解码器，用于聚合patch特征（当前配置use_decoder=False）

---

## 4. 训练流程

### 4.1 训练代码（train.py）

```python
def train():
    # 1. 数据加载
    train_dataset = get_GSVCities()
    train_loader = DataLoader(train_dataset, batch_size=72)
    
    # 2. 模型初始化
    model = network.VPRNet(
        pretrained_foundation=True,
        foundation_model_path="./dinov2_vitb14_pretrain.pth"
    )
    
    # 3. 冻结backbone，只训练adapter
    for name, param in model.backbone.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 4. 初始化D_fc2为0
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    nn.init.constant_(m2.weight, 0.)
                    nn.init.constant_(m2.bias, 0.)
    
    # 5. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 6. 训练循环
    for epoch in range(epochs_num):
        model.train()
        for images, place_id in train_loader:
            images = images.view(-1, 3, 224, 224)
            labels = place_id.view(-1)
            
            descriptors = model(images)
            loss = loss_function(descriptors, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 7. 验证
        recalls, _ = test.test(val_ds, model)
        
        # 8. 保存最好的模型
        if R@1 提升:
            save_checkpoint(best_model)
```

### 4.2 损失函数

**文件**: `loss.py`

```python
from pytorch_metric_learning import losses, miners

loss_fn = losses.MultiSimilarityLoss(
    alpha=1.0, beta=50, base=0.0, 
    distance=DotProductSimilarity()
)
miner = miners.MultiSimilarityMiner(
    epsilon=0.1, 
    distance=CosineSimilarity()
)

def loss_function(descriptors, labels):
    miner_outputs = miner(descriptors, labels)
    loss = loss_fn(descriptors, labels, miner_outputs)
    return loss
```

**作用**:
- **MultiSimilarityLoss**: 度量学习损失，拉近同类样本，推远异类样本
- **MultiSimilarityMiner**: 在线挖掘难样本对/三元组

---

## 5. 测试流程

### 5.1 测试代码（test.py）

```python
def test(eval_ds, model):
    model.eval()
    
    # 1. 提取database特征
    database_features = []
    for inputs in database_loader:
        features = model(inputs)
        database_features.append(features)
    database_features = np.concatenate(database_features)
    
    # 2. 提取query特征
    queries_features = []
    for inputs in queries_loader:
        features = model(inputs)
        queries_features.append(features)
    queries_features = np.concatenate(queries_features)
    
    # 3. FAISS索引建立
    index = faiss.IndexFlatL2(4096)
    index.add(database_features)
    
    # 4. 检索
    distances, predictions = index.search(queries_features, k=20)
    
    # 5. 计算Recall
    recalls = compute_recall(predictions, positives_per_query)
    
    return recalls
```

---

## 6. 参数统计（ViT-B/14）

| 模块 | 参数数量 |
|------|--------|
| PatchEmbed | 3×14×14×768 = 473,088 |
| Pos Embed | 257×768 = 197,376 |
| 12×Block | 约 85M |
| 12×Adapter | 约 74K |
| channel_proj | 768×256 = 196,608 |
| **总计** | 约 86M（不含预训练权重） |
| **可训练** | ~270K（仅adapter + head） |

---

## 7. 训练命令

```bash
python3 train.py \
    --eval_datasets_folder=/root/data/Pittsburgh \
    --eval_dataset_name=pitts30k \
    --foundation_model_path=./dinov2_vitb14_pretrain.pth \
    --epochs_num=100 \
    --patience=25
```

---

## 8. 核心技术要点总结

1. **度量学习**：不使用分类头，直接学习图像间的相似性度量
2. **预训练模型微调**：基于DINOv2冻结backbone，训练轻量级adapter
3. **并行适配器**：每个Transformer block添加并行的低秩adapter
4. **自适应池化**：处理不同patch数量的输入（金字塔结构）
5. **软正样本**：基于UTM坐标确定正样本（25米半径内）

---

## 9. 文件结构

| 文件 | 功能 |
|------|------|
| `train.py` | 训练主循环 |
| `test.py` | 测试/验证，特征提取+Recall计算 |
| `eval.py` | 评估脚本 |
| `network_1.py` | VPRNet模型 |
| `SACA.py` | 交叉注意力模块 |
| `loss.py` | MultiSimilarityLoss + Miner |
| `parser.py` | 命令行参数解析 |
| `datasets_ws.py` | 测试数据集加载 |
| `dataloaders/train/GSVCitiesDataset.py` | 训练数据集加载 |
| `backbone/vision_transformer.py` | DINOv2 ViT实现 |
| `backbone/LadderAdapter.py` | 并行适配器 |
| `backbone/dinov2/*.py` | DINOv2基础组件 |
| `scripts/train.sh` | 训练脚本 |
| `scripts/eval.sh` | 评估脚本 |

---

## 10. 参考资料

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [CricaVPR](https://github.com/Lu-Feng/CricaVPR)
- [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
- [GSV-Cities](https://github.com/amaralibey/gsv-cities)
- [EDTformer论文](https://ieeexplore.ieee.org/document/3559084)