# Geometry-Constrained Assignment 实现说明

## 概述

本实现将 **a²GC (Adaptive Assignment with Geometry Constraints)** 的几何约束思想移植到 Transformer 的 Query↔Patch 分配机制中，实现了**方案A：几何偏置直接加到 attention logits**。

## 核心思想

### a²GC 的几何约束机制

a²GC 指出：现有 OT（Optimal Transport）聚合忽略空间关系是 missed opportunity，空间邻近往往与语义一致相关，可提升 assignment 质量。

**具体做法：**
1. 将 patch 坐标 (x,y) 归一化到 [-1,1]
2. 投影成几何嵌入 `g_xy`
3. 与每个 cluster 的几何向量 `c_j^g` 做点积得到几何兼容分数 `S_{ij}^g`
4. 与特征相似度融合：`S_{ij} = S_{ij}^f + λ_g · S_{ij}^g`

### 移植到 Transformer Attention

在 query-based 聚合（EDTformer/BoQ 类）中，核心分配发生在 query 对 patch 的 cross-attention 权重上。我们将 a²GC 的几何兼容性显式注入到 attention logits：

```
logit_{j,i} = (q_j^T · k_i) / sqrt(d) + λ_g · S_{j,i}^g
```

其中：
- `(q_j^T · k_i) / sqrt(d)` 是标准的内容相似度
- `S_{j,i}^g = g_xy,i^T · q_j^g` 是几何兼容分数
- `λ_g` 是几何约束权重（可调超参数）

## 实现架构

### 1. GeometryConstrainedAttention (`backbone/GeometryConstrainedAttention.py`)

核心组件，实现带几何约束的多头注意力：

**关键组件：**
- **几何嵌入投影** (`geometry_embed`): 将归一化坐标 (x,y) ∈ [-1,1]² 投影到几何嵌入空间
- **Query 几何向量学习** (`query_geometry_proj`): 从 query 特征学习几何偏好向量 `q_j^g`
- **几何兼容分数计算**: `S_{j,i}^g = g_xy,i^T · q_j^g`
- **融合机制**: `attn_logits = content_sim + λ_g · geometry_compat`

**参数：**
- `geometry_dim`: 几何嵌入维度（默认64）
- `lambda_g`: 几何约束权重（默认0.1）

### 2. SA_CA 修改 (`SACA.py`)

修改了 `SA_CA` 类以支持几何约束：

- 添加 `use_geometry_constraint` 参数（默认 True）
- 在 `_mha_block` 中传递几何约束参数
- 支持向后兼容（可关闭几何约束）

### 3. CustomTransformerDecoder (`backbone/CustomTransformerDecoder.py`)

自定义 TransformerDecoder，支持向 decoder layer 传递额外参数（几何约束参数）。

### 4. VPRNet 修改 (`network.py`)

在 `VPRNet.forward()` 中：

1. **计算 patch 坐标**：
   - 从 backbone 获取 patch 网格信息
   - 生成归一化坐标（仅对 patch tokens，不包括 cls token）
   - 坐标归一化到 [-1, 1] 范围

2. **传递几何约束参数**：
   - 将 `patch_grid_h`, `patch_grid_w`, `patch_coords` 传递给 decoder

## 使用方法

### 基本使用

代码已默认启用几何约束。在 `network.py` 中：

```python
decoderlayer = SA_CA(
    d_model=768, 
    nhead=16, 
    batch_first=True,
    use_geometry_constraint=True,  # 启用几何约束
    geometry_dim=64,               # 几何嵌入维度
    lambda_g=0.1,                  # 几何约束权重
)
```

### 调整超参数

**几何约束权重 `lambda_g`**：
- 默认值：0.1
- 建议范围：0.05 ~ 0.3
- 过小：几何约束影响微弱
- 过大：可能抑制内容相似度的作用

**几何嵌入维度 `geometry_dim`**：
- 默认值：64
- 建议范围：32 ~ 128
- 过小：表达能力不足
- 过大：增加计算开销

### 禁用几何约束

如果需要对比实验，可以禁用几何约束：

```python
decoderlayer = SA_CA(
    d_model=768, 
    nhead=16, 
    batch_first=True,
    use_geometry_constraint=False,  # 禁用几何约束
)
```

## 技术细节

### 坐标归一化

采用与 a²GC 和 RoPE 一致的归一化方式：

```python
# 使用0.5偏移，使坐标位于patch中心
coords_h = torch.arange(0.5, patch_grid_h) / patch_grid_h
coords_w = torch.arange(0.5, patch_grid_w) / patch_grid_w
# 归一化到[-1, 1]
coords = 2.0 * coords - 1.0
```

### Query 几何向量学习

不同于 a²GC 中为每个 cluster 学习固定的几何向量，我们使用**可学习的投影层**从 query 特征动态生成几何向量：

```python
q_geometry = query_geometry_proj(query)  # [B, num_queries, num_heads * geometry_dim]
```

这样每个 query 可以根据其内容学习不同的几何偏好，增强表达能力。

### 处理 cls token

当 memory 包含 cls token 时，几何约束只应用到 patch tokens 位置：

```python
# 创建全零的几何偏置
geometry_bias = torch.zeros(B, H, num_queries, num_keys)
# 将几何偏置填充到最后的num_patches个位置（patch tokens）
geometry_bias[:, :, :, -num_patches:] = geometry_bias_patches
```

## 预期效果

### 理论优势

1. **空间感知增强**：显式建模空间邻近关系，提升对重复结构/城市街景的判别能力
2. **减少误检**：在 perceptual aliasing（外观相似但几何组织不同）场景下，几何约束有助于区分
3. **提升 Recall@1**：特别是在具有强空间结构的场景中

### 可检验假设

**If** 在 query→patch 的 attention logits 上加入 a²GC 风格的几何兼容项，**then** 在 perceptual aliasing 更强的数据上（城市街景/重复结构）会减少误检、提升 Recall@1；**otherwise**（无提升或退化）假设被否证。

## 文件结构

```
backbone/
├── GeometryConstrainedAttention.py  # 几何约束注意力实现
├── CustomTransformerDecoder.py    # 自定义Decoder（支持传递额外参数）
└── ...

SACA.py                             # 修改后的SA_CA（支持几何约束）
network.py                          # 修改后的VPRNet（计算并传递坐标）
```

## 注意事项

1. **patch 网格推断**：如果无法从 backbone 获取 patch 网格信息，代码会尝试推断（假设正方形网格）。如果推断失败，几何约束将被跳过。

2. **内存开销**：几何约束增加了少量计算和内存开销（主要是几何嵌入投影和兼容分数计算），但通常可忽略。

3. **训练稳定性**：建议从较小的 `lambda_g`（如 0.05）开始，逐步调整。

## 参考文献

- a²GC: Adaptive Assignment with Geometry Constraints（相关论文）
- RoPE: Rotary Position Embedding（坐标归一化参考）
