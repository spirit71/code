# Geometry-Constrained Attention 精度下降问题分析与修复

## 问题分析

经过详细分析，发现以下几个导致精度下降的关键问题：

### 1. **尺度不匹配问题（最关键）**

**问题描述：**
- 内容相似度 `(q_j^T · k_i) / sqrt(head_dim)` 经过了 `1/sqrt(48) ≈ 0.144` 的缩放
- 几何兼容分数 `g_xy^T · q_j^g` **没有经过任何缩放**
- 如果几何嵌入 `g_xy` 和几何向量 `q_j^g` 的尺度较大（比如10-100），几何偏置会远大于内容相似度，导致几何约束主导attention，抑制了内容信息

**影响：**
- 几何约束过度主导，内容相似度被压制
- 模型无法充分利用语义信息，导致泛化能力下降

**修复方案：**
```python
# 添加几何兼容分数的归一化因子
self.geometry_scale = (geometry_dim ** -0.5)  # 类似于attention的scale

# 在计算几何兼容分数时应用归一化
geometry_bias_patches = torch.matmul(q_geometry, g_xy.transpose(-2, -1)) * self.geometry_scale
```

### 2. **几何嵌入初始化问题**

**问题描述：**
- 几何嵌入MLP使用默认初始化，可能导致输出尺度过大
- `query_geometry_proj` 使用 `std=0.02` 的初始化，可能太小，导致几何向量学习不足

**修复方案：**
```python
# 几何嵌入MLP使用较小的初始化
for module in self.geometry_embed:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# query_geometry_proj使用Xavier初始化，gain=0.1
nn.init.xavier_uniform_(self.query_geometry_proj.weight, gain=0.1)
```

### 3. **lambda_g权重过大**

**问题描述：**
- 即使添加了归一化，`lambda_g=0.1` 可能仍然过大
- 需要与归一化后的尺度匹配

**修复方案：**
- 将 `lambda_g` 从 `0.1` 降低到 `0.05`
- 建议根据实际效果进一步调整（范围：0.01 ~ 0.1）

## 修复总结

### 已修复的问题：

1. ✅ **添加几何兼容分数归一化**：`geometry_scale = 1/sqrt(geometry_dim)`
2. ✅ **改进几何嵌入初始化**：使用 `std=0.01` 的较小初始化
3. ✅ **改进query_geometry_proj初始化**：使用 `Xavier(gain=0.1)` 初始化
4. ✅ **降低lambda_g权重**：从 `0.1` 降低到 `0.05`

### 预期效果：

1. **尺度匹配**：几何兼容分数与内容相似度尺度一致，不会过度主导
2. **平衡学习**：内容信息和几何信息能够平衡学习
3. **更好的泛化**：模型能够充分利用语义信息，提升泛化能力

## 建议的进一步优化

1. **超参数调优**：
   - `lambda_g`: 建议在 [0.01, 0.1] 范围内调整
   - `geometry_dim`: 可以尝试 32, 64, 128 等不同值

2. **训练策略**：
   - 可以考虑使用warmup策略，逐渐增加 `lambda_g`
   - 或者使用可学习的 `lambda_g`（作为可训练参数）

3. **监控指标**：
   - 监控几何偏置和内容相似度的尺度比例
   - 监控attention权重的分布，确保不会过度集中在某些位置

## 代码变更位置

1. `backbone/GeometryConstrainedAttention.py`:
   - 添加 `geometry_scale` 属性
   - 改进几何嵌入和query_geometry_proj的初始化
   - 在几何兼容分数计算中应用归一化

2. `network.py`:
   - 将 `lambda_g` 从 `0.1` 降低到 `0.05`

## 验证方法

训练后检查：
1. 几何偏置的尺度是否与内容相似度匹配（应该在相似的数量级）
2. Attention权重的分布是否合理（不应该过度集中在某些位置）
3. 验证集和测试集的精度是否提升

## 错误样本与注意力热图分析（amstertime / tokyo247 / Nordland）

当 Geometry-Constrained-Assignment 在 amstertime（domain variants）、tokyo247（illumination）有提升，但在 Nordland 上下降时，可筛选各数据集的错误样本并查看**最终层 queries 的注意力热图**以分析原因。

### 运行方式

```bash
# 使用默认 checkpoint 与输出目录
./scripts/run_error_analysis_geometry.sh

# 或指定 checkpoint 与输出目录
./scripts/run_error_analysis_geometry.sh ./logs/default/2026-01-31_10-39-54/best_model.pth ./logs/geometry_error_analysis
```

或直接调用（需在项目根目录，且 `PYTHONPATH` 包含项目根）：

```bash
PYTHONPATH=. python scripts/analyze_errors_attnmap.py --geometry_datasets --resume <checkpoint.pth> --save_dir <输出目录>
```

### 输出内容

- **error_sample_images_\***：每个数据集的错误样本**查询图片**（按 query 文件夹分子目录），便于人工筛选与对比。
- **retrieval_errors_\***：各数据集的错误样本索引、预测索引、正样本等统计（`error_samples_info.txt`）。
- **error_visualization_\***：每个错误样本对应：
  - `*_stacked.png`：各 decoder 层 + 聚合的叠加热力图；
  - `*_final_layer_heatmap.png`：**仅最终层** decoder 的 queries 注意力热图，便于分析模型在错误样本上关注的位置。

### 数据集路径配置

脚本内置三数据集路径（可在 `scripts/analyze_errors_attnmap.py` 中修改 `GEOMETRY_DATASET_CONFIGS`）：

| 数据集   | 说明               | 默认 eval_folder / name |
|----------|--------------------|--------------------------|
| amstertime | domain variants   | `.../datasets`, `amstertime/images` |
| tokyo247   | illumination changes | `/root/data`, `Tokyo247/images` |
| Nordland  | winter as queries | `.../datasets/Nordland`, `images_winter_as_quries` |
