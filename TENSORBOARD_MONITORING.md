# TensorBoard监控和错误分析使用指南

## 功能概述

已添加以下功能：
1. **TensorBoard监控**：实时监控训练过程中的各种统计信息
2. **错误样本分析**：分析模型预测错误的样本
3. **Attention热力图导出**：可视化模型的attention权重分布

## TensorBoard监控

### 监控的指标

训练过程中会自动记录以下指标到TensorBoard：

#### 1. 训练指标
- `Train/Loss`: 每个batch的损失值
- `Train/LearningRate`: 学习率变化

#### 2. 几何约束Attention统计（每50个batch记录一次）
- `GeometryAttention/Layer{idx}/content_sim_mean`: 内容相似度均值
- `GeometryAttention/Layer{idx}/content_sim_std`: 内容相似度标准差
- `GeometryAttention/Layer{idx}/geometry_bias_mean`: 几何偏置均值
- `GeometryAttention/Layer{idx}/geometry_bias_std`: 几何偏置标准差
- `GeometryAttention/Layer{idx}/scale_ratio`: 内容相似度与几何偏置的尺度比例
- `GeometryAttention/Layer{idx}/attn_entropy`: Attention权重熵（衡量分布均匀性）
- `GeometryAttention/Layer{idx}/attn_weights_mean`: Attention权重均值
- `GeometryAttention/Layer{idx}/attn_weights_std`: Attention权重标准差

#### 3. Attention热力图（每100个batch记录一次）
- `AttentionHeatmap/Layer{idx}`: 每个layer的attention权重热力图

#### 4. Epoch级别统计
- `Epoch/Loss`: 每个epoch的平均损失
- `Epoch/GeometryAttention/{metric}`: 每个epoch的几何约束统计
- `Val/Recall@1`, `Val/Recall@5`, etc.: 验证集召回率

### 使用方法

1. **启动训练**：
```bash
python train.py --your-args
```

2. **启动TensorBoard**：
```bash
tensorboard --logdir=logs/your_experiment_name
```

3. **在浏览器中查看**：
打开 `http://localhost:6006`

### 关键指标解读

#### scale_ratio（尺度比例）
- **含义**：内容相似度与几何偏置的尺度比例
- **理想值**：应该在 0.5 ~ 2.0 之间
- **过大**：说明几何偏置过小，可能没有发挥作用
- **过小**：说明几何偏置过大，可能过度主导attention

#### attn_entropy（Attention熵）
- **含义**：衡量attention权重的分布均匀性
- **高熵**：attention权重分布均匀，模型关注所有位置
- **低熵**：attention权重集中，模型只关注少数位置
- **建议**：应该适中，既不过于均匀也不过于集中

## 错误样本分析

### 功能
分析模型预测错误的样本，帮助理解模型的失败模式。

### 使用方法
```bash
python scripts/analyze_errors.py --resume --save_dir=logs/your_experiment
```

### 输出
- 错误样本的详细信息
- 错误原因分析
- 可视化结果

## Attention热力图导出

### 功能
导出模型的attention权重热力图，可视化模型关注的位置。

### 使用方法
```bash
python scripts/analyze_errors.py --resume --save_dir=logs/your_experiment
```

脚本会自动：
1. 从验证集中采样样本
2. 提取每个layer的attention权重
3. 生成热力图并保存

### 输出位置
`logs/your_experiment/attention_heatmaps/`

### 文件命名
- `sample_{idx}_layer_{layer_idx}_query_{query_idx}.png`
- 每个样本会为每个layer和每个query（最多4个）生成热力图

## 监控建议

### 训练初期
1. 检查 `scale_ratio` 是否在合理范围（0.5 ~ 2.0）
2. 检查 `geometry_bias_mean` 和 `content_sim_mean` 的尺度是否匹配
3. 观察 `attn_entropy` 的变化趋势

### 训练过程中
1. 监控 `Train/Loss` 是否正常下降
2. 检查 `Val/Recall@1` 是否提升
3. 观察attention热力图，确保模型关注合理的位置

### 发现问题时
1. **scale_ratio异常**：
   - 如果过小，考虑降低 `lambda_g`
   - 如果过大，考虑增加 `lambda_g` 或检查归一化

2. **attn_entropy过低**：
   - 说明attention过度集中
   - 可能需要调整几何约束的强度

3. **验证集精度不提升**：
   - 查看attention热力图，检查模型是否关注错误的位置
   - 分析错误样本，找出失败模式

## 代码修改说明

### 新增功能
1. `GeometryConstrainedAttention.forward()`: 添加 `return_stats` 参数
2. `SA_CA.forward()`: 支持返回统计信息
3. `CustomTransformerDecoder.forward()`: 支持返回统计信息
4. `VPRNet.forward()`: 支持返回统计信息
5. `train.py`: 添加TensorBoard监控代码
6. `scripts/analyze_errors.py`: 新增错误分析和热力图导出脚本

### 性能影响
- 统计信息收集只在每50个batch进行一次，对训练速度影响很小
- Attention热力图导出只在每100个batch进行一次
- 可以通过调整 `log_interval` 来控制监控频率

## 故障排除

### TensorBoard无法启动
- 检查 `--logdir` 路径是否正确
- 确保有写入权限

### 统计信息为空
- 检查模型是否使用了几何约束（`use_geometry_constraint=True`）
- 检查 `return_stats` 是否正确传递

### 热力图无法显示
- 检查图像保存路径
- 确保有足够的磁盘空间
- 检查matplotlib后端设置
