# 错误样本分析和叠加热力图使用指南

## 功能概述

新增的错误分析工具包含以下功能：

1. **叠加热力图可视化**：将多个layer的attention权重叠加显示，类似论文中的可视化效果
2. **检索错误分析**：自动找出检索失败的样本
3. **数据集性能对比**：对比不同数据集（如amstertime和Nordland）的性能差异

## 使用方法

### 1. 运行错误分析脚本

```bash
python scripts/analyze_errors_attnmap.py --resume --save_dir=logs/your_experiment \
    --eval_datasets_folder=path/to/datasets \
    --eval_dataset_name=amstertime
```

### 2. 对比多个数据集

脚本会自动对比配置的数据集（默认是amstertime和Nordland）。如果需要修改，编辑脚本中的：

```python
dataset_names = ["amstertime", "Nordland"]
```

## 输出文件说明

### 1. 检索错误信息 (`retrieval_errors_{dataset_name}/error_samples_info.txt`)

包含每个数据集的错误样本详细信息：
- 查询索引
- 预测索引
- 真实正样本
- Top-K预测结果
- 距离信息

### 2. 叠加热力图 (`error_visualization_{dataset_name}/{folder_name}/error_*.png`)

每个错误样本生成一个叠加热力图，包含：
- **第1列**：原始图像
- **中间列**：每个layer的attention热力图
- **最后列**：聚合后的attention热力图（所有layer和query的平均）

### 3. 数据集对比结果 (`dataset_comparison/dataset_comparison.txt`)

包含所有数据集的性能对比：
- 每个数据集的综合召回率
- 每个查询文件夹的详细召回率

## 分析建议

### 1. 对比amstertime和Nordland的差异

1. **查看对比结果**：
   ```bash
   cat logs/your_experiment/dataset_comparison/dataset_comparison.txt
   ```

2. **分析错误样本**：
   - 查看两个数据集的错误样本数量
   - 对比错误样本的特征（距离、预测位置等）

3. **可视化attention差异**：
   - 查看错误样本的attention热力图
   - 对比amstertime和Nordland上模型关注的位置差异

### 2. 诊断Nordland性能下降的原因

可能的原因和检查方法：

#### a) 数据集特性差异
- **检查**：查看两个数据集的图像特征
- **方法**：对比错误样本的原始图像，看是否有明显的视觉差异

#### b) Attention模式差异
- **检查**：对比两个数据集上错误样本的attention热力图
- **方法**：查看模型在不同数据集上关注的位置是否合理

#### c) 几何约束过度/不足
- **检查**：查看TensorBoard中的`scale_ratio`指标
- **方法**：如果Nordland上的`scale_ratio`异常，可能需要调整`lambda_g`

#### d) 特征分布差异
- **检查**：对比两个数据集上特征的距离分布
- **方法**：查看错误样本的距离信息，看是否有明显差异

### 3. 改进建议

根据分析结果，可以尝试：

1. **调整几何约束权重**：
   - 如果Nordland上的`scale_ratio`异常，调整`lambda_g`
   - 可以为不同数据集使用不同的`lambda_g`

2. **数据增强**：
   - 如果发现数据集特性差异，考虑针对性的数据增强

3. **模型微调**：
   - 如果发现attention模式不合理，可能需要调整模型结构

## 代码结构

### 主要函数

1. **`extract_attention_weights()`**: 提取所有layer的attention权重
2. **`visualize_stacked_heatmaps()`**: 可视化叠加热力图
3. **`analyze_retrieval_errors()`**: 分析检索错误样本
4. **`visualize_error_samples()`**: 可视化错误样本的attention
5. **`compare_datasets_performance()`**: 对比不同数据集的性能

### 关键参数

- `num_samples`: 每个文件夹要可视化的错误样本数量（默认10）
- `num_queries_to_show`: 热力图中显示的query数量（默认4）

## 故障排除

### 1. 数据集加载失败
- 检查数据集路径是否正确
- 确认数据集名称是否匹配

### 2. 内存不足
- 减少`num_samples`参数
- 减少batch size

### 3. 可视化失败
- 检查图像格式是否正确
- 确认matplotlib后端设置

## 示例输出

运行脚本后，你会得到类似以下结构的输出：

```
logs/your_experiment/
├── dataset_comparison/
│   └── dataset_comparison.txt
├── retrieval_errors_amstertime/
│   └── error_samples_info.txt
├── retrieval_errors_Nordland/
│   └── error_samples_info.txt
├── error_visualization_amstertime/
│   └── folder_name/
│       ├── error_0_query_123_stacked.png
│       ├── error_1_query_456_stacked.png
│       └── ...
└── error_visualization_Nordland/
    └── folder_name/
        ├── error_0_query_789_stacked.png
        └── ...
```

## 下一步

1. 运行分析脚本，获取错误样本和热力图
2. 对比amstertime和Nordland的错误样本特征
3. 查看attention热力图，找出模型关注位置的差异
4. 根据分析结果调整模型参数或训练策略
