"""
可视化 MultiSimilarityMiner 输出的数据流
展示 descriptors[8,6] 和 labels[8] 情况下的数据流
"""

# 模拟数据
descriptors_shape = (8, 6)
labels_example = [0, 0, 1, 1, 2, 2, 3, 3]

# miner_outputs
tensor_a = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 7, 7, 7]
tensor_b = [2, 3, 1, 2, 3, 0, 0, 1, 3, 0, 6, 5, 7, 6, 4, 5, 4, 6]
tensor_c = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 5, 7, 7, 7, 7]
tensor_d = [7, 5, 5, 4, 7, 7, 5, 7, 5, 1, 3, 1, 0, 2, 0, 3, 2, 1]

print("=" * 100)
print("MultiSimilarityMiner 数据流可视化")
print("=" * 100)

print("\n【输入数据】")
print(f"descriptors 形状: {descriptors_shape}")
print(f"labels: {labels_example}")
print("\n样本索引映射:")
for i, label in enumerate(labels_example):
    print(f"  样本[{i}]: 标签={label}")

print("\n" + "=" * 100)
print("【Step 1: MultiSimilarityMiner 挖掘困难样本对】")
print("=" * 100)
print("使用余弦相似度筛选困难样本对...")
print(f"返回 miner_outputs = (tensor_a, tensor_b, tensor_c, tensor_d)")

print("\n【正样本对挖掘结果】")
print(f"tensor_a (锚点索引): {tensor_a}")
print(f"tensor_b (正样本索引): {tensor_b}")
print(f"共挖掘出 {len(tensor_a)} 个正样本对")

print("\n正样本对详情:")
print("索引 | 锚点 | 正样本 | 锚点标签 | 正样本标签 | 说明")
print("-" * 70)
for i in range(len(tensor_a)):
    anchor_idx = tensor_a[i]
    pos_idx = tensor_b[i]
    anchor_label = labels_example[anchor_idx]
    pos_label = labels_example[pos_idx]
    status = "✓ 同标签" if anchor_label == pos_label else "✗ 不同标签"
    print(f" {i+1:2d}  |  {anchor_idx}   |   {pos_idx}    |    {anchor_label}     |     {pos_label}     | {status}")

print("\n【负样本对挖掘结果】")
print(f"tensor_c (负样本锚点索引): {tensor_c}")
print(f"tensor_d (负样本索引): {tensor_d}")
print(f"共挖掘出 {len(tensor_c)} 个负样本对")

print("\n负样本对详情:")
print("索引 | 负锚点 | 负样本 | 负锚点标签 | 负样本标签 | 说明")
print("-" * 70)
for i in range(len(tensor_c)):
    neg_anchor_idx = tensor_c[i]
    neg_idx = tensor_d[i]
    neg_anchor_label = labels_example[neg_anchor_idx]
    neg_label = labels_example[neg_idx]
    status = "✓ 不同标签" if neg_anchor_label != neg_label else "✗ 同标签"
    print(f" {i+1:2d}  |   {neg_anchor_idx}    |   {neg_idx}    |      {neg_anchor_label}      |      {neg_label}      | {status}")

print("\n" + "=" * 100)
print("【Step 2: MultiSimilarityLoss 计算损失】")
print("=" * 100)

print("\n【数据提取阶段】")
print("1. 提取正样本对特征:")
print("   anchor_features = descriptors[tensor_a]  # 形状: [18, 6]")
print("   positive_features = descriptors[tensor_b]  # 形状: [18, 6]")
print("\n   示例:")
print("   - 第1个正样本对: descriptors[0] (锚点) 和 descriptors[2] (正样本)")
print("   - 第2个正样本对: descriptors[0] (锚点) 和 descriptors[3] (正样本)")
print("   - 第3个正样本对: descriptors[0] (锚点) 和 descriptors[1] (正样本)")

print("\n2. 提取负样本对特征:")
print("   neg_anchor_features = descriptors[tensor_c]  # 形状: [18, 6]")
print("   negative_features = descriptors[tensor_d]  # 形状: [18, 6]")
print("\n   示例:")
print("   - 第1个负样本对: descriptors[0] (负锚点) 和 descriptors[7] (负样本)")
print("   - 第2个负样本对: descriptors[0] (负锚点) 和 descriptors[5] (负样本)")

print("\n【相似度计算阶段】")
print("使用点积相似度 (DotProductSimilarity):")
print("  S_ap[i] = descriptors[tensor_a[i]] · descriptors[tensor_b[i]]^T")
print("  S_an[i] = descriptors[tensor_c[i]] · descriptors[tensor_d[i]]^T")
print("\n  对于 descriptors[8, 6]:")
print("  - 每个特征向量是6维: [f0, f1, f2, f3, f4, f5]")
print("  - 点积计算: S = f0_a*f0_b + f1_a*f1_b + ... + f5_a*f5_b")

print("\n【损失计算阶段】")
print("MultiSimilarityLoss 公式:")
print("  L_pos = (1/α) * log(1 + Σ exp(-α * (S_ap - λ_pos)))")
print("  L_neg = (1/β) * log(1 + Σ exp(β * (S_an - λ_neg)))")
print("  Loss = L_pos + L_neg")
print("\n  参数:")
print("  - α = 1.0 (正样本对权重)")
print("  - β = 50 (负样本对权重，较大值使模型更关注推远负样本)")
print("  - λ_pos, λ_neg: 由损失函数内部计算的相似度阈值")

print("\n" + "=" * 100)
print("【数据流总结】")
print("=" * 100)

print("""
输入: descriptors[8, 6] + labels[8]
  ↓
MultiSimilarityMiner (余弦相似度挖掘)
  ├─ 筛选困难正样本对 → (tensor_a, tensor_b) [18对]
  └─ 筛选困难负样本对 → (tensor_c, tensor_d) [18对]
  ↓
MultiSimilarityLoss (点积相似度计算)
  ├─ 提取正样本对特征: descriptors[tensor_a], descriptors[tensor_b]
  ├─ 提取负样本对特征: descriptors[tensor_c], descriptors[tensor_d]
  ├─ 计算点积相似度: S_ap, S_an
  └─ 应用损失公式: L_pos + L_neg
  ↓
输出: loss (标量)
""")

print("\n【关键点】")
print("1. Miner 和 Loss 使用不同的相似度度量:")
print("   - Miner: 余弦相似度 (用于筛选困难样本)")
print("   - Loss: 点积相似度 (用于计算损失值)")
print("\n2. 只使用挖掘出的困难样本对，提高训练效率:")
print("   - 不使用所有可能的样本对组合")
print("   - 只关注'困难'的正样本对和负样本对")
print("\n3. 困难样本的定义:")
print("   - 困难正样本对: 相似度较低但标签相同的对")
print("   - 困难负样本对: 相似度较高但标签不同的对")

print("\n" + "=" * 100)

