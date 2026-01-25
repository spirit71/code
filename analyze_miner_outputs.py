"""
分析 MultiSimilarityMiner 输出的数据流和 MultiSimilarityLoss 的计算原理
"""
import torch
import numpy as np

# 模拟调试状态下的数据
descriptors = torch.randn(8, 6)  # [8, 6] 特征向量
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # [8] 标签

# miner_outputs 返回值
tensor_a = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 7, 7, 7])  # 锚点索引
tensor_b = torch.tensor([2, 3, 1, 2, 3, 0, 0, 1, 3, 0, 6, 5, 7, 6, 4, 5, 4, 6])  # 正样本索引
tensor_c = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5, 5, 7, 7, 7, 7])  # 负样本锚点索引
tensor_d = torch.tensor([7, 5, 5, 4, 7, 7, 5, 7, 5, 1, 3, 1, 0, 2, 0, 3, 2, 1])  # 负样本索引

miner_outputs = (tensor_a, tensor_b, tensor_c, tensor_d)

print("=" * 80)
print("MultiSimilarityMiner 输出分析")
print("=" * 80)

print(f"\n输入数据:")
print(f"  descriptors 形状: {descriptors.shape}")
print(f"  labels: {labels.tolist()}")
print(f"  标签分布: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")

print(f"\nminer_outputs 结构:")
print(f"  tensor_a (锚点索引): {tensor_a.tolist()}")
print(f"  tensor_b (正样本索引): {tensor_b.tolist()}")
print(f"  tensor_c (负样本锚点索引): {tensor_c.tolist()}")
print(f"  tensor_d (负样本索引): {tensor_d.tolist()}")

print(f"\n各tensor长度:")
print(f"  正样本对数量: {len(tensor_a)} (锚点-正样本对)")
print(f"  负样本对数量: {len(tensor_c)} (负样本锚点-负样本对)")

print("\n" + "=" * 80)
print("数据流分析 - 正样本对 (Anchor-Positive Pairs)")
print("=" * 80)

print("\n正样本对详情 (前10个):")
for i in range(min(10, len(tensor_a))):
    anchor_idx = tensor_a[i].item()
    pos_idx = tensor_b[i].item()
    anchor_label = labels[anchor_idx].item()
    pos_label = labels[pos_idx].item()
    match = "✓" if anchor_label == pos_label else "✗"
    print(f"  对 {i+1}: 锚点[{anchor_idx}](标签={anchor_label}) - 正样本[{pos_idx}](标签={pos_label}) {match}")

print(f"\n正样本对统计:")
unique_anchors = torch.unique(tensor_a)
print(f"  唯一锚点数量: {len(unique_anchors)}")
print(f"  唯一锚点索引: {unique_anchors.tolist()}")

# 统计每个锚点对应的正样本数量
anchor_pos_count = {}
for anchor, pos in zip(tensor_a, tensor_b):
    anchor = anchor.item()
    if anchor not in anchor_pos_count:
        anchor_pos_count[anchor] = []
    anchor_pos_count[anchor].append(pos.item())

print(f"\n每个锚点的正样本分布:")
for anchor in sorted(anchor_pos_count.keys()):
    pos_list = anchor_pos_count[anchor]
    print(f"  锚点[{anchor}](标签={labels[anchor].item()}): {len(pos_list)}个正样本 {pos_list}")

print("\n" + "=" * 80)
print("数据流分析 - 负样本对 (Negative Anchor-Negative Pairs)")
print("=" * 80)

print("\n负样本对详情 (前10个):")
for i in range(min(10, len(tensor_c))):
    neg_anchor_idx = tensor_c[i].item()
    neg_idx = tensor_d[i].item()
    neg_anchor_label = labels[neg_anchor_idx].item()
    neg_label = labels[neg_idx].item()
    match = "✓" if neg_anchor_label != neg_label else "✗"
    print(f"  对 {i+1}: 负锚点[{neg_anchor_idx}](标签={neg_anchor_label}) - 负样本[{neg_idx}](标签={neg_label}) {match}")

print(f"\n负样本对统计:")
unique_neg_anchors = torch.unique(tensor_c)
print(f"  唯一负样本锚点数量: {len(unique_neg_anchors)}")
print(f"  唯一负样本锚点索引: {unique_neg_anchors.tolist()}")

# 统计每个负样本锚点对应的负样本数量
neg_anchor_neg_count = {}
for neg_anchor, neg in zip(tensor_c, tensor_d):
    neg_anchor = neg_anchor.item()
    if neg_anchor not in neg_anchor_neg_count:
        neg_anchor_neg_count[neg_anchor] = []
    neg_anchor_neg_count[neg_anchor].append(neg.item())

print(f"\n每个负样本锚点的负样本分布:")
for neg_anchor in sorted(neg_anchor_neg_count.keys()):
    neg_list = neg_anchor_neg_count[neg_anchor]
    print(f"  负锚点[{neg_anchor}](标签={labels[neg_anchor].item()}): {len(neg_list)}个负样本 {neg_list}")

print("\n" + "=" * 80)
print("MultiSimilarityLoss 计算原理")
print("=" * 80)

print("""
MultiSimilarityLoss 使用点积相似度 (DotProductSimilarity) 计算损失，公式为:

对于正样本对 (anchor, positive):
  L_pos = (1/α) * log(1 + Σ exp(-α * (S_ap - λ_pos)))
  
对于负样本对 (negative_anchor, negative):
  L_neg = (1/β) * log(1 + Σ exp(β * (S_an - λ_neg)))

总损失 = L_pos + L_neg

其中:
  - S_ap: 锚点与正样本的点积相似度
  - S_an: 负样本锚点与负样本的点积相似度
  - α = 1.0 (正样本对的权重参数)
  - β = 50 (负样本对的权重参数)
  - λ_pos: 正样本对的相似度阈值
  - λ_neg: 负样本对的相似度阈值
  - base = 0.0 (基础相似度偏移)

点积相似度计算:
  S = descriptors[i] @ descriptors[j]^T
""")

# 模拟计算点积相似度
print("\n模拟相似度计算 (使用随机descriptors):")
print("\n正样本对相似度示例 (前5个):")
for i in range(min(5, len(tensor_a))):
    anchor_idx = tensor_a[i].item()
    pos_idx = tensor_b[i].item()
    similarity = torch.dot(descriptors[anchor_idx], descriptors[pos_idx]).item()
    print(f"  锚点[{anchor_idx}] · 正样本[{pos_idx}] = {similarity:.4f}")

print("\n负样本对相似度示例 (前5个):")
for i in range(min(5, len(tensor_c))):
    neg_anchor_idx = tensor_c[i].item()
    neg_idx = tensor_d[i].item()
    similarity = torch.dot(descriptors[neg_anchor_idx], descriptors[neg_idx]).item()
    print(f"  负锚点[{neg_anchor_idx}] · 负样本[{neg_idx}] = {similarity:.4f}")

print("\n" + "=" * 80)
print("数据流总结")
print("=" * 80)

print("""
1. 输入阶段:
   - descriptors: [8, 6] - 8个样本，每个6维特征向量
   - labels: [8] - 8个标签

2. MultiSimilarityMiner 挖掘阶段 (使用余弦相似度):
   - 挖掘困难正样本对: 找到18个 (anchor, positive) 对
   - 挖掘困难负样本对: 找到18个 (negative_anchor, negative) 对
   - 返回 indices_tuple = (tensor_a, tensor_b, tensor_c, tensor_d)

3. MultiSimilarityLoss 计算阶段 (使用点积相似度):
   - 提取正样本对的特征: descriptors[tensor_a] 和 descriptors[tensor_b]
   - 提取负样本对的特征: descriptors[tensor_c] 和 descriptors[tensor_d]
   - 计算点积相似度矩阵
   - 应用 MultiSimilarityLoss 公式计算最终损失

4. 关键点:
   - Miner 使用余弦相似度筛选困难样本对
   - Loss 使用点积相似度计算损失值
   - 只使用挖掘出的困难样本对参与损失计算，提高训练效率
""")

print("\n" + "=" * 80)
print("验证标签匹配情况")
print("=" * 80)

# 验证正样本对的标签匹配
pos_matches = 0
for anchor, pos in zip(tensor_a, tensor_b):
    if labels[anchor] == labels[pos]:
        pos_matches += 1

print(f"正样本对标签匹配: {pos_matches}/{len(tensor_a)} ({100*pos_matches/len(tensor_a):.1f}%)")

# 验证负样本对的标签不匹配
neg_matches = 0
for neg_anchor, neg in zip(tensor_c, tensor_d):
    if labels[neg_anchor] != labels[neg]:
        neg_matches += 1

print(f"负样本对标签不匹配: {neg_matches}/{len(tensor_c)} ({100*neg_matches/len(tensor_c):.1f}%)")

