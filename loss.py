from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
# 使用点积相似度作为损失函数的距离度量
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
## 使用余弦相似度作为挖掘器的距离度量
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        #miner = MultiSimilarityMiner(  多相似度损失
        #   (distance): CosineSimilarity()
        # ),
        miner_outputs = miner(descriptors, labels)  ## 返回(indices_tuple)
#         print(f"锚点数量: {len(miner_outputs[0])}")
        # 锚点数量: 841
        # print(f"正样本对数量: {len(miner_outputs[1])}")
        # 正样本对数量: 841
        # print(f"负样本对数量: {len(miner_outputs[2])}")
        # 负样本对数量: 28294
        # print(f"负样本锚点: {len(miner_outputs[3])}")
        # 负样本锚点: 28294
        loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        # 计算无效样本对的比例(不贡献损失的样本)
        nb_samples = descriptors.shape[0]  #nb_samples=288
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss