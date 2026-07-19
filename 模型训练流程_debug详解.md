# BoQ 模型训练流程 Debug 详解

本文基于当前仓库 `/home/code_qy_7_28/01_BoQ_inva_260716` 的真实代码整理，按“像 debug 一样从入口往下走”的方式说明模型训练流程、模块流通、函数调用、输入输出、数据变化、特征训练和 loss 约束。

## 1. 从哪里开始

程序入口是 `train.py` 的 `if __name__ == "__main__"`。

运行：

```bash
python train.py
```

以后执行顺序是：

```text
train.py __main__
  -> parse_args()
  -> hparams = HyperParams()
  -> 用命令行参数覆盖默认超参数
  -> train(hparams, ...)
```

默认超参数在 `train.py` 的 `HyperParams`：

```text
backbone_name = dinov2_vitb14
unfreeze_n_blocks = 2
channel_proj = 512
num_queries = 64
num_layers = 2
output_dim = 8192
batch_size = 128
img_per_place = 4
max_epochs = 60
lr = 1e-4
```

所以默认训练的是：

```text
DINOv2 backbone + BoQ aggregator + MultiSimilarity metric learning loss
```

## 2. train() 里先搭模型

核心函数是 `train.py` 的 `train(hparams, ...)`。

第一步固定随机种子：

```python
seed_everything(hparams.seed, workers=True)
```

意义：让数据采样、worker、初始化尽量可复现。

然后根据 `backbone_name` 选择主干网络：

```text
如果是 dinov2:
  backbone = DinoV2(...)
  train_img_size = 224 x 224
  val_img_size = 322 x 322

如果是 resnet:
  backbone = ResNet(...)
  train_img_size = 320 x 320
  val_img_size = 384 x 384
```

当前默认是 `dinov2_vitb14`，所以会创建 DINOv2 backbone。

然后创建 BoQ 聚合器：

```python
aggregator = BoQ(
    in_channels=backbone.out_channels,
    proj_channels=512,
    num_queries=64,
    num_layers=2,
    row_dim=8192 // 512,
)
```

这里：

```text
row_dim = 16
最终输出维度 = proj_channels * row_dim = 512 * 16 = 8192
```

再把 backbone 和 aggregator 包进 Lightning 模型：

```python
model = BoQModel(backbone, aggregator, ...)
```

整体模型结构是：

```text
输入图片
  -> Backbone(DINOv2 或 ResNet)
  -> BoQ 聚合器
  -> 8192 维全局 descriptor
  -> loss / recall
```

## 3. 数据从 DataModule 进入

数据模块是 `src/dataloaders/datamodule.py` 的 `VPRDataModule`。

在 `train.py` 中创建：

```python
datamodule = VPRDataModule(
    gsv_cities_path=...,
    img_per_place=4,
    val_sets={msls-val, pitts30k-val},
    test_sets={pitts30k-test, nordland, sped, ...},
    batch_size=128,
)
```

它管理三类 loader：

```text
train_dataloader()  -> GSV-Cities 训练集
val_dataloader()    -> MSLS-val + Pitts30k-val
test_dataloader()   -> Pitts30k-test / Nordland / SPED / AmsterTime / Tokyo / SVOX
```

训练 transform：

```text
Resize
RandAugment
ToDtype float32
Normalize(ImageNet mean/std)
```

验证/测试 transform：

```text
Resize
ToDtype float32
Normalize
```

区别是：训练有随机增强，验证/测试没有随机增强，保证评估稳定。

## 4. 训练数据长什么样

训练集是 `src/dataloaders/gsv_cities.py` 的 `GSVCitiesDataset`。

这个数据集非常关键：它不是分类任务那种“一张图一个类别”，而是“一次取一个地点的多张图”。

`__getitem__()` 做的事：

```text
输入 index
  -> 找到一个 place_id
  -> 从这个地点随机采样 img_per_place 张图片，默认 4 张
  -> 读图
  -> transform
  -> 返回:
       imgs:   [4, 3, H, W]
       labels: [place_id, place_id, place_id, place_id]
```

DataLoader 再把多个地点拼成 batch。默认 `batch_size=128`，所以一个训练 batch 是：

```text
images: [128, 4, 3, 224, 224]
labels: [128, 4]
```

含义：

```text
128 个地点
每个地点 4 张图
同一个地点的 4 张图 label 相同
不同地点 label 不同
```

这就是后面 metric learning loss 的基础：同地点图片应该近，不同地点图片应该远。

## 5. Lightning 开始训练

真正启动训练：

```python
trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)
```

Lightning 会自动调用：

```text
datamodule.setup("fit")
datamodule.train_dataloader()
model.configure_optimizers()
每个 batch 调 model.training_step()
每个 epoch 后调 validation_step()
```

## 6. training_step 的数据变化

训练核心是 `src/model.py` 的 `BoQModel.training_step()`。

输入 batch：

```text
images: [places, images_per_place, C, H, W]
labels: [places, images_per_place]
```

默认就是：

```text
images: [128, 4, 3, 224, 224]
labels: [128, 4]
```

代码先 flatten：

```python
images = images.flatten(0, 1)
labels = labels.flatten()
```

变成：

```text
images: [512, 3, 224, 224]
labels: [512]
```

为什么要 flatten？

因为 backbone 一次处理的是普通图片 batch：

```text
[B, C, H, W]
```

而不是：

```text
[地点数, 每地点图片数, C, H, W]
```

flatten 后仍然保留 label 关系：

```text
前 4 张可能 label 都是地点 A
再 4 张 label 都是地点 B
...
```

然后进入模型：

```python
descriptors, attentions = self(images)
```

这里 `self(images)` 会调用 `BoQModel.forward()`：

```python
x = self.backbone(x)
x, attns = self.aggregator(x)
```

## 7. Backbone 做什么

默认 backbone 是 `src/backbones.py` 的 `DinoV2`。

初始化时会：

```text
加载 DINOv2 预训练权重
冻结大部分参数
只解冻最后 unfreeze_n_blocks 个 transformer block
```

默认 `unfreeze_n_blocks=2`，意思是：

```text
前面大部分 DINOv2 保持预训练能力
最后 2 个 block 参与 VPR 任务微调
```

forward 里输入：

```text
x: [512, 3, 224, 224]
```

DINOv2 patch size 是 14，所以图片会被切成：

```text
224 / 14 = 16
16 * 16 = 256 个 patch
```

DINOv2 内部会有 CLS token，所以中间 token 类似：

```text
[512, 257, 768]
```

然后代码去掉 CLS token：

```python
x = x[:, 1:]
```

剩下 patch token：

```text
[512, 256, 768]
```

再 reshape 回空间特征图：

```text
[512, 768, 16, 16]
```

这就是 backbone 输出。意义是：

```text
每张图片不再是 RGB 像素，而是一张 16x16 的高级语义特征图；
每个位置有 768 维特征。
```

## 8. BoQ 聚合器做什么

BoQ 在 `src/boq.py`。

它的功能是：

```text
把 backbone 的局部特征图 [B, C, H, W]
压缩成一个全局图像描述子 [B, output_dim]
```

默认输入：

```text
x: [512, 768, 16, 16]
```

第一步 3x3 卷积投影通道：

```python
x = self.proj_c(x)
```

从：

```text
[512, 768, 16, 16]
```

变成：

```text
[512, 512, 16, 16]
```

第二步 flatten 空间位置：

```python
x = x.flatten(2).permute(0, 2, 1)
```

变成：

```text
[512, 256, 512]
```

含义：

```text
512 张图
每张图 256 个 patch token
每个 token 512 维
```

然后进入多个 `BoQBlock`。默认 `num_layers=2`，所以跑 2 层。

每个 `BoQBlock` 里有一组可学习 query：

```text
queries: [1, num_queries, 512]
默认 num_queries = 64
```

每张图复制一份 query：

```text
q: [512, 64, 512]
```

然后做 cross attention：

```text
query 去“询问”图片的 256 个 patch token
找出对地点识别有用的局部区域
```

每层输出：

```text
out: [512, 64, 512]
attn: attention map
```

两层拼起来：

```text
out: [512, 128, 512]
```

然后：

```python
out = self.fc(out.permute(0, 2, 1))
```

`out.permute(0, 2, 1)`：

```text
[512, 512, 128]
```

`fc` 把 `128` 压到 `row_dim=16`：

```text
[512, 512, 16]
```

最后 flatten：

```text
[512, 8192]
```

再 L2 normalize：

```python
normalize(out, p=2, dim=-1)
```

最终 descriptor：

```text
descriptors: [512, 8192]
```

意义是：

```text
每张图变成一个 8192 维向量；
向量方向代表地点特征；
相似地点的向量应该距离近，不同地点距离远。
```

## 9. loss 怎么约束特征

loss 在 `src/model.py`：

```python
self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.0)
self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)
```

训练时调用：

```python
mined_pairs = self.ms_miner(descriptors, labels)
loss = self.ms_loss(descriptors, labels, mined_pairs)
```

输入是：

```text
descriptors: [512, 8192]
labels:      [512]
```

`MultiSimilarityMiner` 先挖难样本：

```text
正样本对:
  label 相同，比如同一个 place_id 的不同图片

负样本对:
  label 不同，比如不同地点的图片
```

它不会平均看所有 pair，而是找更有训练价值的 pair：

```text
难正样本: 明明同地点，但 descriptor 距离还比较远
难负样本: 明明不同地点，但 descriptor 距离却比较近
```

`MultiSimilarityLoss` 的约束可以白话理解为：

```text
同一个地点的图片 descriptor 要靠近
不同地点的图片 descriptor 要分开
特别是那些“容易混淆”的 pair，要重点惩罚
```

所以这个模型不是学分类器：

```text
不是输出“这是第 123 类”
```

而是学检索特征：

```text
输出一个向量；
以后拿 query 向量去图库里找最近的 reference 向量。
```

loss 返回后：

```python
return loss
```

Lightning 自动做：

```text
loss.backward()
optimizer.step()
scheduler.step()
```

## 10. 优化器怎么更新

优化器在 `src/model.py` 的 `configure_optimizers()`。

它分两组参数：

```text
backbone 参数:
  lr = lr * lr_mul = 1e-4 * 0.1 = 1e-5

aggregator 参数:
  lr = lr = 1e-4
```

意义：

```text
预训练 DINOv2 少动一点，避免破坏通用视觉特征；
新加的 BoQ 聚合器多学一点。
```

优化器是：

```text
AdamW
```

学习率策略：

```text
前 warmup_epochs 线性 warmup
到 milestone=[10,20] 时 MultiStepLR 衰减
```

所以训练早期学习率慢慢升，避免一开始梯度太猛把预训练特征冲坏。

## 11. 每个 epoch 结束做什么

训练 epoch 结束时调用：

```python
self.trainer.train_dataloader.dataset._refresh_dataframes()
```

GSV-Cities 训练集内部每次会重新 shuffle dataframe。意义是：

```text
下一轮同一个 place 可能采到不同图片组合；
让模型见到更多同地点视角变化；
减少机械重复。
```

## 12. 验证阶段怎么走

验证不算 loss。核心是 `BoQModel.validation_step()`。

每个 val batch：

```python
self._collect_descriptors(...)
```

里面做：

```python
images, _ = batch
descriptors, _ = self(images)
descriptors = descriptors.detach().cpu()
storage[dataloader_idx].append(descriptors)
```

也就是说验证阶段只做：

```text
图片 -> backbone -> BoQ -> descriptor
```

不做：

```text
loss.backward()
optimizer.step()
```

等一个验证集全部 batch 跑完，汇总：

```text
把所有 batch 的 descriptor cat 起来
调用 compute_recall_performance()
算 R@1 / R@5 / R@10 / R@20
```

## 13. 验证/测试数据的顺序非常重要

验证集和测试集 dataset 都把图片组织成：

```text
image_paths = reference images + query images
```

也就是：

```text
前 num_references 张是图库 reference
后 num_queries 张是查询 query
```

比如 MSLS-val / Pitts30k-val 会读取：

```text
dbImages
qImages
ground_truth
```

测试集也类似。

这点很重要，因为 recall 函数默认：

```text
descriptors[:num_references] 是图库
descriptors[num_references:] 是 query
```

如果 dataset 顺序错了，Recall 就会错。

## 14. Recall@K 怎么算

Recall 函数在 `src/utils.py` 的 `compute_recall_performance()`。

输入：

```text
descriptors:      [num_references + num_queries, dim]
num_references:   图库数量
num_queries:      查询数量
ground_truth:     每个 query 对应哪些 reference 算正确
k_values:         [1, 5, 10, 20]
```

流程：

```text
1. 创建 FAISS IndexFlatL2
2. 把 reference descriptors 加入索引
3. 用 query descriptors 去搜索最近邻
4. 对每个 query:
     看 top-1 是否命中 ground_truth -> R@1
     看 top-5 是否命中 ground_truth -> R@5
     看 top-10 是否命中 ground_truth -> R@10
     看 top-20 是否命中 ground_truth -> R@20
5. 所有 query 平均
```

白话：

```text
如果 query 图片的正确地点出现在模型检索返回的前 K 个结果中，
这个 query 在 R@K 上就算成功。
```

## 15. 不同测试集 ground truth 不一样

测试集构造在 `src/dataloaders/datamodule.py` 的 `TEST_DATASET_BUILDERS`。

当前代码明确区分不同 benchmark：

```text
pitts30k-test:
  CoordinateRadiusTestDataset
  25 米半径内 reference 算正确

tokyo247:
  CoordinateRadiusTestDataset
  25 米半径内算正确

svox-all:
  CoordinateRadiusTestDataset
  25 米半径内算正确，可以丢掉无正样本 query

nordland:
  NordlandSequenceDataset
  query 第 i 帧，对 reference [i-10, i+10] 都算正确

sped / amstertime:
  PairedNameTestDataset
  query 文件名和 reference 文件名相同才算正确
```

所以 Recall 的意义不是所有数据集都一样，它依赖每个 dataset 的 `ground_truth` 定义。

## 16. checkpoint 怎么保存

checkpoint callback 在 `train.py`。

主 checkpoint 监控：

```text
msls-val/R@1
```

保存策略：

```text
save_top_k = 3
save_last = True
mode = max
```

也就是：

```text
按 msls-val 的 R@1 选最好的 3 个 checkpoint；
另外保存最后一个 checkpoint。
```

文件名里会带：

```text
epoch
R@1
R@5
R@10
R@20
```

还有一个 `epoch_checkpointing`，每个 epoch 都保存：

```text
archive-epoch[xx].ckpt
```

这个是为了后面 `--test-every-epoch` 可以逐 epoch 回测。

## 17. 训练结束后最终怎样

正常训练结束后：

```text
trainer.fit()
  -> 每个 epoch 训练 + 验证
  -> 保存 top checkpoint

然后如果没有 --no-test:
  trainer.test(..., ckpt_path="best")
```

默认 `test_ckpt_path="best"`，所以最终测试用的是：

```text
验证集 msls-val/R@1 最好的 checkpoint
```

测试阶段输出：

```text
pitts30k-test R@1/R@5/R@10/R@20
nordland R@1/R@5/R@10/R@20
sped R@1/R@5/R@10/R@20
amstertime R@1/R@5/R@10/R@20
tokyo247 R@1/R@5/R@10/R@20
svox-all R@1/R@5/R@10/R@20
```

并写入：

```text
logs/<backbone>/version_x/evaluation_summary_epoch_xx.json
logs/<backbone>/version_x/evaluation_summary_epoch_xx.md
logs/<backbone>/version_x/evaluation_summary_epoch_xx.xlsx
logs/<backbone>/version_x/evaluation_history.csv
logs/<backbone>/version_x/evaluation_history_r1.png
```

## 18. 一句话总流程

```text
python train.py
  -> 读取默认参数和命令行参数
  -> 创建 DINOv2 backbone
  -> 创建 BoQ aggregator
  -> 包成 BoQModel
  -> 创建 VPRDataModule
  -> GSV-Cities 输出 [地点, 每地点多图]
  -> training_step flatten 成普通图片 batch
  -> backbone 提取局部特征图
  -> BoQ 用 learnable queries 聚合成 8192 维 descriptor
  -> MultiSimilarityMiner 找难正/负样本
  -> MultiSimilarityLoss 拉近同地点、推远不同地点
  -> AdamW 更新 backbone 最后几层和 BoQ
  -> 每个 epoch 用 val set 编码 descriptor
  -> FAISS 检索算 Recall@K
  -> 按 msls-val/R@1 保存最优 checkpoint
  -> 训练结束用 best checkpoint 跑 test set
  -> 导出 summary/history/曲线
```

最核心的理解是：这个模型训练的不是“分类标签”，而是“地点检索向量”。训练时用同地点多图的 label 关系约束 descriptor 空间；测试时用 query descriptor 去 reference gallery 里找最近邻，看正确地点是否出现在 Top-K。
