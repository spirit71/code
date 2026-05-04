# DINOv3 适配说明与验证手册

## 1. 目标
在不改动 BoQ 主体训练与评测逻辑的前提下，将 `dinov3` 作为 `backbone` 接入，并保持与原有 `dinov2` 一致的接口契约：

- 输入 `x: [B, 3, H, W]`
- 输出特征图 `feat: [B, C, H', W']`
- 暴露 `out_channels` 供 BoQ 聚合器构建
- 支持冻结前层 + 解冻最后 `N` 个 transformer blocks

---

## 2. 代码改动总览

### 2.1 `src/backbones.py`
新增一个共享基类 `_DinoBackbone`，抽象了 DINO 系列通用逻辑：

- 统一 `torch.hub.load(...)` 加载
- 统一模型合法性检查（`AVAILABLE_MODELS`）
- 统一冻结策略（先全冻结，再解冻最后 `unfreeze_n_blocks`）
- 统一 `patch_size` 读取与输出 reshape

在此基础上实现：

- `DinoV2(_DinoBackbone)`：`REPO_OR_DIR = "facebookresearch/dinov2"`
- `DinoV3(_DinoBackbone)`：`REPO_OR_DIR = "facebookresearch/dinov3"`

### 2.2 `train.py`

- 导入 `DinoV3`
- 在 backbone 路由中增加 `elif "dinov3" in hparams.backbone_name`
- 增加 `hparams.dino_weights` 与 CLI 参数 `--dino_weights`

这样可以直接使用：

```bash
python train.py --backbone dinov3_vitb16
python train.py --backbone dinov3_vitb16 --dino_weights <weights_name>
```

---

## 3. 适配原理

### 3.1 为什么要抽 `_DinoBackbone`
`dinov2` 和 `dinov3` 在训练代码所需接口上高度一致，核心差异是 hub repo 与模型名。抽基类可以：

- 避免重复实现 forward/freeze/reshape
- 保证两者行为一致，方便对照验证
- 后续扩展新 DINO 变体时改动最小

### 3.2 输出维度如何对齐 BoQ
BoQ 需要卷积式特征图输入。DINO 原始 token 输出会移除 CLS token 后，按 `patch_size` reshape 回 `[B, C, H/ps, W/ps]`：

- `out_channels = embed_dim`
- `patch_size` 来自 `self.dino.patch_embed.patch_size`
- `reshape_output=True` 时自动还原二维空间结构

### 3.3 冻结与解冻策略如何保持一致

- 默认先冻结全部参数
- 仅解冻最后 `unfreeze_n_blocks` 个 transformer blocks
- `unfreeze_n_blocks` 做范围校验，避免越界

该策略与旧版 `dinov2` 训练行为一致，保证迁移可控。

---

## 4. 如何验证 `dinov3` 与 `dinov2` 适配一致

以下步骤按“从快到慢”执行。

### 4.1 语法与导入检查

```bash
python -m py_compile src/backbones.py train.py
```

期望：无报错。

### 4.2 Backbone 加载与输出 shape 对照

```bash
python - <<'PY'
import torch
from src.backbones import DinoV2, DinoV3
# 1. 自动检测并使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 创建输入并指定设备
x = torch.randn(2, 3, 322, 322, device=device)
# 3. 加载模型并迁移到设备
b2 = DinoV2("dinov2_vitb14", unfreeze_n_blocks=2).to(device)
# 4. 前向传播
y2 = b2(x)
print("dinov2:", y2.shape, "out_channels:", b2.out_channels, "patch:", b2.patch_size)

b3 = DinoV3("dinov3_vitb16", unfreeze_n_blocks=2).to(device)
y3 = b3(x)
print("dinov3:", y3.shape, "out_channels:", b3.out_channels, "patch:", b3.patch_size)
PY
```

检查点：

- 两者均成功前向
- 输出均为 4D 张量 `[B, C, H', W']`
- `out_channels` 与输出 `C` 一致

### 4.3 解冻策略检查

```bash
python - <<'PY'
from src.backbones import DinoV2, DinoV3

def trainable_ratio(m):
    t = sum(p.numel() for p in m.parameters())
    r = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return r, t, r / t

for cls, name in [(DinoV2, "dinov2_vitb14"), (DinoV3, "dinov3_vitb16")]:
    m = cls(name, unfreeze_n_blocks=2)
    r, t, ratio = trainable_ratio(m)
    print(cls.__name__, "trainable:", r, "total:", t, "ratio:", round(ratio, 6))
PY
```

检查点：

- 可训练参数占比明显小于 1（说明冻结生效）
- 改大 `unfreeze_n_blocks` 后占比上升

### 4.4 端到端 smoke test（推荐）

```bash
python train.py --backbone dinov2_vitb14 --dev
python train.py --backbone dinov3_vitb16 --dev
```

检查点：

- 两条命令都能走通一个最小训练/验证迭代
- 不出现维度或 dataloader 结构错误

### 4.5 测试流程回归（可选）

```bash
python train.py --backbone dinov3_vitb16 --test_only --ckpt_path <your_ckpt>
```

检查点：

- `trainer.test` 可执行
- Recall 指标能正常输出

---

## 5. 差异与注意事项

- `dinov2` 常见 patch size 为 14，`dinov3` 常见为 16，输出空间分辨率会不同，这是预期行为。
- 当前 `train_img_size/val_img_size` 沿用 DINO 分支默认配置；若后续观察到性能敏感，可单独为 `dinov3` 调参。
- 若 hub 入口需要显式权重参数，使用 `--dino_weights` 传入。

---

## 6. 结论

本次适配通过“统一骨干抽象 + 最小训练入口改动”实现了 `dinov3` 的低侵入接入。  
验证重点是三件事：

- 接口契约一致（输入输出、`out_channels`、reshape）
- 训练行为一致（冻结/解冻策略）
- 端到端流程可跑（`--dev` 与 `--test_only`）
