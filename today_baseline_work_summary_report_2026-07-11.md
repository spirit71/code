# BoQ Baseline复现实验工作汇报

## 1. 今日目标
基于原生 BoQ 代码，补齐完整 baseline 复现所需的测试、结果记录与可视化能力，保证训练过程可追踪、测试结果可导出、实验结论可复盘。

## 2. 今日完成内容

### 2.1 修复原始代码运行问题
- 修复了验证阶段在新版 NumPy 环境下的兼容性报错。
- 结果是模型现在可以正常跑完训练后验证阶段，不再因 `np.in1d` 接口问题中断。

### 2.2 在原生 BoQ 上补齐测试集评估能力
新增并接入了 6 套测试集：
- `pitts30k-test`
- `nordland`
- `sped`
- `amstertime`
- `tokyo247`
- `svox-all`

同时根据不同 benchmark 的标注方式，补齐了 3 类 ground truth 逻辑：
- 同名精确配对型
- 序列容忍型
- 坐标半径型

### 2.3 增强训练入口与实验控制能力
在 `python train.py` 主入口上增加了显式参数：
- `--no-test`
- `--test-only`
- `--checkpoint`
- `--quick`
- `--test-every-epoch`

其中 `--test-every-epoch` 支持“每训练完一个 epoch，就自动跑一次验证集 + 全部测试集”。

### 2.4 增强实验结果记录与可视化能力
每轮评估后自动保存：
- `evaluation_summary_epoch_XX.json`
- `evaluation_summary_epoch_XX.md`
- `evaluation_summary_epoch_XX.xlsx`
- `evaluation_history.csv`
- `evaluation_history_r1.png`

这样可以同时满足：
- 结构化保存结果
- 人工快速查看结果
- Excel 汇总整理
- 按 epoch 画曲线做实验分析

### 2.5 增强训练过程可解释性
补充了关键文件的中文注释，重点解释：
- 训练 / 验证 / 测试流程
- 不同测试集的匹配逻辑
- 每轮测试结果如何保存
- 为什么 `--test-every-epoch` 不直接在训练环里硬插 `trainer.test()`

## 3. 关键代码改动位置
- `train.py`
  - 新增测试参数与周期性测试入口
- `src/dataloaders/datamodule.py`
  - 新增测试集装载与每轮评估拼接逻辑
- `src/dataloaders/test_datasets.py`
  - 新增 3 类测试 ground truth 实现
- `src/model.py`
  - 新增每轮 val/test 结果汇总、阶段计时、历史记录导出
- `src/utils.py`
  - 新增 Recall 汇总保存、Excel 自动导出、历史曲线记录
- `scripts/export_eval_summary_to_excel.py`
  - 新增结果导出为 Excel 的独立脚本

## 4. 当前可直接使用的正式命令
```bash
python train.py --test-every-epoch 2>&1 | tee baseline_test_every_epoch.log
```

## 5. 当前产出能力
当前代码已经支持：
- 原生 BoQ 完整训练
- 验证集评估
- 6 套测试集评估
- 每个 epoch 自动保存结果
- 自动导出 Excel
- 自动保存跨 epoch 历史记录与曲线图
- 新手友好中文注释

## 6. 结果意义
这次改动后，BoQ baseline 不再只是“能训练”，而是具备了完整实验闭环：
- 能跑
- 能测
- 能记录
- 能导出
- 能复盘

对于后续 baseline 对齐、消融实验、结果汇总和论文表格整理，都会明显更方便。

## 7. 后续建议
下一步建议按以下顺序推进：
1. 先用 `--test-every-epoch` 跑完整 baseline。
2. 记录 `version_x` 下每轮与最终结果。
3. 对照论文或官方结果，判断是否达到复现目标。
4. 若存在差距，再进入超参数、训练轮数或实现细节层面的排查。
