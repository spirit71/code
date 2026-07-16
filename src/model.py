# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import time

import torch
import lightning as L
from rich.console import Console
from pytorch_metric_learning import losses, miners

from src import utils


class BoQModel(L.LightningModule):
    def __init__(
        self,
        backbone,
        aggregator,
        lr=1e-4,
        lr_mul=0.1,
        weight_decay=1e-3,
        warmup_epochs=10,
        milestones=[10, 20],
        silent=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.lr = lr
        self.lr_mul = lr_mul
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.silent = silent

        # stage_recalls: 保存当前 epoch 的验证/测试 recall 结果。
        # stage_timings: 保存 train / val / test 三段耗时。
        # _stage_starts: 记录某个阶段开始时间，用于 perf_counter 计时。
        self.stage_recalls = {}
        self.stage_timings = {}
        self._stage_starts = {}
        # 如果开启了“每个 epoch 跑全测试”，验证阶段内部其实混有 val loader 和 test loader。
        # 这两个变量用来分别累计 val/test 两类 loader 的耗时。
        self._validation_batch_times = {}
        self._current_validation_batch_start = None
        self.forced_summary_epoch_index = None

        self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.0)
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)

    def configure_optimizers(self):
        # backbone 和 aggregator 一起训练。
        # 虽然当前 lr 一样，但拆成两个 param group，后续单独调 backbone 会更方便。
        optimizer_params = [
            {"params": self.backbone.parameters(), "lr": self.lr*self.lr_mul, "weight_decay": self.weight_decay},
            {"params": self.aggregator.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # warmup 阶段按 step 线性增大学习率。
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = (self.trainer.global_step + 1) / total_warmup_steps
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                initial_lr = pg.get("initial_lr", self.lr)
                pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)
        self.log("_LR", optimizer.param_groups[-1]["lr"], prog_bar=False, logger=True)

    @torch.compiler.disable()
    def compute_loss(self, descriptors, labels):
        mined_pairs = self.ms_miner(descriptors, labels)
        loss = self.ms_loss(descriptors, labels, mined_pairs)
        return loss

    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns

    def training_step(self, batch, batch_idx):
        # 训练 loader 输出形状是 [places, images_per_place, C, H, W]。
        # metric learning loss 需要平铺后的图像批次，所以这里把前两维合并。
        images, labels = batch
        images = images.flatten(0, 1)
        labels = labels.flatten()

        descriptors, attentions = self(images)
        loss = self.compute_loss(descriptors, labels)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        self._stage_starts["train"] = time.perf_counter()

    def on_train_epoch_end(self):
        # GSV-Cities 的训练 tuple 会按 epoch 重采样。
        # 这里刷新数据表，保证下一轮训练不是机械重复同一批 tuple。
        self.trainer.train_dataloader.dataset._refresh_dataframes()

    # 验证/测试阶段不算 loss，只需要把图片编码成 descriptor，
    # 后面统一做检索并计算 Recall@K。
    def _collect_descriptors(self, storage: dict, batch, dataloader_idx: int):
        images, _ = batch
        descriptors, _ = self(images)
        descriptors = descriptors.detach().cpu()
        storage.setdefault(dataloader_idx, []).append(descriptors)

    def _finalize_recall_stage(self, storage: dict, dataloaders, stage_name: str):
        recalls = {}
        for dataloader_idx, descriptors_list in storage.items():
            # 先把一个数据集所有 batch 的 descriptor 拼起来，
            # 再一次性送进 FAISS 计算 Recall@1/5/10/20。
            descriptors = torch.cat(descriptors_list, dim=0)
            dataset = dataloaders[dataloader_idx].dataset

            recalls_dict = utils.compute_recall_performance(
                descriptors,
                dataset.num_references,
                dataset.num_queries,
                dataset.ground_truth,
                k_values=[1, 5, 10, 20],
            )
            recalls[dataset.dataset_name] = recalls_dict

            # 验证集日志沿用原始命名风格；测试集日志显式带 test 前缀，避免混淆。
            if stage_name == "val":
                recalls_log = {
                    f"{dataset.dataset_name}/R@1": recalls_dict[1],
                    f"{dataset.dataset_name}/R@5": recalls_dict[5],
                }
            else:
                recalls_log = {
                    f"{dataset.dataset_name}/{stage_name}_R@1": recalls_dict[1],
                    f"{dataset.dataset_name}/{stage_name}_R@5": recalls_dict[5],
                }
            self.log_dict(recalls_log, prog_bar=False, logger=True)

        self.stage_recalls[stage_name] = recalls

        # 控制台表格仅用于阅读，不参与训练逻辑。
        if recalls and not self.silent:
            utils.display_recall_performance(
                list(recalls.values()),
                list(recalls.keys()),
                title=f"{stage_name.capitalize()} Recall@k Performance",
            )
        storage.clear()
        return recalls

    def _write_summary_report(self, epoch_index=None):
        if not self.stage_recalls:
            return
        logger = getattr(self.trainer, "logger", None)
        log_dir = getattr(logger, "log_dir", None)
        if not log_dir:
            return

        # 每轮评估结束后，把结果写成 json / md / xlsx / csv / png。
        if epoch_index is None:
            epoch_index = self._resolve_summary_epoch_index()

        summary_name = f"evaluation_summary_epoch_{epoch_index:02d}"
        saved_paths = utils.save_recall_summary(
            log_dir,
            summary_name,
            self.stage_recalls,
            self.stage_timings,
        )
        excel_path = utils.export_recall_summary_excel(saved_paths["json"])
        history_path = utils.update_recall_history(
            log_dir,
            epoch_index,
            self.stage_recalls,
            self.stage_timings,
        )
        plot_path = utils.save_recall_history_plot(log_dir)

        if not self.silent:
            utils.display_stage_timing(self.stage_timings)
            console = Console()
            console.print(f"[bold green]Saved evaluation summary:[/bold green] {saved_paths['md']}")
            if excel_path is not None:
                console.print(f"[bold green]Saved Excel summary:[/bold green] {excel_path}")
            console.print(f"[bold green]Updated history:[/bold green] {history_path}")
            if plot_path is not None:
                console.print(f"[bold green]Saved history plot:[/bold green] {plot_path}")

    def _resolve_summary_epoch_index(self):
        if self.forced_summary_epoch_index is not None:
            return max(int(self.forced_summary_epoch_index), 1)
        return max(int(self.current_epoch) + 1, 1)

    # 前 num_val_datasets 个 loader 属于验证集，后面的 loader 属于“周期性测试集”。
    def _num_val_dataloaders(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        return getattr(datamodule, "num_val_datasets", len(self.trainer.val_dataloaders))

    def _resolve_validation_stage_name(self, dataloader_idx: int) -> str:
        return "val" if dataloader_idx < self._num_val_dataloaders() else "test"

    def on_validation_epoch_start(self):
        self.validation_outputs = {}
        self.stage_recalls.pop("val", None)
        self.stage_recalls.pop("test", None)
        self.stage_timings.pop("val", None)
        self.stage_timings.pop("test", None)
        self._validation_batch_times = {"val": 0.0, "test": 0.0}
        self._current_validation_batch_start = None

        # train 阶段一结束，就在这里补上训练耗时。
        if "train" in self._stage_starts:
            self.stage_timings["train"] = time.perf_counter() - self._stage_starts.pop("train")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._current_validation_batch_start = time.perf_counter()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._collect_descriptors(self.validation_outputs, batch, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # 这里把验证阶段的 batch 耗时按 val/test 两类分别累计。
        if self._current_validation_batch_start is None:
            return
        stage_name = self._resolve_validation_stage_name(dataloader_idx)
        elapsed = time.perf_counter() - self._current_validation_batch_start
        self._validation_batch_times[stage_name] = self._validation_batch_times.get(stage_name, 0.0) + elapsed
        self._current_validation_batch_start = None

    def on_validation_epoch_end(self):
        if self.trainer.fast_dev_run:
            print("\nFast dev run: skipping recall@k computation\n")
            self.validation_outputs.clear()
            return

        # 如果打开了 --test-every-epoch，val_dataloaders 实际上是:
        # [验证集 loaders] + [测试集 loaders]
        # 这里再按数量拆回两组，分别记为 val 和 test。
        num_val = self._num_val_dataloaders()
        all_eval_dataloaders = list(self.trainer.val_dataloaders)
        val_dataloaders = all_eval_dataloaders[:num_val]
        test_dataloaders = all_eval_dataloaders[num_val:]

        val_outputs = {}
        test_outputs = {}
        for dataloader_idx, descriptors_list in self.validation_outputs.items():
            if dataloader_idx < num_val:
                val_outputs[dataloader_idx] = descriptors_list
            else:
                test_outputs[dataloader_idx - num_val] = descriptors_list

        if val_outputs:
            val_finalize_start = time.perf_counter()
            self._finalize_recall_stage(val_outputs, val_dataloaders, "val")
            self.stage_timings["val"] = self._validation_batch_times.get("val", 0.0) + (time.perf_counter() - val_finalize_start)

        if test_outputs:
            test_finalize_start = time.perf_counter()
            self._finalize_recall_stage(test_outputs, test_dataloaders, "test")
            self.stage_timings["test"] = self._validation_batch_times.get("test", 0.0) + (time.perf_counter() - test_finalize_start)

        self._write_summary_report(epoch_index=self._resolve_summary_epoch_index())

    def on_test_epoch_start(self):
        # 这是“独立 test 模式”用的，不是每轮 periodic test 用的。
        self.test_outputs = {}
        self.stage_recalls.pop("test", None)
        self.stage_timings.pop("test", None)
        self._stage_starts["test"] = time.perf_counter()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._collect_descriptors(self.test_outputs, batch, dataloader_idx)

    def on_test_epoch_end(self):
        if "test" in self._stage_starts:
            self.stage_timings["test"] = time.perf_counter() - self._stage_starts.pop("test")
        # 单独 test 模式下，也沿用同样的 summary/history 导出逻辑。
        self._finalize_recall_stage(self.test_outputs, self.trainer.test_dataloaders, "test")
        self._write_summary_report(epoch_index=self._resolve_summary_epoch_index())
