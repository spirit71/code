# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from typing import List, Dict
from pathlib import Path
import csv
import json

import numpy as np
import faiss

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box


def compute_recall_performance(
    descriptors,
    num_references,
    num_queries,
    ground_truth,
    k_values=[1, 5, 10, 20],
):
    """使用 FAISS 计算某个数据集的 Recall@K。"""

    assert num_references + num_queries == len(
        descriptors
    ), "Number of references and queries do not match the number of descriptors. THERE IS A BUG!"

    embed_size = descriptors.shape[1]
    faiss_index = faiss.IndexFlatL2(embed_size)

    # 前 num_references 个 descriptor 视为 reference gallery。
    faiss_index.add(descriptors[:num_references])

    # 后 num_queries 个 descriptor 视为 query，去 gallery 里做最近邻检索。
    _, predictions = faiss_index.search(descriptors[num_references:], max(k_values))

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # 这里用 np.isin 而不是老的 np.in1d，兼容新版 NumPy。
            if np.any(np.isin(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    return {k: v for (k, v) in zip(k_values, correct_at_k)}


def format_duration(seconds: float | None) -> str:
    """把秒数格式化成 HH:MM:SS，方便控制台和 summary 展示。"""
    if seconds is None:
        return "N/A"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def display_recall_performance(
    recalls_list: List[Dict[int, float]],
    val_set_names: List[str],
    title: str = "Recall@k Performance",
) -> None:
    # 纯控制台展示，不影响训练逻辑。
    if not recalls_list:
        return
    console = Console()
    console.print("\n")
    table = Table(title=None, box=box.SIMPLE, header_style="bold")
    k_values = list(recalls_list[0].keys())

    table.add_column("Dataset", justify="left")
    for k in k_values:
        table.add_column(f"R@{k}", justify="center")

    for i, recalls in enumerate(recalls_list):
        table.add_row(val_set_names[i], *[f"{100 * v:.2f}" for v in recalls.values()])

    console.print(Panel(table, expand=False, title=title))
    console.print("\n")


def display_stage_timing(stage_timings: Dict[str, float]) -> None:
    """在控制台打印 train / val / test 三段耗时。"""
    if not stage_timings:
        return
    console = Console()
    table = Table(title=None, box=box.SIMPLE, header_style="bold")
    table.add_column("Stage", justify="left")
    table.add_column("Seconds", justify="right")
    table.add_column("HH:MM:SS", justify="center")

    ordered_names = [stage for stage in ["train", "val", "test"] if stage in stage_timings]
    ordered_names.extend(sorted(stage for stage in stage_timings if stage not in ordered_names))

    for stage_name in ordered_names:
        seconds = stage_timings[stage_name]
        table.add_row(stage_name, f"{seconds:.2f}", format_duration(seconds))

    console.print(Panel(table, expand=False, title="Stage Timing"))
    console.print("\n")


def save_recall_summary(
    output_dir: str | Path,
    summary_name: str,
    stage_recalls: Dict[str, Dict[str, Dict[int, float]]],
    stage_timings: Dict[str, float] | None = None,
) -> Dict[str, Path]:
    # 同一份评估信息同时保存成 json 和 md:
    # json 给脚本读，md 给人直接看。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_timings = stage_timings or {}

    serializable = {
        stage: {
            dataset: {f"R@{k}": float(v) for k, v in recalls.items()}
            for dataset, recalls in dataset_recalls.items()
        }
        for stage, dataset_recalls in stage_recalls.items()
    }
    serializable["timings"] = {
        stage_name: {
            "seconds": float(seconds),
            "hhmmss": format_duration(seconds),
        }
        for stage_name, seconds in stage_timings.items()
    }

    json_path = output_dir / f"{summary_name}.json"
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

        existing_timings = existing.get("timings", {})
        new_timings = serializable.get("timings", {})
        existing.update({k: v for k, v in serializable.items() if k != "timings"})
        existing["timings"] = {**existing_timings, **new_timings}
        serializable = existing

    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    merged_timings = serializable.get("timings", {})

    md_lines = ["# Evaluation Summary", ""]
    if merged_timings:
        md_lines.extend([
            "## Timing",
            "",
            "| Stage | Seconds | HH:MM:SS |",
            "| --- | ---: | --- |",
        ])
        ordered_names = [stage for stage in ["train", "val", "test"] if stage in merged_timings]
        ordered_names.extend(sorted(stage for stage in merged_timings if stage not in ordered_names))
        for stage_name in ordered_names:
            seconds = merged_timings[stage_name]["seconds"]
            hhmmss = merged_timings[stage_name]["hhmmss"]
            md_lines.append(f"| {stage_name} | {seconds:.2f} | {hhmmss} |")
        md_lines.append("")

    for stage, dataset_recalls in serializable.items():
        if stage == "timings":
            continue
        md_lines.append(f"## {stage.capitalize()}")
        md_lines.append("")
        md_lines.append("| Dataset | R@1 | R@5 | R@10 | R@20 |")
        md_lines.append("| --- | ---: | ---: | ---: | ---: |")
        for dataset, recalls in dataset_recalls.items():
            md_lines.append(
                f"| {dataset} | "
                f"{100 * recalls.get('R@1', 0.0):.2f} | "
                f"{100 * recalls.get('R@5', 0.0):.2f} | "
                f"{100 * recalls.get('R@10', 0.0):.2f} | "
                f"{100 * recalls.get('R@20', 0.0):.2f} |"
            )
        md_lines.append("")

    md_path = output_dir / f"{summary_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"json": json_path, "md": md_path}


def export_recall_summary_excel(json_path: str | Path) -> Path | None:
    # 这里复用独立的 Excel 导出脚本，让每个 epoch 自动多产出一个 xlsx 文件。
    json_path = Path(json_path)
    try:
        from scripts.export_eval_summary_to_excel import export_summary
    except Exception:
        return None

    output_path = json_path.with_suffix(".xlsx")
    export_summary(json_path, output_path)
    return output_path


def update_recall_history(
    output_dir: str | Path,
    epoch_index: int,
    stage_recalls: Dict[str, Dict[str, Dict[int, float]]],
    stage_timings: Dict[str, float] | None = None,
) -> Path:
    # evaluation_history.csv 是跨 epoch 的长表，
    # 后面画曲线、做对比、导出报表都靠它。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "evaluation_history.csv"
    stage_timings = stage_timings or {}

    fieldnames = [
        "epoch",
        "stage",
        "dataset",
        "R@1",
        "R@5",
        "R@10",
        "R@20",
        "seconds",
        "hhmmss",
    ]

    existing_rows = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))

    # 同一个 epoch / stage / dataset 如果重复写入，就用新结果覆盖旧结果，避免 CSV 越堆越乱。
    replacement_keys = set()
    new_rows = []
    for stage_name, dataset_recalls in stage_recalls.items():
        for dataset_name, recalls in dataset_recalls.items():
            replacement_keys.add((str(epoch_index), stage_name, dataset_name))
            new_rows.append({
                "epoch": str(epoch_index),
                "stage": stage_name,
                "dataset": dataset_name,
                "R@1": f"{100 * recalls.get(1, 0.0):.6f}",
                "R@5": f"{100 * recalls.get(5, 0.0):.6f}",
                "R@10": f"{100 * recalls.get(10, 0.0):.6f}",
                "R@20": f"{100 * recalls.get(20, 0.0):.6f}",
                "seconds": f"{stage_timings.get(f'{stage_name}:{dataset_name}', stage_timings.get(stage_name, 0.0)):.6f}",
                "hhmmss": format_duration(stage_timings.get(f'{stage_name}:{dataset_name}', stage_timings.get(stage_name, 0.0))),
            })

    kept_rows = [
        row for row in existing_rows
        if (row.get("epoch", ""), row.get("stage", ""), row.get("dataset", "")) not in replacement_keys
    ]
    all_rows = kept_rows + new_rows
    all_rows.sort(key=lambda row: (int(row["epoch"]), row["stage"], row["dataset"]))

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return history_path


def save_recall_history_plot(output_dir: str | Path) -> Path | None:
    # 如果环境里有 matplotlib，就自动把 history.csv 画成 R@1 曲线图。
    # 如果没有，也不报错，直接跳过。
    output_dir = Path(output_dir)
    history_path = output_dir / "evaluation_history.csv"
    if not history_path.exists():
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    grouped = {}
    for row in rows:
        stage_name = row["stage"]
        dataset_name = row["dataset"]
        grouped.setdefault(stage_name, {}).setdefault(dataset_name, []).append(
            (int(row["epoch"]), float(row["R@1"]))
        )

    ordered_stages = [stage for stage in ["val", "test"] if stage in grouped]
    if not ordered_stages:
        return None

    fig, axes = plt.subplots(len(ordered_stages), 1, figsize=(12, 4 * len(ordered_stages)), constrained_layout=True)
    if len(ordered_stages) == 1:
        axes = [axes]

    for ax, stage_name in zip(axes, ordered_stages):
        for dataset_name, points in sorted(grouped[stage_name].items()):
            points = sorted(points, key=lambda item: item[0])
            epochs = [epoch for epoch, _ in points]
            r1_scores = [score for _, score in points]
            ax.plot(epochs, r1_scores, marker="o", linewidth=2, label=dataset_name)
        ax.set_title(f"{stage_name.capitalize()} R@1 over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R@1 (%)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)

    plot_path = output_dir / "evaluation_history_r1.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def display_datasets_stats(datamodule):
    # 这里只负责把训练/验证/测试数据规模打印得更直观。
    console = Console()
    console.print("\n")

    train_dataset = datamodule.train_dataset
    train_table = Table(box=None, show_header=False)
    train_table.add_column("Setting", justify="left", no_wrap=True)
    train_table.add_column("Value", style="green")

    train_table.add_row("Number of cities", str(len(train_dataset.cities)))
    train_table.add_row("Number of places", str(len(train_dataset)))
    train_table.add_row("Number of images", str(train_dataset.total_nb_images))

    train_panel = Panel(train_table, title="[bold]Training Dataset Stats[/bold]", padding=(1, 2), expand=False)
    console.print(train_panel)

    config_table = Table(title=None, title_justify="center", box=None, show_header=False)
    config_table.add_column("Setting", justify="left", no_wrap=True)
    config_table.add_column("Value", style="green")

    config_table.add_row("Iterations per epoch", str(len(datamodule.train_dataset) // datamodule.batch_size))
    config_table.add_row("Train batch size (PxK)", f"{datamodule.batch_size}x{datamodule.img_per_place}")
    config_table.add_row("Training image size", f"{datamodule.train_img_size[0]}x{datamodule.train_img_size[1]}")
    config_table.add_row("Validation image size", f"{datamodule.val_img_size[0]}x{datamodule.val_img_size[1]}")
    config_panel = Panel(config_table, title="[bold]Training Configuration[/bold]", padding=(1, 2), expand=False)
    console.print(config_panel)

    val_tree = Tree("Validation Datasets", hide_root=True)
    for val_set in datamodule.val_datasets:
        val_branch = val_tree.add(f"{val_set.dataset_name}")
        val_branch.add(f"Queries    [green]{val_set.num_queries}[/green]")
        val_branch.add(f"References [green]{val_set.num_references}[/green]")

    tree_panel = Panel(val_tree, title="[bold]Validation Datasets[/bold]", padding=(1, 2), expand=False)
    console.print(tree_panel)

    # 如果当前流程里包含测试集，也一起打印，避免你看日志时不知道这轮测了什么。
    if getattr(datamodule, "test_datasets", None):
        test_tree = Tree("Test Datasets", hide_root=True)
        for test_set in datamodule.test_datasets:
            test_branch = test_tree.add(f"{test_set.dataset_name}")
            test_branch.add(f"Queries    [green]{test_set.num_queries}[/green]")
            test_branch.add(f"References [green]{test_set.num_references}[/green]")

        test_panel = Panel(test_tree, title="[bold]Test Datasets[/bold]", padding=(1, 2), expand=False)
        console.print(test_panel)
