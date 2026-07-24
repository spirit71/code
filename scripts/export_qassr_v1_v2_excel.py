#!/usr/bin/env python
"""把 QASSR V1/V2 已落盘结果汇总成一个可继续编辑的 Excel 工作簿。

数据来源仅限每个实验目录中的 summary.json 和 resolved_config.json；
脚本不会根据日志中的打印文本猜测或补齐不存在的实验。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = Path("/root/data/qassr_outputs")
V2_ROOT = Path("/root/data/qassr_outputs_v2")
OUTPUT = PROJECT_ROOT / "2026-07-23_QASSR_V1_V2_完整实验结果.xlsx"

DATASETS = [
    "amstertime",
    "sped",
    "tokyo247",
    "pitts30k-test",
    "nordland",
    "svox-all",
]
KS = ("1", "5", "10", "20")

V2_MAIN_NAME = {
    dataset: f"v2_{dataset}_attention64_residual025_spatial"
    for dataset in DATASETS
}

# 第一版同一类配置在部分数据集的目录命名不完全一致，按优先级解析。
V1_METHOD_CANDIDATES = {
    "V1 all529 mutual": ["{d}_all529_mutual"],
    "V1 attention64 feature": ["{d}_attention64_feature", "{d}_attention64"],
    "V1 attention64 product-role": [
        "{d}_attention64_product_role",
        "{d}_attention64_all_query",
    ],
    "V1 attention64 combined-gate": ["{d}_attention64_combined_gate"],
}


def load_experiments(root: Path, version: str) -> list[dict[str, Any]]:
    """读取一个版本全部成功落盘的实验。"""
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        experiment = summary_path.parents[1].name
        dataset = summary_path.parent.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "resolved_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        rows.append(
            {
                "version": version,
                "experiment": experiment,
                "dataset": dataset,
                "summary": summary,
                "config": config,
                "summary_path": summary_path,
            }
        )
    return rows


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["experiment"], row["dataset"]): row for row in rows}


def metric(row: dict[str, Any], group: str, k: str) -> float:
    """Excel 内使用百分数数值，例如 0.6556 写成 65.56。"""
    return float(row["summary"][group][k]) * 100.0


def delta(row: dict[str, Any], k: str) -> float:
    return metric(row, "reranked_recall", k) - metric(row, "baseline_recall", k)


NAVY = "17365D"
BLUE = "5B9BD5"
LIGHT_BLUE = "D9EAF7"
LIGHT_ORANGE = "FCE4D6"
LIGHT_GREEN = "E2F0D9"
LIGHT_GRAY = "E7E6E6"
WHITE = "FFFFFF"
THIN_GRAY = Side(style="thin", color="A6A6A6")


def title(ws, text: str, width: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=width)
    cell = ws.cell(1, 1, text)
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.font = Font(color=WHITE, bold=True, size=14)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26


def style_header(cell, fill=BLUE) -> None:
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.font = Font(color=WHITE, bold=True)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = Border(left=THIN_GRAY, right=THIN_GRAY, top=THIN_GRAY, bottom=THIN_GRAY)


def style_body(ws, min_row: int, max_row: int, max_col: int) -> None:
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            cell.border = Border(
                left=THIN_GRAY,
                right=THIN_GRAY,
                top=THIN_GRAY,
                bottom=THIN_GRAY,
            )
            cell.alignment = Alignment(vertical="center", wrap_text=True)


def autosize(ws, minimum=10, maximum=38) -> None:
    for column in range(1, ws.max_column + 1):
        length = max(
            len(str(ws.cell(row, column).value or ""))
            for row in range(1, ws.max_row + 1)
        )
        ws.column_dimensions[get_column_letter(column)].width = min(
            max(length + 2, minimum),
            maximum,
        )


def add_wide_header(ws, start_row: int) -> None:
    ws.cell(start_row, 1, "实验")
    ws.cell(start_row, 2, "配置说明")
    for cell in ws[start_row][:2]:
        style_header(cell)
    column = 3
    for dataset in DATASETS:
        ws.merge_cells(
            start_row=start_row,
            start_column=column,
            end_row=start_row,
            end_column=column + 3,
        )
        ws.cell(start_row, column, dataset)
        style_header(ws.cell(start_row, column))
        for offset, k in enumerate(KS):
            ws.cell(start_row + 1, column + offset, f"R@{k}")
            style_header(ws.cell(start_row + 1, column + offset), fill=NAVY)
        column += 4
    ws.merge_cells(
        start_row=start_row,
        start_column=1,
        end_row=start_row + 1,
        end_column=1,
    )
    ws.merge_cells(
        start_row=start_row,
        start_column=2,
        end_row=start_row + 1,
        end_column=2,
    )
    ws.cell(start_row, 1).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(start_row, 2).alignment = Alignment(horizontal="center", vertical="center")


def build_main_sheet(wb: Workbook, all_index: dict[tuple[str, str], dict[str, Any]]) -> None:
    ws = wb.active
    ws.title = "主实验"
    width = 2 + 4 * len(DATASETS)
    title(ws, "QASSR-V2 六测试集统一主配置结果", width)
    ws.cell(
        2,
        1,
        "固定配置：Top-50 + X2 + Attention Top-64 + Residual Role(0.25) "
        "+ Translation-Inlier Spatial(0.15) + Global/Local 0.7/0.3 + All-Query",
    )
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=width)
    ws.cell(2, 1).fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    ws.cell(2, 1).alignment = Alignment(wrap_text=True)
    add_wide_header(ws, 4)

    rows = [
        ("BoQ Baseline", "固定 archive-epoch[17].ckpt；FAISS IndexFlatL2"),
        (
            "QASSR-V2",
            "Attention64 + ResidualRole(0.25) + TranslationInlier(0.15) + All-Query",
        ),
        ("Δ V2-Baseline", "百分点（percentage points）"),
    ]
    for row_no, (name, description) in enumerate(rows, start=6):
        ws.cell(row_no, 1, name)
        ws.cell(row_no, 2, description)
        ws.cell(row_no, 1).font = Font(bold=True)
        if name == "QASSR-V2":
            fill = LIGHT_GREEN
        elif name.startswith("Δ"):
            fill = LIGHT_ORANGE
        else:
            fill = LIGHT_GRAY
        for cell in ws[row_no]:
            cell.fill = PatternFill("solid", fgColor=fill)

    column = 3
    for dataset in DATASETS:
        row = all_index[(V2_MAIN_NAME[dataset], dataset)]
        for offset, k in enumerate(KS):
            ws.cell(6, column + offset, metric(row, "baseline_recall", k))
            ws.cell(7, column + offset, metric(row, "reranked_recall", k))
            ws.cell(8, column + offset, delta(row, k))
            for row_no in (6, 7, 8):
                ws.cell(row_no, column + offset).number_format = "0.00"
        column += 4
    style_body(ws, 4, 8, width)
    ws.freeze_panes = "C6"
    ws.auto_filter.ref = f"A4:{get_column_letter(width)}8"
    autosize(ws)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 55


def build_v2_ablation_sheet(
    wb: Workbook,
    v2_rows: list[dict[str, Any]],
) -> None:
    ws = wb.create_sheet("V2消融实验")
    headers = [
        "类型",
        "数据集",
        "实验",
        "Token选择",
        "Token数",
        "Matching",
        "Role权重",
        "Spatial",
        "Spatial Sigma",
        "Gate",
        "R@1",
        "R@5",
        "R@10",
        "R@20",
        "ΔR@1",
        "ΔR@5",
        "ΔR@10",
        "ΔR@20",
        "Fixed",
        "NewError",
        "Gate覆盖率",
        "局部匹配秒",
        "Summary路径",
    ]
    title(ws, "QASSR-V2 主实验与消融实验", len(headers))
    for column, name in enumerate(headers, start=1):
        ws.cell(3, column, name)
        style_header(ws.cell(3, column))

    ordered = sorted(
        v2_rows,
        key=lambda row: (
            DATASETS.index(row["dataset"]) if row["dataset"] in DATASETS else 99,
            row["experiment"],
        ),
    )
    for row_no, row in enumerate(ordered, start=4):
        config = row["config"]
        summary = row["summary"]
        is_main = row["experiment"] == V2_MAIN_NAME.get(row["dataset"])
        values = [
            "六测试集主实验" if is_main else "SPED消融",
            row["dataset"],
            row["experiment"],
            config.get("token_selection"),
            config.get("token_count"),
            config.get("matching"),
            config.get("role_weight"),
            config.get("spatial"),
            config.get("spatial_sigma"),
            config.get("gate"),
            *[metric(row, "reranked_recall", k) for k in KS],
            *[delta(row, k) for k in KS],
            summary["transitions"]["fixed"],
            summary["transitions"]["new_error"],
            float(summary["rerank_coverage"]) * 100.0,
            float(summary["latency"]["local_matching_seconds"]),
            str(row["summary_path"]),
        ]
        for column, value in enumerate(values, start=1):
            ws.cell(row_no, column, value)
        fill = LIGHT_GREEN if is_main else LIGHT_ORANGE
        for cell in ws[row_no]:
            cell.fill = PatternFill("solid", fgColor=fill)
        for column in range(11, 19):
            ws.cell(row_no, column).number_format = "0.00"
        ws.cell(row_no, 21).number_format = "0.00"
        ws.cell(row_no, 22).number_format = "0.00"
    style_body(ws, 3, ws.max_row, len(headers))
    ws.freeze_panes = "K4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{ws.max_row}"
    autosize(ws)
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["W"].width = 75


def resolve_v1_row(
    all_index: dict[tuple[str, str], dict[str, Any]],
    dataset: str,
    candidates: list[str],
) -> dict[str, Any] | None:
    for template in candidates:
        experiment = template.format(d=dataset)
        row = all_index.get((experiment, dataset))
        if row is not None:
            return row
    return None


def build_comparison_sheet(
    wb: Workbook,
    all_index: dict[tuple[str, str], dict[str, Any]],
) -> None:
    ws = wb.create_sheet("V1_V2对比")
    width = 2 + 4 * len(DATASETS)
    title(ws, "QASSR V1/V2 多数据集 R@K 对比", width)
    add_wide_header(ws, 3)

    methods: list[tuple[str, str, Any]] = [
        ("BoQ Baseline", "原始Global检索", "baseline"),
        *[
            (label, label, candidates)
            for label, candidates in V1_METHOD_CANDIDATES.items()
        ],
        (
            "QASSR-V2",
            "Attention64 + ResidualRole(0.25) + TranslationInlier(0.15)",
            "v2",
        ),
    ]
    for row_no, (label, description, selector) in enumerate(methods, start=5):
        ws.cell(row_no, 1, label)
        ws.cell(row_no, 2, description)
        ws.cell(row_no, 1).font = Font(bold=True)
        fill = (
            LIGHT_GREEN
            if selector == "v2"
            else LIGHT_GRAY
            if selector == "baseline"
            else LIGHT_BLUE
        )
        for cell in ws[row_no]:
            cell.fill = PatternFill("solid", fgColor=fill)

        column = 3
        for dataset in DATASETS:
            if selector == "baseline":
                row = all_index[(V2_MAIN_NAME[dataset], dataset)]
                group = "baseline_recall"
            elif selector == "v2":
                row = all_index[(V2_MAIN_NAME[dataset], dataset)]
                group = "reranked_recall"
            else:
                row = resolve_v1_row(all_index, dataset, selector)
                group = "reranked_recall"
            for offset, k in enumerate(KS):
                if row is None:
                    ws.cell(row_no, column + offset, "—")
                else:
                    ws.cell(row_no, column + offset, metric(row, group, k))
                    ws.cell(row_no, column + offset).number_format = "0.00"
            column += 4
    style_body(ws, 3, ws.max_row, width)
    ws.freeze_panes = "C5"
    ws.auto_filter.ref = f"A3:{get_column_letter(width)}{ws.max_row}"
    autosize(ws)
    ws.column_dimensions["A"].width = 31
    ws.column_dimensions["B"].width = 55


def build_all_results_sheet(
    wb: Workbook,
    all_rows: list[dict[str, Any]],
) -> None:
    ws = wb.create_sheet("全部原始结果")
    headers = [
        "版本",
        "数据集",
        "实验",
        "R@1",
        "R@5",
        "R@10",
        "R@20",
        "Baseline R@1",
        "Baseline R@5",
        "Baseline R@10",
        "Baseline R@20",
        "ΔR@1",
        "ΔR@5",
        "ΔR@10",
        "ΔR@20",
        "Token选择",
        "Token数",
        "Matching",
        "Role权重",
        "Spatial",
        "Gate",
        "Fixed",
        "NewError",
        "Summary路径",
    ]
    title(ws, "V1/V2 全部成功落盘实验（真实 summary.json）", len(headers))
    for column, name in enumerate(headers, start=1):
        ws.cell(3, column, name)
        style_header(ws.cell(3, column))

    ordered = sorted(
        all_rows,
        key=lambda row: (
            row["version"],
            row["dataset"],
            row["experiment"],
        ),
    )
    for row_no, row in enumerate(ordered, start=4):
        summary, config = row["summary"], row["config"]
        values = [
            row["version"],
            row["dataset"],
            row["experiment"],
            *[metric(row, "reranked_recall", k) for k in KS],
            *[metric(row, "baseline_recall", k) for k in KS],
            *[delta(row, k) for k in KS],
            config.get("token_selection"),
            config.get("token_count"),
            config.get("matching"),
            config.get("role_weight"),
            config.get("spatial"),
            config.get("gate"),
            summary["transitions"]["fixed"],
            summary["transitions"]["new_error"],
            str(row["summary_path"]),
        ]
        for column, value in enumerate(values, start=1):
            ws.cell(row_no, column, value)
        fill = LIGHT_GREEN if row["version"] == "V2" else LIGHT_BLUE
        for cell in ws[row_no]:
            cell.fill = PatternFill("solid", fgColor=fill)
        for column in range(4, 16):
            ws.cell(row_no, column).number_format = "0.00"
    style_body(ws, 3, ws.max_row, len(headers))
    ws.freeze_panes = "D4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(headers))}{ws.max_row}"
    autosize(ws)
    ws.column_dimensions["C"].width = 52
    ws.column_dimensions["X"].width = 75


def validate_workbook(path: Path, expected_all_rows: int) -> None:
    """重新打开文件，避免只验证内存中的 Workbook。"""
    wb = load_workbook(path, data_only=False, read_only=True)
    required = {"主实验", "V2消融实验", "V1_V2对比", "全部原始结果"}
    if set(wb.sheetnames) != required:
        raise RuntimeError(f"Unexpected sheets: {wb.sheetnames}")
    # “全部原始结果”前3行是标题/空行/表头。
    actual = wb["全部原始结果"].max_row - 3
    if actual != expected_all_rows:
        raise RuntimeError(f"Expected {expected_all_rows} raw rows, got {actual}")
    for dataset in DATASETS:
        expected = V2_MAIN_NAME[dataset]
        if not any(
            wb["主实验"].cell(row, 1).value == "QASSR-V2"
            for row in range(1, wb["主实验"].max_row + 1)
        ):
            raise RuntimeError(f"Missing main experiment row for {expected}")
    wb.close()


def main() -> None:
    v1_rows = load_experiments(V1_ROOT, "V1")
    v2_rows = load_experiments(V2_ROOT, "V2")
    if len(v1_rows) != 47 or len(v2_rows) != 10:
        raise RuntimeError(
            f"Expected 47 V1 and 10 V2 summaries, got {len(v1_rows)} and {len(v2_rows)}"
        )
    all_rows = v1_rows + v2_rows
    all_index = index_rows(all_rows)
    for dataset in DATASETS:
        key = (V2_MAIN_NAME[dataset], dataset)
        if key not in all_index:
            raise RuntimeError(f"Missing V2 main result: {key}")

    wb = Workbook()
    build_main_sheet(wb, all_index)
    build_v2_ablation_sheet(wb, v2_rows)
    build_comparison_sheet(wb, all_index)
    build_all_results_sheet(wb, all_rows)
    wb.save(OUTPUT)
    validate_workbook(OUTPUT, len(all_rows))
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "v1_experiments": len(v1_rows),
                "v2_experiments": len(v2_rows),
                "total_experiments": len(all_rows),
                "sheets": ["主实验", "V2消融实验", "V1_V2对比", "全部原始结果"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
