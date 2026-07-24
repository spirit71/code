#!/usr/bin/env python3
"""从 QASSR 运行日志导出 4 实验 x 6 数据集的 Recall 对比长表。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATASETS = (
    "amstertime",
    "sped",
    "tokyo247",
    "pitts30k-test",
    "nordland",
    "svox-all",
)
EXPERIMENTS = (
    ("all529_mutual", "全量529-token Mutual"),
    ("attention64_feature", "Attention Top-64 + Feature-only"),
    ("attention64_product_role", "Attention Top-64 + Feature×Role"),
    ("attention64_combined_gate", "Attention Top-64 + Feature×Role + Combined Gate"),
)
RECALL_K = (1, 5, 10, 20)
MARKER = re.compile(r"\[QASSR\] dataset=(\S+) experiment=(\S+)")


def extract_records(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = list(MARKER.finditer(text))
    decoder = json.JSONDecoder()
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        chunk = text[marker.end() : end]
        pos = 0
        result = None
        while True:
            start = chunk.find("{", pos)
            if start < 0:
                break
            try:
                candidate, consumed = decoder.raw_decode(chunk[start:])
                pos = start + consumed
            except json.JSONDecodeError:
                pos = start + 1
                continue
            if isinstance(candidate, dict) and {
                "baseline_recall",
                "reranked_recall",
            }.issubset(candidate):
                result = candidate
                break
        if result is not None:
            yield marker.group(1), marker.group(2), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/run_logs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("VPR多数据集reranking四实验六数据集Recall对比.csv"),
    )
    args = parser.parse_args()

    records = {}
    for log_path in sorted(args.log_dir.glob("*.log")):
        for dataset, full_name, result in extract_records(log_path):
            for suffix, _ in EXPERIMENTS:
                if full_name == f"{dataset}_{suffix}":
                    key = (dataset, suffix)
                    if key in records:
                        raise RuntimeError(f"重复实验结果: {key}")
                    records[key] = (result, log_path)

    expected = {(dataset, suffix) for dataset in DATASETS for suffix, _ in EXPERIMENTS}
    missing = sorted(expected - records.keys())
    if missing:
        raise RuntimeError(f"缺少 {len(missing)} 组实验结果: {missing}")

    # 同一数据集的四个实验必须使用完全一致的 baseline，防止混用 checkpoint。
    for dataset in DATASETS:
        baselines = [records[(dataset, suffix)][0]["baseline_recall"] for suffix, _ in EXPERIMENTS]
        if any(baseline != baselines[0] for baseline in baselines[1:]):
            raise RuntimeError(f"{dataset} 的四个实验 baseline 不一致")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "实验",
        "实验说明",
        "数据集",
        "指标",
        "Baseline(%)",
        "实验结果(%)",
        "相对Baseline变化(pp)",
        "结果格式_实验值(Δpp)",
        "来源日志",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for suffix, description in EXPERIMENTS:
            for dataset in DATASETS:
                result, log_path = records[(dataset, suffix)]
                for k in RECALL_K:
                    baseline = 100.0 * float(result["baseline_recall"][str(k)])
                    reranked = 100.0 * float(result["reranked_recall"][str(k)])
                    delta_pp = reranked - baseline
                    writer.writerow(
                        {
                            "实验": suffix,
                            "实验说明": description,
                            "数据集": dataset,
                            "指标": f"R@{k}",
                            "Baseline(%)": f"{baseline:.6f}",
                            "实验结果(%)": f"{reranked:.6f}",
                            "相对Baseline变化(pp)": f"{delta_pp:+.6f}",
                            "结果格式_实验值(Δpp)": f"{reranked:.2f} ({delta_pp:+.2f})",
                            "来源日志": str(log_path.resolve()),
                        }
                    )
    print(f"已导出: {args.output.resolve()}")
    print(f"记录数: {len(EXPERIMENTS) * len(DATASETS) * len(RECALL_K)}")


if __name__ == "__main__":
    main()
