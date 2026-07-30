#!/usr/bin/env python3
"""汇总第三轮 query utility mechanism 分析。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.analysis_root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        correlations = summary["correlations"]
        groups = {
            bool(item["baseline_top1_correct"]): item
            for item in summary["correct_wrong"]
        }
        wrong, correct = groups[False], groups[True]
        rows.append({
            "dataset": summary["dataset"],
            "model_variant": summary["model_variant"],
            "gate_mode": summary["gate_mode"],
            "redundancy_top5_contribution_per_image": correlations[
                "redundancy_top5_vs_contribution"
            ]["per_image_mean"],
            "redundancy_top5_contribution_pooled": correlations[
                "redundancy_top5_vs_contribution"
            ]["pooled"],
            "margin_negative_ratio_spearman": correlations[
                "margin_vs_negative_query_ratio"
            ]["pooled"],
            "sign_switching_index_ratio": summary[
                "query_indices_with_sign_switching_ratio"
            ],
            "harmful_negative_similarity_dominant_ratio": summary[
                "harmful_dominant_mechanism_ratio"
            ].get("negative_similarity", 0.0),
            "wrong_num_images": wrong["num_images"],
            "wrong_negative_query_ratio": wrong["negative_query_ratio"],
            "wrong_mean_negative_contribution": wrong["mean_negative_contribution"],
            "correct_num_images": correct["num_images"],
            "correct_negative_query_ratio": correct["negative_query_ratio"],
            "correct_mean_negative_contribution": correct["mean_negative_contribution"],
            "source": str(path),
        })
    if not rows:
        raise RuntimeError("no summary.json files found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "combined_mechanism_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Query Utility Mechanism 统一汇总", "",
        "> 使用测试 GT contribution，仅用于机制诊断，不可部署。", "",
        "| Dataset | Model | Gate | Redundancy-C per-image | Margin-NegRatio | "
        "Sign-switch | Neg-sim dominant | Wrong harmful | Correct harmful |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model_variant']} | {row['gate_mode']} | "
            f"{row['redundancy_top5_contribution_per_image']:.4f} | "
            f"{row['margin_negative_ratio_spearman']:.4f} | "
            f"{100*row['sign_switching_index_ratio']:.2f}% | "
            f"{100*row['harmful_negative_similarity_dominant_ratio']:.2f}% | "
            f"{100*row['wrong_negative_query_ratio']:.2f}% | "
            f"{100*row['correct_negative_query_ratio']:.2f}% |"
        )
    md_path = args.output_root / "combined_mechanism_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
