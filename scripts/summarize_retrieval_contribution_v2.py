#!/usr/bin/env python3
"""汇总多个 LOO contribution 诊断目录为统一 CSV/Markdown。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def masking_value(rows, strategy, ratio):
    values = [row["R@1"] for row in rows
              if row["strategy"] == strategy and float(row["mask_ratio"]) == ratio]
    return values[0] if values else ""


def random_mean(rows, ratio):
    values = [row["R@1"] for row in rows
              if row["strategy"] == "random" and float(row["mask_ratio"]) == ratio]
    return statistics.mean(values) if values else ""


def row_from_summary(path: Path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("experiment") != "LOO retrieval contribution diagnostic":
        return None
    correlations = summary["correlations"]
    contribution = summary["contribution"]
    rows = summary["masking_rows"]
    best = contribution.get("best", contribution)
    oracle_prefix = (
        "contribution_best" if any(
            row["strategy"].startswith("contribution_best") for row in rows
        ) else "contribution"
    )
    baseline_candidates = [row["R@1"] for row in rows if float(row["mask_ratio"]) == 0.0]
    return {
        "dataset": summary["dataset"],
        "model_variant": summary.get("model_variant", "e2"),
        "gate_mode": summary.get("gate_mode", "on"),
        "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
        "num_references": summary["num_references"],
        "num_queries": summary["num_queries"],
        "gt_positive_min": summary.get("gt_positive_count", {}).get("min", ""),
        "gt_positive_mean": summary.get("gt_positive_count", {}).get("mean", ""),
        "gt_positive_median": summary.get("gt_positive_count", {}).get("median", ""),
        "gt_positive_max": summary.get("gt_positive_count", {}).get("max", ""),
        "negative_contribution_ratio_best": best["negative_fraction"],
        "contribution_mean_best": best["mean"],
        "contribution_std_best": best["std"],
        "spearman_pred_repeat": correlations.get("pred_repeat", {}).get("per_image_mean", ""),
        "spearman_pred_contrib_best": correlations.get("pred_contrib_best", {}).get(
            "per_image_mean", correlations.get("pred_contrib", {}).get("per_image_mean", "")
        ),
        "spearman_repeat_contrib_best": correlations.get("repeat_contrib_best", {}).get(
            "per_image_mean", correlations.get("repeat_contrib", {}).get("per_image_mean", "")
        ),
        "spearman_contrib_best_hard": correlations.get("contrib_best_hard", {}).get(
            "per_image_mean", ""
        ),
        "R@1_baseline": baseline_candidates[0] if baseline_candidates else "",
        "R@1_random_25_mean": random_mean(rows, 0.25),
        "R@1_oracle_low_25": masking_value(rows, f"{oracle_prefix}_low", 0.25),
        "R@1_oracle_high_25": masking_value(rows, f"{oracle_prefix}_high", 0.25),
        "R@1_random_50_mean": random_mean(rows, 0.5),
        "R@1_oracle_low_50": masking_value(rows, f"{oracle_prefix}_low", 0.5),
        "R@1_oracle_high_50": masking_value(rows, f"{oracle_prefix}_high", 0.5),
        "oracle_uses_test_gt": summary.get("oracle_uses_test_gt", True),
        "deployable": summary.get("deployable", False),
        "elapsed_seconds": summary["elapsed_seconds"],
        "source": str(path),
    }


def main():
    args = parse_args()
    output_csv = args.output_csv or args.root / "combined_summary.csv"
    output_md = args.output_md or args.root / "combined_report.md"
    rows = [
        row for path in sorted(args.root.rglob("summary.json"))
        if (row := row_from_summary(path)) is not None
    ]
    if not rows:
        raise RuntimeError(f"no contribution summary.json found under {args.root}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Query Retrieval Contribution 统一汇总", "",
        "> Oracle 使用测试 GT，仅用于诊断上界，不是可部署结果。", "",
        "| Dataset | Model | Gate | Baseline R@1 | Neg C | Random25 | Oracle-low25 | "
        "Oracle-high25 | Random50 | Oracle-low50 | Oracle-high50 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        percent = lambda key: (
            "" if row[key] == "" else f"{100 * float(row[key]):.2f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['model_variant']} | {row['gate_mode']} | "
            f"{percent('R@1_baseline')} | {percent('negative_contribution_ratio_best')} | "
            f"{percent('R@1_random_25_mean')} | {percent('R@1_oracle_low_25')} | "
            f"{percent('R@1_oracle_high_25')} | {percent('R@1_random_50_mean')} | "
            f"{percent('R@1_oracle_low_50')} | {percent('R@1_oracle_high_50')} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows to {output_csv} and {output_md}")


if __name__ == "__main__":
    main()
