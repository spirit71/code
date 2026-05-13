#!/usr/bin/env python
"""Batch-evaluate all checkpoints in one run/version and build ONE consolidated comparison report.

Example:
python scripts/eval_all_checkpoints.py \
  --run_name dinov3_vitb16 \
  --version version_6 \
  --backbone dinov3_vitb16
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _parse_epoch_from_name(name: str) -> int:
    try:
        start = name.index("epoch[") + len("epoch[")
        end = name.index("]", start)
        return int(name[start:end])
    except Exception:
        return 10**9


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _collect_reports(report_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for p in report_dir.glob("*.json"):
        d = _load_json(p)
        if d is None:
            continue
        d["_report_path"] = str(p)
        reports.append(d)
    return reports


def _select_latest_report_for_ckpt(
    reports: list[dict[str, Any]],
    run_name: str,
    version: str,
    ckpt_path: Path,
) -> dict[str, Any] | None:
    ckpt_resolved = str(ckpt_path.resolve())
    candidates: list[dict[str, Any]] = []
    for rep in reports:
        rep_run = rep.get("run", {}).get("run_name")
        rep_ver = rep.get("run", {}).get("version")
        rep_ckpt = rep.get("checkpoint", {}).get("resolved_ckpt_path")
        if rep_run == run_name and rep_ver == version and rep_ckpt == ckpt_resolved:
            candidates.append(rep)
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.get("generated_at_utc", ""), reverse=True)
    return candidates[0]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def _extract_datasets_and_ks(reports_for_ckpts: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    datasets = set()
    ks = set()
    for rep in reports_for_ckpts:
        ds_recalls = rep.get("evaluation", {}).get("dataset_recalls", {}) or {}
        for ds_name, recalls in ds_recalls.items():
            datasets.add(ds_name)
            for k in recalls.keys():
                try:
                    ks.add(int(k))
                except Exception:
                    continue
    return sorted(datasets), sorted(ks)


def _get_recall(ds_recalls: dict[str, Any], dataset: str, k: int) -> float | None:
    recalls = ds_recalls.get(dataset, {})
    return _safe_float(recalls.get(k, recalls.get(str(k))))


def _build_rows(
    reports_for_ckpts: list[dict[str, Any]],
    datasets: list[str],
    ks: list[int],
    primary_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in reports_for_ckpts:
        ckpt = rep.get("checkpoint", {})
        parsed = ckpt.get("parsed", {})
        ds_recalls = rep.get("evaluation", {}).get("dataset_recalls", {}) or {}
        flat = rep.get("evaluation", {}).get("flat_metrics", {}) or {}

        dataset_table: dict[str, dict[str, float | None]] = {}
        primary_values: list[float] = []

        for ds in datasets:
            dataset_table[ds] = {}
            for k in ks:
                v = _get_recall(ds_recalls, ds, k)
                dataset_table[ds][f"R@{k}"] = v
                if k == primary_k and v is not None:
                    primary_values.append(v)

        all_values = [_safe_float(v) for v in flat.values()]
        all_values = [v for v in all_values if v is not None]

        row = {
            "checkpoint_path": ckpt.get("resolved_ckpt_path"),
            "checkpoint_file": parsed.get("filename"),
            "epoch": parsed.get("epoch_from_name"),
            "generated_at_utc": rep.get("generated_at_utc"),
            "avg_primary": _mean(primary_values),
            "avg_all_metrics": _mean(all_values),
            "dataset_table": dataset_table,
            "report_json": rep.get("_report_path"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r.get("avg_primary") is not None, r.get("avg_primary") or -1.0), reverse=True)
    return rows


def _best_by_dataset(rows: list[dict[str, Any]], datasets: list[str], ks: list[int]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        best[ds] = {}
        for k in ks:
            metric_name = f"R@{k}"
            winner = None
            winner_val = None
            for row in rows:
                v = row.get("dataset_table", {}).get(ds, {}).get(metric_name)
                if v is None:
                    continue
                if winner is None or float(v) > float(winner_val):
                    winner = row
                    winner_val = float(v)
            best[ds][metric_name] = {
                "value": winner_val,
                "checkpoint_file": winner.get("checkpoint_file") if winner else None,
                "epoch": winner.get("epoch") if winner else None,
            }
    return best


def _build_summary(
    run_name: str,
    version: str,
    reports_for_ckpts: list[dict[str, Any]],
    primary_k: int,
) -> dict[str, Any]:
    datasets, ks = _extract_datasets_and_ks(reports_for_ckpts)
    rows = _build_rows(reports_for_ckpts, datasets, ks, primary_k)
    best_checkpoint = rows[0] if rows else None

    summary = {
        "schema_version": 2,
        "generated_at_utc": _utc_now_iso(),
        "run_name": run_name,
        "version": version,
        "primary_k": primary_k,
        "rank_metric": f"avg(R@{primary_k}) across datasets",
        "datasets": datasets,
        "k_values": ks,
        "num_checkpoints": len(rows),
        "best_checkpoint_by_rank_metric": best_checkpoint,
        "best_by_dataset": _best_by_dataset(rows, datasets, ks),
        "rows": rows,
    }
    return summary


def _write_summary_files(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _sanitize_filename(f"batch_compare_{summary['run_name']}_{summary['version']}")
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    datasets = summary["datasets"]
    ks = summary["k_values"]
    rows = summary["rows"]

    lines: list[str] = []
    lines.append("# Batch Checkpoint Comparison")
    lines.append("")
    lines.append(f"- Generated (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Run: `{summary['run_name']}/{summary['version']}`")
    lines.append(f"- Rank metric: `{summary['rank_metric']}`")
    lines.append(f"- Num checkpoints: `{summary['num_checkpoints']}`")
    lines.append("")

    best = summary.get("best_checkpoint_by_rank_metric")
    if best:
        lines.append("## Best Checkpoint (by average generalization)")
        lines.append("")
        lines.append(f"- Checkpoint: `{best.get('checkpoint_file')}`")
        lines.append(f"- Epoch: `{best.get('epoch')}`")
        lines.append(f"- avg rank metric: `{best.get('avg_primary')}`")
        lines.append("")

    lines.append("## Ranking by Average")
    lines.append("")
    lines.append("| Rank | Epoch | Checkpoint | avg(rank) | avg(all metrics) |")
    lines.append("|---|---:|---|---:|---:|")
    for i, row in enumerate(rows, start=1):
        avg_primary = row.get("avg_primary")
        avg_all = row.get("avg_all_metrics")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    str(row.get("epoch")),
                    str(row.get("checkpoint_file")),
                    f"{avg_primary*100.0:.2f}" if avg_primary is not None else "NA",
                    f"{avg_all*100.0:.2f}" if avg_all is not None else "NA",
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Dataset-wise Recall@k (Checkpoint × Dataset)")
    lines.append("")
    for ds in datasets:
        lines.append(f"### {ds}")
        lines.append("")
        header = ["Rank", "Epoch", "Checkpoint"] + [f"R@{k}" for k in ks]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for i, row in enumerate(rows, start=1):
            ds_row = row.get("dataset_table", {}).get(ds, {})
            vals = []
            for k in ks:
                v = ds_row.get(f"R@{k}")
                vals.append(f"{float(v)*100.0:.2f}" if v is not None else "NA")
            base_cols = [str(i), str(row.get("epoch")), str(row.get("checkpoint_file"))]
            lines.append("| " + " | ".join(base_cols + vals) + " |")
        lines.append("")

    lines.append("## Best Checkpoint per Dataset/Recall")
    lines.append("")
    lines.append("| Dataset | Metric | Value | Epoch | Checkpoint |")
    lines.append("|---|---|---:|---:|---|")
    for ds in datasets:
        for k in ks:
            metric = f"R@{k}"
            info = summary.get("best_by_dataset", {}).get(ds, {}).get(metric, {})
            val = info.get("value")
            lines.append(
                "| "
                + " | ".join(
                    [
                        ds,
                        metric,
                        f"{float(val)*100.0:.2f}" if val is not None else "NA",
                        str(info.get("epoch")),
                        str(info.get("checkpoint_file")),
                    ]
                )
                + " |"
            )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _cleanup_single_reports_for_version(report_dir: Path, run_name: str, version: str) -> int:
    removed = 0
    for p in report_dir.glob("*.json"):
        d = _load_json(p)
        if d is None:
            continue
        rep_run = d.get("run", {}).get("run_name")
        rep_ver = d.get("run", {}).get("version")
        if rep_run == run_name and rep_ver == version and p.name.startswith(f"{run_name}_{version}_"):
            md_path = p.with_suffix(".md")
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
            try:
                md_path.unlink(missing_ok=True)
            except Exception:
                pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints in one version and compare generalization.")
    parser.add_argument("--run_name", required=True, help="Run folder under logs, e.g. dinov3_vitb16")
    parser.add_argument("--version", required=True, help="Version folder, e.g. version_6")
    parser.add_argument("--logs_root", default="./logs", help="Logs root path")
    parser.add_argument("--report_dir", default="./logs/eval_reports", help="Eval report directory used by train.py")
    parser.add_argument("--primary_k", type=int, default=1, help="Use R@k as the cross-checkpoint ranking basis.")
    parser.add_argument("--backbone", default="dinov3_vitb16", help="Backbone passed to train.py during test_only")
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoint if a report already exists for it.")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used to run train.py")
    parser.add_argument("--silent_test", action="store_true", help="Pass --silent when running train.py test_only")
    parser.add_argument("--cleanup_single_reports", action="store_true", help="Delete per-checkpoint JSON/MD reports for this version after consolidated summary is generated.")
    parser.add_argument("--use_qtr", action="store_true")
    parser.add_argument("--qtr_layers", type=str, default="last")
    parser.add_argument("--qtr_hidden_dim", type=str, default="128")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    ckpt_dir = logs_root / args.run_name / args.version / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: (_parse_epoch_from_name(p.name), p.name))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BatchEval] Found {len(ckpts)} checkpoints in {ckpt_dir}")

    existing_reports = _collect_reports(report_dir)

    for idx, ckpt in enumerate(ckpts, start=1):
        latest = _select_latest_report_for_ckpt(existing_reports, args.run_name, args.version, ckpt)
        if latest is not None and args.skip_existing:
            print(f"[BatchEval][{idx}/{len(ckpts)}] Skip existing: {ckpt.name}")
            continue

        cmd = [
            args.python_bin,
            "train.py",
            "--test_only",
            "--backbone",
            args.backbone,
            "--ckpt_path",
            str(ckpt),
            "--eval_report_dir",
            str(report_dir),
        ]
        if args.silent_test:
            cmd.append("--silent")
        
        if args.use_qtr:
            cmd += [
                "--use_qtr",
                "--qtr_layers", args.qtr_layers,
                "--qtr_hidden_dim", str(args.qtr_hidden_dim),
            ]

        print(f"[BatchEval][{idx}/{len(ckpts)}] Evaluating: {ckpt.name}")
        subprocess.run(cmd, check=True)

        # refresh cache after each run
        existing_reports = _collect_reports(report_dir)

    reports = _collect_reports(report_dir)
    selected_reports: list[dict[str, Any]] = []
    for ckpt in ckpts:
        rep = _select_latest_report_for_ckpt(reports, args.run_name, args.version, ckpt)
        if rep is not None:
            selected_reports.append(rep)

    if not selected_reports:
        raise RuntimeError("No evaluation reports found for selected checkpoints.")

    summary = _build_summary(
        run_name=args.run_name,
        version=args.version,
        reports_for_ckpts=selected_reports,
        primary_k=args.primary_k,
    )
    json_path, md_path = _write_summary_files(summary, report_dir)

    if args.cleanup_single_reports:
        removed = _cleanup_single_reports_for_version(report_dir, args.run_name, args.version)
        print(f"[BatchEval] Removed per-checkpoint reports: {removed}")

    print("[BatchEval] Done.")
    print(f"[BatchEval] Consolidated JSON: {json_path}")
    print(f"[BatchEval] Consolidated Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
