# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def parse_checkpoint_metadata(ckpt_path: str) -> dict[str, Any]:
    path = Path(ckpt_path)
    stem = path.stem
    epoch_match = re.search(r"epoch\[(\d+)\]", stem)
    r_at_k_matches = re.findall(r"R@(\d+)\[([0-9]*\.?[0-9]+)\]", stem)
    metrics_in_name = {f"R@{k}": float(v) for k, v in r_at_k_matches}
    return {
        "filename": path.name,
        "stem": stem,
        "epoch_from_name": int(epoch_match.group(1)) if epoch_match else None,
        "metrics_from_name": metrics_in_name,
    }


def infer_run_identity(ckpt_path: str) -> dict[str, str | None]:
    """
    Try to infer run_name/version from a checkpoint path like:
    logs/<run_name>/<version>/checkpoints/<file>.ckpt
    """
    path = Path(ckpt_path).resolve()
    parts = list(path.parts)
    try:
        idx = parts.index("logs")
        run_name = parts[idx + 1] if idx + 1 < len(parts) else None
        version = parts[idx + 2] if idx + 2 < len(parts) else None
        return {"run_name": run_name, "version": version}
    except Exception:
        return {"run_name": None, "version": None}


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or yaml is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_ckpt_hparams(ckpt_path: str) -> tuple[str | None, dict[str, Any] | None]:
    """
    Return (hparams_path, dict) if a sibling run hparams.yaml can be found.
    """
    path = Path(ckpt_path).resolve()
    version_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    hparams_path = version_dir / "hparams.yaml"
    data = _load_yaml(hparams_path)
    return (str(hparams_path) if hparams_path.exists() else None, data)


def flatten_dataset_recalls(dataset_recalls: dict[str, dict[int, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for dataset_name, recalls in dataset_recalls.items():
        for k, v in recalls.items():
            flat[f"{dataset_name}/R@{k}"] = float(v)
    return flat


def build_eval_report(
    *,
    ckpt_path: str,
    requested_ckpt_path: str,
    dataset_recalls: dict[str, dict[int, float]],
    hparams_dict: dict[str, Any],
    cli_args_dict: dict[str, Any],
) -> dict[str, Any]:
    ckpt = Path(ckpt_path).resolve()
    run_identity = infer_run_identity(str(ckpt))
    ckpt_meta = parse_checkpoint_metadata(str(ckpt))
    hparams_path, ckpt_hparams = load_ckpt_hparams(str(ckpt))
    flat_metrics = flatten_dataset_recalls(dataset_recalls)
    monitor = str(hparams_dict.get("top1_monitor", "pitts30k-val/R@1"))

    report = {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "checkpoint": {
            "requested_ckpt_path": requested_ckpt_path,
            "resolved_ckpt_path": str(ckpt),
            "exists": ckpt.exists(),
            "size_bytes": ckpt.stat().st_size if ckpt.exists() else None,
            "mtime_utc": (
                datetime.fromtimestamp(ckpt.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
                if ckpt.exists()
                else None
            ),
            "parsed": ckpt_meta,
        },
        "run": run_identity,
        "evaluation": {
            "monitor_metric": monitor,
            "monitor_value": _safe_float(flat_metrics.get(monitor)),
            "dataset_recalls": dataset_recalls,
            "flat_metrics": flat_metrics,
        },
        "config": {
            "runtime_hparams": hparams_dict,
            "ckpt_hparams_path": hparams_path,
            "ckpt_hparams": ckpt_hparams,
            "cli_args": cli_args_dict,
        },
    }
    return report


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def save_eval_report(report: dict[str, Any], report_dir: str = "./logs/eval_reports") -> tuple[str, str]:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = report.get("run", {}).get("run_name") or "unknown_run"
    version = report.get("run", {}).get("version") or "unknown_version"
    ckpt_stem = report.get("checkpoint", {}).get("parsed", {}).get("stem", "unknown_ckpt")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _sanitize_filename(f"{run_name}_{version}_{ckpt_stem}_{ts}")

    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _write_single_report_markdown(report, md_path)
    _refresh_global_reports_index(out_dir)
    return str(json_path), str(md_path)


def _write_single_report_markdown(report: dict[str, Any], md_path: Path) -> None:
    ckpt = report["checkpoint"]
    run = report["run"]
    eval_block = report["evaluation"]

    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- Run: `{run.get('run_name')}/{run.get('version')}`")
    lines.append(f"- Requested checkpoint: `{ckpt.get('requested_ckpt_path')}`")
    lines.append(f"- Resolved checkpoint: `{ckpt.get('resolved_ckpt_path')}`")
    lines.append(f"- Parsed epoch: `{ckpt.get('parsed', {}).get('epoch_from_name')}`")
    lines.append("")

    name_metrics = ckpt.get("parsed", {}).get("metrics_from_name", {})
    if name_metrics:
        lines.append("## Checkpoint Name Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k, v in sorted(name_metrics.items(), key=lambda x: x[0]):
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

    lines.append("## Recall@k (Dataset View)")
    lines.append("")
    dataset_recalls = eval_block.get("dataset_recalls", {}) or {}
    k_values = set()
    for recalls in dataset_recalls.values():
        for k in recalls.keys():
            try:
                k_values.add(int(k))
            except Exception:
                continue
    sorted_k = sorted(k_values)

    if dataset_recalls and sorted_k:
        header = ["Dataset"] + [f"R@{k}" for k in sorted_k]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for dataset_name in sorted(dataset_recalls.keys()):
            recalls = dataset_recalls[dataset_name]
            row = [dataset_name]
            for k in sorted_k:
                v = recalls.get(k, recalls.get(str(k)))
                row.append(f"{100.0 * float(v):.2f}" if v is not None else "NA")
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("_No test metrics available._")
    lines.append("")

    lines.append("## Flat Metrics (Raw)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    flat = eval_block.get("flat_metrics", {})
    for k, v in sorted(flat.items(), key=lambda x: x[0]):
        lines.append(f"| {k} | {float(v):.4f} |")
    lines.append("")

    monitor = eval_block.get("monitor_metric")
    monitor_value = eval_block.get("monitor_value")
    lines.append("## Monitor")
    lines.append("")
    lines.append(f"- Monitor metric: `{monitor}`")
    lines.append(f"- Monitor value: `{monitor_value}`")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def _refresh_global_reports_index(report_dir: Path) -> None:
    reports: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data["_report_json"] = path.name
            reports.append(data)
        except Exception:
            continue

    # Newest first.
    reports.sort(key=lambda d: d.get("generated_at_utc", ""), reverse=True)

    index_json = report_dir / "reports_index.json"
    with index_json.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    lines: list[str] = []
    lines.append("# Evaluation Reports Index")
    lines.append("")
    lines.append("| Time (UTC) | Run | Version | Epoch(from name) | Monitor | Monitor Value | Checkpoint | Report JSON |")
    lines.append("|---|---|---|---:|---|---:|---|---|")
    for rep in reports:
        ts = rep.get("generated_at_utc", "")
        run = rep.get("run", {}).get("run_name", "")
        version = rep.get("run", {}).get("version", "")
        epoch = rep.get("checkpoint", {}).get("parsed", {}).get("epoch_from_name", "")
        monitor = rep.get("evaluation", {}).get("monitor_metric", "")
        monitor_val = rep.get("evaluation", {}).get("monitor_value", "")
        ckpt_file = rep.get("checkpoint", {}).get("parsed", {}).get("filename", "")
        report_json = rep.get("_report_json", "")
        lines.append(
            f"| {ts} | {run} | {version} | {epoch} | {monitor} | {monitor_val} | {ckpt_file} | {report_json} |"
        )

    lines.append("")
    lines.append("## Latest Leaderboard Snapshot (R@1)")
    lines.append("")
    lines.append("| Run | Version | Dataset | R@1 | Time (UTC) |")
    lines.append("|---|---|---|---:|---|")
    for rep in reports:
        run = rep.get("run", {}).get("run_name", "")
        version = rep.get("run", {}).get("version", "")
        ts = rep.get("generated_at_utc", "")
        flat = rep.get("evaluation", {}).get("flat_metrics", {})
        r1_items = [(k, v) for k, v in flat.items() if k.endswith("/R@1")]
        for metric, value in sorted(r1_items, key=lambda x: x[0]):
            dataset = metric.rsplit("/R@1", 1)[0]
            lines.append(f"| {run} | {version} | {dataset} | {float(value):.4f} | {ts} |")

    (report_dir / "reports_index.md").write_text("\n".join(lines), encoding="utf-8")
