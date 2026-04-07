#!/usr/bin/env python3
"""Build a live status table for provisional validation883 assigned runs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Root directory, e.g. outputs/provisional/validation883_assigned")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prediction_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def collect_rows(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        summary_path = model_dir / "summary.json"
        report_path = model_dir / "report.csv"
        predictions_path = model_dir / "predictions.jsonl"
        row: Dict[str, Any] = {
            "model_label": model_dir.name,
            "status": "pending",
            "predictions_lines": prediction_lines(predictions_path),
            "runtime_success": None,
            "format_ok": None,
            "valid_parse": None,
            "strict_em": None,
            "strict_tm": None,
            "relaxed_em": None,
            "relaxed_tm": None,
            "relaxed_gap_tm": None,
            "truncation_without_answer_rate": None,
            "avg_latency_ms": None,
            "tok_per_sec": None,
            "peak_vram": None,
            "mean_output_tokens": None,
            "notes": "",
        }
        if summary_path.exists():
            summary = load_json(summary_path)
            row.update(
                {
                    "status": "ok",
                    "runtime_success": summary.get("runtime_success"),
                    "format_ok": summary.get("format_ok"),
                    "valid_parse": summary.get("valid_parse"),
                    "strict_em": summary.get("strict_em", summary.get("em")),
                    "strict_tm": summary.get("strict_tm", summary.get("tm")),
                    "relaxed_em": summary.get("relaxed_em"),
                    "relaxed_tm": summary.get("relaxed_tm"),
                    "relaxed_gap_tm": summary.get("relaxed_gap_tm"),
                    "truncation_without_answer_rate": summary.get("truncation_without_answer_rate"),
                    "avg_latency_ms": summary.get("avg_latency_ms"),
                    "tok_per_sec": summary.get("tok_per_sec"),
                    "peak_vram": summary.get("peak_vram"),
                    "mean_output_tokens": summary.get("mean_output_tokens"),
                    "notes": "" if report_path.exists() else "summary_exists_report_pending",
                }
            )
        elif row["predictions_lines"] > 0:
            row["status"] = "running_no_summary"
            row["notes"] = f"predictions_lines={row['predictions_lines']}"
        rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")
    lines = [
        "# validation883 进度表",
        "",
        f"- 生成时间：{generated_at}",
        "- 运行根目录：`outputs/provisional/validation883_assigned/`",
        "",
        "| 模型 | 状态 | PredLines | Strict TM | Relaxed TM | Gap | FormatOK | ValidParse | Truncation | AvgLatency(ms) | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {status} | {preds} | {strict_tm} | {relaxed_tm} | {gap} | {format_ok} | {valid_parse} | {trunc} | {latency} | {notes} |".format(
                model=row["model_label"],
                status=row["status"],
                preds=row["predictions_lines"],
                strict_tm="-" if row["strict_tm"] is None else f"{row['strict_tm']:.6f}",
                relaxed_tm="-" if row["relaxed_tm"] is None else f"{row['relaxed_tm']:.6f}",
                gap="-" if row["relaxed_gap_tm"] is None else f"{row['relaxed_gap_tm']:.6f}",
                format_ok="-" if row["format_ok"] is None else f"{row['format_ok']:.6f}",
                valid_parse="-" if row["valid_parse"] is None else f"{row['valid_parse']:.6f}",
                trunc="-" if row["truncation_without_answer_rate"] is None else f"{row['truncation_without_answer_rate']:.6f}",
                latency="-" if row["avg_latency_ms"] is None else f"{row['avg_latency_ms']:.2f}",
                notes=row["notes"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows = collect_rows(run_root)
    write_json(args.output_json.resolve(), rows)
    write_markdown(args.output_md.resolve(), rows)
    print(f"Wrote validation883 table to {args.output_json.resolve()} and {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
