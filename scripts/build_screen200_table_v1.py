#!/usr/bin/env python3
"""Build a per-model val_screen200 status table from summary.json outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_THRESHOLDS = {
    "runtime_success": 0.95,
    "format_ok": 0.80,
    "valid_parse": 0.60,
    "truncation_without_answer_rate": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[],
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--runtime-success-threshold", type=float, default=DEFAULT_THRESHOLDS["runtime_success"])
    parser.add_argument("--format-ok-threshold", type=float, default=DEFAULT_THRESHOLDS["format_ok"])
    parser.add_argument("--valid-parse-threshold", type=float, default=DEFAULT_THRESHOLDS["valid_parse"])
    parser.add_argument("--truncation-threshold", type=float, default=DEFAULT_THRESHOLDS["truncation_without_answer_rate"])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def filter_registry(registry: List[Dict[str, Any]], labels: List[str]) -> List[Dict[str, Any]]:
    if not labels:
        return list(registry)
    allowed = set(labels)
    return [row for row in registry if row["label"] in allowed]


def qualified(summary: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return (
        summary["runtime_success"] >= thresholds["runtime_success"]
        and summary["format_ok"] >= thresholds["format_ok"]
        and summary["valid_parse"] >= thresholds["valid_parse"]
        and summary["truncation_without_answer_rate"] <= thresholds["truncation_without_answer_rate"]
    )


def build_rows(registry: List[Dict[str, Any]], run_root: Path, max_new_tokens: int, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in registry:
        summary_path = run_root / "screen" / model["label"] / "summary.json"
        predictions_path = run_root / "screen" / model["label"] / "predictions.jsonl"
        row: Dict[str, Any] = {
            "model_label": model["label"],
            "group": model["group"],
            "repo_id": model["repo_id"],
            "registry_status": model["status"],
            "snapshot_path": model["snapshot_path"],
            "max_new_tokens": max_new_tokens,
            "run_status": "pending",
            "qualified_by_frozen_rule": "pending",
            "runtime_success": None,
            "format_ok": None,
            "valid_parse": None,
            "truncation_without_answer_rate": None,
            "em": None,
            "tm": None,
            "avg_latency_ms": None,
            "tok_per_sec": None,
            "peak_vram": None,
            "mean_output_tokens": None,
            "note": "",
        }
        if model["status"] not in {"available_local", "available_or_downloaded"} or not model["snapshot_path"]:
            row["run_status"] = "skipped_missing_model"
            row["qualified_by_frozen_rule"] = "no"
            row["note"] = model.get("error", "") or "model_not_available"
            rows.append(row)
            continue
        if not summary_path.exists():
            if predictions_path.exists():
                with predictions_path.open("r", encoding="utf-8") as f:
                    num_lines = sum(1 for _ in f)
                row["run_status"] = "running_no_summary"
                row["note"] = f"predictions_lines={num_lines}"
            rows.append(row)
            continue
        summary = load_json(summary_path)
        row.update(
            {
                "run_status": "ok",
                "runtime_success": summary["runtime_success"],
                "format_ok": summary["format_ok"],
                "valid_parse": summary["valid_parse"],
                "truncation_without_answer_rate": summary["truncation_without_answer_rate"],
                "em": summary["em"],
                "tm": summary["tm"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "tok_per_sec": summary["tok_per_sec"],
                "peak_vram": summary["peak_vram"],
                "mean_output_tokens": summary["mean_output_tokens"],
                "qualified_by_frozen_rule": "yes" if qualified(summary, thresholds) else "no",
                "note": "",
            }
        )
        rows.append(row)
    return rows


def build_markdown(rows: List[Dict[str, Any]], thresholds: Dict[str, float], max_new_tokens: int) -> str:
    lines = [
        "# val_screen200 进度表",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 冻结参数：`max_new_tokens={max_new_tokens}`",
        "- 冻结门槛："
        f" `runtime_success >= {thresholds['runtime_success']:.2f}`,"
        f" `format_ok >= {thresholds['format_ok']:.2f}`,"
        f" `valid_parse >= {thresholds['valid_parse']:.2f}`,"
        f" `truncation_without_answer_rate <= {thresholds['truncation_without_answer_rate']:.2f}`",
        "",
        "| 模型 | 状态 | 通过冻结门槛 | RuntimeSuccess | FormatOK | ValidParse | Truncation | EM | TM | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {status} | {qualified} | {runtime} | {format_ok} | {valid_parse} | {trunc} | {em} | {tm} | {note} |".format(
                model=row["model_label"],
                status=row["run_status"],
                qualified=row["qualified_by_frozen_rule"],
                runtime="-" if row["runtime_success"] is None else f"{row['runtime_success']:.3f}",
                format_ok="-" if row["format_ok"] is None else f"{row['format_ok']:.3f}",
                valid_parse="-" if row["valid_parse"] is None else f"{row['valid_parse']:.3f}",
                trunc="-" if row["truncation_without_answer_rate"] is None else f"{row['truncation_without_answer_rate']:.3f}",
                em="-" if row["em"] is None else f"{row['em']:.3f}",
                tm="-" if row["tm"] is None else f"{row['tm']:.3f}",
                note=row["note"] or "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    registry = filter_registry(load_json(args.model_registry.resolve()), args.labels)
    thresholds = {
        "runtime_success": args.runtime_success_threshold,
        "format_ok": args.format_ok_threshold,
        "valid_parse": args.valid_parse_threshold,
        "truncation_without_answer_rate": args.truncation_threshold,
    }
    rows = build_rows(
        registry=registry,
        run_root=args.run_root.resolve(),
        max_new_tokens=args.max_new_tokens,
        thresholds=thresholds,
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "max_new_tokens": args.max_new_tokens,
        "thresholds": thresholds,
        "rows": rows,
    }
    write_json(args.output_json.resolve(), payload)
    write_text(args.output_md.resolve(), build_markdown(rows, thresholds, args.max_new_tokens))
    print(f"Wrote screen200 status table to {args.output_json.resolve()} and {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
