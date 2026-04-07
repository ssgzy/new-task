#!/usr/bin/env python3
"""Run validation883 for one assigned model using the project registry and frozen decode."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--label", type=str, required=True, help="Exact model label from the registry.")
    parser.add_argument("--download-missing", action="store_true", help="Download the model if not present in local cache.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_report_csv(path: Path, label: str, summary: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "model_label",
        "stage",
        "max_new_tokens",
        "runtime_success",
        "format_ok",
        "valid_parse",
        "em",
        "tm",
        "strict_em",
        "strict_tm",
        "relaxed_em",
        "relaxed_tm",
        "relaxed_gap_tm",
        "truncation_without_answer_rate",
        "avg_latency_ms",
        "tok_per_sec",
        "peak_vram",
        "mean_output_tokens",
    ]
    row = {
        "model_label": label,
        "stage": "validation883",
        "max_new_tokens": summary.get("max_new_tokens"),
        "runtime_success": summary.get("runtime_success"),
        "format_ok": summary.get("format_ok"),
        "valid_parse": summary.get("valid_parse"),
        "em": summary.get("em"),
        "tm": summary.get("tm"),
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
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    label_slug = slugify(args.label)
    registry_path = project_root / "outputs" / "metadata" / "provisional" / f"model_registry.labels-{label_slug}.json"

    ensure_cmd = [
        sys.executable,
        str(project_root / "scripts" / "ensure_candidate_models.py"),
        "--labels",
        args.label,
        "--provisional",
    ]
    if args.download_missing:
        ensure_cmd.append("--download-missing")
    subprocess.run(ensure_cmd, cwd=project_root, check=True)

    registry = load_json(registry_path)
    row = next((item for item in registry if item["label"] == args.label), None)
    if row is None:
        raise SystemExit(f"Label not found in registry: {args.label}")
    if row.get("status") not in {"available_or_downloaded", "available_local"} or not row.get("snapshot_path"):
        raise SystemExit(f"Model not available locally for validation883: {args.label}")

    run_dir = project_root / "outputs" / "provisional" / "validation883_assigned" / args.label
    ensure_dir(run_dir)
    summary_json = run_dir / "summary.json"
    predictions_jsonl = run_dir / "predictions.jsonl"
    report_csv = run_dir / "report.csv"

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_finqa_local_benchmark_v1.py"),
        "--model-label",
        args.label,
        "--model-path",
        row["snapshot_path"],
        "--manifest",
        str(project_root / "data" / "manifests" / "validation883.jsonl"),
        "--output-jsonl",
        str(predictions_jsonl),
        "--summary-json",
        str(summary_json),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.resume_existing:
        cmd.append("--resume-existing")
    subprocess.run(cmd, cwd=project_root, check=True)

    summary = load_json(summary_json)
    write_report_csv(report_csv, args.label, summary)
    print(json.dumps({"summary_json": str(summary_json), "predictions_jsonl": str(predictions_jsonl), "report_csv": str(report_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
