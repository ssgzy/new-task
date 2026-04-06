#!/usr/bin/env python3
"""Run protocol v1 length calibration across candidate models and aggregate a CSV report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains outputs/metadata and data/manifests.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "metadata" / "model_registry.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests" / "val_calib50.jsonl",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[128, 192, 256],
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "calibration_report.csv",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=[],
        help="Optional model groups to include, e.g. main appendix.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=[],
        help="Optional exact model labels to include, e.g. Lion-7B Orca-2-7B.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-run summary.json files when available.",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Write aggregate CSV to provisional path instead of the canonical report path.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "all"


def build_scope_name(groups: List[str], labels: List[str]) -> str:
    parts: List[str] = []
    if groups:
        parts.append("groups-" + "-".join(slugify(group) for group in groups))
    if labels:
        parts.append("labels-" + "-".join(slugify(label) for label in labels))
    return ".".join(parts) if parts else "all"


def resolve_report_path(project_root: Path, report_csv: Path, groups: List[str], labels: List[str], provisional: bool) -> tuple[Path, str]:
    canonical_path = (project_root / "outputs" / "calibration_report.csv").resolve()
    requested_path = report_csv.resolve()
    provisional_mode = provisional or bool(labels)
    if provisional_mode and requested_path == canonical_path:
        scope = build_scope_name(groups, labels)
        return (
            (project_root / "outputs" / "provisional" / f"calibration_report.{scope}.provisional.csv").resolve(),
            "provisional",
        )
    return requested_path, "provisional" if provisional_mode else "canonical"


def filter_registry(registry: List[Dict[str, Any]], groups: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    selected = list(registry)

    if groups:
        allowed_groups = set(groups)
        selected = [model for model in selected if model["group"] in allowed_groups]

    if labels:
        available_labels = {model["label"] for model in registry}
        missing_labels = [label for label in labels if label not in available_labels]
        if missing_labels:
            raise SystemExit(f"Unknown model labels: {', '.join(missing_labels)}")
        allowed_labels = set(labels)
        selected = [model for model in selected if model["label"] in allowed_labels]

    if not selected:
        raise SystemExit("No models matched the provided --groups/--labels filters.")
    return selected


def run_benchmark(project_root: Path, model_label: str, model_path: str, manifest: Path, max_new_tokens: int, resume: bool) -> Dict[str, Any]:
    run_dir = project_root / "outputs" / "calibration_runs" / model_label / f"max_new_tokens_{max_new_tokens}"
    ensure_dir(run_dir)
    summary_path = run_dir / "summary.json"
    if resume and summary_path.exists():
        return {
            "status": "ok",
            "summary": load_json(summary_path),
            "stdout": "resumed_from_existing_summary",
        }
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_finqa_local_benchmark_v1.py"),
        "--model-label",
        model_label,
        "--model-path",
        model_path,
        "--manifest",
        str(manifest),
        "--output-jsonl",
        str(run_dir / "predictions.jsonl"),
        "--summary-json",
        str(summary_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--resume-existing",
    ]
    completed = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "status": "run_failed",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    return {
        "status": "ok",
        "summary": load_json(run_dir / "summary.json"),
        "stdout": completed.stdout.strip(),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry = load_json(args.model_registry.resolve())
    manifest = args.manifest.resolve()
    selected_models = filter_registry(registry, args.groups, args.labels)
    report_csv_path, output_mode = resolve_report_path(
        project_root=project_root,
        report_csv=args.report_csv,
        groups=args.groups,
        labels=args.labels,
        provisional=args.provisional,
    )

    rows: List[Dict[str, Any]] = []
    for model in selected_models:
        base_row = {
            "model_label": model["label"],
            "group": model["group"],
            "repo_id": model["repo_id"],
            "registry_status": model["status"],
            "snapshot_path": model["snapshot_path"],
        }
        for max_new_tokens in args.max_new_tokens:
            row = {**base_row, "max_new_tokens": max_new_tokens}
            if model["status"] not in {"available_or_downloaded", "available_local"} or not model["snapshot_path"]:
                row.update(
                    {
                        "run_status": "skipped_missing_model",
                        "answer_present_rate": None,
                        "valid_parse_rate": None,
                        "truncation_without_answer_rate": None,
                        "mean_new_tokens": None,
                        "p95_new_tokens": None,
                        "error": model.get("error", ""),
                    }
                )
                rows.append(row)
                continue

            result = run_benchmark(
                project_root=project_root,
                model_label=model["label"],
                model_path=model["snapshot_path"],
                manifest=manifest,
                max_new_tokens=max_new_tokens,
                resume=args.resume,
            )
            if result["status"] != "ok":
                row.update(
                    {
                        "run_status": result["status"],
                        "answer_present_rate": None,
                        "valid_parse_rate": None,
                        "truncation_without_answer_rate": None,
                        "mean_new_tokens": None,
                        "p95_new_tokens": None,
                        "error": result["error"],
                    }
                )
                rows.append(row)
                continue

            summary = result["summary"]
            row.update(
                {
                    "run_status": "ok",
                    "answer_present_rate": summary["answer_present_rate"],
                    "valid_parse_rate": summary["valid_parse_rate"],
                    "truncation_without_answer_rate": summary["truncation_without_answer_rate"],
                    "mean_new_tokens": summary["mean_new_tokens"],
                    "p95_new_tokens": summary["p95_new_tokens"],
                    "error": "",
                }
            )
            rows.append(row)

    write_csv(
        report_csv_path,
        rows,
        fieldnames=[
            "model_label",
            "group",
            "repo_id",
            "registry_status",
            "snapshot_path",
            "max_new_tokens",
            "run_status",
            "answer_present_rate",
            "valid_parse_rate",
            "truncation_without_answer_rate",
            "mean_new_tokens",
            "p95_new_tokens",
            "error",
        ],
    )
    print(f"Wrote {output_mode} calibration report to {report_csv_path}")


if __name__ == "__main__":
    main()
