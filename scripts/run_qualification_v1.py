#!/usr/bin/env python3
"""Run qualification screening on val_screen200 and full validation on qualified models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


THRESHOLDS = {
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
        help="Project root that contains outputs/metadata and data/manifests.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "metadata" / "model_registry.json",
    )
    parser.add_argument(
        "--screen-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests" / "val_screen200.jsonl",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests" / "validation883.jsonl",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "qualification_summary.csv",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional root directory for per-model qualification runs. Defaults to outputs/qualification_runs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
        help="Reuse existing per-stage summary.json files when available.",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="Write aggregate CSV to provisional path instead of the canonical report path.",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Run only val_screen200 and skip full validation even for qualified models.",
    )
    parser.add_argument(
        "--runtime-success-threshold",
        type=float,
        default=THRESHOLDS["runtime_success"],
    )
    parser.add_argument(
        "--format-ok-threshold",
        type=float,
        default=THRESHOLDS["format_ok"],
    )
    parser.add_argument(
        "--valid-parse-threshold",
        type=float,
        default=THRESHOLDS["valid_parse"],
    )
    parser.add_argument(
        "--truncation-threshold",
        type=float,
        default=THRESHOLDS["truncation_without_answer_rate"],
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
    canonical_path = (project_root / "outputs" / "qualification_summary.csv").resolve()
    requested_path = report_csv.resolve()
    provisional_mode = provisional or bool(labels)
    if provisional_mode and requested_path == canonical_path:
        scope = build_scope_name(groups, labels)
        return (
            (project_root / "outputs" / "provisional" / f"qualification_summary.{scope}.provisional.csv").resolve(),
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


def run_benchmark(
    project_root: Path,
    run_root: Path,
    stage: str,
    model_label: str,
    model_path: str,
    manifest: Path,
    max_new_tokens: int,
    resume: bool,
) -> Dict[str, Any]:
    run_dir = run_root / stage / model_label
    ensure_dir(run_dir)
    summary_path = run_dir / "summary.json"
    if resume and summary_path.exists():
        return {"status": "ok", "summary": load_json(summary_path)}
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
        return {"status": "run_failed", "error": completed.stderr.strip() or completed.stdout.strip()}
    return {"status": "ok", "summary": load_json(run_dir / "summary.json")}


def passed_thresholds(summary: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return (
        summary["runtime_success"] >= thresholds["runtime_success"]
        and summary["format_ok"] >= thresholds["format_ok"]
        and summary["valid_parse"] >= thresholds["valid_parse"]
        and summary["truncation_without_answer_rate"] <= thresholds["truncation_without_answer_rate"]
    )


def summary_row(model: Dict[str, Any], stage: str, max_new_tokens: int, run_status: str, summary: Dict[str, Any] | None = None, error: str = "", qualified: str = "") -> Dict[str, Any]:
    row = {
        "model_label": model["label"],
        "group": model["group"],
        "repo_id": model["repo_id"],
        "registry_status": model["status"],
        "snapshot_path": model["snapshot_path"],
        "stage": stage,
        "max_new_tokens": max_new_tokens,
        "run_status": run_status,
        "qualified_for_full_validation": qualified,
        "runtime_success": None,
        "format_ok": None,
        "valid_parse": None,
        "em": None,
        "tm": None,
        "avg_latency_ms": None,
        "tok_per_sec": None,
        "peak_vram": None,
        "mean_output_tokens": None,
        "truncation_without_answer_rate": None,
        "error": error,
    }
    if summary is not None:
        row.update(
            {
                "runtime_success": summary["runtime_success"],
                "format_ok": summary["format_ok"],
                "valid_parse": summary["valid_parse"],
                "em": summary["em"],
                "tm": summary["tm"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "tok_per_sec": summary["tok_per_sec"],
                "peak_vram": summary["peak_vram"],
                "mean_output_tokens": summary["mean_output_tokens"],
                "truncation_without_answer_rate": summary["truncation_without_answer_rate"],
            }
        )
    return row


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_root = (
        args.run_root.resolve()
        if args.run_root is not None
        else (project_root / "outputs" / "qualification_runs").resolve()
    )
    registry = load_json(args.model_registry.resolve())
    screen_manifest = args.screen_manifest.resolve()
    validation_manifest = args.validation_manifest.resolve()
    thresholds = {
        "runtime_success": args.runtime_success_threshold,
        "format_ok": args.format_ok_threshold,
        "valid_parse": args.valid_parse_threshold,
        "truncation_without_answer_rate": args.truncation_threshold,
    }
    selected_models = filter_registry(registry, args.groups, args.labels)
    report_csv_path, output_mode = resolve_report_path(
        project_root=project_root,
        report_csv=args.report_csv,
        groups=args.groups,
        labels=args.labels,
        provisional=args.provisional,
    )

    rows: List[Dict[str, Any]] = []
    qualified_models: List[Dict[str, Any]] = []
    for model in selected_models:
        if model["status"] not in {"available_or_downloaded", "available_local"} or not model["snapshot_path"]:
            rows.append(
                summary_row(
                    model=model,
                    stage="screen",
                    max_new_tokens=args.max_new_tokens,
                    run_status="skipped_missing_model",
                    error=model.get("error", ""),
                    qualified="no",
                )
            )
            continue

        result = run_benchmark(
            project_root=project_root,
            run_root=run_root,
            stage="screen",
            model_label=model["label"],
            model_path=model["snapshot_path"],
            manifest=screen_manifest,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )
        if result["status"] != "ok":
            rows.append(
                summary_row(
                    model=model,
                    stage="screen",
                    max_new_tokens=args.max_new_tokens,
                    run_status=result["status"],
                    error=result["error"],
                    qualified="no",
                )
            )
            continue

        summary = result["summary"]
        qualified = "yes" if passed_thresholds(summary, thresholds) else "no"
        rows.append(
            summary_row(
                model=model,
                stage="screen",
                max_new_tokens=args.max_new_tokens,
                run_status="ok",
                summary=summary,
                qualified=qualified,
            )
        )
        if qualified == "yes":
            qualified_models.append(model)

    if args.screen_only:
        qualified_models = []

    for model in qualified_models:
        result = run_benchmark(
            project_root=project_root,
            run_root=run_root,
            stage="validation",
            model_label=model["label"],
            model_path=model["snapshot_path"],
            manifest=validation_manifest,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )
        if result["status"] != "ok":
            rows.append(
                summary_row(
                    model=model,
                    stage="validation",
                    max_new_tokens=args.max_new_tokens,
                    run_status=result["status"],
                    error=result["error"],
                    qualified="yes",
                )
            )
            continue
        rows.append(
            summary_row(
                model=model,
                stage="validation",
                max_new_tokens=args.max_new_tokens,
                run_status="ok",
                summary=result["summary"],
                qualified="yes",
            )
        )

    write_csv(
        report_csv_path,
        rows,
        fieldnames=[
            "model_label",
            "group",
            "repo_id",
            "registry_status",
            "snapshot_path",
            "stage",
            "max_new_tokens",
            "run_status",
            "qualified_for_full_validation",
            "runtime_success",
            "format_ok",
            "valid_parse",
            "em",
            "tm",
            "avg_latency_ms",
            "tok_per_sec",
            "peak_vram",
            "mean_output_tokens",
            "truncation_without_answer_rate",
            "error",
        ],
    )
    print(f"Wrote {output_mode} qualification summary to {report_csv_path}")


if __name__ == "__main__":
    main()
