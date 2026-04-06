#!/usr/bin/env python3
"""Resume canonical val_calib50, freeze max_new_tokens, then run main-table val_screen200."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


SAFE_TRUNCATION_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing outputs/, data/, and scripts/.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "metadata" / "model_registry.json",
        help="Canonical model registry path.",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "calibration_report.csv",
        help="Canonical calibration report path.",
    )
    parser.add_argument(
        "--qualification-report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "qualification_summary.csv",
        help="Canonical qualification summary path.",
    )
    parser.add_argument(
        "--freeze-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "protocol_v1" / "frozen_length_selection_v1.json",
        help="Machine-readable record of the frozen max_new_tokens decision.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_cmd(cmd: List[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def freeze_max_new_tokens(rows: List[Dict[str, str]], main_labels: List[str]) -> int:
    by_tokens: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        if row["group"] != "main":
            continue
        token_budget = int(row["max_new_tokens"])
        by_tokens.setdefault(token_budget, {})[row["model_label"]] = row

    for token_budget in sorted(by_tokens):
        model_rows = by_tokens[token_budget]
        if any(label not in model_rows for label in main_labels):
            continue
        if any(model_rows[label]["run_status"] != "ok" for label in main_labels):
            continue
        if all(float(model_rows[label]["truncation_without_answer_rate"]) <= SAFE_TRUNCATION_THRESHOLD for label in main_labels):
            return token_budget

    raise SystemExit(
        "No max_new_tokens candidate satisfied the frozen safety rule "
        f"(all main models with truncation_without_answer_rate <= {SAFE_TRUNCATION_THRESHOLD:.2f})."
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry = load_json(args.model_registry.resolve())
    main_labels = [row["label"] for row in registry if row["group"] == "main"]

    calibration_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_length_calibration_v1.py"),
        "--groups",
        "main",
        "--resume",
    ]
    run_cmd(calibration_cmd, project_root)

    rows = load_csv_rows(args.calibration_report.resolve())
    chosen = freeze_max_new_tokens(rows, main_labels)
    freeze_payload = {
        "scope": "main",
        "freeze_rule": "smallest max_new_tokens where every main model has truncation_without_answer_rate <= 0.02",
        "chosen_max_new_tokens": chosen,
        "main_labels": main_labels,
        "calibration_report": str(args.calibration_report.resolve()),
    }
    write_json(args.freeze_json.resolve(), freeze_payload)
    print(json.dumps(freeze_payload, ensure_ascii=False, indent=2))

    qualification_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_qualification_v1.py"),
        "--groups",
        "main",
        "--max-new-tokens",
        str(chosen),
        "--resume",
        "--screen-only",
        "--report-csv",
        str(args.qualification_report.resolve()),
    ]
    run_cmd(qualification_cmd, project_root)


if __name__ == "__main__":
    main()
