#!/usr/bin/env python3
"""Resolve and optionally download candidate models into the Hugging Face cache."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE


CANDIDATE_MODELS = [
    {
        "label": "Lion-7B",
        "group": "historical_anchor",
        "model_family": "historical_anchor",
        "repo_id": "YuxinJiang/lion-7b",
        "notes": "historical anchor / protocol failure case; keep for continuity but do not count toward formal instruction main table",
    },
    {
        "label": "Orca-2-7B",
        "group": "main",
        "model_family": "instruction_distill",
        "repo_id": "microsoft/Orca-2-7b",
        "notes": "current formal instruction candidate; ChatML + LlamaTokenizer path recovered on 1-sample smoke",
    },
    {
        "label": "Zephyr-7B-beta",
        "group": "main",
        "model_family": "alignment",
        "repo_id": "HuggingFaceH4/zephyr-7b-beta",
        "notes": "current formal instruction candidate; plain wrapper is the currently valid parser-compatible path",
    },
    {
        "label": "MiniLLM-Llama-7B",
        "group": "base_distilled_baseline",
        "model_family": "base_distilled",
        "repo_id": "MiniLLM/MiniLLM-Llama-7B",
        "notes": "base_distilled baseline; keep as non-instruction comparator and do not count toward formal instruction main table",
    },
    {
        "label": "Xwin-LM-7B",
        "group": "main_candidate_expansion",
        "model_family": "alignment",
        "repo_id": "Xwin-LM/Xwin-LM-7B-V0.1",
        "notes": "new expansion candidate; official model card uses Vicuna-style prompts",
    },
    {
        "label": "LaMini-LLaMA-7B",
        "group": "main_candidate_expansion",
        "model_family": "instruction_distill",
        "repo_id": "MBZUAI/LaMini-LLaMA-7B",
        "skip_remote_resolution": True,
        "placeholder_status": "pending_repo_id",
        "notes": "repo placeholder only; MBZUAI/LaMini-LLaMA-7B is currently not publicly resolvable via HF API, so do not download or smoke-test this label in the current round",
    },
    {
        "label": "DeepSeek-R1-Distill-Qwen-7B",
        "group": "main_candidate_expansion",
        "model_family": "reasoning_distill",
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "notes": "new expansion candidate; reasoning-distilled Qwen2 model, first pass stays under direct-answer parser v1 without CoT-specific parser changes",
    },
    {
        "label": "OpenR1-Distill-7B",
        "group": "main_candidate_expansion",
        "model_family": "long_reasoning_distill",
        "repo_id": "open-r1/OpenR1-Distill-7B",
        "notes": "new expansion candidate; reasoning-oriented Qwen2 model, first pass compares ChatML/plain wrappers without changing parser v1",
    },
]

# Restrict downloads to model/tokenizer/config artifacts to keep cache clean.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.model",
    "*.txt",
    "*.py",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains outputs/metadata.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download models that are not already available in local Hugging Face cache.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Resolve snapshot paths from local Hugging Face cache only; do not call remote HF APIs.",
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
        "--provisional",
        action="store_true",
        help="Write registry outputs to provisional paths instead of canonical paths.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_models(models: List[Dict[str, str]], groups: List[str], labels: List[str]) -> List[Dict[str, str]]:
    selected = list(models)

    if groups:
        allowed_groups = set(groups)
        selected = [model for model in selected if model["group"] in allowed_groups]

    if labels:
        available_labels = {model["label"] for model in models}
        missing_labels = [label for label in labels if label not in available_labels]
        if missing_labels:
            raise SystemExit(f"Unknown model labels: {', '.join(missing_labels)}")
        allowed_labels = set(labels)
        selected = [model for model in selected if model["label"] in allowed_labels]

    if not selected:
        raise SystemExit("No candidate models matched the provided --groups/--labels filters.")
    return selected


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "all"


def build_scope_name(groups: List[str], labels: List[str]) -> str:
    parts: List[str] = []
    if groups:
        parts.append("groups-" + "-".join(slugify(group) for group in groups))
    if labels:
        parts.append("labels-" + "-".join(slugify(label) for label in labels))
    return ".".join(parts) if parts else "all"


def resolve_registry_paths(project_root: Path, groups: List[str], labels: List[str], provisional: bool) -> tuple[Path, Path, str]:
    metadata_dir = project_root / "outputs" / "metadata"
    if provisional or groups or labels:
        scope = build_scope_name(groups, labels)
        provisional_dir = metadata_dir / "provisional"
        return (
            provisional_dir / f"model_registry.{scope}.json",
            provisional_dir / f"model_registry.{scope}.csv",
            "provisional",
        )
    return (
        metadata_dir / "model_registry.json",
        metadata_dir / "model_registry.csv",
        "canonical",
    )


def resolve_local_snapshot(repo_id: str) -> Dict[str, Any]:
    repo_cache_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    row: Dict[str, Any] = {
        "status": "missing_local_path",
        "snapshot_path": "",
        "resolved_revision": "",
        "num_siblings": "",
        "error": "",
    }

    if not snapshots_dir.exists():
        row["error"] = f"Local snapshots dir not found: {snapshots_dir}"
        return row

    snapshot_dirs = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshot_dirs:
        row["error"] = f"No local snapshot revisions under: {snapshots_dir}"
        return row

    latest_snapshot = max(snapshot_dirs, key=lambda path: path.stat().st_mtime)
    row["snapshot_path"] = str(latest_snapshot)
    row["resolved_revision"] = latest_snapshot.name
    row["num_siblings"] = sum(1 for path in latest_snapshot.rglob("*") if path.is_file())

    incomplete_blobs = sorted(repo_cache_dir.glob("blobs/*.incomplete"))
    if incomplete_blobs:
        row["status"] = "incomplete_local_snapshot"
        row["error"] = (
            f"Found {len(incomplete_blobs)} partial blob(s) under {repo_cache_dir / 'blobs'}; "
            "skip smoke until download resumes on a non-hotspot network"
        )
        return row

    row["status"] = "available_local"
    return row


def resolve_snapshot(model: Dict[str, Any], download_missing: bool, local_only: bool) -> Dict[str, Any]:
    repo_id = model["repo_id"]
    row: Dict[str, Any] = {
        "repo_id": repo_id,
        "status": "unknown",
        "snapshot_path": "",
        "resolved_revision": "",
        "num_siblings": "",
        "error": "",
    }

    if model.get("skip_remote_resolution"):
        row["status"] = model.get("placeholder_status", "skipped_by_policy")
        row["error"] = model.get("notes", "")
        return row

    if local_only:
        row.update(resolve_local_snapshot(repo_id))
        return row

    api = HfApi()

    try:
        info = api.model_info(repo_id)
        row["resolved_revision"] = info.sha or ""
        row["num_siblings"] = len(info.siblings)
    except Exception as exc:  # pragma: no cover - network dependent
        row["status"] = "repo_lookup_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=ALLOW_PATTERNS,
            local_files_only=not download_missing,
        )
        row["status"] = "available_local" if not download_missing else "available_or_downloaded"
        row["snapshot_path"] = snapshot_path
        return row
    except Exception as exc:
        if not download_missing:
            row["status"] = "missing_local_path"
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row
        row["status"] = "download_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    selected_models = select_models(CANDIDATE_MODELS, args.groups, args.labels)
    registry_json_path, registry_csv_path, output_mode = resolve_registry_paths(
        project_root=project_root,
        groups=args.groups,
        labels=args.labels,
        provisional=args.provisional,
    )

    rows: List[Dict[str, Any]] = []
    for model in selected_models:
        resolved = resolve_snapshot(
            model=model,
            download_missing=args.download_missing,
            local_only=args.local_only,
        )
        rows.append(
            {
                "label": model["label"],
                "group": model["group"],
                "model_family": model.get("model_family", ""),
                "notes": model.get("notes", ""),
                **resolved,
            }
        )

    write_json(registry_json_path, rows)
    write_csv(
        registry_csv_path,
        rows,
        fieldnames=[
            "label",
            "group",
            "model_family",
            "repo_id",
            "status",
            "snapshot_path",
            "resolved_revision",
            "num_siblings",
            "error",
            "notes",
        ],
    )

    print(
        "Selected models: "
        + ", ".join(f"{model['label']}[{model['group']}]" for model in selected_models)
    )
    print(f"Registry output mode: {output_mode} | json={registry_json_path} | csv={registry_csv_path}")
    for row in rows:
        print(
            f"{row['label']}: {row['status']} | repo={row['repo_id']} | "
            f"path={row['snapshot_path'] or '-'}"
        )
        if row["error"]:
            print(f"  error: {row['error']}")


if __name__ == "__main__":
    main()
