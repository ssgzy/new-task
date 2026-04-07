#!/usr/bin/env python3
"""Build a unified summary table for all protocol-screened models attempted so far."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_existing_rows(project_root: Path) -> List[Dict[str, Any]]:
    path = project_root / "outputs" / "debug" / "protocol_screening_registry" / "current_candidate_status_table.json"
    payload = load_json(path)
    return payload["rows"]


def map_existing_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_label": row["model_label"],
        "source_batch": "existing_protocol_screening",
        "repo_id": "",
        "method_family": row.get("model_family", ""),
        "base_model_family": "",
        "current_positioning": row.get("current_positioning", ""),
        "effective_wrapper": row.get("effective_wrapper", ""),
        "tokenizer_policy": row.get("tokenizer_policy", ""),
        "entered_effective_input_protocol": row.get("entered_effective_input_protocol", False),
        "smoke_pass": row.get("entered_effective_input_protocol", False),
        "status": "ok" if row.get("entered_effective_input_protocol", False) else "screened_not_entered",
        "success_or_failure_reason": row.get("current_failure_mode_or_strength", ""),
        "improvement_method_or_next_action": row.get("next_minimal_action", ""),
    }


def load_mainstream_registry(project_root: Path) -> Dict[str, Dict[str, Any]]:
    provisional_dir = project_root / "outputs" / "metadata" / "provisional"
    rows_by_label: Dict[str, Dict[str, Any]] = {}

    candidate_paths = sorted(provisional_dir.glob("model_registry.labels-*.json"))
    candidate_paths.extend(sorted(provisional_dir.glob("model_registry.groups-*.json")))

    for path in candidate_paths:
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            label = row.get("label")
            if not label:
                continue
            rows_by_label[label] = row
    return rows_by_label


def load_mainstream_smoke_rows(project_root: Path) -> Dict[str, Dict[str, Any]]:
    smoke_root = project_root / "outputs" / "debug" / "input_smoke_mainstream"
    rows: Dict[str, Dict[str, Any]] = {}
    for path in sorted(smoke_root.glob("*/smoke_mainstream.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        label = payload.get("model_label")
        candidate = payload.get("recommended_candidate", {})
        if not label or not candidate:
            continue
        rows[label] = {
            "model_label": label,
            "status": payload.get("status", ""),
            "effective_wrapper": candidate.get("effective_wrapper", ""),
            "pass_smoke": bool(candidate.get("pass_smoke", False)),
            "answer_appears": candidate.get("answer_appears"),
            "valid_parse": candidate.get("valid_parse"),
            "truncation_suspect": candidate.get("truncation_suspect"),
            "repetition_collapse": candidate.get("repetition_collapse"),
            "parse_error_reason": candidate.get("parse_error_reason"),
            "raw_generation_text": candidate.get("raw_generation_text", ""),
        }
    return rows


def build_mainstream_rows(project_root: Path) -> List[Dict[str, Any]]:
    registry = load_mainstream_registry(project_root)
    smoke_rows = load_mainstream_smoke_rows(project_root)
    labels = [
        "Mistral-7B-Instruct-v0.3",
        "Qwen2.5-7B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Gemma-7B-IT",
        "Yi-1.5-6B-Chat",
        "ChatGLM3-6B",
    ]
    rows: List[Dict[str, Any]] = []
    for label in labels:
        registry_row = registry.get(label, {})
        smoke_row = smoke_rows.get(label, {})
        diagnosis_path = project_root / "outputs" / "debug" / "input_interface_diagnosis_mainstream" / f"{label}.json"
        diagnosis = load_json(diagnosis_path) if diagnosis_path.exists() else {}
        rows.append(
            {
                "model_label": label,
                "source_batch": "mainstream_instruction_prescreen_20260407",
                "repo_id": registry_row.get("repo_id", ""),
                "method_family": registry_row.get("method_family", ""),
                "base_model_family": registry_row.get("base_model_family", ""),
                "current_positioning": "mainstream prescreen candidate",
                "effective_wrapper": smoke_row.get("effective_wrapper", diagnosis.get("default_wrapper", "")),
                "tokenizer_policy": diagnosis.get("tokenizer_load_policy", {}).get("notes", ""),
                "entered_effective_input_protocol": bool(smoke_row.get("pass_smoke", False)),
                "smoke_pass": bool(smoke_row.get("pass_smoke", False)),
                "status": smoke_row.get("status", registry_row.get("status", "missing")),
                "success_or_failure_reason": (
                    diagnosis.get("runtime_error", "")
                    or (
                        ""
                        if not smoke_row
                        else (
                            f"Answer={smoke_row.get('answer_appears')} / "
                            f"valid_parse={smoke_row.get('valid_parse')} / "
                            f"truncation_suspect={smoke_row.get('truncation_suspect')} / "
                            f"repetition_collapse={smoke_row.get('repetition_collapse')}"
                            + (
                                f" / parse_error_reason={smoke_row.get('parse_error_reason')}"
                                if smoke_row.get("parse_error_reason")
                                else ""
                            )
                        )
                    )
                    or registry_row.get("error", "")
                ),
                "improvement_method_or_next_action": "",
            }
        )
    return rows


def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    manual_overrides = {
        "Mistral-7B-Instruct-v0.3": "若 smoke 失败，优先检查官方 tokenizer chat_template 与 `mistral_common`/Transformers 兼容性；不要先改 parser",
        "Qwen2.5-7B-Instruct": "若 smoke 失败，优先检查官方 chat_template 是否稳定输出 `Answer:`，不要改评分逻辑",
        "Llama-3.1-8B-Instruct": "若下载失败，明确记为官方 gated/license 失败；若 smoke 失败，再看 chat_template 是否正确进入 assistant 边界",
        "Gemma-7B-IT": "若下载失败，明确记为官方 gated/license 失败；若 smoke 失败，再看 gemma chat_template 输出风格",
        "Yi-1.5-6B-Chat": "若 smoke 失败，优先检查是否缺少官方 chat 模板或 special-token 对齐异常",
        "ChatGLM3-6B": "若 smoke 失败，优先检查官方 `trust_remote_code` + 非标准 generate/chat 入口，而不是改 parser",
    }
    for row in rows:
        if row["model_label"] in manual_overrides and not row["improvement_method_or_next_action"]:
            row["improvement_method_or_next_action"] = manual_overrides[row["model_label"]]
    return rows


def render_markdown(rows: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
    lines = [
        "# 全量尝试模型汇总",
        "",
        f"- 更新时间：`{payload['updated_at_hkt']}`",
        f"- 已汇总模型数：`{payload['num_models_total']}`",
        f"- 已进入有效输入协议：`{payload['num_models_in_effective_input_protocol']}`",
        "",
        "| 模型 | 批次 | 定位 | wrapper | 是否进入有效输入协议 | 状态 | 成功/失败原因 | 改进方法/下一步 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model_label} | {source_batch} | {current_positioning} | {effective_wrapper} | {entered_effective_input_protocol} | {status} | {reason} | {next_action} |".format(
                model_label=row["model_label"],
                source_batch=row["source_batch"],
                current_positioning=row["current_positioning"],
                effective_wrapper=row["effective_wrapper"] or "-",
                entered_effective_input_protocol=row["entered_effective_input_protocol"],
                status=row["status"],
                reason=(row["success_or_failure_reason"] or "").replace("\n", " "),
                next_action=(row["improvement_method_or_next_action"] or "").replace("\n", " "),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    rows = [map_existing_row(row) for row in load_existing_rows(project_root)]
    rows.extend(build_mainstream_rows(project_root))
    rows = enrich_rows(rows)
    rows = sorted(rows, key=lambda row: row["model_label"].lower())

    payload = {
        "updated_at_hkt": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "num_models_total": len(rows),
        "num_models_in_effective_input_protocol": sum(1 for row in rows if row["entered_effective_input_protocol"]),
        "rows": rows,
    }

    output_dir = project_root / "outputs" / "debug" / "protocol_screening_registry"
    write_json(output_dir / "all_attempted_models_summary.json", payload)
    write_text(output_dir / "all_attempted_models_summary.md", render_markdown(rows, payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
