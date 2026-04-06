#!/usr/bin/env python3
"""Diagnose and smoke-test main-candidate expansion models in a provisional-only path."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from finqa_protocol_v1 import (
    PROMPT_RENDER_MODE_CHATML,
    PROMPT_RENDER_MODE_PLAIN,
    PROMPT_RENDER_MODE_REGISTRY,
    DecodeConfig,
    align_model_tokenizer_special_tokens,
    load_tokenizer_with_policy,
    parse_prediction,
    render_prompt_for_tokenizer,
    resolve_instruction_wrapper_spec,
    resolve_tokenizer_load_policy,
    special_token_snapshot,
)


DEFAULT_LABELS = [
    "Xwin-LM-7B",
    "LaMini-LLaMA-7B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "OpenR1-Distill-7B",
]

RUNNABLE_STATUSES = {"available_local", "available_or_downloaded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=project_root
        / "outputs"
        / "metadata"
        / "provisional"
        / "model_registry.groups-main-candidate-expansion.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "val_calib50.jsonl",
    )
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=DecodeConfig().repetition_penalty)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_manifest_sample(path: Path, sample_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                return json.loads(line)
    raise SystemExit(f"sample_index={sample_index} out of range for {path}")


def read_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_json(path)
    return {row["label"]: row for row in rows}


def resolve_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def detect_repetition_collapse(text: str) -> bool:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < 20:
        return False
    for n in range(1, 6):
        counts: Dict[tuple[str, ...], int] = {}
        for idx in range(0, len(tokens) - n + 1):
            key = tuple(tokens[idx : idx + n])
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            continue
        top_count = max(counts.values())
        if top_count >= 10 and top_count * n >= len(tokens) * 0.5:
            return True
    return False


def candidate_prompt_modes(label: str) -> List[str]:
    if label == "OpenR1-Distill-7B":
        return [PROMPT_RENDER_MODE_CHATML, PROMPT_RENDER_MODE_PLAIN]
    return [PROMPT_RENDER_MODE_REGISTRY]


def smoke_score(row: Dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["valid_parse"]),
        int(row["answer_appears"]),
        int(not row["truncation_suspect"]),
        int(not row["repetition_collapse"]),
    )


def smoke_row_text(label: str, diagnosis_payload: Dict[str, Any], smoke_payload: Dict[str, Any]) -> str:
    recommended = smoke_payload.get("recommended_candidate")
    lines = [
        f"# 扩展模型 smoke：{label}",
        "",
        f"- registry status：`{diagnosis_payload['registry_status']}`",
        f"- model_family：`{diagnosis_payload.get('model_family', '')}`",
        f"- wrapper family：`{diagnosis_payload['wrapper_family']}`",
        f"- tokenizer class：`{diagnosis_payload['tokenizer'].get('class_name')}`",
        f"- tokenizer is_fast：`{diagnosis_payload['tokenizer'].get('is_fast')}`",
        f"- tokenizer.chat_template：`{diagnosis_payload['tokenizer'].get('chat_template_exists')}`",
        "",
    ]
    if diagnosis_payload["registry_status"] not in RUNNABLE_STATUSES:
        lines.extend(
            [
                f"- skip reason：{diagnosis_payload.get('skip_reason', '')}",
                "",
            ]
        )
        return "\n".join(lines)

    if recommended:
        lines.extend(
            [
                f"- recommended wrapper mode：`{recommended['requested_prompt_mode']}`",
                f"- effective wrapper：`{recommended['effective_wrapper']}`",
                f"- Answer: 是否出现：`{recommended['answer_appears']}`",
                f"- valid_parse：`{recommended['valid_parse']}`",
                f"- truncation_suspect：`{recommended['truncation_suspect']}`",
                f"- repetition_collapse：`{recommended['repetition_collapse']}`",
                f"- parse_error_reason：`{recommended['parse_error_reason']}`",
                "",
                "## rendered prompt head",
                "```text",
                recommended["rendered_prompt_head_500"],
                "```",
                "",
                "## rendered prompt tail",
                "```text",
                recommended["rendered_prompt_tail_300"],
                "```",
                "",
                "## raw generation",
                "```text",
                recommended["raw_generation_text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_skip_payload(label: str, registry_row: Dict[str, Any], skip_reason: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    wrapper_spec = resolve_instruction_wrapper_spec(label)
    diagnosis_payload = {
        "model_label": label,
        "group": registry_row.get("group", ""),
        "model_family": registry_row.get("model_family", ""),
        "repo_id": registry_row.get("repo_id", ""),
        "registry_status": registry_row.get("status", "missing_registry_row"),
        "snapshot_path": registry_row.get("snapshot_path", ""),
        "wrapper_family": wrapper_spec.model_family,
        "default_wrapper": wrapper_spec.default_wrapper,
        "allow_chat_template": wrapper_spec.allow_chat_template,
        "fallback_wrapper": wrapper_spec.fallback_wrapper,
        "wrapper_notes": wrapper_spec.notes,
        "tokenizer_load_policy": {
            "requested_use_fast": None,
            "requested_tokenizer_class_name": "",
            "effective_tokenizer_class_name": "",
            "effective_tokenizer_is_fast": None,
            "trust_remote_code": False,
            "notes": "",
        },
        "tokenizer": {
            "class_name": None,
            "is_fast": None,
            "chat_template_exists": None,
            "pad_token": None,
            "pad_token_id": None,
            "eos_token": None,
            "eos_token_id": None,
            "bos_token": None,
            "bos_token_id": None,
            "unk_token": None,
            "unk_token_id": None,
            "special_tokens_map": {},
        },
        "model_config": {
            "model_type": None,
            "pad_token_id": None,
            "eos_token_id": None,
            "bos_token_id": None,
            "token_id_alignment_reasonable": None,
        },
        "prompt_candidates": [],
        "skip_reason": skip_reason,
    }
    smoke_payload = {
        "model_label": label,
        "registry_status": registry_row.get("status", "missing_registry_row"),
        "status": "skipped",
        "skip_reason": skip_reason,
        "recommended_candidate": None,
        "candidate_rows": [],
    }
    return diagnosis_payload, smoke_payload


def build_runnable_diagnosis(
    label: str,
    registry_row: Dict[str, Any],
    tokenizer: Any,
    config: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    wrapper_spec = resolve_instruction_wrapper_spec(label)
    tokenizer_policy = resolve_tokenizer_load_policy(model_label=label, trust_remote_code=False)
    prompt_candidates: List[Dict[str, Any]] = []
    for prompt_mode in candidate_prompt_modes(label):
        prompt_info = render_prompt_for_tokenizer(
            record=record,
            tokenizer=tokenizer,
            prompt_render_mode=prompt_mode,
            model_label=label,
        )
        prompt_candidates.append(
            {
                "requested_prompt_mode": prompt_mode,
                "resolved_mode": prompt_info["resolved_mode"],
                "effective_mode": prompt_info["effective_mode"],
                "fallback_reason": prompt_info["fallback_reason"],
                "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
                "rendered_prompt_head_500": prompt_info["prompt"][:500],
                "rendered_prompt_tail_300": prompt_info["prompt"][-300:],
            }
        )

    tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    model_pad_token_id = getattr(config, "pad_token_id", None)
    model_eos_token_id = getattr(config, "eos_token_id", None)
    return {
        "model_label": label,
        "group": registry_row.get("group", ""),
        "model_family": registry_row.get("model_family", ""),
        "repo_id": registry_row.get("repo_id", ""),
        "registry_status": registry_row.get("status", ""),
        "snapshot_path": registry_row.get("snapshot_path", ""),
        "wrapper_family": wrapper_spec.model_family,
        "default_wrapper": wrapper_spec.default_wrapper,
        "allow_chat_template": wrapper_spec.allow_chat_template,
        "fallback_wrapper": wrapper_spec.fallback_wrapper,
        "wrapper_notes": wrapper_spec.notes,
        "tokenizer_load_policy": {
            "requested_use_fast": tokenizer_policy.use_fast,
            "requested_tokenizer_class_name": tokenizer_policy.tokenizer_class_name,
            "effective_tokenizer_class_name": type(tokenizer).__name__,
            "effective_tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
            "trust_remote_code": tokenizer_policy.trust_remote_code,
            "notes": tokenizer_policy.notes,
        },
        "tokenizer": {
            "class_name": type(tokenizer).__name__,
            "is_fast": getattr(tokenizer, "is_fast", None),
            "chat_template_exists": bool(getattr(tokenizer, "chat_template", None)),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": tokenizer_pad_token_id,
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": tokenizer_eos_token_id,
            "bos_token": getattr(tokenizer, "bos_token", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "unk_token": getattr(tokenizer, "unk_token", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None),
            "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        },
        "model_config": {
            "model_type": getattr(config, "model_type", None),
            "pad_token_id": model_pad_token_id,
            "eos_token_id": model_eos_token_id,
            "bos_token_id": getattr(config, "bos_token_id", None),
            "token_id_alignment_reasonable": (
                (model_pad_token_id in {None, tokenizer_pad_token_id})
                and (model_eos_token_id in {None, tokenizer_eos_token_id})
            ),
        },
        "prompt_candidates": prompt_candidates,
        "skip_reason": "",
    }


def run_candidate_generation(
    label: str,
    prompt_mode: str,
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    prompt_info = render_prompt_for_tokenizer(
        record=record,
        tokenizer=tokenizer,
        prompt_render_mode=prompt_mode,
        model_label=label,
    )
    inputs = tokenizer(prompt_info["prompt"], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

    generated_ids = generated[0, prompt_len:]
    raw_generation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parse = parse_prediction(
        raw_generation_text,
        max_new_tokens_hit=int(generated_ids.shape[-1]) >= max_new_tokens,
    )
    return {
        "requested_prompt_mode": prompt_mode,
        "effective_wrapper": prompt_info["effective_mode"],
        "wrapper_family": prompt_info["wrapper_family"],
        "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
        "prompt_fallback_reason": prompt_info["fallback_reason"],
        "rendered_prompt_head_500": prompt_info["prompt"][:500],
        "rendered_prompt_tail_300": prompt_info["prompt"][-300:],
        "tokenized_input_length": prompt_len,
        "raw_generation_text": raw_generation_text,
        "answer_appears": parse["format_ok"],
        "valid_parse": parse["valid_parse"],
        "truncation_suspect": parse["truncated_suspect"],
        "repetition_collapse": detect_repetition_collapse(raw_generation_text),
        "parse_error_reason": parse["parse_error_reason"],
        "answer_line_raw": parse["answer_line_raw"],
        "parse": parse,
        "new_tokens": int(generated_ids.shape[-1]),
        "latency_ms": latency_ms,
    }


def run_model_smoke(
    label: str,
    registry_row: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    record: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    candidate_rows = [
        run_candidate_generation(
            label=label,
            prompt_mode=prompt_mode,
            model=model,
            tokenizer=tokenizer,
            record=record,
            device=device,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        for prompt_mode in candidate_prompt_modes(label)
    ]
    recommended_candidate = max(candidate_rows, key=smoke_score) if candidate_rows else None
    return {
        "model_label": label,
        "repo_id": registry_row.get("repo_id", ""),
        "registry_status": registry_row.get("status", ""),
        "status": "ok",
        "candidate_rows": candidate_rows,
        "recommended_candidate": recommended_candidate,
    }


def registry_text(payload: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Wrapper Registry Expansion：{payload['model_label']}",
            "",
            f"- group：`{payload['group']}`",
            f"- model_family：`{payload['model_family']}`",
            f"- wrapper family：`{payload['wrapper_family']}`",
            f"- default wrapper：`{payload['default_wrapper']}`",
            f"- allow chat_template：`{payload['allow_chat_template']}`",
            f"- fallback wrapper：`{payload['fallback_wrapper']}`",
            f"- notes：{payload['wrapper_notes']}",
            "",
        ]
    )


def diagnosis_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"# 输入接口诊断 Expansion：{payload['model_label']}",
        "",
        f"- registry status：`{payload['registry_status']}`",
        f"- group：`{payload['group']}`",
        f"- model_family：`{payload['model_family']}`",
        f"- wrapper family：`{payload['wrapper_family']}`",
        f"- default wrapper：`{payload['default_wrapper']}`",
        f"- tokenizer class：`{payload['tokenizer']['class_name']}`",
        f"- tokenizer is_fast：`{payload['tokenizer']['is_fast']}`",
        f"- tokenizer.chat_template：`{payload['tokenizer']['chat_template_exists']}`",
        f"- tokenizer pad/eos/bos/unk：`{payload['tokenizer']['pad_token']}` / `{payload['tokenizer']['eos_token']}` / `{payload['tokenizer']['bos_token']}` / `{payload['tokenizer']['unk_token']}`",
        f"- tokenizer pad/eos id：`{payload['tokenizer']['pad_token_id']}` / `{payload['tokenizer']['eos_token_id']}`",
        f"- model config pad/eos id：`{payload['model_config']['pad_token_id']}` / `{payload['model_config']['eos_token_id']}`",
        f"- token id 对齐是否合理：`{payload['model_config']['token_id_alignment_reasonable']}`",
        "",
    ]
    if payload["skip_reason"]:
        lines.extend([f"- skip reason：{payload['skip_reason']}", ""])
        return "\n".join(lines)

    for prompt_info in payload["prompt_candidates"]:
        lines.extend(
            [
                f"## Prompt Candidate：{prompt_info['requested_prompt_mode']}",
                f"- effective mode：`{prompt_info['effective_mode']}`",
                f"- generation boundary：`{prompt_info['has_assistant_generation_boundary']}`",
                f"- fallback reason：`{prompt_info['fallback_reason']}`",
                "",
                "### rendered prompt head",
                "```text",
                prompt_info["rendered_prompt_head_500"],
                "```",
                "",
                "### rendered prompt tail",
                "```text",
                prompt_info["rendered_prompt_tail_300"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def emit_outputs(project_root: Path, label: str, diagnosis_payload: Dict[str, Any], smoke_payload: Dict[str, Any]) -> None:
    wrapper_dir = project_root / "outputs" / "debug" / "wrapper_registry_expansion"
    diagnosis_dir = project_root / "outputs" / "debug" / "input_interface_diagnosis_expansion"
    smoke_dir = project_root / "outputs" / "debug" / "input_smoke_expansion" / label

    write_json(wrapper_dir / f"{label}.json", diagnosis_payload)
    write_text(wrapper_dir / f"{label}.txt", registry_text(diagnosis_payload))
    write_json(diagnosis_dir / f"{label}.json", diagnosis_payload)
    write_text(diagnosis_dir / f"{label}.txt", diagnosis_text(diagnosis_payload))
    write_json(smoke_dir / "smoke_expansion.json", smoke_payload)
    write_text(smoke_dir / "smoke_expansion.txt", smoke_row_text(label, diagnosis_payload, smoke_payload))

    for candidate in smoke_payload.get("candidate_rows", []):
        prompt_mode = candidate["requested_prompt_mode"]
        write_json(smoke_dir / f"smoke_{prompt_mode}.json", candidate)
        write_text(smoke_dir / f"raw_generation_{prompt_mode}.txt", candidate["raw_generation_text"])


def process_one_label(
    label: str,
    registry: Dict[str, Dict[str, Any]],
    record: Dict[str, Any],
    project_root: Path,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Dict[str, Any]:
    registry_row = registry.get(label)
    if registry_row is None:
        diagnosis_payload, smoke_payload = build_skip_payload(
            label=label,
            registry_row={},
            skip_reason="missing_registry_row",
        )
        emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
        return {
            "model_label": label,
            "status": "skipped",
            "skip_reason": "missing_registry_row",
            "recommended_wrapper": None,
            "effective_wrapper": None,
            "tokenizer_effective_class": None,
            "answer_appears": False,
            "valid_parse": False,
            "truncation_suspect": None,
            "repetition_collapse": None,
            "parse_error_reason": "missing_registry_row",
        }

    if registry_row.get("status") not in RUNNABLE_STATUSES or not registry_row.get("snapshot_path"):
        skip_reason = registry_row.get("error") or f"model status={registry_row.get('status')} is not runnable"
        diagnosis_payload, smoke_payload = build_skip_payload(
            label=label,
            registry_row=registry_row,
            skip_reason=skip_reason,
        )
        emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
        return {
            "model_label": label,
            "status": "skipped",
            "skip_reason": skip_reason,
            "recommended_wrapper": None,
            "effective_wrapper": None,
            "tokenizer_effective_class": None,
            "answer_appears": False,
            "valid_parse": False,
            "truncation_suspect": None,
            "repetition_collapse": None,
            "parse_error_reason": "skipped_missing_or_placeholder_model",
        }

    try:
        tokenizer = load_tokenizer_with_policy(
            model_path=registry_row["snapshot_path"],
            model_label=label,
            trust_remote_code=False,
        )
        config = AutoConfig.from_pretrained(registry_row["snapshot_path"], trust_remote_code=False)
        diagnosis_payload = build_runnable_diagnosis(
            label=label,
            registry_row=registry_row,
            tokenizer=tokenizer,
            config=config,
            record=record,
        )

        model = AutoModelForCausalLM.from_pretrained(
            registry_row["snapshot_path"],
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        special_token_alignment = align_model_tokenizer_special_tokens(model=model, tokenizer=tokenizer)
        diagnosis_payload["special_token_alignment"] = special_token_alignment
        diagnosis_payload["tokenizer_snapshot_after_alignment"] = special_token_snapshot(
            model=model,
            tokenizer=tokenizer,
        )
        model.eval()
        model.to(device)

        smoke_payload = run_model_smoke(
            label=label,
            registry_row=registry_row,
            tokenizer=tokenizer,
            model=model,
            record=record,
            device=device,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        emit_outputs(project_root, label, diagnosis_payload, smoke_payload)

        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        recommended = smoke_payload["recommended_candidate"] or {}
        return {
            "model_label": label,
            "status": smoke_payload["status"],
            "skip_reason": "",
            "recommended_wrapper": recommended.get("requested_prompt_mode"),
            "effective_wrapper": recommended.get("effective_wrapper"),
            "tokenizer_effective_class": diagnosis_payload["tokenizer"]["class_name"],
            "answer_appears": recommended.get("answer_appears"),
            "valid_parse": recommended.get("valid_parse"),
            "truncation_suspect": recommended.get("truncation_suspect"),
            "repetition_collapse": recommended.get("repetition_collapse"),
            "parse_error_reason": recommended.get("parse_error_reason"),
        }
    except Exception as exc:
        diagnosis_payload, smoke_payload = build_skip_payload(
            label=label,
            registry_row=registry_row,
            skip_reason=f"{type(exc).__name__}: {exc}",
        )
        diagnosis_payload["registry_status"] = registry_row.get("status", "")
        smoke_payload["status"] = "run_failed"
        emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
        return {
            "model_label": label,
            "status": "run_failed",
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "recommended_wrapper": None,
            "effective_wrapper": None,
            "tokenizer_effective_class": None,
            "answer_appears": False,
            "valid_parse": False,
            "truncation_suspect": None,
            "repetition_collapse": None,
            "parse_error_reason": "runtime_exception",
        }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry_path = args.model_registry.resolve()
    if not registry_path.exists():
        fallback = project_root / "outputs" / "metadata" / "model_registry.json"
        if fallback.exists():
            registry_path = fallback
        else:
            raise SystemExit(f"Registry not found: {args.model_registry}")

    registry = read_registry(registry_path)
    record = load_manifest_sample(args.manifest.resolve(), sample_index=args.sample_index)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    summary_rows: List[Dict[str, Any]] = []
    for label in args.labels:
        row = process_one_label(
            label=label,
            registry=registry,
            record=record,
            project_root=project_root,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        summary_rows.append(row)
        print(
            f"{label}: status={row['status']} recommended={row['recommended_wrapper'] or '-'} "
            f"valid_parse={row['valid_parse']} answer={row['answer_appears']}",
            flush=True,
        )

    write_json(
        project_root / "outputs" / "debug" / "input_smoke_expansion" / "summary.json",
        {
            "registry_path": str(registry_path),
            "sample_index": args.sample_index,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
