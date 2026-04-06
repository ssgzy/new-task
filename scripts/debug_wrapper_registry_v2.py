#!/usr/bin/env python3
"""Run one-sample wrapper-registry smoke tests and write provisional debug outputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finqa_protocol_v1 import (
    INSTRUCTION_WRAPPER_REGISTRY,
    PROMPT_RENDER_MODE_REGISTRY,
    SUPPORTED_PROMPT_RENDER_MODES,
    DecodeConfig,
    align_model_tokenizer_special_tokens,
    parse_prediction,
    render_prompt_for_tokenizer,
    resolve_instruction_wrapper_spec,
    special_token_snapshot,
)


DEFAULT_LABELS = [
    "Lion-7B",
    "Orca-2-7B",
    "Zephyr-7B-beta",
    "MiniLLM-Llama-7B",
]

LIKELY_INSTRUCTION_TUNED = {
    "Lion-7B": True,
    "Orca-2-7B": True,
    "Zephyr-7B-beta": True,
    "MiniLLM-Llama-7B": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
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
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt-render-mode", type=str, default=PROMPT_RENDER_MODE_REGISTRY, choices=sorted(SUPPORTED_PROMPT_RENDER_MODES))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=DecodeConfig().repetition_penalty)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_manifest_sample(path: Path, sample_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                return json.loads(line)
    raise SystemExit(f"sample_index={sample_index} out of range for {path}")


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


def read_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_json(path)
    return {row["label"]: row for row in rows}


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


def wrapper_registry_text(label: str, payload: Dict[str, Any]) -> str:
    spec = payload["wrapper_spec"]
    prompt = payload["prompt_render"]
    token_align = payload["special_token_alignment"]
    lines = [
        f"# Wrapper Registry 诊断：{label}",
        "",
        f"- 模型路径：`{payload['model_path']}`",
        f"- 是否 instruction/chat tuned：`{payload['likely_instruction_tuned']}`",
        f"- 模型家族：`{spec['model_family']}`",
        f"- registry 默认 wrapper：`{spec['default_wrapper']}`",
        f"- registry 是否允许 chat_template 对照：`{spec['allow_chat_template']}`",
        f"- 当前请求渲染模式：`{prompt['requested_mode']}`",
        f"- 当前实际 wrapper：`{prompt['effective_mode']}`",
        f"- tokenizer.chat_template 是否存在：`{prompt['tokenizer_has_chat_template']}`",
        f"- 是否使用 apply_chat_template：`{prompt['used_chat_template']}`",
        f"- fallback 原因：`{prompt['fallback_reason'] or 'none'}`",
        f"- 是否存在 assistant generation boundary：`{prompt['has_assistant_generation_boundary']}`",
        "",
        "## special-token 对齐修复",
        f"- 修复前：`{json.dumps(token_align['before'], ensure_ascii=False)}`",
        f"- 修复动作：`{json.dumps(token_align['actions'], ensure_ascii=False)}`",
        f"- 修复后：`{json.dumps(token_align['after'], ensure_ascii=False)}`",
        f"- 是否发生修改：`{token_align['changed']}`",
        "",
        "## 渲染后 prompt 头部 500 字符",
        "```text",
        prompt["prompt_head_500"],
        "```",
        "",
        "## 渲染后 prompt 尾部 300 字符",
        "```text",
        prompt["prompt_tail_300"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def smoke_text(label: str, payload: Dict[str, Any]) -> str:
    parse = payload["parse"]
    lines = [
        f"# Wrapper Smoke v2：{label}",
        "",
        f"- 使用 wrapper：`{payload['prompt_render_mode_effective']}`",
        f"- wrapper family：`{payload['wrapper_family']}`",
        f"- fallback 原因：`{payload['prompt_fallback_reason'] or 'none'}`",
        f"- special-token 对齐是否修复：`{payload['special_token_alignment']['changed']}`",
        f"- Answer: 是否出现：`{payload['answer_appears']}`",
        f"- valid_parse：`{parse['valid_parse']}`",
        f"- truncation_suspect：`{payload['truncation_suspect']}`",
        f"- repetition_collapse：`{payload['repetition_collapse']}`",
        f"- new_tokens：`{payload['new_tokens']}`",
        f"- latency_ms：`{payload['latency_ms']}`",
        f"- answer_line_raw：`{parse['answer_line_raw']}`",
        f"- parse_error_reason：`{parse['parse_error_reason']}`",
        "",
        "## raw prompt 头部 500 字符",
        "```text",
        payload["raw_prompt_head_500"],
        "```",
        "",
        "## raw prompt 尾部 300 字符",
        "```text",
        payload["raw_prompt_tail_300"],
        "```",
        "",
        "## raw generation 前 1600 字符",
        "```text",
        payload["raw_generation_text"][:1600],
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_registry_index(project_root: Path) -> None:
    payload = {
        "registry_version": "wrapper_registry_v2_provisional",
        "instruction_wrapper_registry": {
            model_label: asdict(spec) for model_label, spec in INSTRUCTION_WRAPPER_REGISTRY.items()
        },
    }
    output_dir = project_root / "outputs" / "debug" / "wrapper_registry"
    write_json(output_dir / "registry_index.json", payload)
    lines = ["# Wrapper Registry v2", ""]
    for model_label, spec in INSTRUCTION_WRAPPER_REGISTRY.items():
        lines.extend(
            [
                f"## {model_label}",
                f"- model_family: `{spec.model_family}`",
                f"- default_wrapper: `{spec.default_wrapper}`",
                f"- allow_chat_template: `{spec.allow_chat_template}`",
                f"- fallback_wrapper: `{spec.fallback_wrapper}`",
                f"- notes: {spec.notes}",
                "",
            ]
        )
    write_text(output_dir / "registry_index.txt", "\n".join(lines).rstrip() + "\n")


def run_one_model(
    label: str,
    model_path: str,
    record: Dict[str, Any],
    device: str,
    dtype: torch.dtype,
    prompt_render_mode: str,
    max_new_tokens: int,
    repetition_penalty: float,
    project_root: Path,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    special_token_alignment = align_model_tokenizer_special_tokens(model=model, tokenizer=tokenizer)
    model.eval()
    model.to(device)

    prompt_info = render_prompt_for_tokenizer(
        record=record,
        tokenizer=tokenizer,
        prompt_render_mode=prompt_render_mode,
        model_label=label,
    )
    prompt = prompt_info["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[-1])

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
    generated_ids = generated[0, prompt_length:]
    generation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parse = parse_prediction(
        generation_text,
        max_new_tokens_hit=int(generated_ids.shape[-1]) >= max_new_tokens,
    )

    spec = resolve_instruction_wrapper_spec(label)
    wrapper_payload = {
        "model_label": label,
        "model_path": model_path,
        "sample_id": record["id"],
        "likely_instruction_tuned": LIKELY_INSTRUCTION_TUNED.get(label, None),
        "wrapper_spec": asdict(spec),
        "prompt_render": {
            "requested_mode": prompt_info["requested_mode"],
            "resolved_mode": prompt_info["resolved_mode"],
            "effective_mode": prompt_info["effective_mode"],
            "wrapper_family": prompt_info["wrapper_family"],
            "registry_default_wrapper": prompt_info["registry_default_wrapper"],
            "registry_allow_chat_template": prompt_info["registry_allow_chat_template"],
            "registry_fallback_wrapper": prompt_info["registry_fallback_wrapper"],
            "tokenizer_has_chat_template": prompt_info["tokenizer_has_chat_template"],
            "used_chat_template": prompt_info["used_chat_template"],
            "fallback_reason": prompt_info["fallback_reason"],
            "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
            "prompt_head_500": prompt[:500],
            "prompt_tail_300": prompt[-300:],
        },
        "tokenizer_snapshot_after_alignment": special_token_snapshot(model=model, tokenizer=tokenizer),
        "special_token_alignment": special_token_alignment,
    }

    smoke_payload = {
        "model_label": label,
        "sample_id": record["id"],
        "prompt_render_mode_requested": prompt_info["requested_mode"],
        "prompt_render_mode_resolved": prompt_info["resolved_mode"],
        "prompt_render_mode_effective": prompt_info["effective_mode"],
        "wrapper_family": prompt_info["wrapper_family"],
        "prompt_fallback_reason": prompt_info["fallback_reason"],
        "used_chat_template": prompt_info["used_chat_template"],
        "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
        "raw_prompt": prompt,
        "raw_prompt_head_500": prompt[:500],
        "raw_prompt_tail_300": prompt[-300:],
        "tokenized_input_length": prompt_length,
        "tokenizer_snapshot_after_alignment": special_token_snapshot(model=model, tokenizer=tokenizer),
        "special_token_alignment": special_token_alignment,
        "raw_generation_text": generation_text,
        "answer_appears": parse["format_ok"],
        "valid_parse": parse["valid_parse"],
        "truncation_suspect": parse["truncated_suspect"],
        "repetition_collapse": detect_repetition_collapse(generation_text),
        "parse": parse,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "new_tokens": int(generated_ids.shape[-1]),
        "latency_ms": latency_ms,
    }

    registry_dir = project_root / "outputs" / "debug" / "wrapper_registry"
    smoke_dir = project_root / "outputs" / "debug" / "input_smoke_v2" / label
    write_json(registry_dir / f"{label}.json", wrapper_payload)
    write_text(registry_dir / f"{label}.txt", wrapper_registry_text(label, wrapper_payload))
    write_json(smoke_dir / "smoke_v2.json", smoke_payload)
    write_text(smoke_dir / "smoke_v2.txt", smoke_text(label, smoke_payload))
    write_text(smoke_dir / "raw_prompt.txt", prompt)
    write_text(smoke_dir / "raw_generation.txt", generation_text)

    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return smoke_payload


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    model_registry = read_registry(args.model_registry.resolve())
    record = load_manifest_sample(args.manifest.resolve(), sample_index=args.sample_index)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    write_registry_index(project_root=project_root)

    summary_rows: List[Dict[str, Any]] = []
    for label in args.labels:
        if label not in model_registry:
            raise SystemExit(f"Unknown model label in registry: {label}")

        smoke_payload = run_one_model(
            label=label,
            model_path=model_registry[label]["snapshot_path"],
            record=record,
            device=device,
            dtype=dtype,
            prompt_render_mode=args.prompt_render_mode,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            project_root=project_root,
        )
        summary_rows.append(
            {
                "model_label": label,
                "wrapper_family": smoke_payload["wrapper_family"],
                "effective_wrapper": smoke_payload["prompt_render_mode_effective"],
                "answer_appears": smoke_payload["answer_appears"],
                "valid_parse": smoke_payload["valid_parse"],
                "truncation_suspect": smoke_payload["truncation_suspect"],
                "repetition_collapse": smoke_payload["repetition_collapse"],
                "special_token_alignment_changed": smoke_payload["special_token_alignment"]["changed"],
                "parse_error_reason": smoke_payload["parse"]["parse_error_reason"],
            }
        )
        print(f"Wrote wrapper registry and smoke v2 outputs for {label}", flush=True)

    write_json(project_root / "outputs" / "debug" / "input_smoke_v2" / "summary.json", {"rows": summary_rows})


if __name__ == "__main__":
    main()
