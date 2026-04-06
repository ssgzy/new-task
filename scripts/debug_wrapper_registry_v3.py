#!/usr/bin/env python3
"""Run one-sample Alpaca/slow-tokenizer smoke tests for Lion and Orca only."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM

from finqa_protocol_v1 import (
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


DEFAULT_LABELS = ["Lion-7B", "Orca-2-7B"]


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
    parser.add_argument("--prompt-render-mode", type=str, default=PROMPT_RENDER_MODE_REGISTRY)
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


def smoke_text(payload: Dict[str, Any]) -> str:
    parse = payload["parse"]
    tokenizer_policy = payload["tokenizer_load_policy"]
    return "\n".join(
        [
            f"# Smoke v3：{payload['model_label']}",
            "",
            f"- effective wrapper：`{payload['effective_wrapper']}`",
            f"- tokenizer requested class：`{tokenizer_policy['requested_tokenizer_class_name']}`",
            f"- tokenizer effective class：`{tokenizer_policy['effective_tokenizer_class_name']}`",
            f"- tokenizer requested use_fast：`{tokenizer_policy['requested_use_fast']}`",
            f"- tokenizer effective is_fast：`{tokenizer_policy['effective_tokenizer_is_fast']}`",
            f"- tokenizer load notes：{tokenizer_policy['notes']}",
            f"- Answer: 是否出现：`{payload['answer_appears']}`",
            f"- valid_parse：`{payload['valid_parse']}`",
            f"- truncation_suspect：`{payload['truncation_suspect']}`",
            f"- repetition_collapse：`{payload['repetition_collapse']}`",
            f"- parse_error_reason：`{parse['parse_error_reason']}`",
            f"- answer_line_raw：`{parse['answer_line_raw']}`",
            "",
            "## rendered prompt head",
            "```text",
            payload["rendered_prompt_head_500"],
            "```",
            "",
            "## rendered prompt tail",
            "```text",
            payload["rendered_prompt_tail_300"],
            "```",
            "",
            "## raw generation",
            "```text",
            payload["raw_generation_text"],
            "```",
            "",
        ]
    )


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
    tokenizer_policy = resolve_tokenizer_load_policy(model_label=label, trust_remote_code=False)
    tokenizer = load_tokenizer_with_policy(
        model_path=model_path,
        model_label=label,
        trust_remote_code=False,
    )
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
    wrapper_spec = resolve_instruction_wrapper_spec(label)

    payload = {
        "model_label": label,
        "model_path": model_path,
        "sample_id": record["id"],
        "wrapper_family": wrapper_spec.model_family,
        "default_wrapper": wrapper_spec.default_wrapper,
        "effective_wrapper": prompt_info["effective_mode"],
        "prompt_fallback_reason": prompt_info["fallback_reason"],
        "tokenizer_load_policy": {
            "requested_use_fast": tokenizer_policy.use_fast,
            "requested_tokenizer_class_name": tokenizer_policy.tokenizer_class_name,
            "effective_tokenizer_class_name": type(tokenizer).__name__,
            "effective_tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
            "trust_remote_code": tokenizer_policy.trust_remote_code,
            "notes": tokenizer_policy.notes,
        },
        "tokenizer_snapshot_after_alignment": special_token_snapshot(model=model, tokenizer=tokenizer),
        "special_token_alignment": special_token_alignment,
        "rendered_prompt": prompt,
        "rendered_prompt_head_500": prompt[:500],
        "rendered_prompt_tail_300": prompt[-300:],
        "tokenized_input_length": prompt_length,
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

    output_dir = project_root / "outputs" / "debug" / "input_smoke_v3" / label
    write_json(output_dir / "smoke_v3.json", payload)
    write_text(output_dir / "smoke_v3.txt", smoke_text(payload))
    write_text(output_dir / "raw_prompt.txt", prompt)
    write_text(output_dir / "raw_generation.txt", generation_text)

    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return payload


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    model_registry = read_registry(args.model_registry.resolve())
    record = load_manifest_sample(args.manifest.resolve(), sample_index=args.sample_index)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    rows: List[Dict[str, Any]] = []
    for label in args.labels:
        if label not in model_registry:
            raise SystemExit(f"Unknown model label in registry: {label}")
        if label not in DEFAULT_LABELS:
            raise SystemExit(f"debug_wrapper_registry_v3.py only supports Lion-7B and Orca-2-7B, got: {label}")

        payload = run_one_model(
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
        rows.append(
            {
                "model_label": label,
                "effective_wrapper": payload["effective_wrapper"],
                "tokenizer_requested_use_fast": payload["tokenizer_load_policy"]["requested_use_fast"],
                "tokenizer_requested_class": payload["tokenizer_load_policy"]["requested_tokenizer_class_name"],
                "tokenizer_effective_class": payload["tokenizer_load_policy"]["effective_tokenizer_class_name"],
                "tokenizer_effective_is_fast": payload["tokenizer_load_policy"]["effective_tokenizer_is_fast"],
                "answer_appears": payload["answer_appears"],
                "valid_parse": payload["valid_parse"],
                "truncation_suspect": payload["truncation_suspect"],
                "repetition_collapse": payload["repetition_collapse"],
                "parse_error_reason": payload["parse"]["parse_error_reason"],
            }
        )
        print(f"Wrote input_smoke_v3 outputs for {label}", flush=True)

    write_json(project_root / "outputs" / "debug" / "input_smoke_v3" / "summary.json", {"rows": rows})


if __name__ == "__main__":
    main()
