#!/usr/bin/env python3
"""Run the frozen FinQA protocol v1 on a local Hugging Face causal LM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM

from finqa_protocol_v1 import (
    PROMPT_RENDER_MODE_AUTO,
    SUPPORTED_PROMPT_RENDER_MODES,
    DecodeConfig,
    align_model_tokenizer_special_tokens,
    exact_match,
    load_tokenizer_with_policy,
    parse_prediction,
    render_prompt_for_tokenizer,
    resolve_tokenizer_load_policy,
    tolerance_match,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=DecodeConfig().max_new_tokens)
    parser.add_argument("--temperature", type=float, default=DecodeConfig().temperature)
    parser.add_argument("--top-p", type=float, default=DecodeConfig().top_p)
    parser.add_argument("--repetition-penalty", type=float, default=DecodeConfig().repetition_penalty)
    parser.add_argument(
        "--prompt-render-mode",
        type=str,
        default=PROMPT_RENDER_MODE_AUTO,
        choices=sorted(SUPPORTED_PROMPT_RENDER_MODES),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="If output JSONL already exists, skip completed ids and continue appending.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")
        f.flush()


def load_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_manifest(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


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


def current_memory_bytes(device: str) -> Optional[int]:
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    return None


def load_model_and_tokenizer(
    model_label: str,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    tokenizer_load_policy = resolve_tokenizer_load_policy(
        model_label=model_label,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = load_tokenizer_with_policy(
        model_path=model_path,
        model_label=model_label,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    special_token_alignment = align_model_tokenizer_special_tokens(model=model, tokenizer=tokenizer)
    model.eval()
    model.to(device)
    tokenizer_policy_summary = {
        "model_label": tokenizer_load_policy.model_label,
        "requested_use_fast": tokenizer_load_policy.use_fast,
        "requested_tokenizer_class_name": tokenizer_load_policy.tokenizer_class_name,
        "effective_tokenizer_class_name": type(tokenizer).__name__,
        "effective_tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
        "trust_remote_code": tokenizer_load_policy.trust_remote_code,
        "notes": tokenizer_load_policy.notes,
    }
    return model, tokenizer, tokenizer_policy_summary, special_token_alignment


def summarize(
    results: List[Dict[str, Any]],
    model_label: str,
    model_path: str,
    manifest_path: Path,
    device: str,
    dtype: str,
    max_new_tokens: int,
    prompt_render_mode: str,
    repetition_penalty: float,
    tokenizer_load_policy: Dict[str, Any],
    special_token_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    total = len(results)
    runtime_success_count = sum(1 for row in results if row["runtime_success"])
    format_ok_count = sum(1 for row in results if row["parse"]["format_ok"])
    valid_parse_count = sum(1 for row in results if row["parse"]["valid_parse"])
    em_count = sum(1 for row in results if row["em"])
    tm_count = sum(1 for row in results if row["tm"])
    trunc_wo_answer_count = sum(
        1 for row in results if row["parse"]["truncated_suspect"] and not row["parse"]["format_ok"]
    )
    latencies_ms = [row["latency_ms"] for row in results if row["runtime_success"] and row["latency_ms"] is not None]
    output_tokens = [row["new_tokens"] for row in results if row["runtime_success"] and row["new_tokens"] is not None]
    peak_vram_values = [row["peak_vram_bytes"] for row in results if row["peak_vram_bytes"] is not None]

    total_latency_sec = sum(value / 1000.0 for value in latencies_ms)
    total_output_tokens = sum(output_tokens)
    tok_per_sec = (total_output_tokens / total_latency_sec) if total_latency_sec > 0 else None

    return {
        "model_label": model_label,
        "model_path": model_path,
        "manifest": str(manifest_path),
        "num_examples": total,
        "runtime_success": round(runtime_success_count / total, 6) if total else 0.0,
        "format_ok": round(format_ok_count / total, 6) if total else 0.0,
        "valid_parse": round(valid_parse_count / total, 6) if total else 0.0,
        "em": round(em_count / total, 6) if total else 0.0,
        "tm": round(tm_count / total, 6) if total else 0.0,
        "truncation_without_answer_rate": round(trunc_wo_answer_count / total, 6) if total else 0.0,
        "answer_present_rate": round(format_ok_count / total, 6) if total else 0.0,
        "valid_parse_rate": round(valid_parse_count / total, 6) if total else 0.0,
        "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else None,
        "mean_output_tokens": round(sum(output_tokens) / len(output_tokens), 2) if output_tokens else None,
        "mean_new_tokens": round(sum(output_tokens) / len(output_tokens), 2) if output_tokens else None,
        "p95_new_tokens": sorted(output_tokens)[max(0, int(0.95 * len(output_tokens)) - 1)] if output_tokens else None,
        "tok_per_sec": round(tok_per_sec, 4) if tok_per_sec is not None else None,
        "peak_vram": max(peak_vram_values) if peak_vram_values else None,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "prompt_render_mode": prompt_render_mode,
        "repetition_penalty": repetition_penalty,
        "tokenizer_load_policy": tokenizer_load_policy,
        "special_token_alignment": special_token_alignment,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    records = load_manifest(args.manifest.resolve(), limit=args.limit)
    output_jsonl = args.output_jsonl.resolve()
    summary_json = args.summary_json.resolve()

    existing_rows = load_existing_jsonl(output_jsonl) if args.resume_existing else []
    completed_ids = {row["id"] for row in existing_rows}
    pending_records = [record for record in records if record["id"] not in completed_ids]

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    model, tokenizer, tokenizer_load_policy, special_token_alignment = load_model_and_tokenizer(
        model_label=args.model_label,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    rows: List[Dict[str, Any]] = list(existing_rows)
    for record in pending_records:
        prompt_info = render_prompt_for_tokenizer(
            record=record,
            tokenizer=tokenizer,
            prompt_render_mode=args.prompt_render_mode,
            model_label=args.model_label,
        )
        prompt = prompt_info["prompt"]
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
        prompt_length = int(prompt_inputs["input_ids"].shape[-1])

        row: Dict[str, Any] = {
            "id": record["id"],
            "split": record["split"],
            "raw_answer": record["raw_answer"],
            "gold_numeric": record["gold_numeric"],
            "runtime_success": False,
            "error": None,
            "prompt_render_mode_requested": prompt_info["requested_mode"],
            "prompt_wrapper_family": prompt_info["wrapper_family"],
            "prompt_registry_default_wrapper": prompt_info["registry_default_wrapper"],
            "prompt_render_mode_effective": prompt_info["effective_mode"],
            "prompt_uses_chat_template": prompt_info["used_chat_template"],
            "prompt_fallback_reason": prompt_info["fallback_reason"],
            "prompt_length_tokens": prompt_length,
            "tokenizer_load_policy": tokenizer_load_policy,
            "special_token_alignment": special_token_alignment,
            "new_tokens": None,
            "latency_ms": None,
            "peak_vram_bytes": current_memory_bytes(device),
            "prediction_text": "",
            "parse": {
                "format_ok": False,
                "valid_parse": False,
                "pred_value": None,
                "pred_unit_type": None,
                "answer_line_raw": "",
                "multiple_answer_lines": False,
                "truncated_suspect": False,
                "parse_error_reason": "not_run",
            },
            "em": False,
            "tm": False,
        }

        try:
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False if args.temperature == 0 else True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            latency_ms = (time.perf_counter() - start) * 1000.0
            generated_ids = generated[0, prompt_length:]
            prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parse = parse_prediction(
                prediction_text,
                max_new_tokens_hit=int(generated_ids.shape[-1]) >= args.max_new_tokens,
            )
            pred_value = parse["pred_value"]

            row["runtime_success"] = True
            row["latency_ms"] = round(latency_ms, 2)
            row["new_tokens"] = int(generated_ids.shape[-1])
            row["peak_vram_bytes"] = current_memory_bytes(device)
            row["prediction_text"] = prediction_text
            row["parse"] = parse
            row["em"] = exact_match(pred_value, record["gold_numeric"]) if parse["valid_parse"] else False
            row["tm"] = tolerance_match(pred_value, record["gold_numeric"]) if parse["valid_parse"] else False
        except Exception as exc:  # pragma: no cover - runtime dependent
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)
        append_jsonl_row(output_jsonl, row)

    summary = summarize(
        results=rows,
        model_label=args.model_label,
        model_path=args.model_path,
        manifest_path=args.manifest.resolve(),
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        max_new_tokens=args.max_new_tokens,
        prompt_render_mode=args.prompt_render_mode,
        repetition_penalty=args.repetition_penalty,
        tokenizer_load_policy=tokenizer_load_policy,
        special_token_alignment=special_token_alignment,
    )

    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
