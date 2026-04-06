#!/usr/bin/env python3
"""Write per-model input-interface diagnosis and 1-sample smoke outputs under outputs/debug/."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from finqa_protocol_v1 import (
    PROMPT_RENDER_MODE_AUTO,
    PROMPT_RENDER_MODE_PLAIN,
    DecodeConfig,
    parse_prediction,
    render_prompt_for_tokenizer,
)


MODEL_LABELS = [
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
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
        "--labels",
        nargs="+",
        default=MODEL_LABELS,
        help="Model labels to diagnose and smoke test.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
        for i in range(0, len(tokens) - n + 1):
            key = tuple(tokens[i : i + n])
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            continue
        top_count = max(counts.values())
        if top_count >= 10 and top_count * n >= len(tokens) * 0.5:
            return True
    return False


def token_summary(tokenizer: Any) -> Dict[str, Any]:
    return {
        "chat_template_exists": bool(getattr(tokenizer, "chat_template", None)),
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
    }


def model_config_summary(config: Any, tokenizer: Any) -> Dict[str, Any]:
    eos_token_id = getattr(config, "eos_token_id", None)
    pad_token_id = getattr(config, "pad_token_id", None)
    bos_token_id = getattr(config, "bos_token_id", None)
    unk_token_id = getattr(config, "unk_token_id", None)

    checks = {
        "eos_token_id_match_tokenizer": eos_token_id == tokenizer.eos_token_id,
        "pad_token_id_reasonable": pad_token_id is None or pad_token_id == tokenizer.pad_token_id,
        "bos_token_id_match_tokenizer": bos_token_id == tokenizer.bos_token_id,
        "unk_token_id_reasonable": unk_token_id is None or unk_token_id == tokenizer.unk_token_id,
    }

    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "model_config_eos_token_id": eos_token_id,
        "model_config_pad_token_id": pad_token_id,
        "model_config_bos_token_id": bos_token_id,
        "model_config_unk_token_id": unk_token_id,
        "token_id_alignment_checks": checks,
        "token_id_alignment_reasonable": all(checks.values()),
    }


def diagnosis_text(payload: Dict[str, Any]) -> str:
    lines = [
        f"# 输入接口诊断：{payload['model_label']}",
        "",
        f"- 模型路径：`{payload['model_path']}`",
        f"- 是否 chat/instruction tuned：`{payload['likely_instruction_tuned']}`",
        f"- tokenizer.chat_template 是否存在：`{payload['tokenizer']['chat_template_exists']}`",
        f"- 当前 runner 请求渲染模式：`{payload['prompt_render']['requested_mode']}`",
        f"- 当前 runner 实际渲染模式：`{payload['prompt_render']['effective_mode']}`",
        f"- 是否使用 chat_template：`{payload['prompt_render']['used_chat_template']}`",
        f"- fallback 策略：`{payload['prompt_render']['fallback_reason'] or 'none'}`",
        f"- 是否出现 assistant generation boundary：`{payload['prompt_render']['has_assistant_generation_boundary']}`",
        "",
        "## tokenizer special tokens",
        f"- bos_token: `{payload['tokenizer']['bos_token']}` / id=`{payload['tokenizer']['bos_token_id']}`",
        f"- eos_token: `{payload['tokenizer']['eos_token']}` / id=`{payload['tokenizer']['eos_token_id']}`",
        f"- pad_token: `{payload['tokenizer']['pad_token']}` / id=`{payload['tokenizer']['pad_token_id']}`",
        f"- unk_token: `{payload['tokenizer']['unk_token']}` / id=`{payload['tokenizer']['unk_token_id']}`",
        f"- special_tokens_map: `{json.dumps(payload['tokenizer']['special_tokens_map'], ensure_ascii=False)}`",
        "",
        "## model config tokens",
        f"- model_type: `{payload['model_config']['model_type']}`",
        f"- architectures: `{payload['model_config']['architectures']}`",
        f"- model.config.eos_token_id: `{payload['model_config']['model_config_eos_token_id']}`",
        f"- model.config.pad_token_id: `{payload['model_config']['model_config_pad_token_id']}`",
        f"- model.config.bos_token_id: `{payload['model_config']['model_config_bos_token_id']}`",
        f"- model.config.unk_token_id: `{payload['model_config']['model_config_unk_token_id']}`",
        f"- token id 对齐是否整体合理：`{payload['model_config']['token_id_alignment_reasonable']}`",
        f"- token id 对齐检查：`{json.dumps(payload['model_config']['token_id_alignment_checks'], ensure_ascii=False)}`",
        "",
        "## tokenizer fallback 状态",
        f"- pad_token 是否在加载后由 eos_token 回填：`{payload['tokenizer_runtime_patch']['pad_token_filled_from_eos']}`",
        f"- 原始 pad_token: `{payload['tokenizer_runtime_patch']['original_pad_token']}`",
        f"- 原始 pad_token_id: `{payload['tokenizer_runtime_patch']['original_pad_token_id']}`",
        "",
        "## 渲染后 prompt 前 500 字符",
        "```text",
        payload["prompt_render"]["prompt_head_500"],
        "```",
        "",
        "## 渲染后 prompt 后 300 字符",
        "```text",
        payload["prompt_render"]["prompt_tail_300"],
        "```",
    ]
    return "\n".join(lines) + "\n"


def run_smoke_once(
    model: Any,
    tokenizer: Any,
    record: Dict[str, Any],
    device: str,
    prompt_render_mode: str,
    repetition_penalty: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt_info = render_prompt_for_tokenizer(
        record=record,
        tokenizer=tokenizer,
        prompt_render_mode=prompt_render_mode,
    )
    prompt = prompt_info["prompt"]
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    prompt_length = int(prompt_inputs["input_ids"].shape[-1])

    started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **prompt_inputs,
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
    return {
        "sample_id": record["id"],
        "prompt_render_mode_requested": prompt_info["requested_mode"],
        "prompt_render_mode_effective": prompt_info["effective_mode"],
        "used_chat_template": prompt_info["used_chat_template"],
        "fallback_reason": prompt_info["fallback_reason"],
        "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
        "raw_prompt": prompt,
        "raw_prompt_head_500": prompt[:500],
        "raw_prompt_tail_300": prompt[-300:],
        "tokenized_input_length": prompt_length,
        "special_tokens_info": token_summary(tokenizer),
        "new_tokens": int(generated_ids.shape[-1]),
        "raw_generation_text": generation_text,
        "answer_appears": parse["format_ok"],
        "truncation_suspect": parse["truncated_suspect"],
        "repetition_collapse": detect_repetition_collapse(generation_text),
        "parse": parse,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "latency_ms": latency_ms,
    }


def smoke_text(model_label: str, rows: List[Dict[str, Any]]) -> str:
    lines = [f"# 输入有效性 Smoke：{model_label}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['prompt_render_mode_requested']} / repetition_penalty={row['repetition_penalty']}",
                f"- 实际渲染模式：`{row['prompt_render_mode_effective']}`",
                f"- 是否使用 chat_template：`{row['used_chat_template']}`",
                f"- fallback 策略：`{row['fallback_reason'] or 'none'}`",
                f"- tokenized input length：`{row['tokenized_input_length']}`",
                f"- Answer: 是否出现：`{row['answer_appears']}`",
                f"- truncation_suspect：`{row['truncation_suspect']}`",
                f"- repetition_collapse：`{row['repetition_collapse']}`",
                f"- new_tokens：`{row['new_tokens']}`",
                f"- latency_ms：`{row['latency_ms']}`",
                "",
                "### raw prompt 前 500 字符",
                "```text",
                row["raw_prompt_head_500"],
                "```",
                "",
                "### raw prompt 后 300 字符",
                "```text",
                row["raw_prompt_tail_300"],
                "```",
                "",
                "### raw generation text 前 1200 字符",
                "```text",
                row["raw_generation_text"][:1200],
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry = read_registry(args.model_registry.resolve())
    record = load_manifest_sample(args.manifest.resolve(), sample_index=args.sample_index)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    diagnosis_dir = project_root / "outputs" / "debug" / "input_interface_diagnosis"
    smoke_root = project_root / "outputs" / "debug" / "input_smoke"

    for label in args.labels:
        if label not in registry:
            raise SystemExit(f"Unknown model label in registry: {label}")

        model_path = registry[label]["snapshot_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        original_pad_token = tokenizer.pad_token
        original_pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_path)
        prompt_info = render_prompt_for_tokenizer(
            record=record,
            tokenizer=tokenizer,
            prompt_render_mode=PROMPT_RENDER_MODE_AUTO,
        )

        diagnosis_payload = {
            "model_label": label,
            "model_path": model_path,
            "sample_id": record["id"],
            "likely_instruction_tuned": LIKELY_INSTRUCTION_TUNED.get(label, None),
            "tokenizer": token_summary(tokenizer),
            "tokenizer_runtime_patch": {
                "pad_token_filled_from_eos": original_pad_token is None and tokenizer.pad_token is not None,
                "original_pad_token": original_pad_token,
                "original_pad_token_id": original_pad_token_id,
                "effective_pad_token": tokenizer.pad_token,
                "effective_pad_token_id": tokenizer.pad_token_id,
            },
            "model_config": model_config_summary(config, tokenizer),
            "prompt_render": {
                "requested_mode": prompt_info["requested_mode"],
                "effective_mode": prompt_info["effective_mode"],
                "used_chat_template": prompt_info["used_chat_template"],
                "fallback_reason": prompt_info["fallback_reason"],
                "has_assistant_generation_boundary": prompt_info["has_assistant_generation_boundary"],
                "prompt_head_500": prompt_info["prompt"][:500],
                "prompt_tail_300": prompt_info["prompt"][-300:],
            },
        }
        write_json(diagnosis_dir / f"{label}.json", diagnosis_payload)
        write_text(diagnosis_dir / f"{label}.txt", diagnosis_text(diagnosis_payload))

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        model.eval()
        model.to(device)

        smoke_rows = [
            run_smoke_once(
                model=model,
                tokenizer=tokenizer,
                record=record,
                device=device,
                prompt_render_mode=PROMPT_RENDER_MODE_PLAIN,
                repetition_penalty=1.0,
                max_new_tokens=args.max_new_tokens,
            ),
            run_smoke_once(
                model=model,
                tokenizer=tokenizer,
                record=record,
                device=device,
                prompt_render_mode=PROMPT_RENDER_MODE_AUTO,
                repetition_penalty=DecodeConfig().repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            ),
        ]
        model_smoke_dir = smoke_root / label
        write_json(model_smoke_dir / "smoke_compare.json", {"model_label": label, "rows": smoke_rows})
        write_text(model_smoke_dir / "smoke_compare.txt", smoke_text(label, smoke_rows))

        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        if device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        print(f"Wrote debug diagnosis and smoke outputs for {label}")


if __name__ == "__main__":
    main()
