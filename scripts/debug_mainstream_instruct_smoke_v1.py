#!/usr/bin/env python3
"""Run a provisional 1-sample smoke prescreen for mainstream instruct/chat models."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.generation.utils import GenerationConfig, GenerationMixin

from finqa_protocol_v1 import (
    PROMPT_RENDER_MODE_REGISTRY,
    DecodeConfig,
    align_model_tokenizer_special_tokens,
    load_tokenizer_with_policy,
    parse_prediction,
    render_prompt_for_tokenizer,
    resolve_instruction_wrapper_spec,
    resolve_tokenizer_load_policy,
)


DEFAULT_LABELS = [
    "Mistral-7B-Instruct-v0.3",
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Gemma-7B-IT",
    "Yi-1.5-6B-Chat",
    "ChatGLM3-6B",
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
        / "model_registry.groups-mainstream-instruction-prescreen.json",
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


def candidate_prompt_modes(_label: str) -> List[str]:
    return [PROMPT_RENDER_MODE_REGISTRY]


def smoke_score(row: Dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("valid_parse", False)),
        int(row.get("answer_appears", False)),
        int(not row.get("truncation_suspect", True)),
        int(not row.get("repetition_collapse", True)),
    )


def update_chatglm_generation_kwargs(outputs: Any, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(model_kwargs)
    updated["past_key_values"] = getattr(outputs, "past_key_values", None)

    attention_mask = updated.get("attention_mask")
    if attention_mask is not None:
        updated["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )

    position_ids = updated.get("position_ids")
    if position_ids is not None:
        new_position_id = position_ids[..., -1:].clone()
        new_position_id += 1
        updated["position_ids"] = torch.cat([position_ids, new_position_id], dim=-1)

    updated["is_first_forward"] = False
    return updated


def build_skip_payload(label: str, registry_row: Dict[str, Any], skip_reason: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    wrapper_spec = resolve_instruction_wrapper_spec(label)
    tokenizer_policy = resolve_tokenizer_load_policy(model_label=label, trust_remote_code=False)
    diagnosis_payload = {
        "model_label": label,
        "group": registry_row.get("group", ""),
        "model_family": registry_row.get("model_family", ""),
        "method_family": registry_row.get("method_family", ""),
        "base_model_family": registry_row.get("base_model_family", ""),
        "repo_id": registry_row.get("repo_id", ""),
        "registry_status": registry_row.get("status", "missing_registry_row"),
        "snapshot_path": registry_row.get("snapshot_path", ""),
        "wrapper_family": wrapper_spec.model_family,
        "default_wrapper": wrapper_spec.default_wrapper,
        "allow_chat_template": wrapper_spec.allow_chat_template,
        "fallback_wrapper": wrapper_spec.fallback_wrapper,
        "wrapper_notes": wrapper_spec.notes,
        "tokenizer_load_policy": {
            "requested_use_fast": tokenizer_policy.use_fast,
            "requested_tokenizer_class_name": tokenizer_policy.tokenizer_class_name,
            "trust_remote_code": tokenizer_policy.trust_remote_code,
            "notes": tokenizer_policy.notes,
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
        "model_loader": {"loader_name": None, "effective_model_class_name": None},
        "prompt_candidates": [],
        "special_token_alignment": {},
        "skip_reason": skip_reason,
        "runtime_error": "",
    }
    smoke_payload = {
        "model_label": label,
        "repo_id": registry_row.get("repo_id", ""),
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
    tokenizer_policy: Dict[str, Any],
    model_loader: Dict[str, Any],
    special_token_alignment: Dict[str, Any],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    wrapper_spec = resolve_instruction_wrapper_spec(label)
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
        "method_family": registry_row.get("method_family", ""),
        "base_model_family": registry_row.get("base_model_family", ""),
        "repo_id": registry_row.get("repo_id", ""),
        "registry_status": registry_row.get("status", ""),
        "snapshot_path": registry_row.get("snapshot_path", ""),
        "wrapper_family": wrapper_spec.model_family,
        "default_wrapper": wrapper_spec.default_wrapper,
        "allow_chat_template": wrapper_spec.allow_chat_template,
        "fallback_wrapper": wrapper_spec.fallback_wrapper,
        "wrapper_notes": wrapper_spec.notes,
        "tokenizer_load_policy": tokenizer_policy,
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
        "model_loader": model_loader,
        "prompt_candidates": prompt_candidates,
        "special_token_alignment": special_token_alignment,
        "skip_reason": "",
        "runtime_error": "",
    }


def load_model_bundle(
    model_path: str,
    model_label: str,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    tokenizer_policy = resolve_tokenizer_load_policy(model_label=model_label, trust_remote_code=False)
    tokenizer = load_tokenizer_with_policy(
        model_path=model_path,
        model_label=model_label,
        trust_remote_code=tokenizer_policy.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=tokenizer_policy.trust_remote_code)
    config_patch_notes: List[str] = []
    if model_label == "ChatGLM3-6B" and not hasattr(tokenizer, "batch_encode_plus"):
        tokenizer.batch_encode_plus = tokenizer.__call__
        config_patch_notes.append("set tokenizer.batch_encode_plus = tokenizer.__call__ for ChatGLMTokenizer compatibility")
    if model_label == "ChatGLM3-6B" and not hasattr(config, "max_length") and hasattr(config, "seq_length"):
        config.max_length = config.seq_length
        config_patch_notes.append("set config.max_length = config.seq_length for ChatGLM3 compatibility")

    if model_label == "ChatGLM3-6B" and tokenizer_policy.trust_remote_code:
        model_cls = get_class_from_dynamic_module(
            "modeling_chatglm.ChatGLMForConditionalGeneration",
            model_path,
        )
        compat_model_cls = model_cls
        if not issubclass(model_cls, GenerationMixin):
            compat_model_cls = type(
                "ChatGLMCompatForConditionalGeneration",
                (model_cls, GenerationMixin),
                {},
            )
            config_patch_notes.append("wrap ChatGLMForConditionalGeneration with GenerationMixin for Transformers compatibility")
        if not hasattr(compat_model_cls, "all_tied_weights_keys"):
            compat_model_cls.all_tied_weights_keys = {}
            config_patch_notes.append("set ChatGLMForConditionalGeneration.all_tied_weights_keys = {} for Transformers compatibility")
        model = compat_model_cls.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=tokenizer_policy.trust_remote_code,
        )
        if not hasattr(model, "generation_config") or model.generation_config is None:
            model.generation_config = GenerationConfig.from_model_config(config)
            config_patch_notes.append("set model.generation_config = GenerationConfig.from_model_config(config)")
        model_loader = {
            "loader_name": "dynamic_module.ChatGLMCompatForConditionalGeneration",
            "effective_model_class_name": type(model).__name__,
        }
        if config_patch_notes:
            model_loader["config_patch_notes"] = config_patch_notes

        special_token_alignment = align_model_tokenizer_special_tokens(model=model, tokenizer=tokenizer)
        tokenizer_policy_summary = {
            "model_label": tokenizer_policy.model_label,
            "requested_use_fast": tokenizer_policy.use_fast,
            "requested_tokenizer_class_name": tokenizer_policy.tokenizer_class_name,
            "effective_tokenizer_class_name": type(tokenizer).__name__,
            "effective_tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
            "trust_remote_code": tokenizer_policy.trust_remote_code,
            "notes": tokenizer_policy.notes,
            "config_patch_notes": config_patch_notes,
        }
        return model, tokenizer, config, tokenizer_policy_summary, special_token_alignment | {"model_loader": model_loader}

    model_loader = {"loader_name": "AutoModelForCausalLM", "effective_model_class_name": ""}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=tokenizer_policy.trust_remote_code,
        )
        model_loader["effective_model_class_name"] = type(model).__name__
    except Exception as primary_exc:
        if not tokenizer_policy.trust_remote_code:
            raise primary_exc
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=tokenizer_policy.trust_remote_code,
        )
        model_loader = {
            "loader_name": "AutoModel",
            "effective_model_class_name": type(model).__name__,
            "fallback_reason": f"AutoModelForCausalLM_failed: {type(primary_exc).__name__}: {primary_exc}",
        }
    if config_patch_notes:
        model_loader["config_patch_notes"] = config_patch_notes

    special_token_alignment = align_model_tokenizer_special_tokens(model=model, tokenizer=tokenizer)
    tokenizer_policy_summary = {
        "model_label": tokenizer_policy.model_label,
        "requested_use_fast": tokenizer_policy.use_fast,
        "requested_tokenizer_class_name": tokenizer_policy.tokenizer_class_name,
        "effective_tokenizer_class_name": type(tokenizer).__name__,
        "effective_tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
        "trust_remote_code": tokenizer_policy.trust_remote_code,
        "notes": tokenizer_policy.notes,
        "config_patch_notes": config_patch_notes,
    }
    return model, tokenizer, config, tokenizer_policy_summary, special_token_alignment | {"model_loader": model_loader}


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

    if label == "ChatGLM3-6B" and hasattr(tokenizer, "build_chat_input") and hasattr(model, "prepare_inputs_for_generation"):
        messages = prompt_info["messages"]
        history = [message for message in messages[:-1]]
        query = messages[-1]["content"]
        prompt_inputs = tokenizer.build_chat_input(query, history=history, role="user")
        prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        generated = prompt_inputs["input_ids"]
        model_kwargs = {
            "past_key_values": None,
            "attention_mask": prompt_inputs.get("attention_mask"),
            "position_ids": model.get_position_ids(generated, device=generated.device),
            "use_cache": True,
            "is_first_forward": True,
        }
        eos_token_ids = {
            tokenizer.eos_token_id,
            tokenizer.get_command("<|user|>"),
            tokenizer.get_command("<|observation|>"),
        }
        repetition_processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)

        started_at = time.perf_counter()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                model_inputs = model.prepare_inputs_for_generation(generated, **model_kwargs)
                outputs = model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                next_token_scores = outputs.logits[:, -1, :]
                next_token_scores = repetition_processor(generated, next_token_scores)
                next_tokens = torch.argmax(next_token_scores, dim=-1)
                generated = torch.cat([generated, next_tokens[:, None]], dim=-1)
                model_kwargs = update_chatglm_generation_kwargs(outputs, model_kwargs)
                if int(next_tokens[0].item()) in eos_token_ids:
                    break
        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        response_token_ids = generated[0, prompt_len:].tolist()
        if response_token_ids and response_token_ids[-1] in eos_token_ids:
            response_token_ids = response_token_ids[:-1]
        decoded_response = tokenizer.decode(response_token_ids)
        history_for_process = [dict(item) for item in history]
        history_for_process.append({"role": "user", "content": query})
        processed_response, _processed_history = model.process_response(decoded_response, history_for_process)
        raw_generation_text = processed_response if isinstance(processed_response, str) else json.dumps(processed_response, ensure_ascii=False)
        new_tokens = len(response_token_ids)
    else:
        if not hasattr(model, "generate"):
            raise RuntimeError(f"{type(model).__name__} has no generate() method")

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
        new_tokens = int(generated_ids.shape[-1])

    parse = parse_prediction(
        raw_generation_text,
        max_new_tokens_hit=new_tokens >= max_new_tokens,
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
        "new_tokens": new_tokens,
        "latency_ms": latency_ms,
        "pass_smoke": bool(parse["format_ok"] and parse["valid_parse"] and not parse["truncated_suspect"]),
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
            f"# 主流预筛 Wrapper Registry：{payload['model_label']}",
            "",
            f"- group：`{payload['group']}`",
            f"- model_family：`{payload['model_family']}`",
            f"- method_family：`{payload['method_family']}`",
            f"- base_model_family：`{payload['base_model_family']}`",
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
        f"# 主流预筛输入接口诊断：{payload['model_label']}",
        "",
        f"- registry status：`{payload['registry_status']}`",
        f"- group：`{payload['group']}`",
        f"- model_family：`{payload['model_family']}`",
        f"- method_family：`{payload['method_family']}`",
        f"- base_model_family：`{payload['base_model_family']}`",
        f"- wrapper family：`{payload['wrapper_family']}`",
        f"- default wrapper：`{payload['default_wrapper']}`",
        f"- tokenizer class：`{payload['tokenizer']['class_name']}`",
        f"- tokenizer is_fast：`{payload['tokenizer']['is_fast']}`",
        f"- tokenizer.chat_template：`{payload['tokenizer']['chat_template_exists']}`",
        f"- tokenizer pad/eos/bos/unk：`{payload['tokenizer']['pad_token']}` / `{payload['tokenizer']['eos_token']}` / `{payload['tokenizer']['bos_token']}` / `{payload['tokenizer']['unk_token']}`",
        f"- tokenizer pad/eos id：`{payload['tokenizer']['pad_token_id']}` / `{payload['tokenizer']['eos_token_id']}`",
        f"- model config pad/eos id：`{payload['model_config']['pad_token_id']}` / `{payload['model_config']['eos_token_id']}`",
        f"- token id 对齐是否合理：`{payload['model_config']['token_id_alignment_reasonable']}`",
        f"- model loader：`{payload['model_loader']['loader_name']}` -> `{payload['model_loader']['effective_model_class_name']}`",
        "",
    ]
    if payload["skip_reason"]:
        lines.extend([f"- skip reason：{payload['skip_reason']}", ""])
        return "\n".join(lines)

    if payload["runtime_error"]:
        lines.extend([f"- runtime error：{payload['runtime_error']}", ""])

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


def smoke_row_text(label: str, diagnosis_payload: Dict[str, Any], smoke_payload: Dict[str, Any]) -> str:
    recommended = smoke_payload.get("recommended_candidate")
    lines = [
        f"# 主流预筛 smoke：{label}",
        "",
        f"- registry status：`{diagnosis_payload['registry_status']}`",
        f"- model_family：`{diagnosis_payload.get('model_family', '')}`",
        f"- method_family：`{diagnosis_payload.get('method_family', '')}`",
        f"- base_model_family：`{diagnosis_payload.get('base_model_family', '')}`",
        f"- wrapper family：`{diagnosis_payload['wrapper_family']}`",
        f"- tokenizer class：`{diagnosis_payload['tokenizer'].get('class_name')}`",
        f"- tokenizer is_fast：`{diagnosis_payload['tokenizer'].get('is_fast')}`",
        f"- tokenizer.chat_template：`{diagnosis_payload['tokenizer'].get('chat_template_exists')}`",
        "",
    ]
    if diagnosis_payload["skip_reason"]:
        lines.extend([f"- skip reason：{diagnosis_payload['skip_reason']}", ""])
        return "\n".join(lines)
    if diagnosis_payload["runtime_error"]:
        lines.extend([f"- runtime error：{diagnosis_payload['runtime_error']}", ""])
    if recommended:
        lines.extend(
            [
                f"- requested prompt mode：`{recommended['requested_prompt_mode']}`",
                f"- effective wrapper：`{recommended['effective_wrapper']}`",
                f"- tokenized input length：`{recommended['tokenized_input_length']}`",
                f"- Answer: 是否出现：`{recommended['answer_appears']}`",
                f"- valid_parse：`{recommended['valid_parse']}`",
                f"- truncation_suspect：`{recommended['truncation_suspect']}`",
                f"- repetition_collapse：`{recommended['repetition_collapse']}`",
                f"- parse_error_reason：`{recommended['parse_error_reason']}`",
                f"- smoke pass：`{recommended['pass_smoke']}`",
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


def emit_outputs(project_root: Path, label: str, diagnosis_payload: Dict[str, Any], smoke_payload: Dict[str, Any]) -> None:
    wrapper_dir = project_root / "outputs" / "debug" / "wrapper_registry_mainstream"
    diagnosis_dir = project_root / "outputs" / "debug" / "input_interface_diagnosis_mainstream"
    smoke_dir = project_root / "outputs" / "debug" / "input_smoke_mainstream" / label

    write_json(wrapper_dir / f"{label}.json", diagnosis_payload)
    write_text(wrapper_dir / f"{label}.txt", registry_text(diagnosis_payload))
    write_json(diagnosis_dir / f"{label}.json", diagnosis_payload)
    write_text(diagnosis_dir / f"{label}.txt", diagnosis_text(diagnosis_payload))
    write_json(smoke_dir / "smoke_mainstream.json", smoke_payload)
    write_text(smoke_dir / "smoke_mainstream.txt", smoke_row_text(label, diagnosis_payload, smoke_payload))

    for candidate in smoke_payload.get("candidate_rows", []):
        prompt_mode = candidate["requested_prompt_mode"]
        write_json(smoke_dir / f"smoke_{prompt_mode}.json", candidate)


def summary_text(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# 主流 instruct/chat 预筛汇总",
        "",
        "| 模型 | 状态 | wrapper | tokenizer | Answer | valid_parse | truncation | repetition | 通过 smoke | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {status} | {wrapper} | {tokenizer} | {answer} | {valid_parse} | {truncation} | {repetition} | {passed} | {note} |".format(
                label=row["model_label"],
                status=row["status"],
                wrapper=row.get("effective_wrapper", "-"),
                tokenizer=row.get("effective_tokenizer_class_name", "-"),
                answer=row.get("answer_appears", "-"),
                valid_parse=row.get("valid_parse", "-"),
                truncation=row.get("truncation_suspect", "-"),
                repetition=row.get("repetition_collapse", "-"),
                passed=row.get("pass_smoke", "-"),
                note=row.get("note", ""),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry = read_registry(args.model_registry.resolve())
    record = load_manifest_sample(args.manifest.resolve(), args.sample_index)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    summary_rows: List[Dict[str, Any]] = []
    for label in args.labels:
        registry_row = registry.get(
            label,
            {
                "label": label,
                "group": "",
                "model_family": "",
                "method_family": "",
                "base_model_family": "",
                "repo_id": "",
                "status": "missing_registry_row",
                "snapshot_path": "",
            },
        )

        if registry_row.get("status") not in RUNNABLE_STATUSES:
            diagnosis_payload, smoke_payload = build_skip_payload(
                label=label,
                registry_row=registry_row,
                skip_reason=f"registry_status={registry_row.get('status', 'missing_registry_row')}",
            )
            emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
            summary_rows.append(
                {
                    "model_label": label,
                    "status": smoke_payload["status"],
                    "effective_wrapper": "-",
                    "effective_tokenizer_class_name": "-",
                    "answer_appears": "-",
                    "valid_parse": "-",
                    "truncation_suspect": "-",
                    "repetition_collapse": "-",
                    "pass_smoke": False,
                    "note": smoke_payload["skip_reason"],
                }
            )
            continue

        try:
            model, tokenizer, config, tokenizer_policy, alignment_plus_loader = load_model_bundle(
                model_path=registry_row["snapshot_path"],
                model_label=label,
                dtype=dtype,
            )
            special_token_alignment = {
                "before": alignment_plus_loader.get("before", {}),
                "after": alignment_plus_loader.get("after", {}),
                "actions": alignment_plus_loader.get("actions", []),
                "changed": alignment_plus_loader.get("changed", False),
            }
            model_loader = alignment_plus_loader.get(
                "model_loader",
                {"loader_name": "unknown", "effective_model_class_name": type(model).__name__},
            )
            diagnosis_payload = build_runnable_diagnosis(
                label=label,
                registry_row=registry_row,
                tokenizer=tokenizer,
                config=config,
                tokenizer_policy=tokenizer_policy,
                model_loader=model_loader,
                special_token_alignment=special_token_alignment,
                record=record,
            )
        except Exception as exc:
            diagnosis_payload, smoke_payload = build_skip_payload(
                label=label,
                registry_row=registry_row,
                skip_reason="load_failed",
            )
            diagnosis_payload["runtime_error"] = f"{type(exc).__name__}: {exc}"
            smoke_payload["status"] = "load_failed"
            smoke_payload["skip_reason"] = diagnosis_payload["runtime_error"]
            emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
            summary_rows.append(
                {
                    "model_label": label,
                    "status": smoke_payload["status"],
                    "effective_wrapper": "-",
                    "effective_tokenizer_class_name": "-",
                    "answer_appears": "-",
                    "valid_parse": "-",
                    "truncation_suspect": "-",
                    "repetition_collapse": "-",
                    "pass_smoke": False,
                    "note": smoke_payload["skip_reason"],
                }
            )
            continue

        try:
            model.eval()
            model.to(device)
            smoke_payload = run_model_smoke(
                label=label,
                registry_row=registry_row,
                tokenizer=tokenizer,
                model=model,
                record=record,
                device=device,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as exc:
            diagnosis_payload["runtime_error"] = f"{type(exc).__name__}: {exc}"
            smoke_payload = {
                "model_label": label,
                "repo_id": registry_row.get("repo_id", ""),
                "registry_status": registry_row.get("status", ""),
                "status": "runtime_failed",
                "skip_reason": diagnosis_payload["runtime_error"],
                "recommended_candidate": None,
                "candidate_rows": [],
            }

        emit_outputs(project_root, label, diagnosis_payload, smoke_payload)
        recommended = smoke_payload.get("recommended_candidate") or {}
        summary_rows.append(
            {
                "model_label": label,
                "status": smoke_payload["status"],
                "effective_wrapper": recommended.get("effective_wrapper", "-"),
                "effective_tokenizer_class_name": diagnosis_payload.get("tokenizer_load_policy", {}).get(
                    "effective_tokenizer_class_name",
                    "-",
                ),
                "answer_appears": recommended.get("answer_appears", "-"),
                "valid_parse": recommended.get("valid_parse", "-"),
                "truncation_suspect": recommended.get("truncation_suspect", "-"),
                "repetition_collapse": recommended.get("repetition_collapse", "-"),
                "pass_smoke": recommended.get("pass_smoke", False),
                "note": smoke_payload.get("skip_reason", "") or recommended.get("parse_error_reason", ""),
            }
        )

    summary_payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_index": args.sample_index,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "rows": summary_rows,
        "passed_labels": [row["model_label"] for row in summary_rows if row.get("pass_smoke")],
    }
    summary_dir = project_root / "outputs" / "debug" / "input_smoke_mainstream"
    write_json(summary_dir / "summary.json", summary_payload)
    write_text(summary_dir / "summary.md", summary_text(summary_rows))
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
