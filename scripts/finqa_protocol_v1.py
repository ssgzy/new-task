#!/usr/bin/env python3
"""Frozen prompt, parser, and scoring helpers for FinQA benchmark protocol v1."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, LlamaTokenizer


SYSTEM_PROMPT = """You are a precise financial question answering assistant.
Use only the provided table and text.
Do not show your reasoning.
Do not explain.
Output exactly one final line in this format:
Answer: <numeric value>"""

USER_PROMPT_TEMPLATE = """### Context
{context}

### Question
{question}"""

PROMPT_RENDER_MODE_AUTO = "auto"
PROMPT_RENDER_MODE_REGISTRY = "registry"
PROMPT_RENDER_MODE_PLAIN = "plain"
PROMPT_RENDER_MODE_CHAT_TEMPLATE = "chat_template"
PROMPT_RENDER_MODE_ALPACA = "alpaca"
PROMPT_RENDER_MODE_VICUNA = "vicuna"
PROMPT_RENDER_MODE_CHATML = "chatml"
PROMPT_RENDER_MODE_COMPLETION = "completion"

MODEL_FAMILY_ALPACA = "alpaca"
MODEL_FAMILY_VICUNA = "vicuna"
MODEL_FAMILY_CHATML = "chatml"
MODEL_FAMILY_PLAIN = "plain"
MODEL_FAMILY_BASE_DISTILLED = "base_distilled"

NUMERIC_LITERAL_RE = re.compile(r"^\s*\(?[-+]?(?:\d[\d,]*\.?\d*|\.\d+)\)?\s*%?\s*$")
NUMBER_EXTRACT_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


@dataclass(frozen=True)
class DecodeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1


@dataclass(frozen=True)
class InstructionWrapperSpec:
    model_label: str
    model_family: str
    default_wrapper: str
    allow_chat_template: bool = False
    fallback_wrapper: str = PROMPT_RENDER_MODE_PLAIN
    notes: str = ""


@dataclass(frozen=True)
class TokenizerLoadPolicy:
    model_label: str
    use_fast: Optional[bool] = None
    tokenizer_class_name: str = "AutoTokenizer"
    trust_remote_code: bool = False
    notes: str = ""


INSTRUCTION_WRAPPER_REGISTRY: Dict[str, InstructionWrapperSpec] = {
    "Lion-7B": InstructionWrapperSpec(
        model_label="Lion-7B",
        model_family=MODEL_FAMILY_ALPACA,
        default_wrapper=PROMPT_RENDER_MODE_ALPACA,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_ALPACA,
        notes="instruction-tuned anchor baseline; use Alpaca-style ### Instruction / ### Input / ### Response serializer from the official Lion training script",
    ),
    "Orca-2-7B": InstructionWrapperSpec(
        model_label="Orca-2-7B",
        model_family=MODEL_FAMILY_CHATML,
        default_wrapper=PROMPT_RENDER_MODE_CHATML,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_CHATML,
        notes="instruction-tuned model; use ChatML serializer when tokenizer has no builtin template",
    ),
    "Zephyr-7B-beta": InstructionWrapperSpec(
        model_label="Zephyr-7B-beta",
        model_family=MODEL_FAMILY_PLAIN,
        default_wrapper=PROMPT_RENDER_MODE_PLAIN,
        allow_chat_template=True,
        fallback_wrapper=PROMPT_RENDER_MODE_PLAIN,
        notes="mainline plain serializer, with chat_template retained as a debug comparison path",
    ),
    "MiniLLM-Llama-7B": InstructionWrapperSpec(
        model_label="MiniLLM-Llama-7B",
        model_family=MODEL_FAMILY_BASE_DISTILLED,
        default_wrapper=PROMPT_RENDER_MODE_COMPLETION,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_COMPLETION,
        notes="base distilled baseline; expose completion-style serializer",
    ),
    "Xwin-LM-7B": InstructionWrapperSpec(
        model_label="Xwin-LM-7B",
        model_family=MODEL_FAMILY_VICUNA,
        default_wrapper=PROMPT_RENDER_MODE_VICUNA,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_VICUNA,
        notes="main-table expansion candidate; use the Vicuna-style USER/ASSISTANT serializer from the official Xwin-LM model card",
    ),
    "LaMini-LLaMA-7B": InstructionWrapperSpec(
        model_label="LaMini-LLaMA-7B",
        model_family=MODEL_FAMILY_ALPACA,
        default_wrapper=PROMPT_RENDER_MODE_ALPACA,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_ALPACA,
        notes="placeholder expansion candidate; repo_id is unresolved in the current round, so Alpaca-style serializer is only a provisional fallback assumption and should not be used as final evidence",
    ),
    "DeepSeek-R1-Distill-Qwen-7B": InstructionWrapperSpec(
        model_label="DeepSeek-R1-Distill-Qwen-7B",
        model_family=MODEL_FAMILY_CHATML,
        default_wrapper=PROMPT_RENDER_MODE_CHAT_TEMPLATE,
        allow_chat_template=True,
        fallback_wrapper=PROMPT_RENDER_MODE_CHATML,
        notes="main-table expansion candidate; local tokenizer exposes chat_template, fallback to ChatML if a future snapshot removes it",
    ),
    "OpenR1-Distill-7B": InstructionWrapperSpec(
        model_label="OpenR1-Distill-7B",
        model_family=MODEL_FAMILY_CHATML,
        default_wrapper=PROMPT_RENDER_MODE_CHATML,
        allow_chat_template=False,
        fallback_wrapper=PROMPT_RENDER_MODE_CHATML,
        notes="main-table expansion candidate; local tokenizer has no chat_template, so compare ChatML versus plain in provisional smoke before freezing a wrapper",
    ),
}

SUPPORTED_PROMPT_RENDER_MODES = {
    PROMPT_RENDER_MODE_AUTO,
    PROMPT_RENDER_MODE_REGISTRY,
    PROMPT_RENDER_MODE_PLAIN,
    PROMPT_RENDER_MODE_CHAT_TEMPLATE,
    PROMPT_RENDER_MODE_ALPACA,
    PROMPT_RENDER_MODE_VICUNA,
    PROMPT_RENDER_MODE_CHATML,
    PROMPT_RENDER_MODE_COMPLETION,
}

TOKENIZER_LOAD_POLICIES: Dict[str, TokenizerLoadPolicy] = {
    "Orca-2-7B": TokenizerLoadPolicy(
        model_label="Orca-2-7B",
        use_fast=False,
        tokenizer_class_name="LlamaTokenizer",
        trust_remote_code=False,
        notes="Official Orca-2-7b model card asks to use the slow tokenizer; in this local Transformers stack, AutoTokenizer(..., use_fast=False) still resolves to TokenizersBackend, so Orca is loaded with LlamaTokenizer.from_pretrained for the debug/provisional path.",
    ),
}


def normalize_string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        return [str(v) for v in values]
    return [str(values)]


def render_table(table: Any) -> str:
    if not isinstance(table, list):
        return str(table) if table is not None else ""
    lines: List[str] = []
    for row in table:
        if isinstance(row, list):
            lines.append(" | ".join(str(cell) for cell in row))
        else:
            lines.append(str(row))
    return "\n".join(lines)


def build_context(pre_text: Any, table: Any, post_text: Any) -> str:
    parts: List[str] = []
    pre_lines = normalize_string_list(pre_text)
    if pre_lines:
        parts.append("### Pre Text\n" + "\n".join(pre_lines))
    table_text = render_table(table).strip()
    if table_text:
        parts.append("### Table\n" + table_text)
    post_lines = normalize_string_list(post_text)
    if post_lines:
        parts.append("### Post Text\n" + "\n".join(post_lines))
    return "\n\n".join(parts).strip()


def build_user_prompt(context: str, question: str) -> str:
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_chat_messages(context: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context=context, question=question)},
    ]


def build_plain_prompt(context: str, question: str) -> str:
    user_prompt = build_user_prompt(context=context, question=question)
    return f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n"


def build_alpaca_prompt(context: str, question: str) -> str:
    user_prompt = build_user_prompt(context=context, question=question)
    return f"### Instruction:\n{SYSTEM_PROMPT}\n\n### Input:\n{user_prompt}\n\n### Response:\n"


def build_vicuna_prompt(context: str, question: str) -> str:
    user_prompt = build_user_prompt(context=context, question=question)
    return f"{SYSTEM_PROMPT}\n\nUSER: {user_prompt}\nASSISTANT:"


def build_chatml_prompt(context: str, question: str) -> str:
    user_prompt = build_user_prompt(context=context, question=question)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_completion_prompt(context: str, question: str) -> str:
    user_prompt = build_user_prompt(context=context, question=question)
    return f"### Instruction\n{SYSTEM_PROMPT}\n\n### Input\n{user_prompt}\n\n### Response\n"


def build_prompt_from_record(record: Dict[str, Any]) -> str:
    context = build_context(record.get("pre_text"), record.get("table"), record.get("post_text"))
    return build_plain_prompt(context=context, question=str(record.get("question", "")).strip())


def build_chat_messages_from_record(record: Dict[str, Any]) -> List[Dict[str, str]]:
    context = build_context(record.get("pre_text"), record.get("table"), record.get("post_text"))
    return build_chat_messages(context=context, question=str(record.get("question", "")).strip())


def build_serialized_prompt(context: str, question: str, wrapper_mode: str) -> str:
    if wrapper_mode == PROMPT_RENDER_MODE_PLAIN:
        return build_plain_prompt(context=context, question=question)
    if wrapper_mode == PROMPT_RENDER_MODE_ALPACA:
        return build_alpaca_prompt(context=context, question=question)
    if wrapper_mode == PROMPT_RENDER_MODE_VICUNA:
        return build_vicuna_prompt(context=context, question=question)
    if wrapper_mode == PROMPT_RENDER_MODE_CHATML:
        return build_chatml_prompt(context=context, question=question)
    if wrapper_mode == PROMPT_RENDER_MODE_COMPLETION:
        return build_completion_prompt(context=context, question=question)
    raise ValueError(f"Unsupported serializer wrapper mode: {wrapper_mode}")


def resolve_instruction_wrapper_spec(model_label: Optional[str]) -> InstructionWrapperSpec:
    if model_label and model_label in INSTRUCTION_WRAPPER_REGISTRY:
        return INSTRUCTION_WRAPPER_REGISTRY[model_label]
    return InstructionWrapperSpec(
        model_label=model_label or "unknown",
        model_family=MODEL_FAMILY_PLAIN,
        default_wrapper=PROMPT_RENDER_MODE_PLAIN,
        allow_chat_template=True,
        fallback_wrapper=PROMPT_RENDER_MODE_PLAIN,
        notes="fallback plain serializer for unregistered models",
    )


def resolve_tokenizer_load_policy(model_label: Optional[str], trust_remote_code: bool = False) -> TokenizerLoadPolicy:
    if model_label and model_label in TOKENIZER_LOAD_POLICIES:
        policy = TOKENIZER_LOAD_POLICIES[model_label]
        return TokenizerLoadPolicy(
            model_label=policy.model_label,
            use_fast=policy.use_fast,
            tokenizer_class_name=policy.tokenizer_class_name,
            trust_remote_code=trust_remote_code or policy.trust_remote_code,
            notes=policy.notes,
        )
    return TokenizerLoadPolicy(
        model_label=model_label or "unknown",
        use_fast=None,
        tokenizer_class_name="AutoTokenizer",
        trust_remote_code=trust_remote_code,
        notes="default AutoTokenizer.from_pretrained behavior",
    )


def tokenizer_from_pretrained_kwargs(model_label: Optional[str], trust_remote_code: bool = False) -> Dict[str, Any]:
    policy = resolve_tokenizer_load_policy(model_label=model_label, trust_remote_code=trust_remote_code)
    kwargs: Dict[str, Any] = {"trust_remote_code": policy.trust_remote_code}
    if policy.use_fast is not None:
        kwargs["use_fast"] = policy.use_fast
    return kwargs


def load_tokenizer_with_policy(model_path: str, model_label: Optional[str], trust_remote_code: bool = False) -> Any:
    policy = resolve_tokenizer_load_policy(model_label=model_label, trust_remote_code=trust_remote_code)
    kwargs = tokenizer_from_pretrained_kwargs(model_label=model_label, trust_remote_code=trust_remote_code)
    if policy.tokenizer_class_name == "LlamaTokenizer":
        return LlamaTokenizer.from_pretrained(model_path, **kwargs)
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def special_token_snapshot(model: Any, tokenizer: Any) -> Dict[str, Any]:
    config = getattr(model, "config", model)
    return {
        "tokenizer_pad_token": getattr(tokenizer, "pad_token", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "tokenizer_eos_token": getattr(tokenizer, "eos_token", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "model_config_pad_token_id": getattr(config, "pad_token_id", None),
        "model_config_eos_token_id": getattr(config, "eos_token_id", None),
    }


def align_model_tokenizer_special_tokens(model: Any, tokenizer: Any) -> Dict[str, Any]:
    before = special_token_snapshot(model=model, tokenizer=tokenizer)
    actions: List[str] = []

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        actions.append("set tokenizer.pad_token = tokenizer.eos_token")

    if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id
        actions.append("set model.config.pad_token_id = tokenizer.pad_token_id")

    if tokenizer.eos_token_id is not None and getattr(model.config, "eos_token_id", None) != tokenizer.eos_token_id:
        model.config.eos_token_id = tokenizer.eos_token_id
        actions.append("set model.config.eos_token_id = tokenizer.eos_token_id")

    after = special_token_snapshot(model=model, tokenizer=tokenizer)
    return {
        "before": before,
        "after": after,
        "actions": actions,
        "changed": bool(actions),
    }


def render_prompt_for_tokenizer(
    record: Dict[str, Any],
    tokenizer: Any,
    prompt_render_mode: str = PROMPT_RENDER_MODE_AUTO,
    model_label: Optional[str] = None,
) -> Dict[str, Any]:
    if prompt_render_mode not in SUPPORTED_PROMPT_RENDER_MODES:
        raise ValueError(f"Unsupported prompt_render_mode: {prompt_render_mode}")

    context = build_context(record.get("pre_text"), record.get("table"), record.get("post_text"))
    question = str(record.get("question", "")).strip()
    messages = build_chat_messages_from_record(record)
    tokenizer_has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    wrapper_spec = resolve_instruction_wrapper_spec(model_label)
    resolved_mode = prompt_render_mode
    fallback_reason = ""

    if prompt_render_mode == PROMPT_RENDER_MODE_REGISTRY:
        resolved_mode = wrapper_spec.default_wrapper
        if (
            wrapper_spec.allow_chat_template
            and tokenizer_has_chat_template
            and wrapper_spec.default_wrapper == PROMPT_RENDER_MODE_CHAT_TEMPLATE
        ):
            resolved_mode = PROMPT_RENDER_MODE_CHAT_TEMPLATE
        elif resolved_mode == PROMPT_RENDER_MODE_CHAT_TEMPLATE and not tokenizer_has_chat_template:
            resolved_mode = wrapper_spec.fallback_wrapper
            fallback_reason = f"tokenizer.chat_template_missing_use_{wrapper_spec.fallback_wrapper}"
    elif prompt_render_mode == PROMPT_RENDER_MODE_AUTO:
        if tokenizer_has_chat_template:
            resolved_mode = PROMPT_RENDER_MODE_CHAT_TEMPLATE
        elif model_label and model_label in INSTRUCTION_WRAPPER_REGISTRY:
            resolved_mode = wrapper_spec.default_wrapper
            fallback_reason = f"tokenizer.chat_template_missing_use_{wrapper_spec.default_wrapper}"
        else:
            resolved_mode = PROMPT_RENDER_MODE_PLAIN
            fallback_reason = "tokenizer.chat_template_missing_use_plain_prompt"

    if resolved_mode == PROMPT_RENDER_MODE_CHAT_TEMPLATE and tokenizer_has_chat_template:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        effective_mode = PROMPT_RENDER_MODE_CHAT_TEMPLATE
        has_assistant_generation_boundary = True
    else:
        if resolved_mode == PROMPT_RENDER_MODE_CHAT_TEMPLATE and not tokenizer_has_chat_template:
            resolved_mode = wrapper_spec.fallback_wrapper
            fallback_reason = fallback_reason or f"tokenizer.chat_template_missing_use_{wrapper_spec.fallback_wrapper}"

        rendered_prompt = build_serialized_prompt(
            context=context,
            question=question,
            wrapper_mode=resolved_mode,
        )
        effective_mode = resolved_mode
        has_assistant_generation_boundary = effective_mode in {
            PROMPT_RENDER_MODE_ALPACA,
            PROMPT_RENDER_MODE_VICUNA,
            PROMPT_RENDER_MODE_CHATML,
            PROMPT_RENDER_MODE_COMPLETION,
        }

    return {
        "prompt": rendered_prompt,
        "messages": messages,
        "requested_mode": prompt_render_mode,
        "resolved_mode": resolved_mode,
        "effective_mode": effective_mode,
        "model_label": wrapper_spec.model_label,
        "wrapper_family": wrapper_spec.model_family,
        "registry_default_wrapper": wrapper_spec.default_wrapper,
        "registry_allow_chat_template": wrapper_spec.allow_chat_template,
        "registry_fallback_wrapper": wrapper_spec.fallback_wrapper,
        "registry_notes": wrapper_spec.notes,
        "tokenizer_has_chat_template": tokenizer_has_chat_template,
        "used_chat_template": effective_mode == PROMPT_RENDER_MODE_CHAT_TEMPLATE,
        "has_assistant_generation_boundary": has_assistant_generation_boundary,
        "fallback_reason": fallback_reason,
    }


def parse_numeric_literal(text: str) -> Optional[float]:
    candidate = text.strip().replace("$", "")
    if not NUMERIC_LITERAL_RE.match(candidate):
        return None
    candidate = candidate.replace("%", "").replace(",", "")
    candidate = candidate.replace("(", "-").replace(")", "")
    match = NUMBER_EXTRACT_RE.search(candidate)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def answer_line_candidates(text: str) -> List[str]:
    lines = [line.rstrip("\r") for line in text.splitlines()]
    return [line.strip() for line in lines if line.strip().startswith("Answer:")]


def detect_truncation_suspect(text: str, answer_line_raw: str, format_ok: bool, max_new_tokens_hit: bool = False) -> bool:
    stripped = text.rstrip()
    if max_new_tokens_hit and not format_ok:
        return True
    if not format_ok:
        return stripped.endswith("Answer") or stripped.endswith("Answer:")
    payload = answer_line_raw.partition("Answer:")[2].strip()
    return payload in {"", "+", "-", ".", "%"}


def parse_prediction(text: str, max_new_tokens_hit: bool = False) -> Dict[str, Any]:
    answer_lines = answer_line_candidates(text)
    format_ok = bool(answer_lines)
    answer_line_raw = answer_lines[-1] if answer_lines else ""
    multiple_answer_lines = len(answer_lines) > 1
    truncated_suspect = detect_truncation_suspect(
        text=text,
        answer_line_raw=answer_line_raw,
        format_ok=format_ok,
        max_new_tokens_hit=max_new_tokens_hit,
    )

    result = {
        "format_ok": format_ok,
        "valid_parse": False,
        "pred_value": None,
        "pred_unit_type": None,
        "answer_line_raw": answer_line_raw,
        "multiple_answer_lines": multiple_answer_lines,
        "truncated_suspect": truncated_suspect,
        "parse_error_reason": None,
    }

    if not format_ok:
        result["parse_error_reason"] = "missing_answer_line"
        return result

    payload = answer_line_raw.partition("Answer:")[2].strip()
    if not payload:
        result["parse_error_reason"] = "empty_answer_line"
        return result

    pred_value = parse_numeric_literal(payload)
    result["pred_unit_type"] = "percent" if "%" in payload else "plain"
    if pred_value is None:
        result["parse_error_reason"] = "invalid_numeric_literal"
        return result

    result["valid_parse"] = True
    result["pred_value"] = pred_value
    return result


def exact_match(pred_value: Optional[float], gold_value: Optional[float]) -> bool:
    if pred_value is None or gold_value is None:
        return False
    return math.isclose(pred_value, gold_value, rel_tol=0.0, abs_tol=1e-9)


def tolerance_match(pred_value: Optional[float], gold_value: Optional[float], rel_tol: float = 0.01, abs_tol: float = 0.01) -> bool:
    if pred_value is None or gold_value is None:
        return False
    return math.isclose(pred_value, gold_value, rel_tol=rel_tol, abs_tol=abs_tol)


def protocol_bundle() -> Dict[str, Any]:
    return {
        "protocol_version": "v1",
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "decode_config": asdict(DecodeConfig()),
        "prompt_rendering": {
            "default_mode": PROMPT_RENDER_MODE_AUTO,
            "supported_modes": sorted(SUPPORTED_PROMPT_RENDER_MODES),
            "instruction_wrapper_registry": {
                model_label: asdict(spec) for model_label, spec in INSTRUCTION_WRAPPER_REGISTRY.items()
            },
            "tokenizer_load_policies": {
                model_label: asdict(policy) for model_label, policy in TOKENIZER_LOAD_POLICIES.items()
            },
            "auto_policy": "use tokenizer.apply_chat_template(..., add_generation_prompt=True) if tokenizer.chat_template exists; otherwise prefer model registry default wrapper when model_label is registered; fallback plain for unknown models",
        },
        "parser_rules": {
            "answer_prefix": "Answer:",
            "last_answer_line_only": True,
            "allow_multiple_answer_lines": True,
            "multiple_answer_lines_policy": "use_last_and_record_flag",
            "numeric_literal_required": True,
            "unit_types": ["plain", "percent"],
            "truncation_suspect_rule": "text_heuristics OR (hit_max_new_tokens AND missing_answer_line)",
        },
        "metric_rules": {
            "em": "numeric exact match after parser v1",
            "tm": "math.isclose(pred, gold, rel_tol=0.01, abs_tol=0.01)",
        },
    }


def export_protocol_bundle(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_system_v1.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (output_dir / "prompt_user_v1.txt").write_text(USER_PROMPT_TEMPLATE + "\n", encoding="utf-8")
    (output_dir / "decode_config_v1.json").write_text(
        json.dumps(asdict(DecodeConfig()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "protocol_v1.json").write_text(
        json.dumps(protocol_bundle(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "protocol_v1",
        help="Directory to write protocol bundle files.",
    )
    args = parser.parse_args()
    export_protocol_bundle(args.output_dir.resolve())
    print(f"Exported protocol bundle to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
