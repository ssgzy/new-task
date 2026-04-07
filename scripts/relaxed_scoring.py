#!/usr/bin/env python3
"""Supplemental relaxed scoring for FinQA direct-answer outputs.

This module does not change parser v1. Strict scoring still reuses the frozen
`parse_prediction()` logic from `finqa_protocol_v1.py`. Relaxed scoring is only
used as a supplemental reporting view to estimate how much accuracy is lost to
formatting or answer-line serialization issues.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from finqa_protocol_v1 import exact_match, parse_prediction, tolerance_match


def _find_answer_line(text: str) -> Optional[str]:
    for line in reversed(text.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            return line.strip().split(":", 1)[1].strip()
    return None


def relaxed_extract(text: str) -> Optional[float]:
    """Extract a final numeric answer from the output with relaxed heuristics.

    Priority:
    1. If an `Answer:` line exists, prefer it.
    2. If `=` exists in the chosen span, use the suffix after the last `=`.
    3. Otherwise take the last numeric-looking token in the chosen span.
    4. If no `Answer:` line exists, fall back to the full output.
    """

    answer_content = _find_answer_line(text)
    target = answer_content if answer_content else text

    if "=" in target:
        target = target.split("=")[-1]

    target = target.replace(",", "").replace("$", "")
    candidates = re.findall(r"-?\d+\.?\d*%?", target)
    if not candidates:
        return None

    raw = candidates[-1].replace("%", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None

    bracket_pattern = r"\(" + re.escape(candidates[-1].replace("%", "")) + r"%?\)"
    if re.search(bracket_pattern, answer_content or text):
        value = -abs(value)
    return value


def strict_score(prediction_text: str, gold_numeric: Optional[float]) -> Dict[str, Any]:
    """Frozen strict scoring using parser v1, unchanged."""

    parse = parse_prediction(prediction_text)
    pred_value = parse.get("pred_value")
    strict_em = bool(parse.get("valid_parse")) and exact_match(pred_value, gold_numeric)
    strict_tm = bool(parse.get("valid_parse")) and tolerance_match(pred_value, gold_numeric)
    return {
        "strict_em": strict_em,
        "strict_tm": strict_tm,
        "strict_parsed": pred_value if parse.get("valid_parse") else None,
    }


def relaxed_score(prediction_text: str, gold_numeric: Optional[float]) -> Dict[str, Any]:
    """Supplemental relaxed scoring with parser-independent numeric extraction."""

    pred_value = relaxed_extract(prediction_text)
    if pred_value is None:
        return {
            "relaxed_em": False,
            "relaxed_tm": False,
            "relaxed_parsed": None,
        }
    return {
        "relaxed_em": exact_match(pred_value, gold_numeric),
        "relaxed_tm": tolerance_match(pred_value, gold_numeric),
        "relaxed_parsed": pred_value,
    }


def score_prediction(prediction_text: str, gold_numeric: Optional[float]) -> Dict[str, Any]:
    strict_payload = strict_score(prediction_text=prediction_text, gold_numeric=gold_numeric)
    relaxed_payload = relaxed_score(prediction_text=prediction_text, gold_numeric=gold_numeric)
    return {
        **strict_payload,
        **relaxed_payload,
        "format_gap": int(relaxed_payload["relaxed_tm"]) - int(strict_payload["strict_tm"]),
    }


if __name__ == "__main__":
    test_cases = [
        ("Answer: 86.47", 86.47),
        ("Answer: 86.47%", 86.47),
        ("Answer: 100 * (5.99 - 3.24) / 3.24 = 86.47%", 86.47),
        ("Answer: (16.34)", -16.34),
        ("To calculate the growth rate...\nThe answer is 86.47", 86.47),
    ]
    for text, gold in test_cases:
        print(score_prediction(text, gold))
