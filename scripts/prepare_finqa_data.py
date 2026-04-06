#!/usr/bin/env python3
"""Download official FinQA splits, inspect fields, standardize records, and build fixed manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OFFICIAL_SPLITS = {
    "train": {
        "official_name": "train",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/train.json",
    },
    "validation": {
        "official_name": "dev",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/dev.json",
    },
    "test": {
        "official_name": "test",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/test.json",
    },
}

NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
BOOLEAN_MAP = {"yes": 1.0, "no": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/, outputs/, and scripts/ directories.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fixed manifests.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload raw files even if cached.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        return

    ensure_dir(destination.parent)
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    tmp_path.replace(destination)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def normalize_gold_numeric(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in BOOLEAN_MAP:
        return BOOLEAN_MAP[lowered]

    normalized = text.replace("$", "").replace("%", "")
    normalized = normalized.replace("(", "-").replace(")", "")
    match = NUMBER_RE.search(normalized)
    if not match:
        return None

    candidate = match.group(0).replace(",", "")
    try:
        value = float(candidate)
    except ValueError:
        return None

    if math.isfinite(value):
        return value
    return None


def answer_unit_type(raw: Any) -> str:
    text = str(raw).strip() if raw is not None else ""
    return "percent" if "%" in text else "plain"


def choose_answer_source(qa: Dict[str, Any]) -> Tuple[str, Any]:
    candidates = ["answer", "answers", "exe_ans"]
    for name in candidates:
        if name in qa and qa[name] not in (None, ""):
            return f"qa.{name}", qa[name]
    return "missing", None


def standardize_example(example: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    qa = example.get("qa", {})
    source_field_name, raw_answer = choose_answer_source(qa)
    question = qa.get("question")
    context = build_context(example.get("pre_text"), example.get("table"), example.get("post_text"))

    return {
        "id": str(example.get("id", "")).strip(),
        "pre_text": example.get("pre_text", []),
        "post_text": example.get("post_text", []),
        "table": example.get("table", []),
        "question": question,
        "raw_answer": raw_answer,
        "gold_numeric": normalize_gold_numeric(raw_answer),
        "split": split_name,
        "source_answer_field_name": source_field_name,
        "context_char_len": len(context),
    }


def compare_numeric(a: Optional[float], b: Optional[float], tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def summarize_split(
    split_name: str,
    official_name: str,
    examples: List[Dict[str, Any]],
    standardized: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top_level_keys = Counter()
    qa_keys = Counter()
    answer_source_counter = Counter()
    unit_type_counter = Counter()
    boolean_answer_counter = Counter()
    raw_parse_success = 0
    exe_parse_success = 0
    raw_exe_numeric_mismatch = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for example, item in zip(examples, standardized):
        qa = example.get("qa", {})
        top_level_keys.update(example.keys())
        qa_keys.update(qa.keys())
        answer_source_counter[item["source_answer_field_name"]] += 1
        unit_type_counter[answer_unit_type(item["raw_answer"])] += 1
        lowered_answer = str(item["raw_answer"]).strip().lower()
        if lowered_answer in BOOLEAN_MAP:
            boolean_answer_counter[lowered_answer] += 1

        raw_numeric = item["gold_numeric"]
        exe_numeric = normalize_gold_numeric(qa.get("exe_ans"))
        if raw_numeric is not None:
            raw_parse_success += 1
        if exe_numeric is not None:
            exe_parse_success += 1
        if raw_numeric is not None and exe_numeric is not None and not compare_numeric(raw_numeric, exe_numeric):
            raw_exe_numeric_mismatch += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "id": item["id"],
                        "question": qa.get("question"),
                        "raw_answer": item["raw_answer"],
                        "raw_numeric": raw_numeric,
                        "exe_ans": qa.get("exe_ans"),
                        "exe_numeric": exe_numeric,
                        "program": qa.get("program"),
                    }
                )

    context_lengths = [row["context_char_len"] for row in standardized]
    gold_present = [row for row in standardized if row["gold_numeric"] is not None]

    return {
        "split": split_name,
        "official_split_name": official_name,
        "num_examples": len(examples),
        "raw_answer_parse_success": raw_parse_success,
        "raw_answer_parse_rate": round(raw_parse_success / len(examples), 6) if examples else 0.0,
        "exe_ans_parse_success": exe_parse_success,
        "exe_ans_parse_rate": round(exe_parse_success / len(examples), 6) if examples else 0.0,
        "raw_exe_numeric_mismatch_count": raw_exe_numeric_mismatch,
        "min_context_char_len": min(context_lengths) if context_lengths else 0,
        "max_context_char_len": max(context_lengths) if context_lengths else 0,
        "mean_context_char_len": round(sum(context_lengths) / len(context_lengths), 2) if context_lengths else 0.0,
        "top_level_keys": dict(sorted(top_level_keys.items())),
        "qa_keys": dict(sorted(qa_keys.items())),
        "answer_source_counter": dict(sorted(answer_source_counter.items())),
        "answer_unit_counter": dict(sorted(unit_type_counter.items())),
        "boolean_answer_counter": dict(sorted(boolean_answer_counter.items())),
        "gold_numeric_non_null": len(gold_present),
        "gold_numeric_null": len(standardized) - len(gold_present),
        "mismatch_examples": mismatch_examples,
    }


def manifest_payload(
    name: str,
    split: str,
    source_ids: List[str],
    rows: List[Dict[str, Any]],
    seed: int,
    description: str,
) -> Dict[str, Any]:
    return {
        "manifest_name": name,
        "split": split,
        "num_examples": len(rows),
        "seed": seed,
        "description": description,
        "ids": [row["id"] for row in rows],
        "source_id_pool_size": len(source_ids),
    }


def build_manifests(
    validation_rows: List[Dict[str, Any]],
    manifest_dir: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    ordered_rows = sorted(validation_rows, key=lambda row: row["id"])
    shuffled_rows = list(ordered_rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled_rows)

    calib_rows = shuffled_rows[:50]
    screen_rows = shuffled_rows[50:250]
    validation_full_rows = ordered_rows

    manifests = [
        (
            "val_calib50",
            calib_rows,
            "Validation split 中按固定 seed 打乱后选出的前 50 条，用于 max_new_tokens 长度校准。",
        ),
        (
            "val_screen200",
            screen_rows,
            "Validation split 中与 val_calib50 不重叠的后续 200 条，用于 benchmark 准入筛选。",
        ),
        (
            "validation883",
            validation_full_rows,
            "Validation split 的完整 883 条标准化样本，用于通过准入后的完整验证。",
        ),
    ]

    manifest_index_rows: List[Dict[str, Any]] = []
    source_ids = [row["id"] for row in ordered_rows]
    for name, rows, description in manifests:
        write_jsonl(manifest_dir / f"{name}.jsonl", rows)
        payload = manifest_payload(name, "validation", source_ids, rows, seed, description)
        write_json(manifest_dir / f"{name}.json", payload)
        manifest_index_rows.append(
            {
                "manifest_name": name,
                "split": "validation",
                "num_examples": len(rows),
                "seed": seed,
                "description": description,
            }
        )

    return manifest_index_rows


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "raw" / "finqa"
    processed_dir = project_root / "data" / "processed" / "finqa"
    manifest_dir = project_root / "data" / "manifests"
    metadata_dir = project_root / "outputs" / "metadata"

    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    ensure_dir(manifest_dir)
    ensure_dir(metadata_dir)

    split_summaries: List[Dict[str, Any]] = []
    standardized_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split_name, config in OFFICIAL_SPLITS.items():
        raw_path = raw_dir / f"{config['official_name']}.json"
        download_file(config["url"], raw_path, overwrite=args.overwrite)
        examples = load_json(raw_path)
        standardized = [standardize_example(example, split_name) for example in examples]
        standardized_by_split[split_name] = standardized

        write_jsonl(processed_dir / f"{split_name}.standardized.jsonl", standardized)
        split_summary = summarize_split(split_name, config["official_name"], examples, standardized)
        split_summaries.append(split_summary)
        write_json(metadata_dir / f"finqa_{split_name}_field_summary.json", split_summary)

    manifest_index_rows = build_manifests(standardized_by_split["validation"], manifest_dir, seed=args.seed)
    write_csv(
        metadata_dir / "finqa_split_summary.csv",
        [
            {
                "split": row["split"],
                "official_split_name": row["official_split_name"],
                "num_examples": row["num_examples"],
                "raw_answer_parse_success": row["raw_answer_parse_success"],
                "raw_answer_parse_rate": row["raw_answer_parse_rate"],
                "exe_ans_parse_success": row["exe_ans_parse_success"],
                "exe_ans_parse_rate": row["exe_ans_parse_rate"],
                "raw_exe_numeric_mismatch_count": row["raw_exe_numeric_mismatch_count"],
                "gold_numeric_non_null": row["gold_numeric_non_null"],
                "gold_numeric_null": row["gold_numeric_null"],
                "min_context_char_len": row["min_context_char_len"],
                "max_context_char_len": row["max_context_char_len"],
                "mean_context_char_len": row["mean_context_char_len"],
            }
            for row in split_summaries
        ],
        fieldnames=[
            "split",
            "official_split_name",
            "num_examples",
            "raw_answer_parse_success",
            "raw_answer_parse_rate",
            "exe_ans_parse_success",
            "exe_ans_parse_rate",
            "raw_exe_numeric_mismatch_count",
            "gold_numeric_non_null",
            "gold_numeric_null",
            "min_context_char_len",
            "max_context_char_len",
            "mean_context_char_len",
        ],
    )
    write_csv(
        metadata_dir / "finqa_manifest_index.csv",
        manifest_index_rows,
        fieldnames=["manifest_name", "split", "num_examples", "seed", "description"],
    )
    write_json(metadata_dir / "finqa_manifest_index.json", manifest_index_rows)

    print("Prepared FinQA raw splits and standardized assets.")
    for row in split_summaries:
        print(
            f"- {row['split']}: {row['num_examples']} examples | "
            f"raw_parse={row['raw_answer_parse_success']}/{row['num_examples']} | "
            f"exe_parse={row['exe_ans_parse_success']}/{row['num_examples']} | "
            f"raw_exe_mismatch={row['raw_exe_numeric_mismatch_count']}"
        )
    for row in manifest_index_rows:
        print(f"- manifest {row['manifest_name']}: {row['num_examples']} examples")


if __name__ == "__main__":
    main()
