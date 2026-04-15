#!/usr/bin/env python3
"""Build post-hoc error taxonomy tables from existing validation883 predictions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


PROTOCOL_PRIMARY_BUCKETS: Sequence[str] = (
    "truncated_without_answer",
    "missing_answer_line",
    "empty_answer_line",
    "invalid_numeric_literal",
    "none",
)

OUTCOME_BUCKETS: Sequence[str] = (
    "strict_pass",
    "strict_fail_but_relaxed_hit",
    "strict_fail_and_relaxed_fail",
)

SAMPLE_LEVEL_COLUMNS: Sequence[str] = (
    "model",
    "id",
    "gold_numeric",
    "prediction_text",
    "new_tokens",
    "strict_tm",
    "relaxed_tm",
    "relaxed_gap_tm",
    "format_ok",
    "valid_parse",
    "answer_line_raw",
    "multiple_answer_lines",
    "truncated_suspect",
    "parse_error_reason",
    "protocol_primary_bucket",
    "multiple_answer_lines_flag",
    "outcome_bucket",
)

COUNT_OUTPUT_COLUMNS: Sequence[str] = (
    "model",
    "n_total",
    "strict_pass_count",
    "strict_fail_count",
    "missing_answer_line_count",
    "empty_answer_line_count",
    "invalid_numeric_literal_count",
    "multiple_answer_lines_count",
    "truncated_without_answer_count",
    "strict_fail_but_relaxed_hit_count",
    "strict_fail_and_relaxed_fail_count",
    "strict_pass_rate",
    "strict_fail_rate",
    "missing_answer_line_rate",
    "empty_answer_line_rate",
    "invalid_numeric_literal_rate",
    "multiple_answer_lines_rate",
    "truncated_without_answer_rate",
    "strict_fail_but_relaxed_hit_rate",
    "strict_fail_and_relaxed_fail_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/provisional/validation883_assigned"),
        help="Root directory containing <MODEL_NAME>/predictions.jsonl outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/provisional/validation883_assigned/error_taxonomy_v1"),
        help="Directory to write taxonomy reports.",
    )
    parser.add_argument(
        "--sample-k",
        type=int,
        default=15,
        help="Maximum manual review samples per model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for manual review sampling.",
    )
    parser.add_argument(
        "--include-models",
        type=str,
        default=None,
        help="Optional comma-separated allowlist of model directory names.",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help="Optional comma-separated denylist of model directory names.",
    )
    args = parser.parse_args()
    if args.sample_k < 0:
        raise ValueError("--sample-k must be >= 0")
    return args


def parse_model_filter(raw: str | None) -> set[str] | None:
    """Parse a comma-separated model filter string."""
    if raw is None:
        return None
    models = {item.strip() for item in raw.split(",") if item.strip()}
    return models or None


def coerce_bool(value: Any) -> bool:
    """Convert common scalar representations into booleans."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def discover_prediction_files(
    input_root: Path,
    include_models: set[str] | None,
    exclude_models: set[str] | None,
) -> List[Path]:
    """Discover predictions.jsonl files under model subdirectories."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    files: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if include_models is not None and child.name not in include_models:
            continue
        if exclude_models is not None and child.name in exclude_models:
            continue
        predictions_path = child / "predictions.jsonl"
        if predictions_path.exists():
            files.append(predictions_path)

    if not files:
        raise FileNotFoundError(
            f"No predictions.jsonl discovered under {input_root} after applying filters."
        )
    return files


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load UTF-8 JSONL rows from disk."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}, got {type(payload)!r}"
                )
            rows.append(payload)
    return rows


def derive_protocol_primary_bucket(
    format_ok: bool,
    truncated_suspect: bool,
    parse_error_reason: str | None,
) -> str:
    """Assign the mutually exclusive protocol bucket with the required priority order."""
    if truncated_suspect and not format_ok:
        return "truncated_without_answer"

    reason = (parse_error_reason or "").strip()
    if reason == "missing_answer_line":
        return "missing_answer_line"
    if reason == "empty_answer_line":
        return "empty_answer_line"
    if reason == "invalid_numeric_literal":
        return "invalid_numeric_literal"
    return "none"


def derive_outcome_bucket(strict_tm: bool, relaxed_tm: bool) -> str:
    """Assign the mutually exclusive scoring outcome bucket."""
    if strict_tm:
        return "strict_pass"
    if relaxed_tm:
        return "strict_fail_but_relaxed_hit"
    return "strict_fail_and_relaxed_fail"


def flatten_prediction_row(raw_row: Mapping[str, Any], default_model: str) -> Dict[str, Any]:
    """Flatten a single raw prediction row into taxonomy-ready fields."""
    parse_payload = raw_row.get("parse")
    parse_data: Mapping[str, Any] = (
        parse_payload if isinstance(parse_payload, Mapping) else {}
    )

    row_model = raw_row.get("model")
    model = default_model
    if row_model is not None and str(row_model).strip():
        model = str(row_model).strip()

    format_ok = coerce_bool(parse_data.get("format_ok"))
    valid_parse = coerce_bool(parse_data.get("valid_parse"))
    multiple_answer_lines = coerce_bool(parse_data.get("multiple_answer_lines"))
    truncated_suspect = coerce_bool(parse_data.get("truncated_suspect"))
    strict_tm = coerce_bool(raw_row.get("strict_tm"))
    relaxed_tm = coerce_bool(raw_row.get("relaxed_tm"))
    parse_error_reason = parse_data.get("parse_error_reason")
    if parse_error_reason is not None:
        parse_error_reason = str(parse_error_reason)

    flattened: Dict[str, Any] = {
        "model": model,
        "id": raw_row.get("id"),
        "gold_numeric": raw_row.get("gold_numeric"),
        "prediction_text": raw_row.get("prediction_text"),
        "new_tokens": raw_row.get("new_tokens"),
        "strict_tm": strict_tm,
        "relaxed_tm": relaxed_tm,
        "relaxed_gap_tm": raw_row.get("relaxed_gap_tm"),
        "format_ok": format_ok,
        "valid_parse": valid_parse,
        "answer_line_raw": parse_data.get("answer_line_raw"),
        "multiple_answer_lines": multiple_answer_lines,
        "truncated_suspect": truncated_suspect,
        "parse_error_reason": parse_error_reason,
    }
    flattened["protocol_primary_bucket"] = derive_protocol_primary_bucket(
        format_ok=format_ok,
        truncated_suspect=truncated_suspect,
        parse_error_reason=parse_error_reason,
    )
    flattened["multiple_answer_lines_flag"] = multiple_answer_lines
    flattened["outcome_bucket"] = derive_outcome_bucket(
        strict_tm=strict_tm,
        relaxed_tm=relaxed_tm,
    )
    return flattened


def build_sample_level_dataframe(prediction_files: Iterable[Path]) -> pd.DataFrame:
    """Build the sample-level taxonomy dataframe from discovered predictions."""
    rows: List[Dict[str, Any]] = []
    for predictions_path in prediction_files:
        default_model = predictions_path.parent.name
        raw_rows = load_jsonl(predictions_path)
        if not raw_rows:
            raise ValueError(f"No JSONL rows found in {predictions_path}")
        rows.extend(flatten_prediction_row(raw_row, default_model) for raw_row in raw_rows)

    if not rows:
        raise ValueError("No prediction rows loaded from discovered model outputs.")

    dataframe = pd.DataFrame(rows)
    for column in SAMPLE_LEVEL_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    dataframe = dataframe.loc[:, list(SAMPLE_LEVEL_COLUMNS)].copy()
    dataframe = dataframe.sort_values(["model", "id"], kind="stable").reset_index(drop=True)
    return dataframe


def build_count_row(model: str, group: pd.DataFrame) -> Dict[str, Any]:
    """Build one model-level count/rate row."""
    n_total = int(len(group))
    strict_pass_count = int((group["outcome_bucket"] == "strict_pass").sum())
    strict_fail_count = n_total - strict_pass_count
    missing_answer_line_count = int(
        (group["protocol_primary_bucket"] == "missing_answer_line").sum()
    )
    empty_answer_line_count = int(
        (group["protocol_primary_bucket"] == "empty_answer_line").sum()
    )
    invalid_numeric_literal_count = int(
        (group["protocol_primary_bucket"] == "invalid_numeric_literal").sum()
    )
    multiple_answer_lines_count = int(group["multiple_answer_lines_flag"].sum())
    truncated_without_answer_count = int(
        (group["protocol_primary_bucket"] == "truncated_without_answer").sum()
    )
    strict_fail_but_relaxed_hit_count = int(
        (group["outcome_bucket"] == "strict_fail_but_relaxed_hit").sum()
    )
    strict_fail_and_relaxed_fail_count = int(
        (group["outcome_bucket"] == "strict_fail_and_relaxed_fail").sum()
    )

    row: Dict[str, Any] = {
        "model": model,
        "n_total": n_total,
        "strict_pass_count": strict_pass_count,
        "strict_fail_count": strict_fail_count,
        "missing_answer_line_count": missing_answer_line_count,
        "empty_answer_line_count": empty_answer_line_count,
        "invalid_numeric_literal_count": invalid_numeric_literal_count,
        "multiple_answer_lines_count": multiple_answer_lines_count,
        "truncated_without_answer_count": truncated_without_answer_count,
        "strict_fail_but_relaxed_hit_count": strict_fail_but_relaxed_hit_count,
        "strict_fail_and_relaxed_fail_count": strict_fail_and_relaxed_fail_count,
    }

    if n_total == 0:
        raise ValueError(f"Cannot build count row for empty group: {model}")

    for key in (
        "strict_pass",
        "strict_fail",
        "missing_answer_line",
        "empty_answer_line",
        "invalid_numeric_literal",
        "multiple_answer_lines",
        "truncated_without_answer",
        "strict_fail_but_relaxed_hit",
        "strict_fail_and_relaxed_fail",
    ):
        row[f"{key}_rate"] = row[f"{key}_count"] / n_total

    return row


def build_by_model_counts(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts and rates per model."""
    rows = [
        build_count_row(model=str(model), group=group.copy())
        for model, group in sample_df.groupby("model", sort=True, dropna=False)
    ]
    counts_df = pd.DataFrame(rows)
    counts_df = counts_df.loc[:, list(COUNT_OUTPUT_COLUMNS)].copy()
    counts_df = counts_df.sort_values("model", kind="stable").reset_index(drop=True)
    return counts_df


def build_overall_counts(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts and rates across all models."""
    overall_row = build_count_row(model="ALL_MODELS", group=sample_df.copy())
    return pd.DataFrame([overall_row], columns=list(COUNT_OUTPUT_COLUMNS))


def build_manual_review_candidates(
    sample_df: pd.DataFrame,
    sample_k: int,
    seed: int,
) -> pd.DataFrame:
    """Sample strict-fail-and-relaxed-fail rows for manual review."""
    target_df = sample_df.loc[
        sample_df["outcome_bucket"] == "strict_fail_and_relaxed_fail"
    ].copy()

    sampled_rows: List[Dict[str, Any]] = []
    for model, group in target_df.groupby("model", sort=True, dropna=False):
        group = group.sort_values(["model", "id"], kind="stable").reset_index(drop=True)
        records = group.to_dict(orient="records")
        if sample_k == 0:
            selected_records: List[Dict[str, Any]] = []
        elif len(records) <= sample_k:
            selected_records = records
        else:
            rng = random.Random(f"{seed}:{model}")
            selected_indices = sorted(rng.sample(range(len(records)), sample_k))
            selected_records = [records[index] for index in selected_indices]

        for record in selected_records:
            record["manual_label"] = ""
            record["review_notes"] = ""
            sampled_rows.append(record)

    columns = list(SAMPLE_LEVEL_COLUMNS) + ["manual_label", "review_notes"]
    if not sampled_rows:
        return pd.DataFrame(columns=columns)

    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df = sampled_df.loc[:, columns].copy()
    sampled_df = sampled_df.sort_values(["model", "id"], kind="stable").reset_index(drop=True)
    return sampled_df


def ensure_output_dir(path: Path) -> None:
    """Create output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def write_dataframe_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Write a dataframe to CSV using UTF-8."""
    dataframe.to_csv(path, index=False, encoding="utf-8")


def determine_top_protocol_issue(row: pd.Series) -> tuple[str, float]:
    """Find the highest-rate protocol issue for reporting."""
    issue_columns = (
        "missing_answer_line_rate",
        "empty_answer_line_rate",
        "invalid_numeric_literal_rate",
        "truncated_without_answer_rate",
    )
    top_issue = "none"
    top_rate = 0.0
    for column in issue_columns:
        rate = float(row[column])
        if rate > top_rate:
            top_rate = rate
            top_issue = column.removesuffix("_rate")
    return top_issue, top_rate


def write_summary_markdown(path: Path, by_model_counts: pd.DataFrame) -> None:
    """Write a compact markdown summary of the error taxonomy."""
    lines = [
        "# validation883 error taxonomy summary",
        "",
        "- 本文档基于 `validation883` 现有 `predictions.jsonl` 做后验错误分类。",
        "- `Strict TM` 仍是主指标；本摘要只补充解释 strict 失败的结构。",
        "",
        "| 模型 | strict_fail_but_relaxed_hit_rate | strict_fail_and_relaxed_fail_rate | top protocol issue by rate | top protocol issue rate |",
        "| --- | --- | --- | --- | --- |",
    ]

    for _, row in by_model_counts.sort_values("model", kind="stable").iterrows():
        top_issue, top_rate = determine_top_protocol_issue(row)
        lines.append(
            "| {model} | {relaxed_hit:.6f} | {relaxed_fail:.6f} | {issue} | {issue_rate:.6f} |".format(
                model=row["model"],
                relaxed_hit=row["strict_fail_but_relaxed_hit_rate"],
                relaxed_fail=row["strict_fail_and_relaxed_fail_rate"],
                issue=top_issue,
                issue_rate=top_rate,
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(sample_df: pd.DataFrame, by_model_counts: pd.DataFrame) -> None:
    """Validate aggregation invariants."""
    protocol_bucket_total_series = sample_df["protocol_primary_bucket"].value_counts()
    protocol_bucket_total = int(
        sum(int(protocol_bucket_total_series.get(bucket, 0)) for bucket in PROTOCOL_PRIMARY_BUCKETS)
    )
    if protocol_bucket_total != len(sample_df):
        raise AssertionError(
            "Protocol bucket counts do not sum to sample count: "
            f"{protocol_bucket_total} != {len(sample_df)}"
        )

    for _, row in by_model_counts.iterrows():
        model = str(row["model"])
        n_total = int(row["n_total"])
        strict_total = int(row["strict_pass_count"]) + int(row["strict_fail_count"])
        if strict_total != n_total:
            raise AssertionError(
                f"strict_pass_count + strict_fail_count != n_total for {model}: "
                f"{strict_total} != {n_total}"
            )

        protocol_total = (
            int(row["missing_answer_line_count"])
            + int(row["empty_answer_line_count"])
            + int(row["invalid_numeric_literal_count"])
            + int(row["truncated_without_answer_count"])
            + int((sample_df["model"] == model).sum())
            - int(
                (
                    sample_df.loc[sample_df["model"] == model, "protocol_primary_bucket"] != "none"
                ).sum()
            )
        )
        if protocol_total != n_total:
            raise AssertionError(
                f"protocol_primary_bucket counts + none != n_total for {model}: "
                f"{protocol_total} != {n_total}"
            )


def print_console_summary(
    sample_df: pd.DataFrame,
    by_model_counts: pd.DataFrame,
    output_paths: Sequence[Path],
) -> None:
    """Print the requested short console summary."""
    discovered_models = list(by_model_counts["model"])
    print("Discovered models:")
    for model in discovered_models:
        print(f"- {model}")

    print("\nPer-model totals:")
    for _, row in by_model_counts.iterrows():
        model = row["model"]
        n_total = int(row["n_total"])
        strict_total = int(row["strict_pass_count"]) + int(row["strict_fail_count"])
        none_count = int(
            (
                sample_df.loc[sample_df["model"] == model, "protocol_primary_bucket"] == "none"
            ).sum()
        )
        protocol_total = (
            int(row["missing_answer_line_count"])
            + int(row["empty_answer_line_count"])
            + int(row["invalid_numeric_literal_count"])
            + int(row["truncated_without_answer_count"])
            + none_count
        )
        print(
            f"- {model}: n_total={n_total}, strict_total={strict_total}, "
            f"protocol_total_with_none={protocol_total}"
        )

    print("\nOutput files:")
    for path in output_paths:
        print(f"- {path}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    include_models = parse_model_filter(args.include_models)
    exclude_models = parse_model_filter(args.exclude_models)

    prediction_files = discover_prediction_files(
        input_root=input_root,
        include_models=include_models,
        exclude_models=exclude_models,
    )
    sample_df = build_sample_level_dataframe(prediction_files)
    by_model_counts = build_by_model_counts(sample_df)
    overall_counts = build_overall_counts(sample_df)
    manual_review_df = build_manual_review_candidates(
        sample_df=sample_df,
        sample_k=args.sample_k,
        seed=args.seed,
    )

    run_sanity_checks(sample_df, by_model_counts)

    by_model_path = output_dir / "by_model_error_counts.csv"
    overall_path = output_dir / "overall_error_counts.csv"
    sample_level_path = output_dir / "sample_level_taxonomy.csv"
    manual_review_path = output_dir / "manual_review_candidates.csv"
    summary_md_path = output_dir / "error_taxonomy_summary.md"

    write_dataframe_csv(by_model_path, by_model_counts)
    write_dataframe_csv(overall_path, overall_counts)
    write_dataframe_csv(sample_level_path, sample_df)
    write_dataframe_csv(manual_review_path, manual_review_df)
    write_summary_markdown(summary_md_path, by_model_counts)

    print_console_summary(
        sample_df=sample_df,
        by_model_counts=by_model_counts,
        output_paths=(
            by_model_path,
            overall_path,
            sample_level_path,
            manual_review_path,
            summary_md_path,
        ),
    )


if __name__ == "__main__":
    main()
