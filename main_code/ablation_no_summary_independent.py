from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

for path in (str(CURRENT_DIR), str(WORKSPACE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from summarizer import Summarizer  # noqa: E402
from pythonProject1.API_invocation.qwen_gen import (  # noqa: E402
    get_shared_usage,
    reset_shared_usage,
    shared_qwen_client,
)


DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Run_Results_Ablation_Strict"
QWEN_MODEL = "qwen2.5-72b-instruct"
QWEN_MAX_TOKENS = 8192
QWEN_TEMPERATURE = 0.0
REQUEST_TIMEOUT = int(os.getenv("LAED_DETECTION_REQUEST_TIMEOUT", "120"))
REQUEST_RETRIES = int(os.getenv("LAED_DETECTION_RETRIES", "1"))
MAX_VALUES_PER_COLUMN = int(os.getenv("LAED_DIRECT_MAX_VALUES_PER_COLUMN", "120"))
DIRECT_BATCH_LIMIT = int(os.getenv("LAED_DIRECT_BATCH_LIMIT", "30"))

ErrorCell = Tuple[int, str]

MISSING_LIKE_VALUES = {
    "",
    "nan",
    "na",
    "n/a",
    "null",
    "none",
    "missing",
    "unknown",
    "not available",
    "not applicable",
}


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def configure_qwen() -> None:
    shared_qwen_client.model = QWEN_MODEL
    shared_qwen_client.max_tokens = QWEN_MAX_TOKENS
    shared_qwen_client.temperature = QWEN_TEMPERATURE


def dump_json(value, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)


def resolve_dataset_paths(dataset: str, data_dir: Path) -> tuple[str, Path, Path]:
    data_dir = data_dir.resolve()
    key = dataset.strip().lower()
    aliases = {
        "hospital": "hospital",
        "hospitals": "hospital",
        "flights": "flights",
        "flight": "flights",
        "beers": "beers",
        "beer": "beers",
        "rayyan": "rayyan",
        "movies": "movies",
        "movie": "movies",
    }
    name = aliases.get(key, key)
    dirty = data_dir / f"{name}_error.csv"
    clean = data_dir / f"{name}_clean.csv"
    if not dirty.exists() or not clean.exists():
        candidates = {p.name.lower(): p for p in data_dir.glob("*.csv")}
        dirty = candidates.get(f"{name}_error.csv", dirty)
        clean = candidates.get(f"{name}_clean.csv", clean)
    if not dirty.exists():
        raise FileNotFoundError(f"Dirty CSV not found for dataset '{dataset}': {dirty}")
    if not clean.exists():
        raise FileNotFoundError(f"Clean CSV not found for dataset '{dataset}': {clean}")
    return name, dirty.resolve(), clean.resolve()


def load_csv(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    del chunksize
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, keep_default_na=False, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def load_ground_truth_errors(clean_path: Path, dirty_path: Path) -> Set[ErrorCell]:
    clean = load_csv(clean_path)
    dirty = load_csv(dirty_path)
    if clean.shape != dirty.shape:
        raise ValueError(f"Clean and dirty files have different shapes: {clean.shape} vs {dirty.shape}.")
    if list(clean.columns) != list(dirty.columns):
        raise ValueError("Clean and dirty files have different columns.")
    errors: Set[ErrorCell] = set()
    for row_idx in clean.index:
        for col in clean.columns:
            if str(clean.at[row_idx, col]) != str(dirty.at[row_idx, col]):
                errors.add((int(row_idx), str(col)))
    return errors


def save_error_cells(errors: Iterable[ErrorCell], output_path: Path) -> None:
    dump_json([[row, col] for row, col in sorted(errors)], output_path)


def compute_metrics(ground_truth: Set[ErrorCell], detected: Set[ErrorCell]) -> dict:
    tp = len(ground_truth & detected)
    fp = len(detected - ground_truth)
    fn = len(ground_truth - detected)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ground_truth_count": len(ground_truth),
        "detected_count": len(detected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def detected_errors_from_result(result: dict) -> Set[ErrorCell]:
    detected: Set[ErrorCell] = set()
    for row, col in result.get("final_errors", []):
        detected.add((int(row), str(col)))
    return detected


def evaluate_detection(clean_path: Path, dirty_path: Path, result: dict, output_dir: Path) -> dict:
    ground_truth = load_ground_truth_errors(clean_path, dirty_path)
    detected = detected_errors_from_result(result)
    save_error_cells(ground_truth, output_dir / "ground_truth_errors.json")
    save_error_cells(detected, output_dir / "detected_errors.json")
    metrics = compute_metrics(ground_truth, detected)
    dump_json(metrics, output_dir / "evaluation_metrics.json")
    return metrics


def summarize_error_distribution(detailed_errors: list[dict]) -> tuple[dict, dict]:
    error_types_count: dict[str, int] = {}
    field_errors: dict[str, int] = {}
    for error in detailed_errors:
        for error_type in str(error.get("errorType", "")).split(","):
            if error_type:
                error_types_count[error_type] = error_types_count.get(error_type, 0) + 1
        field = str(error.get("fieldName", ""))
        if field:
            field_errors[field] = field_errors.get(field, 0) + 1
    return error_types_count, field_errors


def is_missing_like(value) -> bool:
    text = "" if value is None else str(value).strip()
    return text.lower() in MISSING_LIKE_VALUES


def text_noise_reasons(value: str) -> list[str]:
    text = "" if value is None else str(value)
    reasons = []
    if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in text):
        reasons.append("control_character")
    if re.search(r"(?:�|Ã|Â|â€|锟)", text):
        reasons.append("mojibake_marker")
    if text and len(re.sub(r"[A-Za-z0-9\u4e00-\u9fff\s.,;:()/_+\-%&'\"#]", "", text)) >= 3:
        reasons.append("symbol_noise")
    return reasons


def value_shape(value: str) -> str:
    text = "" if value is None else str(value).strip()
    if is_missing_like(text):
        return "<missing-like>"
    out = []
    for ch in text:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def compact_value(value, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def infer_dtype(series: pd.Series) -> str:
    values = series.astype(str)
    nonmissing = values[~values.map(is_missing_like)]
    if nonmissing.empty:
        return "string"
    numeric = pd.to_numeric(nonmissing, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return "number"
    date_like = nonmissing.map(lambda value: bool(re.search(r"\d{1,4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}|\b\d{4}\b", str(value))))
    if date_like.mean() >= 0.80:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_dates = pd.to_datetime(nonmissing, errors="coerce", format=None)
        if parsed_dates.notna().mean() >= 0.90 and len(nonmissing) >= 5:
            return "date"
    uniqueness = nonmissing.nunique(dropna=False) / max(len(nonmissing), 1)
    return "category" if uniqueness < 0.20 else "string"


def build_initial_profile(df: pd.DataFrame) -> dict:
    fields = []
    for col in df.columns:
        values = df[col].astype(str)
        nonmissing = values[~values.map(is_missing_like)]
        top_counts = values.value_counts(dropna=False).head(12)
        shape_counts = values.map(value_shape).value_counts(dropna=False).head(12)
        sample_values = (
            nonmissing.drop_duplicates().sample(n=min(30, nonmissing.nunique()), random_state=42).tolist()
            if nonmissing.nunique() else []
        )
        fields.append({
            "column": str(col),
            "dtype": infer_dtype(values),
            "row_count": int(len(values)),
            "missing_like_count": int(values.map(is_missing_like).sum()),
            "num_unique_values": int(values.nunique(dropna=False)),
            "uniqueness_ratio": round(float(values.nunique(dropna=False) / max(len(values), 1)), 4),
            "top_value_counts": {str(k): int(v) for k, v in top_counts.items()},
            "shape_counts": {str(k): int(v) for k, v in shape_counts.items()},
            "sample_values": [compact_value(v) for v in sample_values],
        })
    return {
        "source": "raw_dirty_table_only",
        "summary_module_used": False,
        "fields": fields,
        "field_names": [str(c) for c in df.columns],
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


def extract_json_object(text: str):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        return json.loads(match.group(0))
    raise ValueError("LLM response did not contain a JSON object.")


def fallback_regex_from_shape(field: dict) -> dict:
    dtype = field.get("dtype", "")
    if dtype == "number":
        return {
            "format": "generic numeric surface",
            "regex": r"^-?\d+(?:\.\d+)?$",
            "explanation": "Non-summary fallback from inferred numeric dtype.",
            "source": "nonsemantic_fallback",
        }
    shape_counts = field.get("shape_counts", {}) or {}
    nonmissing_shapes = [(s, int(c)) for s, c in shape_counts.items() if s != "<missing-like>"]
    if not nonmissing_shapes:
        return {"format": "unknown", "regex": "", "explanation": "No non-missing shape support.", "source": "none"}
    total = sum(c for _, c in nonmissing_shapes)
    shape, count = sorted(nonmissing_shapes, key=lambda item: (-item[1], item[0]))[0]
    if total <= 0 or count / total < 0.80 or len(shape) > 80:
        return {
            "format": "unknown",
            "regex": "",
            "explanation": "No dominant nonsemantic shape strong enough for initial screening.",
            "source": "none",
        }
    pieces = []
    idx = 0
    while idx < len(shape):
        ch = shape[idx]
        run = 1
        while idx + run < len(shape) and shape[idx + run] == ch:
            run += 1
        if ch == "9":
            pieces.append(rf"\d{{{run}}}")
        elif ch == "A":
            pieces.append(rf"[A-Za-z]{{{run}}}")
        elif ch == " ":
            pieces.append(r"\s+")
        else:
            pieces.append(re.escape(ch) if run == 1 else re.escape(ch) + f"{{{run}}}")
        idx += run
    return {
        "format": "dominant nonsemantic shape",
        "regex": "^" + "".join(pieces) + "$",
        "explanation": f"Fallback from dominant raw value shape with support {count}/{total}.",
        "source": "nonsemantic_shape_fallback",
    }


def generate_format_rules(profile: dict) -> dict:
    rules = {}
    for field in profile.get("fields", []):
        col = field["column"]
        rule = fallback_regex_from_shape(field)
        rule.setdefault("format", "unknown")
        rule.setdefault("regex", "")
        rule.setdefault(
            "explanation",
            "Deterministic no-summary initial-screening rule generated directly from raw table values.",
        )
        rule["source"] = "deterministic_raw_table_nonsemantic"
        rules[col] = rule
    return rules


def build_llm_format_rule_summary(df: pd.DataFrame) -> dict:
    """
    Build a summary shell for no-summary format-rule generation.

    It intentionally leaves semantic_type and description empty. The LLM sees
    only column names, inferred dtype, compact observed profiles, and samples,
    so format rules are generated without data-summary semantics.
    """
    fields = []
    basic_profile = {field["column"]: field for field in build_initial_profile(df).get("fields", [])}
    for col in df.columns:
        info = basic_profile.get(str(col), {})
        props = {
            "uniqueness_ratio": info.get("uniqueness_ratio", 0.0),
            "dtype": info.get("dtype", "string"),
            "num_unique_values": info.get("num_unique_values", 0),
            "semantic_type": "",
            "description": "",
        }
        fields.append({"column": str(col), "properties": props})
    summary = {
        "name": "",
        "file_name": "",
        "dataset_description": "",
        "fields": fields,
        "field_names": [str(col) for col in df.columns],
        "field_relationships": {
            "hierarchical": {},
            "mathematical": {},
            "temporal": [],
            "associative": {},
        },
        "relationship_evidence": {},
        "format_rules": {},
        "summary_module_used": False,
        "ablation_flags": {
            "no_summary": True,
            "semantic_summary_disabled": True,
        },
    }
    summarizer = Summarizer()
    return summarizer._refresh_profile_metadata(summary, df)


def generate_no_summary_format_rules(df: pd.DataFrame, fallback_profile: dict) -> tuple[dict, dict]:
    summary = build_llm_format_rule_summary(df)
    fallback_rules = generate_format_rules(fallback_profile)
    try:
        rules = Summarizer().extract_format_rules(df, summary)
    except Exception as exc:
        print(f"[Warn] no-summary LLM format-rule generation failed; using deterministic fallback rules: {exc}")
        rules = {}

    merged = {}
    for col in df.columns:
        rule = rules.get(str(col), {}) if isinstance(rules, dict) else {}
        if not isinstance(rule, dict) or not str(rule.get("regex", "") or "").strip():
            rule = fallback_rules.get(str(col), {"format": "unknown", "regex": ""})
        rule = dict(rule)
        rule.setdefault("format", "unknown")
        rule.setdefault("regex", "")
        rule.setdefault("explanation", "No-summary format rule.")
        rule["source"] = "llm_raw_table_nonsemantic" if str(rule.get("regex", "") or "").strip() else rule.get("source", "none")
        merged[str(col)] = rule
    return summary, merged


def infer_relationship_rules(df: pd.DataFrame, max_rules: int = 30) -> list[dict]:
    rules = []
    columns = [str(c) for c in df.columns]
    values = df.astype(str)
    for key_col in columns:
        key_series = values[key_col]
        nonmissing_key = key_series[~key_series.map(is_missing_like)]
        if nonmissing_key.empty:
            continue
        duplicate_groups = nonmissing_key.value_counts()
        duplicate_groups = duplicate_groups[duplicate_groups >= 3]
        if len(duplicate_groups) < 3:
            continue
        for dep_col in columns:
            if dep_col == key_col:
                continue
            pair = values[[key_col, dep_col]]
            clean_pair = pair[
                (~pair[key_col].map(is_missing_like)) &
                (~pair[dep_col].map(is_missing_like))
            ]
            if len(clean_pair) < 20:
                continue
            group_count = 0
            violation_rows = []
            consistent_cells = 0
            total_cells = 0
            examples = []
            for key_value, group in clean_pair.groupby(key_col, sort=False):
                if len(group) < 3:
                    continue
                counts = group[dep_col].value_counts()
                if counts.empty:
                    continue
                majority = str(counts.index[0])
                majority_count = int(counts.iloc[0])
                if majority_count / len(group) < 0.75:
                    continue
                group_count += 1
                consistent_cells += majority_count
                total_cells += len(group)
                bad_rows = group.index[group[dep_col].astype(str) != majority].tolist()
                for row_idx in bad_rows:
                    violation_rows.append(int(row_idx))
                    if len(examples) < 8:
                        examples.append({
                            "row": int(row_idx),
                            "key_value": compact_value(key_value),
                            "candidate_value": compact_value(values.at[row_idx, dep_col]),
                            "majority_value": compact_value(majority),
                        })
            if group_count < 3 or total_cells < 20:
                continue
            violation_ratio = len(set(violation_rows)) / max(total_cells, 1)
            consistency_ratio = consistent_cells / max(total_cells, 1)
            if not violation_rows or violation_ratio > 0.20 or consistency_ratio < 0.80:
                continue
            rules.append({
                "key_column": key_col,
                "dependent_column": dep_col,
                "group_count": group_count,
                "support_cells": int(total_cells),
                "consistency_ratio": round(consistency_ratio, 4),
                "violation_ratio": round(violation_ratio, 4),
                "violation_rows": sorted(set(violation_rows))[:200],
                "examples": examples,
                "source": "raw_table_repeated_key_consistency",
            })
    rules = sorted(rules, key=lambda r: (-r["consistency_ratio"], r["violation_ratio"], r["key_column"], r["dependent_column"]))
    return rules[:max_rules]


def relationship_validator_code(rules: list[dict]) -> str:
    payload = json.dumps(
        [
            {"key_column": r["key_column"], "dependent_column": r["dependent_column"]}
            for r in rules
        ],
        ensure_ascii=False,
    )
    return f"""def validate_relationships(df):
    rules = {payload}
    errors = set()
    for rule in rules:
        key_col = rule.get("key_column")
        dep_col = rule.get("dependent_column")
        if key_col not in df.columns or dep_col not in df.columns:
            continue
        pair = df[[key_col, dep_col]].astype(str)
        for key_value, group in pair.groupby(key_col, sort=False):
            if len(group) < 3:
                continue
            counts = group[dep_col].value_counts()
            if counts.empty:
                continue
            majority = str(counts.index[0])
            majority_count = int(counts.iloc[0])
            if majority_count / max(len(group), 1) < 0.75:
                continue
            for row_idx, value in group[dep_col].items():
                if str(value) != majority:
                    errors.add((int(row_idx), str(dep_col)))
    return errors
"""


def build_initial_screening_rules(df: pd.DataFrame) -> tuple[dict, dict]:
    profile = build_initial_profile(df)
    llm_profile, format_rules = generate_no_summary_format_rules(df, profile)
    relationship_rules = infer_relationship_rules(df)
    rules = {
        "source": "no_summary_raw_table_initial_screening",
        "summary_module_used": False,
        "format_rules": format_rules,
        "relationship_rules": relationship_rules,
        "relationship_validator_code": {
            "validation": {
                "code": relationship_validator_code(relationship_rules),
                "status": "generated_without_summary",
                "explanation": "Generic repeated-key consistency validator built directly from the raw dirty table.",
            }
        },
    }
    profile["llm_format_rule_profile"] = llm_profile
    return profile, rules


def apply_initial_screening(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, pd.DataFrame, float, Set[ErrorCell]]:
    df_processed = df.copy().astype(str)
    correct = pd.DataFrame(True, index=df_processed.index, columns=df_processed.columns)
    relationship_candidates: Set[ErrorCell] = set()

    for col in df_processed.columns:
        rule = (rules.get("format_rules", {}) or {}).get(str(col), {}) or {}
        regex = str(rule.get("regex", "") or "").strip()
        series = df_processed[col].astype(str)
        col_correct = ~series.map(is_missing_like)
        if regex:
            try:
                compiled = re.compile(regex)
                col_correct = col_correct & series.map(lambda value: compiled.fullmatch(str(value).strip()) is not None)
            except re.error:
                print(f"[Warn] no-summary screening ignored invalid regex for '{col}': {regex}")
        correct[col] = col_correct

    for rule in rules.get("relationship_rules", []) or []:
        dep_col = rule.get("dependent_column")
        if dep_col not in df_processed.columns:
            continue
        for row_idx in rule.get("violation_rows", []) or []:
            try:
                rid = int(row_idx)
            except (TypeError, ValueError):
                continue
            if rid in correct.index:
                correct.at[rid, dep_col] = False
                relationship_candidates.add((rid, str(dep_col)))

    correct_count = int(correct.values.sum())
    total_cells = int(df_processed.size)
    pct = correct_count / total_cells * 100 if total_cells else 0.0
    correct.to_csv("correct_cells_mask.csv", index=False, encoding="utf-8-sig")
    df.mask(correct).to_csv("suspicious_cells_for_error_detection.csv", index=False, encoding="utf-8-sig")
    return df_processed, correct, pct, relationship_candidates


NO_SUMMARY_DETECTION_SYSTEM = """
You detect erroneous cells for a LAED no-summary ablation. No data summary,
semantic field description, summary statistics, external knowledge, or hidden
schema may be used. Use only the initial-screening rule/context and shown cell
values. Return JSON only.
"""


NO_SUMMARY_DETECTION_USER = """
Detect erroneous cells in one table column.

Allowed evidence:
- Column name and inferred dtype.
- The initial-screening regex generated directly from the raw dirty table.
- Values shown in this prompt.

Forbidden evidence:
- No data_summary.json.
- No semantic_type, description, dataset description, or summary profile.
- No external domain knowledge.

Allowed error types:
- missing_errors: actual nulls or obvious missing placeholders.
- format_errors: values that visibly violate the regex or the dominant surface pattern shown here.
- outliers: only numeric extremes that are clearly impossible from shown values.
- spelling_errors: only obvious character-level variants of strongly repeated values shown here.

The input rows have already been selected by the no-summary initial-screening
stage. Judge every shown candidate with the rule and local batch evidence. When
the regex is empty, infer only from repeated surface patterns visible in this
batch; when evidence remains weak, leave the value unflagged.

Column: "{col}"
Inferred dtype: {dtype}
Initial-screening regex: `{regex}`
Rule explanation: {explanation}
Values as row index to value:
{samples_json}

Return exactly:
{{
  "errors": [
    {{"row": <row index or list of row indices>, "errorType": "<format_errors|outliers|spelling_errors|missing_errors>"}}
  ]
}}
"""


DIRECT_DETECTION_SYSTEM = """
You detect erroneous cells for a LAED ablation with no summary and no initial
screening. Do not use hidden summaries, rules, semantic descriptions, profile
metadata, inferred profile information, or external knowledge. Use only the
column name and shown values. Return JSON only.
"""


DIRECT_DETECTION_USER = """
Detect erroneous cells in one table column using only the information shown.

Allowed error types:
- missing_errors: actual nulls or obvious missing placeholders.
- format_errors: visibly corrupted surface forms or values that break a strong
  pattern in the shown values.

Do not infer domain-specific rules. If evidence is weak, leave the value
unflagged.

Column: "{col}"
Values as row index to value:
{samples_json}

Return exactly:
{{
  "errors": [
    {{"row": <row index or list of row indices>, "errorType": "<format_errors|missing_errors>"}}
  ]
}}
"""


def normalize_llm_errors(errors: list, col: str, allowed_types: set[str]) -> list[dict]:
    normalized = []
    for error in errors or []:
        if not isinstance(error, dict):
            continue
        etype = str(error.get("errorType", "")).strip()
        if etype not in allowed_types:
            continue
        row_spec = error.get("row", error.get("rows"))
        rows = row_spec if isinstance(row_spec, list) else [row_spec]
        valid_rows = []
        for row in rows:
            try:
                valid_rows.append(int(row))
            except (TypeError, ValueError):
                continue
        if not valid_rows:
            continue
        normalized.append({
            "row": valid_rows if len(valid_rows) > 1 else valid_rows[0],
            "fieldName": col,
            "errorType": etype,
            "description": "No-summary ablation LLM diagnosis.",
            "correctFormat": None,
        })
    return normalized


def expand_representative_errors(errors: list[dict], representative_rows: dict[int, list[int]]) -> list[dict]:
    expanded = []
    for error in errors:
        row_spec = error.get("row")
        rows = row_spec if isinstance(row_spec, list) else [row_spec]
        expanded_rows = []
        for row in rows:
            try:
                rid = int(row)
            except (TypeError, ValueError):
                continue
            expanded_rows.extend(representative_rows.get(rid, [rid]))
        if not expanded_rows:
            continue
        copy = dict(error)
        unique_rows = sorted(set(expanded_rows))
        copy["row"] = unique_rows if len(unique_rows) > 1 else unique_rows[0]
        expanded.append(copy)
    return expanded


def select_direct_prompt_samples(samples: list[dict]) -> list[dict]:
    """
    Deterministic coverage for the no-summary/no-screening direct prompt.

    This ablation must not use summary, initial-screening rules, profile
    evidence, or risk scoring before the LLM. When the column is too large for
    one prompt, keep a stable row-order spread only.
    """
    budget = max(1, MAX_VALUES_PER_COLUMN)
    if len(samples) <= budget:
        return samples
    ordered = sorted(samples, key=lambda item: int(item.get("row", 0)))
    if budget == 1:
        return [ordered[0]]
    step = (len(ordered) - 1) / (budget - 1)
    selected = []
    used_rows = set()
    for pos in range(budget):
        idx = int(round(pos * step))
        sample = ordered[min(idx, len(ordered) - 1)]
        row = int(sample.get("row", 0))
        if row in used_rows:
            continue
        selected.append(sample)
        used_rows.add(row)
    for sample in ordered:
        if len(selected) >= budget:
            break
        row = int(sample.get("row", 0))
        if row in used_rows:
            continue
        selected.append(sample)
        used_rows.add(row)
    return selected


def select_representative_samples(
    samples: list[dict],
    value_to_rows: dict[str, list[int]],
    regex: str,
    direct_only: bool = False,
) -> list[dict]:
    if direct_only:
        return select_direct_prompt_samples(samples)
    budget = max(1, MAX_VALUES_PER_COLUMN)
    if len(samples) <= budget:
        return samples
    del value_to_rows, regex
    selected = sorted(samples, key=lambda item: int(item.get("row", 0)))[:budget]
    print(f"No-summary LLM row-order budget kept {len(selected)}/{len(samples)} distinct suspicious values")
    return selected


def detect_columns(
    df: pd.DataFrame,
    correct_cells: pd.DataFrame,
    profile: dict,
    rules: dict,
    direct_only: bool,
) -> tuple[list[dict], dict]:
    field_profiles = {field["column"]: field for field in profile.get("fields", [])}
    format_rules = rules.get("format_rules", {}) if rules else {}
    errors_all = []
    per_column_counts = {}
    suspicious_mask = ~correct_cells
    for col in df.columns:
        suspicious_idx = suspicious_mask.index[suspicious_mask[col]].tolist()
        if not suspicious_idx:
            per_column_counts[str(col)] = 0
            print(f" -> no suspicious cells in '{col}', skip LLM")
            continue
        series = df.loc[suspicious_idx, col].astype(str)
        value_to_rows: dict[str, list[int]] = {}
        for row_idx, value in series.items():
            value_to_rows.setdefault(str(value), []).append(int(row_idx))
        samples = [
            {
                "row": rows[0],
                "value": compact_value(value),
                "raw_value": value,
            }
            for value, rows in value_to_rows.items()
            if rows
        ]
        fmt = format_rules.get(str(col), {}) if isinstance(format_rules, dict) else {}
        regex = str((fmt or {}).get("regex", "") or "")
        selected = select_representative_samples(
            samples,
            value_to_rows,
            "" if direct_only else regex,
            direct_only=direct_only,
        )
        representative_rows = {
            int(sample["row"]): value_to_rows.get(str(sample.get("raw_value", "")), [int(sample["row"])])
            for sample in selected
        }
        for sample in selected:
            sample.pop("raw_value", None)
        allowed_types = {"format_errors", "missing_errors"} if direct_only else {
            "format_errors",
            "outliers",
            "spelling_errors",
            "missing_errors",
        }
        idx = 0
        col_errors = []
        while idx < len(selected):
            batch = selected[idx:idx + DIRECT_BATCH_LIMIT]
            idx += len(batch)
            if direct_only:
                system_prompt = DIRECT_DETECTION_SYSTEM
                user_prompt = DIRECT_DETECTION_USER.format(
                    col=col,
                    samples_json=json.dumps(batch, ensure_ascii=False, indent=2),
                )
            else:
                system_prompt = NO_SUMMARY_DETECTION_SYSTEM
                user_prompt = NO_SUMMARY_DETECTION_USER.format(
                    col=col,
                    dtype=field_profiles.get(str(col), {}).get("dtype", ""),
                    regex=regex,
                    explanation=str((fmt or {}).get("explanation", "")),
                    samples_json=json.dumps(batch, ensure_ascii=False, indent=2),
                )
            try:
                response = shared_qwen_client.send_message(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=QWEN_MAX_TOKENS,
                    retries=REQUEST_RETRIES,
                    request_timeout=REQUEST_TIMEOUT,
                )
                parsed = extract_json_object(response)
                normalized = normalize_llm_errors(parsed.get("errors", []), str(col), allowed_types)
                col_errors.extend(expand_representative_errors(normalized, representative_rows))
            except Exception as exc:
                print(f"[Warn] no-summary detection batch failed for '{col}': {exc}")
        per_column_counts[str(col)] = sum(
            len(err["row"]) if isinstance(err.get("row"), list) else 1
            for err in col_errors
        )
        errors_all.extend(col_errors)
        print(f" -> found {per_column_counts[str(col)]} cells in '{col}'")
    return errors_all, per_column_counts


RELATIONSHIP_CONFIRM_SYSTEM = """
You confirm relationship-consistency candidates for a no-summary ablation.
Use only the raw repeated-key consistency evidence shown here. Do not use data
summary, semantic descriptions, or external knowledge. Return JSON only.
"""


RELATIONSHIP_CONFIRM_USER = """
The initial-screening logic found repeated-key consistency candidates directly
from the raw dirty table. Confirm only candidates that are clear contradictions
relative to the shown majority mapping evidence.

Candidates:
{candidates_json}

Return exactly:
{{
  "errors": [
    {{"row": <row index>, "fieldName": "<dependent column>"}}
  ]
}}
"""


def confirm_relationship_errors(rules: dict) -> list[dict]:
    relationship_rules = rules.get("relationship_rules", []) if rules else []
    candidates = []
    for rule in relationship_rules:
        for example in rule.get("examples", [])[:8]:
            candidates.append({
                "row": example.get("row"),
                "fieldName": rule.get("dependent_column"),
                "key_column": rule.get("key_column"),
                "dependent_column": rule.get("dependent_column"),
                "key_value": example.get("key_value"),
                "candidate_value": example.get("candidate_value"),
                "majority_value": example.get("majority_value"),
            })
    if not candidates:
        return []
    try:
        response = shared_qwen_client.send_message(
            [
                {"role": "system", "content": RELATIONSHIP_CONFIRM_SYSTEM},
                {
                    "role": "user",
                    "content": RELATIONSHIP_CONFIRM_USER.format(
                        candidates_json=json.dumps(candidates[:80], ensure_ascii=False, indent=2)
                    ),
                },
            ],
            max_tokens=QWEN_MAX_TOKENS,
            retries=REQUEST_RETRIES,
            request_timeout=REQUEST_TIMEOUT,
        )
        parsed = extract_json_object(response)
    except Exception as exc:
        print(f"[Warn] relationship confirmation failed: {exc}")
        return []
    confirmed = []
    for item in parsed.get("errors", []) if isinstance(parsed, dict) else []:
        try:
            rid = int(item.get("row"))
            field = str(item.get("fieldName"))
        except (TypeError, ValueError):
            continue
        confirmed.append({
            "row": rid,
            "fieldName": field,
            "errorType": "logical_errors",
            "description": "No-summary LLM-confirmed repeated-key inconsistency.",
            "correctFormat": None,
        })
    return confirmed


def flatten_errors(errors: list[dict]) -> list[dict]:
    flattened = []
    for error in errors:
        rows = error.get("row")
        row_list = rows if isinstance(rows, list) else [rows]
        for row in row_list:
            try:
                rid = int(row)
            except (TypeError, ValueError):
                continue
            flattened.append({
                "row": rid,
                "fieldName": str(error.get("fieldName")),
                "errorType": str(error.get("errorType")),
                "description": error.get("description"),
                "correctFormat": error.get("correctFormat"),
            })
    merged = {}
    for error in flattened:
        key = (error["row"], error["fieldName"])
        if key not in merged:
            merged[key] = dict(error)
        else:
            old_types = set(str(merged[key].get("errorType", "")).split(","))
            old_types.add(error["errorType"])
            merged[key]["errorType"] = ",".join(sorted(t for t in old_types if t))
            if not merged[key].get("description") and error.get("description"):
                merged[key]["description"] = error.get("description")
    return [merged[key] for key in sorted(merged)]


def write_detection_outputs(result: dict, output_dir: Path) -> None:
    dump_json(result.get("errors", []), output_dir / "detailed_errors.json")
    dump_json(result, output_dir / "errors_with_context.json")
    save_error_cells(detected_errors_from_result(result), output_dir / "detected_errors.json")


def run_no_summary_ablation(
    config_name: str,
    dataset: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    chunksize: int = 50_000,
) -> dict:
    if config_name not in {"no_summary", "no_summary_screening"}:
        raise ValueError(f"Unsupported independent no-summary config: {config_name}")
    configure_qwen()
    dataset_name, dirty_path, clean_path = resolve_dataset_paths(dataset, data_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = output_root.resolve()
    output_dir = output_root / config_name / dataset_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"LAED strict ablation: {config_name}")
    print(f"Dataset: {dataset_name}")
    print("Summary module: disabled; no data_summary.json will be generated or read.")
    print(f"Output directory: {output_dir}")

    reset_shared_usage()
    started = time.perf_counter()
    with working_directory(output_dir):
        df = load_csv(dirty_path, chunksize=chunksize)
        print(f"Loaded dirty data: {df.shape}")

        if config_name == "no_summary":
            initial_profile, rules = build_initial_screening_rules(df)
            profile = initial_profile
            dump_json(profile, output_dir / "initial_screening_profile.json")
            dump_json(rules, output_dir / "initial_screening_rules.json")
            dump_json(
                {
                    "config": config_name,
                    "summary_module_used": False,
                    "detection_uses": [
                        "initial_screening_rules.format_rules",
                        "initial_screening_rules.relationship_rules",
                        "shown suspicious values",
                    ],
                },
                output_dir / "detection_context_no_summary.json",
            )
            df_processed, correct_cells, pct, _relationship_candidates = apply_initial_screening(df, rules)
            direct_only = False
        else:
            profile = {
                "source": "direct_prompt_only_no_summary_no_screening",
                "summary_module_used": False,
                "initial_screening_used": False,
                "fields": [{"column": str(col)} for col in df.columns],
                "field_names": [str(col) for col in df.columns],
            }
            rules = {
                "source": "no_summary_no_screening_direct_llm",
                "summary_module_used": False,
                "format_rules": {},
                "relationship_rules": [],
                "relationship_validator_code": {
                    "validation": {
                        "code": "def validate_relationships(df):\n    return set()",
                        "status": "disabled",
                    }
                },
            }
            dump_json(
                {
                    "config": config_name,
                    "summary_module_used": False,
                    "initial_screening_used": False,
                    "detection_uses": ["column name", "shown values"],
                },
                output_dir / "detection_context_no_summary_screening.json",
            )
            df_processed = df.copy().astype(str)
            correct_cells = pd.DataFrame(False, index=df.index, columns=df.columns)
            pct = 0.0
            correct_cells.to_csv("correct_cells_mask.csv", index=False, encoding="utf-8-sig")
            df.to_csv("suspicious_cells_for_error_detection.csv", index=False, encoding="utf-8-sig")
            direct_only = True

        column_errors, per_column_counts = detect_columns(
            df=df.copy().astype(str),
            correct_cells=correct_cells,
            profile=profile,
            rules=rules,
            direct_only=direct_only,
        )
        relationship_errors = [] if direct_only else confirm_relationship_errors(rules)
        detailed_errors = flatten_errors(column_errors + relationship_errors)
        result = {
            "ablation": config_name,
            "summary_module_used": False,
            "initial_screening_used": not direct_only,
            "format_rules": rules.get("format_rules", {}),
            "relationship_rules": rules.get("relationship_rules", []),
            "errors": detailed_errors,
            "final_errors": [[error["row"], error["fieldName"]] for error in detailed_errors],
            "llm_confirmation_policy": {
                "final_error_requirement": "Every final error is returned or confirmed by the LLM.",
                "summary_forbidden": "No data_summary.json, semantic summary, or summary statistics are generated, read, or passed.",
            },
        }
        write_detection_outputs(result, output_dir)
        metrics = evaluate_detection(clean_path, dirty_path, result, output_dir)

    runtime_seconds = round(time.perf_counter() - started, 2)
    token_usage = get_shared_usage()
    error_types_count, field_errors = summarize_error_distribution(detailed_errors)
    correct_count = int(correct_cells.values.sum())
    report = {
        "run_id": run_id,
        "mode": "full",
        "ablation": {
            "name": config_name,
            "remove_summary": True,
            "remove_screening": direct_only,
            "summary_module_used": False,
            "data_summary_json_generated": False,
        },
        "dataset": dataset_name,
        "model": shared_qwen_client.model,
        "max_tokens": shared_qwen_client.max_tokens,
        "temperature": shared_qwen_client.temperature,
        "data": {
            "dirty_path": str(dirty_path),
            "clean_path_for_evaluation_only": str(clean_path),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
        },
        "initial_screening": {
            "correct_cells": correct_count,
            "total_cells": int(df_processed.size),
            "correct_pct": round(float(pct), 4),
            "suspicious_cells": int(df_processed.size - correct_count),
            "disabled": bool(direct_only),
        },
        "detection": {
            "detailed_error_count": len(detailed_errors),
            "final_error_count": len(result.get("final_errors", [])),
            "error_types_count": error_types_count,
            "field_errors": field_errors,
            "per_column_llm_error_cells": per_column_counts,
            "candidate_scope": "all_table_cells" if direct_only else "initial_screening_suspicious_cells",
            "llm_required": True,
            "direct_prompt_sampling": (
                "deterministic_row_order_spread_without_profile_or_rule_scoring"
                if direct_only
                else ""
            ),
        },
        "metrics": metrics,
        "token_usage": {
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "api_call_count": token_usage.get("api_call_count", 0),
        },
        "runtime_seconds": runtime_seconds,
        "output_dir": str(output_dir),
        "output_files": {
            "initial_screening_profile": "" if direct_only else str(output_dir / "initial_screening_profile.json"),
            "initial_screening_rules": str(output_dir / "initial_screening_rules.json") if not direct_only else "",
            "detection_context": str(
                output_dir / (
                    "detection_context_no_summary_screening.json"
                    if direct_only
                    else "detection_context_no_summary.json"
                )
            ),
            "detailed_errors": str(output_dir / "detailed_errors.json"),
            "errors_with_context": str(output_dir / "errors_with_context.json"),
            "ground_truth_errors": str(output_dir / "ground_truth_errors.json"),
            "detected_errors": str(output_dir / "detected_errors.json"),
            "evaluation_metrics": str(output_dir / "evaluation_metrics.json"),
        },
    }
    report_path = output_dir / f"ablation_metrics_tokens_{run_id}.json"
    latest_path = output_root / config_name / dataset_name / "latest_ablation_metrics_tokens.json"
    latest_pointer = output_root / config_name / dataset_name / "latest_run_path.txt"
    for path in (report_path, latest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(report, path)
    latest_pointer.write_text(str(output_dir), encoding="utf-8")

    print("===== Independent No-Summary Ablation Evaluation Results =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"API calls: {report['token_usage']['api_call_count']}")
    print(f"Saved report: {report_path}")
    print("=" * 80)
    return report


def parse_args(default_config: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent no-summary LAED ablations.")
    if default_config is None:
        parser.add_argument("--config", choices=("no_summary", "no_summary_screening"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--mode", choices=("auto", "full", "replay"), default="full")
    args = parser.parse_args()
    if default_config is not None:
        args.config = default_config
    return args


def main(default_config: Optional[str] = None) -> None:
    args = parse_args(default_config)
    if args.mode == "replay":
        raise ValueError("Independent no-summary ablations do not support replay mode.")
    report = run_no_summary_ablation(
        config_name=args.config,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        chunksize=args.chunksize,
    )
    print(
        f"Done. {report['ablation']['name']} metrics for {report['dataset']} are in "
        f"{report['output_files']['evaluation_metrics']}"
    )


if __name__ == "__main__":
    main()
