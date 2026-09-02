from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
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

from summarizer import Summarizer, _normalize_obj
from Initial_Screening_all import (
    _relationship_columns as screening_relationship_columns,
    _suppress_low_confidence_relationship_mask,
    filter_by_format,
    filter_by_relationships,
    initial_screening,
)
from Error_Detection_update import (
    DETECTION_REQUEST_RETRIES,
    DETECTION_REQUEST_TIMEOUT,
    MAX_RESPONSE_TOKENS,
    PROMPT_OVERHEAD,
    DetectionExplorer,
    count_cells,
    expand_representative_errors,
    is_missing_like_value,
    is_transient_request_error,
    parse_llm_json_response,
    text_noise_reasons,
    text_gen,
)
from LAED_Demo import (
    DEFAULT_DATA_DIR,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    compute_metrics,
    detected_errors_from_result,
    evaluate_detection,
    load_csv,
    load_ground_truth_errors,
    resolve_dataset_paths,
    save_error_cells,
    summarize_error_distribution,
)
from pythonProject1.API_invocation.qwen_gen import (
    get_shared_usage,
    reset_shared_usage,
    shared_qwen_client,
)


DEFAULT_ABLATION_OUTPUT_ROOT = PROJECT_DIR / "Run_Results_Ablation"
REFERENCE_RUNS = {
    "beers": PROJECT_DIR / "Run_Results" / "beers" / "20260511_221549_588710",
    "flights": PROJECT_DIR / "Run_Results" / "flights" / "20260511_224030_855652",
    "hospital": PROJECT_DIR / "Run_Results" / "hospital" / "20260512_020303_191421",
    "movies": PROJECT_DIR / "Run_Results" / "movies" / "20260512_151250_755846",
    "rayyan": PROJECT_DIR / "Run_Results" / "rayyan" / "20260512_084610_493183",
}

ErrorCell = Tuple[int, str]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    remove_summary: bool = False
    remove_rule: bool = False
    remove_screening: bool = False


ABLATION_CONFIGS = {
    "no_summary": AblationConfig("no_summary", remove_summary=True),
    "no_rule": AblationConfig("no_rule", remove_rule=True),
    "no_screening": AblationConfig("no_screening", remove_screening=True),
    "no_summary_screening": AblationConfig("no_summary_screening", remove_summary=True, remove_screening=True),
    "no_summary_rule": AblationConfig("no_summary_rule", remove_summary=True, remove_rule=True),
    "no_screening_rule": AblationConfig("no_screening_rule", remove_screening=True, remove_rule=True),
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


def copy_jsonable(value):
    return json.loads(json.dumps(_normalize_obj(value), ensure_ascii=False, default=str))


def dump_json(value, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(copy_jsonable(value), f, ensure_ascii=False, indent=2)


def normalize_summary_field_properties(summary: dict) -> dict:
    """
    Convert compact public data_summary fields into the runtime shape consumed
    by Initial_Screening_all.py and Error_Detection_update.py.

    LAED_Demo stores public summaries as:
      {"column": "A", "semantic_type": "...", "description": "...", "dtype": "..."}
    while runtime modules expect:
      {"column": "A", "properties": {"semantic_type": "...", ...}}
    This adapter is schema-only; it does not add rules or domain knowledge.
    """
    normalized = copy_jsonable(summary)
    fields = []
    for field in normalized.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        column = field.get("column")
        if not column:
            continue
        props = field.get("properties")
        if not isinstance(props, dict):
            props = {}
        props = dict(props)
        for key in ("dtype", "semantic_type", "description"):
            if key in field and key not in props:
                props[key] = field.get(key, "")
        fields.append({
            "column": column,
            "properties": props,
        })
    normalized["fields"] = fields
    normalized["field_names"] = [field["column"] for field in fields]
    return normalized


def _empty_relationship_validator(note: str) -> dict:
    return {
        "validation": {
            "code": "def validate_relationships(df):\n    return set()",
            "explanation": note,
            "generation_history": [],
            "status": "disabled",
        }
    }


def strip_screening_rules_from_summary(summary: dict, note: str) -> dict:
    """
    Keep the LLM semantic summary, but remove the executable screening rules
    so screening and detection cannot use summary-derived regexes or validators.
    """
    stripped = copy_jsonable(summary)
    stripped["format_rules"] = {}
    stripped["field_relationships"] = {
        "hierarchical": {},
        "mathematical": {},
        "temporal": [],
        "associative": {},
    }
    stripped["relationship_evidence"] = {}
    stripped["relationship_validator_code"] = _empty_relationship_validator(note)
    stripped.setdefault("ablation_flags", {})
    stripped["ablation_flags"]["screening_rules_removed"] = True
    stripped["ablation_note"] = note
    return stripped


def strip_detection_rule_context_from_summary(summary: dict, note: str) -> dict:
    """
    Keep the same non-semantic column profiles/candidate mask context, but do
    not expose screening rules to the LLM detection and rule-execution stage.
    """
    stripped = strip_screening_rules_from_summary(summary, note)
    stripped.setdefault("ablation_flags", {})
    stripped["ablation_flags"]["detection_rule_context_removed"] = True
    return stripped


PROFILE_METADATA_KEYS = {
    "top_value_counts",
    "rare_value_examples",
    "shape_counts",
    "shape_family_counts",
    "date_component_profile",
    "text_noise_examples",
    "numeric_text_skeleton_counts",
    "normalized_numeric_text_skeleton_counts",
    "normalized_numeric_text_skeleton_groups",
    "numeric_value_profiles",
    "dominant_shape_family",
    "dominant_numeric_text_skeleton",
    "dominant_normalized_numeric_text_skeleton",
    "case_punctuation_variant_groups",
    "missing_like_values",
    "num_unique_observed_values",
    "value_counts_truncated",
    "top_value_coverage",
    "value_counts",
}


def strip_profile_metadata_from_summary(summary: dict, note: str) -> dict:
    """
    Remove summary/profile-derived warning evidence for ablations whose
    detection input must be limited to semantic text or simple column metadata.
    """
    stripped = copy_jsonable(summary)
    for field in stripped.get("fields", []):
        if not isinstance(field, dict):
            continue
        props = field.get("properties")
        if not isinstance(props, dict):
            continue
        for key in PROFILE_METADATA_KEYS:
            props.pop(key, None)
    stripped.setdefault("ablation_flags", {})
    stripped["ablation_flags"]["profile_metadata_removed"] = True
    stripped["ablation_note"] = note
    return stripped


def build_simple_detection_summary(df: pd.DataFrame, file_name: str) -> dict:
    """
    Build the direct-LLM ablation summary. It contains no LLM-generated summary,
    no screening rules, no field relationships, and no profile-derived evidence.
    """
    summarizer = Summarizer()
    fields = summarizer.get_column_properties(df)
    for field in fields:
        props = field.get("properties", {})
        for key in PROFILE_METADATA_KEYS:
            props.pop(key, None)
        props["semantic_type"] = ""
        props["description"] = ""
    return {
        "name": file_name,
        "file_name": file_name,
        "dataset_description": "",
        "fields": fields,
        "field_relationships": {
            "hierarchical": {},
            "mathematical": {},
            "temporal": [],
            "associative": {},
        },
        "relationship_evidence": {},
        "format_rules": {},
        "relationship_validator_code": _empty_relationship_validator(
            "No Summary and no Screening ablation: no relationship validator is used."
        ),
        "field_names": df.columns.tolist(),
        "ablation_flags": {
            "no_summary": True,
            "direct_llm_only": True,
            "profile_metadata_removed": True,
            "screening_rules_removed": True,
        },
        "ablation_note": (
            "Direct LLM ablation: no data summary, no summary rules, and no "
            "profile-derived evidence are provided to detection."
        ),
    }


def empty_rules_from_columns(columns: Iterable[str]) -> dict:
    return {
        str(col): {
            "format": "unknown",
            "regex": "",
            "explanation": "No summary ablation: no semantic summary was used to derive this rule.",
        }
        for col in columns
    }


def infer_no_summary_relationships(summary: dict, df: pd.DataFrame) -> dict:
    """
    Keep no-summary screening strictly non-semantic. Relationship candidates are
    inferred from generic repeated-key evidence only, without LLM semantic
    enrichment or field descriptions.
    """
    evidence = Summarizer()._build_relationship_evidence(df, summary)
    associative = {}
    for candidate in evidence.get("candidate_associative_dependencies", []) or []:
        key = candidate.get("key")
        dependents = [
            item.get("field")
            for item in candidate.get("dependents", []) or []
            if isinstance(item, dict) and item.get("field") in df.columns
        ]
        if key in df.columns and dependents:
            associative[str(key)] = sorted({str(dep) for dep in dependents})
    return {
        "hierarchical": {},
        "mathematical": {},
        "temporal": [],
        "associative": associative,
    }


def build_no_summary_summary(df: pd.DataFrame, file_name: str, generate_screening_rules: bool = True) -> dict:
    """
    Build the no-Summary ablation input.

    This intentionally skips the LLM semantic enrichment produced by
    Summarizer.enrich(): field semantic_type/description stay empty. When
    generate_screening_rules is True, the initial-screening format regexes and
    relationship validator are still generated from column names, compact local
    profiles, samples, and relationship evidence, never from per-field semantic
    summary text.
    """
    summarizer = Summarizer()
    fields = summarizer.get_column_properties(df)
    summary = {
        "name": file_name,
        "file_name": file_name,
        "dataset_description": "",
        "fields": fields,
        "field_relationships": {
            "hierarchical": {},
            "mathematical": {},
            "temporal": [],
            "associative": {},
        },
        "format_rules": {},
        "relationship_validator_code": _empty_relationship_validator(
            "No Summary ablation: relationship validator not generated."
        ),
        "field_names": df.columns.tolist(),
        "ablation_flags": {"no_summary": True},
        "ablation_note": (
            "No LLM-generated semantic dataset summary is used. Field "
            "semantic_type and description are intentionally empty."
        ),
    }
    summary = summarizer._refresh_profile_metadata(summary, df)

    if generate_screening_rules:
        relationship_evidence = summarizer._build_relationship_evidence(df, summary)
        summary["relationship_evidence"] = relationship_evidence
        summary["field_relationships"] = infer_no_summary_relationships(summary, df)
        summary["format_rules"] = summarizer.extract_format_rules(df, summary)
        summary["relationship_validator_code"] = summarizer.generate_relationship_validator(df, summary)
        summary["ablation_flags"]["screening_rules_generated_without_summary_semantics"] = True
    else:
        summary["relationship_evidence"] = {}
        summary["ablation_flags"]["screening_rules_removed"] = True
    return summary


def build_no_summary_screening_summary(df: pd.DataFrame, file_name: str) -> dict:
    """
    No-summary screening keeps the initial-screening stage but does not allow
    any summary-derived semantic text to participate in rule generation.
    """
    summarizer = Summarizer()
    fields = summarizer.get_column_properties(df)
    summary = {
        "name": file_name,
        "file_name": file_name,
        "dataset_description": "",
        "fields": fields,
        "field_relationships": {
            "hierarchical": {},
            "mathematical": {},
            "temporal": [],
            "associative": {},
        },
        "relationship_evidence": {},
        "format_rules": empty_rules_from_columns(df.columns),
        "relationship_validator_code": _empty_relationship_validator(
            "No Summary ablation: no semantic relationship validator is used."
        ),
        "field_names": df.columns.tolist(),
        "ablation_flags": {
            "no_summary": True,
            "screening_rules_generated_without_summary_semantics": True,
            "llm_format_rule_generation_disabled": True,
        },
        "ablation_note": (
            "No LLM-generated semantic summary is used. Initial screening uses "
            "generic missing/shape/distribution evidence plus optional nonsemantic "
            "relationship evidence; detection may use only these nonsemantic rules."
        ),
    }
    summary = summarizer._refresh_profile_metadata(summary, df)
    summary["relationship_evidence"] = summarizer._build_relationship_evidence(df, summary)
    summary["field_relationships"] = infer_no_summary_relationships(summary, df)
    return summary


def load_reference_summary(dataset_name: str, fallback_run_dir: Optional[Path] = None) -> tuple[dict, Path]:
    candidates = []
    if fallback_run_dir is not None:
        candidates.append(fallback_run_dir / "data_summary.json")
    if dataset_name in REFERENCE_RUNS:
        candidates.append(REFERENCE_RUNS[dataset_name] / "data_summary.json")

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(
        f"No reusable LLM-generated data_summary.json found for dataset '{dataset_name}'."
    )


def make_no_screening_result(df: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df_processed = df.copy()
    fields = {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }
    for col, props in fields.items():
        if col in df_processed.columns and props.get("dtype") in ("number", "int", "float"):
            df_processed[col] = df_processed[col].astype(str)
    correct_cells = pd.DataFrame(False, index=df_processed.index, columns=df_processed.columns)
    correct_cells.to_csv("correct_cells_mask.csv", index=False)
    df_processed.to_csv("suspicious_cells_for_error_detection.csv", index=False)
    return df_processed, correct_cells, 0.0


def initial_screening_without_profile_evidence(
    df: pd.DataFrame,
    summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Run the no-Summary screening stage with only the generated format regexes
    and relationship validator. This avoids using summary/profile semantic
    warning evidence as a hidden extra screening signal.
    """
    df_processed = df.copy()
    number_cols = [
        f["column"]
        for f in summary.get("fields", [])
        if isinstance(f, dict) and f.get("properties", {}).get("dtype") == "number"
    ]
    for col in number_cols:
        if col in df_processed.columns:
            df_processed[col] = (
                df_processed[col]
                .astype(str)
                .map(lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x)
            )

    format_rules = summary.get("format_rules", {})
    relationships = summary.get("field_relationships", {})
    rel_code = (
        summary.get("relationship_validator_code", {})
        .get("validation", {})
        .get("code", "")
    )
    fmt_mask = filter_by_format(df_processed, format_rules)
    rel_err_mask = filter_by_relationships(df_processed, rel_code, relationships)
    rel_cols = screening_relationship_columns(relationships, df_processed)
    rel_err_mask = _suppress_low_confidence_relationship_mask(
        rel_err_mask,
        rel_cols,
        len(df_processed),
    )

    correct_cells = fmt_mask & (~rel_err_mask)
    correct_count = int(correct_cells.values.sum())
    total_cells = int(df_processed.size)
    pct = correct_count / total_cells * 100 if total_cells else 0.0
    correct_cells.to_csv("correct_cells_mask.csv", index=False)
    df.mask(correct_cells).to_csv("suspicious_cells_for_error_detection.csv", index=False)
    print(
        f"No-summary screening with generated rules only: "
        f"{correct_count}/{total_cells} cells marked correct ({pct:.2f}%)"
    )
    return df_processed, correct_cells, pct


def remove_detection_rules_from_result(result: dict) -> dict:
    """
    Keep LLM-diagnosed cells but remove the rule execution contribution.

    In Error_Detection_update.py, cells added purely by compiled LLM regexes or
    relationship-rule execution have no per-cell LLM explanation. Removing
    those entries models the "no Rule" ablation while preserving direct LLM
    diagnoses from detection batches.
    """
    filtered = []
    for error in result.get("errors", []):
        is_direct_llm_diagnosis = error.get("description") is not None or error.get("correctFormat") is not None
        if not is_direct_llm_diagnosis:
            continue
        filtered.append(error)
    final_errors = [[int(error["row"]), str(error["fieldName"])] for error in filtered]
    ablated = copy_jsonable(result)
    ablated["errors"] = filtered
    ablated["final_errors"] = final_errors
    ablated["compiled_error_rules"] = {
        col: {etype: "Correct" for etype in ("format_errors", "spelling_errors", "outliers", "missing_errors")}
        for col in ablated.get("compiled_error_rules", {})
    }
    return ablated


class StrictAblationDetectionExplorer(DetectionExplorer):
    """
    Detector used only by ablation scripts.

    It disables the profile-warning shortcuts and profile metadata that the
    full LAED detector uses as auxiliary evidence. This keeps ablation inputs
    faithful to the requested module removal instead of letting summary/profile
    side channels compensate for the missing module.
    """

    def column_profile_for_prompt(self, props: dict) -> str:
        return "{}"

    def make_sample_entry(self, row: int, value: str, props: dict, fmt: dict) -> dict:
        return {"row": row, "value": value}

    def confirm_profile_warning_errors_with_llm(self, col, props, fmt, samples, representative_rows):
        return {"errors": [], "error_regex_by_type_list": []}

    def expand_llm_confirmed_profile_warnings(
        self,
        col: str,
        df: pd.DataFrame,
        props: dict,
        fmt: dict,
        indices_to_check: list[int],
        errors: list,
    ) -> list:
        return errors

    def should_probe_free_text(self, props, fmt, distinct_count, suspicious_count) -> bool:
        return False

    def representative_sample_budget(self, props, fmt, distinct_count: int) -> int | None:
        return None

    def select_indices_for_llm(self, col, df, props, fmt, suspicious_idx):
        return list(suspicious_idx or [])

    def filter_summary_unsupported_outliers(self, col, df, props, errors):
        return errors

    def filter_low_evidence_identifier_errors(self, col, df, props, fmt, errors):
        return errors


DIRECT_LLM_SYSTEM_PROMPT = """
You are a data error detection assistant. Do not use any hidden data summary,
schema summary, generated rules, field relationships, or profile metadata.
Use only the column name, dtype, and the values shown in the prompt.
Return JSON only.
"""


DIRECT_LLM_USER_PROMPT = """
Detect erroneous cells in one table column using only the information shown
below. The values may contain correct and incorrect cells.

Allowed error types:
- missing_errors: nulls or obvious missing placeholders such as empty, nan,
  null, n/a, none.
- format_errors: values that visibly break the common surface pattern in this
  batch.

Do not infer domain-specific rules. Do not use external knowledge. Because no
summary or screening evidence is available, label only visually obvious
missingness or surface-format corruption. If evidence is weak, leave the value
unflagged.

Column: "{col}"
Data type: {dtype}
Values as row index to value:
{samples_json}

Return exactly:
{{
  "errors": [
    {{
      "row": <row index or list of row indices>,
      "errorType": "<format_errors|missing_errors>"
    }}
  ]
}}

Do not include explanations, corrected values, markdown, or any extra keys.
"""


NO_SUMMARY_RULE_DIRECT_SYSTEM_PROMPT = """
You are a data error detection assistant for a no-summary ablation. Do not use
semantic summaries, profile metadata, external knowledge, or hidden schema
descriptions. Use only the generated format rule, dtype, column name, and shown
values. Return JSON only.
"""


NO_SUMMARY_RULE_DIRECT_USER_PROMPT = """
Detect erroneous cells in one table column using only the generated nonsemantic
format rule and the shown values.

Allowed evidence:
- Column name and dtype.
- Generated format regex and explanation.
- Visible values in this prompt.

Forbidden evidence:
- No semantic summary or field description.
- No profile metadata or hidden statistics.
- No external domain knowledge.

Allowed error types:
- missing_errors: actual nulls or obvious missing placeholders.
- format_errors: values that visibly violate the generated regex or the dominant
  surface pattern shown here.
- outliers: only numeric extremes that are clearly implausible from the shown values.
- spelling_errors: only minor character-level typos relative to a strongly
  repeated value in the shown values.

If the regex is empty or evidence is weak, leave the value unflagged.

Column: "{col}"
Data type: {dtype}
Generated regex: `{regex}`
Rule explanation: {fmt_expl}
Values as row index to value:
{samples_json}

Return exactly:
{{
  "errors": [
    {{
      "row": <row index or list of row indices>,
      "errorType": "<format_errors|outliers|spelling_errors|missing_errors>"
    }}
  ]
}}

Do not include explanations, corrected values, markdown, or any extra keys.
"""


SEMANTIC_DIRECT_LLM_SYSTEM_PROMPT = """
You are a data error detection assistant for a no-initial-screening ablation.
Use only the field semantic text and the values shown in the prompt. Do not use
generated format rules, field relationship rules, validator code, profile
metadata, or external knowledge. Return JSON only.
"""


SEMANTIC_DIRECT_LLM_USER_PROMPT = """
Detect erroneous cells in one table column using only the field semantics and
values shown below.

Allowed evidence:
- The column name, dtype, semantic_type, and description.
- The visible values in this prompt.

Forbidden evidence:
- No generated regex/format rules.
- No field relationship rules or validator functions.
- No profile metadata, hidden summaries, or external domain knowledge.

Allowed error types:
- missing_errors: actual nulls or obvious semantic missing placeholders.
- format_errors: visibly corrupted values, impossible surface forms, or values
  that plainly contradict the field meaning.
- outliers: only numeric extremes that are clearly impossible from the shown
  values and field meaning.
- spelling_errors: only minor character-level typos relative to a strongly
  repeated canonical value in the shown values.

If evidence is weak, leave the value unflagged.

Column: "{col}"
Data type: {dtype}
Semantic type: {semantic_type}
Description: {description}
Values as row index to value:
{samples_json}

Return exactly:
{{
  "errors": [
    {{
      "row": <row index or list of row indices>,
      "errorType": "<format_errors|outliers|spelling_errors|missing_errors>"
    }}
  ]
}}

Do not include explanations, corrected values, markdown, or any extra keys.
"""


class DirectLLMOnlyDetectionExplorer(StrictAblationDetectionExplorer):
    """
    Direct LLM baseline for the no-summary-and-no-screening ablation. It does
    not ask the LLM to generate reusable regex rules and therefore does not use
    the stage-three rule-expansion path.
    """

    DIRECT_VALUE_MAX_CHARS = int(os.getenv("LAED_DIRECT_VALUE_MAX_CHARS", "160"))
    DIRECT_BATCH_LIMIT = int(os.getenv("LAED_DIRECT_BATCH_LIMIT", "30"))
    DIRECT_PARSE_FAILURE_BUDGET = int(os.getenv("LAED_DIRECT_PARSE_FAILURE_BUDGET", "8"))
    DIRECT_MAX_VALUES_PER_COLUMN = int(os.getenv("LAED_DIRECT_MAX_VALUES_PER_COLUMN", "120"))

    def compact_direct_value(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if len(text) <= self.DIRECT_VALUE_MAX_CHARS:
            return text
        return text[: self.DIRECT_VALUE_MAX_CHARS] + "...[truncated]"

    def allowed_direct_error_types(self) -> set[str]:
        return {"format_errors", "missing_errors"}

    def normalize_direct_errors(self, errs: list, col: str) -> list:
        normalized = []
        allowed = self.allowed_direct_error_types()
        for err in errs:
            if not isinstance(err, dict):
                continue
            error_type = str(err.get("errorType", "")).strip()
            if error_type not in allowed:
                continue
            row_spec = err.get("row", err.get("rows"))
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
                "errorType": error_type,
                "description": None,
                "correctFormat": None,
            })
        return normalized

    def direct_value_score(self, value: str, rows: list[int], shape_counts: dict[str, int], total_rows: int) -> float:
        text = str(value)
        stripped = text.strip()
        score = 0.0
        if is_missing_like_value(stripped):
            score += 100.0
        noise = text_noise_reasons(stripped)
        if noise:
            score += 80.0 + len(noise)
        if stripped != text:
            score += 10.0
        shape = Summarizer._value_shape(stripped)
        shape_count = int(shape_counts.get(shape, 0))
        if stripped and shape_count <= max(2, int(total_rows * 0.003)):
            score += 25.0
        if stripped and len(stripped) >= 120:
            score += 8.0
        if stripped and len(stripped) <= 1 and not is_missing_like_value(stripped):
            score += 6.0
        score += min(len(rows), 10) * 0.01
        return score

    def select_direct_samples(
            self,
            samples: list[dict],
            value_to_rows: dict[str, list[int]],
            total_rows: int,
    ) -> list[dict]:
        del value_to_rows, total_rows
        budget = max(1, self.DIRECT_MAX_VALUES_PER_COLUMN)
        if len(samples) <= budget:
            return samples
        ordered = sorted(samples, key=lambda item: int(item.get("row", 0)))
        if budget == 1:
            selected = [ordered[0]]
        else:
            step = (len(ordered) - 1) / (budget - 1)
            selected = []
            seen_rows = set()
            for pos in range(budget):
                idx = int(round(pos * step))
                sample = ordered[min(idx, len(ordered) - 1)]
                row = int(sample.get("row", 0))
                if row in seen_rows:
                    continue
                selected.append(sample)
                seen_rows.add(row)
            for sample in ordered:
                if len(selected) >= budget:
                    break
                row = int(sample.get("row", 0))
                if row in seen_rows:
                    continue
                selected.append(sample)
                seen_rows.add(row)
        print(f"Direct LLM deterministic row-order spread kept {len(selected)}/{len(samples)} distinct values")
        return selected

    def detect_column_errors(self, col, df, props, fmt, suspicious_idx):
        series = df.loc[suspicious_idx, col].astype(str).copy()
        print(f"Direct LLM cells selected for '{col}': {len(series)}")
        if props.get("dtype") == "number":
            series = series.map(lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x)
        if len(series) == 0:
            return {"errors": [], "error_regex_by_type_list": []}

        value_to_rows = {}
        for row_idx, value in series.items():
            value_to_rows.setdefault(str(value), []).append(int(row_idx))
        representative_rows = {
            rows[0]: rows
            for rows in value_to_rows.values()
            if rows
        }
        effective_fmt = self.effective_format_rule_for_detection(props, fmt)
        samples = [
            {
                "row": rows[0],
                "value": self.compact_direct_value(value),
                "raw_value": value,
                "regex": effective_fmt.get("regex", ""),
            }
            for value, rows in value_to_rows.items()
            if rows
        ]
        samples = self.select_direct_samples(samples, value_to_rows, len(series))
        for sample in samples:
            sample.pop("raw_value", None)
            sample.pop("regex", None)

        errors_all = []
        idx = 0
        inherited_limit = self.initial_batch_limit_for_field(col, props, samples)
        batch_limit = min(
            self.DIRECT_BATCH_LIMIT,
            inherited_limit if inherited_limit is not None else self.DIRECT_BATCH_LIMIT,
        )
        parse_failures = 0
        while idx < len(samples):
            batch = []
            tokens_used = PROMPT_OVERHEAD
            count = 0
            while idx < len(samples):
                if batch_limit is not None and count >= batch_limit:
                    break
                entry = samples[idx]
                est = self.estimate_tokens(json.dumps(entry, ensure_ascii=False))
                if tokens_used + est > MAX_RESPONSE_TOKENS:
                    break
                batch.append(entry)
                tokens_used += est
                idx += 1
                count += 1
            if not batch:
                batch.append(samples[idx])
                idx += 1

            prompt = self.direct_prompt(col, props, fmt, batch)
            resp = None
            try:
                resp = text_gen.send_message(
                    [
                        {"role": "system", "content": self.direct_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_RESPONSE_TOKENS,
                    retries=DETECTION_REQUEST_RETRIES,
                    request_timeout=DETECTION_REQUEST_TIMEOUT,
                )
                parsed = parse_llm_json_response(resp)
                errs = self.normalize_direct_errors(parsed.get("errors", []), col)
                errors_all.extend(expand_representative_errors(errs, representative_rows))
            except Exception as exc:
                transient = resp is None and is_transient_request_error(exc)
                if resp is not None:
                    parse_failures += 1
                    print(f"[Warn] direct LLM parse failure sample for '{col}': {repr(clean_code_snippet(resp)[:500])}")
                    if parse_failures >= self.DIRECT_PARSE_FAILURE_BUDGET:
                        print(
                            f"[Warn] direct parse failure budget reached for '{col}' "
                            f"({parse_failures}); stopping this column"
                        )
                        break
                if batch_limit is None or batch_limit > 1:
                    reason = "request fail" if resp is None else "parse fail"
                    if transient:
                        print(f"[Warn] transient {reason} for '{col}' ({exc}), retrying same batch")
                    else:
                        batch_limit = max(1, len(batch) // 2)
                        print(f"[Warn] {reason} for '{col}' ({exc}), new batch_limit={batch_limit}")
                    idx -= len(batch)
                else:
                    if transient:
                        print(f"[Warn] transient request fail for '{col}' ({exc}), retrying minimal batch")
                        idx -= len(batch)
                    else:
                        print(f"[Error] skip direct minimal batch for '{col}': {repr(exc)}")
                continue

        return {"errors": errors_all, "error_regex_by_type_list": []}

    def direct_system_prompt(self) -> str:
        return DIRECT_LLM_SYSTEM_PROMPT

    def direct_prompt(self, col: str, props: dict, fmt: dict, batch: list[dict]) -> str:
        return DIRECT_LLM_USER_PROMPT.format(
            col=col,
            dtype=props.get("dtype", ""),
            samples_json=json.dumps(batch, ensure_ascii=False, indent=2),
        )


class SemanticOnlyDirectLLMDetectionExplorer(DirectLLMOnlyDetectionExplorer):
    """
    No-screening detector. It sends the full-table candidate space to the
    LLM direct detection stage with semantic field text only. Summary-generated
    regexes, relationship rules, validators, and profile metadata are removed
    before this detector is constructed.
    """

    SEMANTIC_BASE_VALUES_PER_COLUMN = int(os.getenv("LAED_SEMANTIC_DIRECT_BASE_VALUES_PER_COLUMN", "160"))
    SEMANTIC_HIGH_CARD_VALUES_PER_COLUMN = int(os.getenv("LAED_SEMANTIC_DIRECT_HIGH_CARD_VALUES_PER_COLUMN", "500"))
    SEMANTIC_HIGH_CARD_THRESHOLD = int(os.getenv("LAED_SEMANTIC_DIRECT_HIGH_CARD_THRESHOLD", "500"))

    def allowed_direct_error_types(self) -> set[str]:
        return {"format_errors", "outliers", "spelling_errors", "missing_errors"}

    def direct_system_prompt(self) -> str:
        return SEMANTIC_DIRECT_LLM_SYSTEM_PROMPT

    def direct_prompt(self, col: str, props: dict, fmt: dict, batch: list[dict]) -> str:
        del fmt
        return SEMANTIC_DIRECT_LLM_USER_PROMPT.format(
            col=col,
            dtype=props.get("dtype", ""),
            semantic_type=props.get("semantic_type", ""),
            description=props.get("description", ""),
            samples_json=json.dumps(batch, ensure_ascii=False, indent=2),
        )

    def select_direct_samples(
            self,
            samples: list[dict],
            value_to_rows: dict[str, list[int]],
            total_rows: int,
    ) -> list[dict]:
        del value_to_rows, total_rows
        budget = self.SEMANTIC_BASE_VALUES_PER_COLUMN
        if len(samples) >= self.SEMANTIC_HIGH_CARD_THRESHOLD:
            budget = self.SEMANTIC_HIGH_CARD_VALUES_PER_COLUMN
        budget = max(1, int(budget))
        if len(samples) <= budget:
            return samples
        ordered = sorted(samples, key=lambda item: int(item.get("row", 0)))
        if budget == 1:
            selected = [ordered[0]]
        else:
            step = (len(ordered) - 1) / (budget - 1)
            selected = []
            seen_rows = set()
            for pos in range(budget):
                idx = int(round(pos * step))
                sample = ordered[min(idx, len(ordered) - 1)]
                row = int(sample.get("row", 0))
                if row in seen_rows:
                    continue
                selected.append(sample)
                seen_rows.add(row)
            for sample in ordered:
                if len(selected) >= budget:
                    break
                row = int(sample.get("row", 0))
                if row in seen_rows:
                    continue
                selected.append(sample)
                seen_rows.add(row)
        print(f"Semantic full-table deterministic spread kept {len(selected)}/{len(samples)} distinct values")
        return selected


class NoSummaryDetectionExplorer(DirectLLMOnlyDetectionExplorer):
    """No-summary detector: uses generated rules, but no semantic/profile text."""

    def direct_system_prompt(self) -> str:
        return NO_SUMMARY_RULE_DIRECT_SYSTEM_PROMPT

    def allowed_direct_error_types(self) -> set[str]:
        return {"format_errors", "outliers", "spelling_errors", "missing_errors"}

    def select_direct_samples(
            self,
            samples: list[dict],
            value_to_rows: dict[str, list[int]],
            total_rows: int,
    ) -> list[dict]:
        del total_rows
        budget = max(1, self.DIRECT_MAX_VALUES_PER_COLUMN)
        regex = ""
        if samples:
            regex = str(samples[0].get("regex", "") or "")
        if not regex:
            return super().select_direct_samples(samples, value_to_rows, len(value_to_rows))
        selected = []
        for sample in samples:
            raw_value = str(sample.get("raw_value", sample.get("value", ""))).strip()
            try:
                mismatch = re.fullmatch(regex, raw_value) is None
            except re.error:
                mismatch = False
            if mismatch or is_missing_like_value(raw_value) or text_noise_reasons(raw_value):
                selected.append(sample)
        if len(selected) > budget:
            selected = selected[:budget]
        if not selected:
            return []
        print(f"No-summary generated-rule LLM budget kept {len(selected)}/{len(samples)} distinct values")
        return selected

    def direct_prompt(self, col: str, props: dict, fmt: dict, batch: list[dict]) -> str:
        effective_fmt = self.effective_format_rule_for_detection(props, fmt)
        return NO_SUMMARY_RULE_DIRECT_USER_PROMPT.format(
            col=col,
            dtype=props.get("dtype", ""),
            regex=effective_fmt.get("regex", ""),
            fmt_expl=effective_fmt.get("explanation", ""),
            samples_json=json.dumps(batch, ensure_ascii=False, indent=2),
        )


def write_detection_outputs(result: dict, output_dir: Path) -> None:
    dump_json(result.get("errors", []), output_dir / "detailed_errors.json")
    dump_json(result, output_dir / "errors_with_context.json")
    detected = detected_errors_from_result(result)
    save_error_cells(detected, output_dir / "detected_errors.json")


def reference_run_dir_for(dataset_name: str, explicit_run_dir: Optional[Path] = None) -> Path:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir.resolve()
    else:
        try:
            run_dir = REFERENCE_RUNS[dataset_name].resolve()
        except KeyError as exc:
            raise FileNotFoundError(f"No reference run configured for dataset '{dataset_name}'.") from exc
    if not run_dir.exists():
        raise FileNotFoundError(f"Reference run directory not found: {run_dir}")
    return run_dir


def load_reference_result(run_dir: Path) -> dict:
    context_path = run_dir / "errors_with_context.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Reference errors_with_context.json not found: {context_path}")
    with context_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_report(run_dir: Path) -> dict:
    latest_path = run_dir / "latest_metrics_tokens.json"
    if latest_path.exists():
        with latest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def apply_compiled_rules_to_all_cells(df: pd.DataFrame, compiled_error_rules: dict) -> Set[ErrorCell]:
    rule_cells: Set[ErrorCell] = set()
    for col, rules in (compiled_error_rules or {}).items():
        if col not in df.columns or not isinstance(rules, dict):
            continue
        values = df[col].astype(str)
        for pattern in rules.values():
            if not pattern or pattern == "Correct":
                continue
            try:
                regex = re.compile(str(pattern))
            except re.error:
                continue
            for row_idx, value in values.items():
                if regex.match(str(value)):
                    rule_cells.add((int(row_idx), str(col)))
    return rule_cells


def result_from_cells(template: dict, cells: Iterable[ErrorCell], note: str) -> dict:
    errors = [
        {
            "row": int(row),
            "fieldName": str(col),
            "errorType": "ablation_replay",
            "description": note,
            "correctFormat": None,
        }
        for row, col in sorted(set(cells))
    ]
    result = copy_jsonable(template)
    result["errors"] = errors
    result["final_errors"] = [[error["row"], error["fieldName"]] for error in errors]
    return result


def replay_ablation_result(config: AblationConfig, df: pd.DataFrame, reference_result: dict) -> tuple[dict, dict]:
    detailed = reference_result.get("errors", [])
    explicit_cells: Set[ErrorCell] = {
        (int(error["row"]), str(error["fieldName"]))
        for error in detailed
        if error.get("description") is not None or error.get("correctFormat") is not None
    }
    full_cells = detected_errors_from_result(reference_result)

    if config.remove_rule:
        cells = set(explicit_cells)
    else:
        cells = set(full_cells)

    replay_notes = []
    if config.remove_rule:
        replay_notes.append("removed compiled LLM rule expansion and relationship-rule execution")
    if config.remove_screening:
        replay_notes.append("applied compiled LLM rules beyond the initial-screening candidate mask")
        if not config.remove_rule:
            cells |= apply_compiled_rules_to_all_cells(df, reference_result.get("compiled_error_rules", {}))

    if not replay_notes:
        replay_notes.append("replayed reference result")

    result = result_from_cells(reference_result, cells, "; ".join(replay_notes))
    if config.remove_rule:
        result["compiled_error_rules"] = {
            col: {etype: "Correct" for etype in ("format_errors", "spelling_errors", "outliers", "missing_errors")}
            for col in result.get("compiled_error_rules", {})
        }
    result.setdefault("ablation_replay", {})
    result["ablation_replay"].update(
        {
            "source": "reference_run_outputs",
            "notes": replay_notes,
            "explicit_llm_cells": len(explicit_cells),
            "reference_final_cells": len(full_cells),
            "replayed_final_cells": len(cells),
        }
    )

    total_cells = int(df.size)
    if config.remove_screening:
        screening_info = {
            "correct_cells": 0,
            "total_cells": total_cells,
            "correct_pct": 0.0,
            "suspicious_cells": total_cells,
            "disabled": True,
        }
    else:
        screening_info = {
            "correct_cells": None,
            "total_cells": total_cells,
            "correct_pct": None,
            "suspicious_cells": None,
            "disabled": False,
        }
    return result, screening_info


def run_replay_ablation(
    config: AblationConfig,
    dataset_name: str,
    dirty_path: Path,
    clean_path: Path,
    output_root: Path,
    chunksize: int,
    reference_run_dir: Optional[Path],
) -> dict:
    run_dir = reference_run_dir_for(dataset_name, reference_run_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root.resolve() / config.name / dataset_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    df = load_csv(dirty_path, chunksize=chunksize)
    raw_context_path = run_dir / "raw_errors_with_context_before_ablation.json"
    if config.remove_summary and raw_context_path.exists():
        with raw_context_path.open("r", encoding="utf-8") as f:
            reference_result = json.load(f)
    else:
        reference_result = load_reference_result(run_dir)
    reference_report = load_reference_report(run_dir)
    if config.remove_summary and (run_dir / "data_summary.json").exists():
        summary_path = run_dir / "data_summary.json"
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary, summary_path = load_reference_summary(dataset_name, run_dir)

    dump_json(summary, output_dir / "data_summary.json")

    if config.remove_screening:
        pd.DataFrame(False, index=df.index, columns=df.columns).to_csv(output_dir / "correct_cells_mask.csv", index=False)
        df.to_csv(output_dir / "suspicious_cells_for_error_detection.csv", index=False)
    else:
        for filename in ("correct_cells_mask.csv", "suspicious_cells_for_error_detection.csv"):
            src = run_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

    result, screening_info = replay_ablation_result(config, df, reference_result)
    write_detection_outputs(result, output_dir)

    ground_truth = load_ground_truth_errors(clean_path, dirty_path)
    save_error_cells(ground_truth, output_dir / "ground_truth_errors.json")
    metrics = compute_metrics(ground_truth, detected_errors_from_result(result))
    dump_json(metrics, output_dir / "evaluation_metrics.json")

    if not config.remove_screening and reference_report.get("initial_screening"):
        screening_info.update(reference_report["initial_screening"])
        screening_info["disabled"] = False

    detailed_errors = result.get("errors", [])
    error_types_count, field_errors = summarize_error_distribution(detailed_errors)
    runtime_seconds = round(time.perf_counter() - started, 2)
    report = {
        "run_id": run_id,
        "mode": "replay",
        "ablation": {
            "name": config.name,
            "remove_summary": config.remove_summary,
            "remove_rule": config.remove_rule,
            "remove_screening": config.remove_screening,
            "summary_source": str(summary_path),
            "reference_run_dir": str(run_dir),
            "note": (
                "This replay mode reuses the completed LAED run and recomputes "
                "the requested module removal without new LLM calls."
            ),
        },
        "dataset": dataset_name,
        "model": reference_report.get("model", shared_qwen_client.model),
        "data": {
            "dirty_path": str(dirty_path),
            "clean_path_for_evaluation_only": str(clean_path),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
        },
        "initial_screening": screening_info,
        "detection": {
            "detailed_error_count": len(detailed_errors),
            "final_error_count": len(result.get("final_errors", [])),
            "error_types_count": error_types_count,
            "field_errors": field_errors,
        },
        "metrics": metrics,
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "api_call_count": 0,
            "replayed_reference_total_tokens": (reference_report.get("token_usage") or {}).get("total_tokens"),
        },
        "runtime_seconds": runtime_seconds,
        "output_dir": str(output_dir),
        "output_files": {
            "summary": str(output_dir / "data_summary.json"),
            "detailed_errors": str(output_dir / "detailed_errors.json"),
            "errors_with_context": str(output_dir / "errors_with_context.json"),
            "ground_truth_errors": str(output_dir / "ground_truth_errors.json"),
            "detected_errors": str(output_dir / "detected_errors.json"),
            "evaluation_metrics": str(output_dir / "evaluation_metrics.json"),
        },
    }

    report_path = output_dir / f"ablation_metrics_tokens_{run_id}.json"
    latest_path = output_root.resolve() / config.name / dataset_name / "latest_ablation_metrics_tokens.json"
    latest_pointer = output_root.resolve() / config.name / dataset_name / "latest_run_path.txt"
    for path in (report_path, latest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(report, path)
    latest_pointer.write_text(str(output_dir), encoding="utf-8")

    print("===== Replay Ablation Evaluation Results =====")
    print(f"Config:    {config.name}")
    print(f"Dataset:   {dataset_name}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"Saved report: {report_path}")
    return report


def run_detector(
    df: pd.DataFrame,
    summary: dict,
    output_dir: Path,
    remove_rule: bool,
    remove_screening: bool,
    detection_summary: Optional[dict] = None,
    screening_mode: str = "standard",
) -> tuple[dict, dict]:
    if remove_screening:
        screening_result = make_no_screening_result(df, summary)
    elif screening_mode == "rules_only":
        screening_result = initial_screening_without_profile_evidence(df.copy(), summary)
    else:
        screening_result = initial_screening(df.copy(), summary)

    df_processed, correct_cells, pct = screening_result
    detection_input_summary = detection_summary or summary
    if detection_summary is not None:
        dump_json(detection_summary, output_dir / "data_summary_for_detection.json")
    ablation_flags = detection_input_summary.get("ablation_flags", {}) if isinstance(detection_input_summary, dict) else {}
    if ablation_flags.get("direct_llm_only"):
        detector_cls = DirectLLMOnlyDetectionExplorer
    elif ablation_flags.get("semantic_only_direct_llm"):
        detector_cls = SemanticOnlyDirectLLMDetectionExplorer
    elif ablation_flags.get("no_summary"):
        detector_cls = NoSummaryDetectionExplorer
    elif ablation_flags.get("screening_rules_removed"):
        detector_cls = StrictAblationDetectionExplorer
    else:
        detector_cls = DetectionExplorer
    detector = detector_cls(experience_file=str(output_dir / "ErrorDetection_Experience_file.json"))
    result = detector.generate(
        detection_input_summary,
        df.copy(),
        screening_result=screening_result,
    )
    dump_json(result, output_dir / "raw_errors_with_context_before_ablation.json")
    dump_json(result.get("errors", []), output_dir / "raw_detailed_errors_before_ablation.json")
    if remove_rule:
        result = remove_detection_rules_from_result(result)
        write_detection_outputs(result, output_dir)

    correct_count = int(correct_cells.values.sum())
    screening_info = {
        "correct_cells": correct_count,
        "total_cells": int(df_processed.size),
        "correct_pct": round(float(pct), 4),
        "suspicious_cells": int(df_processed.size - correct_count),
        "disabled": bool(remove_screening),
    }
    return result, screening_info


def run_ablation(
    config: AblationConfig,
    dataset: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_root: Path = DEFAULT_ABLATION_OUTPUT_ROOT,
    chunksize: int = 50_000,
    reference_run_dir: Optional[Path] = None,
    mode: str = "auto",
) -> dict:
    configure_qwen()
    dataset_name, dirty_path, clean_path = resolve_dataset_paths(dataset, data_dir)
    if mode not in {"auto", "replay", "full"}:
        raise ValueError(f"Unsupported ablation mode: {mode}")
    if mode == "replay" or (mode == "auto" and not config.remove_summary and not config.remove_screening):
        return run_replay_ablation(
            config=config,
            dataset_name=dataset_name,
            dirty_path=dirty_path,
            clean_path=clean_path,
            output_root=output_root,
            chunksize=chunksize,
            reference_run_dir=reference_run_dir,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = output_root.resolve()
    output_dir = output_root / config.name / dataset_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if reference_run_dir is not None:
        reference_run_dir = reference_run_dir.resolve()

    print("=" * 80)
    print(f"LAED ablation: {config.name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dirty data: {dirty_path}")
    print(f"Output directory: {output_dir}")

    reset_shared_usage()
    started = time.perf_counter()

    with working_directory(output_dir):
        df = load_csv(dirty_path, chunksize=chunksize)
        detection_summary = None
        screening_mode = "standard"
        if config.remove_summary:
            if config.remove_screening:
                summary = build_simple_detection_summary(df, dirty_path.name)
                summary_source = "direct_llm_without_summary_or_screening"
                detection_summary = summary
            else:
                summary = build_no_summary_summary(
                    df,
                    dirty_path.name,
                    generate_screening_rules=True,
                )
                summary_source = "generic_profile_only_with_screening_rules_no_semantics"
                screening_mode = "rules_only"
                detection_summary = strip_profile_metadata_from_summary(
                    summary,
                    "No Summary ablation: detection may use format rules and relationship validators generated "
                    "without LLM semantic summary, but receives no semantic summary or profile-derived evidence.",
                )
                detection_summary.setdefault("ablation_flags", {})
                detection_summary["ablation_flags"]["no_summary"] = True
        else:
            if config.remove_screening:
                # The no-screening ablation keeps Summary as an active module.
                # Generate it freshly for each run, then pass only semantic text
                # to detection so format rules, field-logic rules, validator
                # code, and profile evidence cannot leak through.
                summarizer = Summarizer()
                runtime_summary = summarizer.summarize(df, file_name=dirty_path.name)
                public_summary = summarizer.compact_for_data_summary(runtime_summary)
                summary = normalize_summary_field_properties(public_summary)
                summary_source = "fresh_llm_generated_summary_per_ablation_run"
                detection_summary = strip_profile_metadata_from_summary(
                    summary,
                    "No Screening ablation: detection receives only semantic field summary text; "
                    "format rules, relationship rules, validators, and profile-derived evidence are removed.",
                )
                detection_summary = strip_screening_rules_from_summary(
                    detection_summary,
                    "No Screening ablation: semantic summary is retained, but format rules and field-relationship rules are removed.",
                )
                detection_summary.setdefault("ablation_flags", {})
                detection_summary["ablation_flags"]["semantic_only_direct_llm"] = True
            else:
                summary, summary_path = load_reference_summary(dataset_name, reference_run_dir)
                summary_source = str(summary_path)

        dump_json(summary, output_dir / "data_summary.json")

        result, screening_info = run_detector(
            df=df,
            summary=summary,
            output_dir=output_dir,
            remove_rule=config.remove_rule,
            remove_screening=config.remove_screening,
            detection_summary=detection_summary,
            screening_mode=screening_mode,
        )
        metrics = evaluate_detection(clean_path, dirty_path, result, output_dir)

    runtime_seconds = round(time.perf_counter() - started, 2)
    token_usage = get_shared_usage()
    detailed_errors = result.get("errors", [])
    error_types_count, field_errors = summarize_error_distribution(detailed_errors)

    report = {
        "run_id": run_id,
        "mode": "full",
        "ablation": {
            "name": config.name,
            "remove_summary": config.remove_summary,
            "remove_rule": config.remove_rule,
            "remove_screening": config.remove_screening,
            "summary_source": summary_source,
        },
        "dataset": dataset_name,
        "model": shared_qwen_client.model,
        "max_tokens": shared_qwen_client.max_tokens,
        "data": {
            "dirty_path": str(dirty_path),
            "clean_path_for_evaluation_only": str(clean_path),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
        },
        "initial_screening": screening_info,
        "detection": {
            "detailed_error_count": len(detailed_errors),
            "final_error_count": len(result.get("final_errors", [])),
            "error_types_count": error_types_count,
            "field_errors": field_errors,
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
            "summary": str(output_dir / "data_summary.json"),
            "detailed_errors": str(output_dir / "detailed_errors.json"),
            "errors_with_context": str(output_dir / "errors_with_context.json"),
            "ground_truth_errors": str(output_dir / "ground_truth_errors.json"),
            "detected_errors": str(output_dir / "detected_errors.json"),
            "evaluation_metrics": str(output_dir / "evaluation_metrics.json"),
        },
    }

    report_path = output_dir / f"ablation_metrics_tokens_{run_id}.json"
    latest_path = output_root / config.name / dataset_name / "latest_ablation_metrics_tokens.json"
    latest_pointer = output_root / config.name / dataset_name / "latest_run_path.txt"
    for path in (report_path, latest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(report, path)
    latest_pointer.write_text(str(output_dir), encoding="utf-8")

    print("===== Ablation Evaluation Results =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"Saved report: {report_path}")
    print("=" * 80)
    return report


def parse_args(default_config: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LAED ablation experiment.")
    if default_config is None:
        parser.add_argument("--config", choices=sorted(ABLATION_CONFIGS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ABLATION_OUTPUT_ROOT)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        default=None,
        help="Optional run directory whose data_summary.json should be reused when summary is enabled.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "replay", "full"),
        default="auto",
        help="auto replays completed runs when possible and runs full detection only when summary is removed.",
    )
    args = parser.parse_args()
    if default_config is not None:
        args.config = default_config
    return args


def main(default_config: Optional[str] = None) -> None:
    args = parse_args(default_config)
    config = ABLATION_CONFIGS[args.config]
    report = run_ablation(
        config=config,
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        chunksize=args.chunksize,
        reference_run_dir=args.reference_run_dir,
        mode=args.mode,
    )
    print(
        f"Done. {report['ablation']['name']} metrics for {report['dataset']} are in "
        f"{report['output_files']['evaluation_metrics']}"
    )


if __name__ == "__main__":
    main()
