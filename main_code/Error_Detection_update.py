# Error_Detection_update.py

import contextlib
import io
import ast
import json, os, time, re
import sys
from collections import OrderedDict
from difflib import SequenceMatcher
import pandas as pd
from utils import clean_code_snippet, read_dataframe
from Initial_Screening_all import initial_screening
from summarizer import Summarizer
from pythonProject1.API_invocation.qwen_gen import shared_qwen_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 初始化客户端
text_gen = shared_qwen_client
# 限流与上下文配置
RATE_LIMIT_RPM      = 500
RATE_LIMIT_SLEEP    = 60.0 / RATE_LIMIT_RPM
MAX_RESPONSE_TOKENS = 8192
AVG_TOKEN_PER_CHAR  = 1/4
PROMPT_OVERHEAD     = 1400
MAX_RELATIONSHIP_ERROR_RATIO = 0.05
MAX_STRUCTURED_RELATIONSHIP_ERROR_RATIO = 0.65
MAX_OVERALL_RELATIONSHIP_ERROR_RATIO = 0.50
MAX_RELATIONSHIP_CANDIDATE_CELLS_PER_ROW = 2.00
LOW_CONSISTENCY_RELATIONSHIP_RATIO = float(os.getenv("LAED_LOW_CONSISTENCY_RELATIONSHIP_RATIO", "0.70"))
STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO = float(os.getenv("LAED_STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO", "0.99"))
RELATIONSHIP_REPRESENTATIVE_BATCH_LIMIT = int(os.getenv("LAED_RELATIONSHIP_REP_BATCH_LIMIT", "5"))
DETECTION_REQUEST_TIMEOUT = int(os.getenv("LAED_DETECTION_REQUEST_TIMEOUT", "60"))
DETECTION_REQUEST_RETRIES = int(os.getenv("LAED_DETECTION_RETRIES", "2"))
SINGLE_COLUMN_DETECTION_REQUEST_RETRIES = int(os.getenv("LAED_SINGLE_COLUMN_DETECTION_RETRIES", "1"))
PROFILE_WARNING_EARLY_STOP_RATIO = float(os.getenv("LAED_PROFILE_WARNING_EARLY_STOP_RATIO", "0.75"))
MAX_LLM_REPRESENTATIVE_VALUES_PER_COLUMN = int(os.getenv("LAED_MAX_LLM_REPRESENTATIVE_VALUES_PER_COLUMN", "40"))
TRANSIENT_REQUEST_ERROR_TOKENS = (
    "SSLError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "ReadTimeout",
    "TimeoutError",
    "timed out",
    "exceeded",
    "ProxyError",
    "RemoteDisconnected",
    "does not contain any choices",
    "temporarily unavailable",
)
SINGLE_COLUMN_ERROR_TYPES = {"format_errors", "spelling_errors", "outliers", "missing_errors"}
BROAD_LLM_CONFIRMED_WARNING_FAMILIES = {
    "missing_like_candidate",
    "placeholder_text_candidate",
    "text_noise_candidate",
    "list_duplicate_candidate",
    "date_component_order_candidate",
    "date_granularity_candidate",
    "numeric_surface_variant_candidate",
    "numeric_representation_variant_candidate",
}

SYSTEM_INSTRUCTIONS = """
Task: You are a data detective. The dataset has been preliminarily screened; remaining non-blank cells may contain errors.
**Error detection must be strictly data-driven**: leverage the column’s overall value distribution, its dtype, semantic_type, description, and any provided regex. Do NOT rely on subjective or invented criteria.
For fields with a provided regex, use it as a primary guideline to identify likely errors, but verify against the column’s actual distribution—values that slightly deviate from the regex may still be valid if they align with common patterns.
For all fields (with or without regex), compare each candidate against the column’s common patterns (distribution, length, character set) to identify spelling_errors or outliers.
Return, for each batch:
- A list of error objects (with "row" as int or list of ints if grouped).
- If applicable, an `"error_regex_by_type"` object that maps each errorType (`format_errors`, `spelling_errors`, `outliers`, `missing_errors`) to its own anchored regex. If a given errorType has no matches, return an empty string for that type.
Only these single-column error types are allowed:
- format_errors
- spelling_errors
- outliers
- missing_errors
Do NOT report duplicate_errors or logical_errors here.
Do NOT mark a value as erroneous merely because it is rare, short, long, or
from a different real-world source/name. Rarity is only evidence that deserves
checking; the final error label must be supported by missingness, a clear
structural violation, a distributional extreme, or a typo relative to a
well-supported canonical value in this same column.
For identifier/code/source fields, a rare but structurally well-formed value is
not an error unless it violates the column's canonical format or relationship
evidence.
"""

USER_PROMPT_WITH_REGEX = """
**Error detection must be strictly data-driven**. Please carefully understand the meaning of the fields to formulate detection rules and make sure that reasonable free text is not regarded as an error. For example: 'Example Corp.' and 'General Item Name' are both legal free-text values.     They are not marked as errors unless the values are obviously junk (such as pure symbols, random control characters). For example, 12345 and 42 are all legal ids and do not require a fixed length. Mark only non-integer strings.
Do not treat different identifier digit lengths, source names, abbreviations, or
valid-looking category labels as errors solely because they are uncommon. The
column evidence must show a strict canonical surface form before you call such
values format_errors.
Each error type is defined as follows:
- **missing_errors**: actual nulls or semantic placeholders like "N/A", "nul", "nan", "empty".
- **format_errors**: clear structural violations relative to `{regex}`, for instance, unexpected characters, wrong delimiters, or completely different patterns.   Minor deviations that still follow the general pattern (but might be flagged as spelling errors later) should not be marked as `format_errors`. Explain exactly how the structure violates the expected format.
- **outliers**: values matching the correct pattern but lying far outside the normal distribution.
- **spelling_errors**: values matching the majority pattern but containing minor character-level typos.

Column: "{col}"
Data type: {dtype}
Semantic type: {semantic_type}
Description: {description}
Expected format (regex): `{regex}`
Format explanation: {fmt_expl}
Column profile evidence from data_summary.json:
{profile_json}
Values in this batch (row index → value). Some entries may include `profile_warnings`; these are high-priority generic candidate signals from data_summary/profile evidence:
{samples_json}
Task:
1. First, label any values that match the “missing” definition (actual nulls, `"N/A"`, `"nul"`, `"nan"`, `"empty"`, etc.) as **missing_errors**, and explain why each is considered missing.
2. From the remaining non-missing values, label only those with a clear structural or pattern violation relative to `{regex}` as **format_errors**, and explain how each deviates (unexpected characters, wrong delimiters, wrong arrangement). Do not label values here merely because they fail to match `{regex}` if they still follow the general pattern.
3. From the remaining values that passed steps 1–2, label any entries that match the correct pattern but lie far outside the normal distribution (numeric extremes, extremely long or short text) as **outliers**, and explain why.
4. From the remaining values that passed steps 1–3, label any that match the majority pattern but contain minor character-level misspellings as **spelling_errors**, and explain the specific typo.
For date columns, use the profile evidence to check component-order consistency. Values that look like a cyclic month/day/year or month/year/day transposition can be format_errors when the column profile supports a different component order.
If a value has `date_component_order_candidate`, label it as a format_errors unless the profile evidence clearly shows that the same component order is the dominant valid order for this column. Do not exempt ambiguous compact dates just because two components are equal; use the warning and the column-level component profile.
If a value has `date_granularity_candidate`, label it as format_errors when it is visibly much coarser than the dominant date granularity in the profile.
For free-text columns, replacement characters, embedded null tokens, control characters, and abnormal repeated whitespace are candidate evidence of format_errors, but only label them when the value itself shows such corruption.
For list-like text fields, exact duplicate list items inside one cell are candidate evidence of format_errors because they are redundant structural corruption, but only label them when the value visibly repeats the same item.
If a value has `missing_like_candidate` or `placeholder_text_candidate`, label it as missing_errors. If it has `text_noise_candidate:*`, label it as format_errors when the visible value contains the cited corruption marker.
If a value has `list_duplicate_candidate`, label it as format_errors when the visible comma/semicolon-separated list repeats the same normalized item.
If a category/list-like value has `category_variant_candidate`, label it only when the column profile supports a different canonical category vocabulary and the value visibly uses an alternate taxonomy or compound category label.
5. Construct an `"error_regex_by_type"` dictionary mapping each errorType (`format_errors`, `spelling_errors`, `outliers`, `missing_errors`) to an anchored regex that matches exactly the values flagged for that errorType in this batch. If no values of a certain type were flagged, set that type’s regex to `""`.
6. If multiple rows share the exact same `errorType`, `description`, and `correctFormat`, group their row indices into a list rather than repeating entries.
** Please note that all errors must have reasons, and these reasons should be analyzed based on the data set.      Do not imagine them out of thin air.      There should be data support.**
Return exactly a JSON object:
{{
  "errors": [
    {{
      "row": <row index>,
      "fieldName": "{col}",
      "errorType": "<format_errors|outliers|spelling_errors|missing_errors>",
      "description": "<why it deviates>",
      "correctFormat": "<brief description or corrected example>"
    }},
    …
  ],
  "error_regex_by_type": {{
    "format_errors": "<anchored regex or empty>",
    "spelling_errors": "<anchored regex or empty>",
    "outliers": "<anchored regex or empty>",
    "missing_errors": "<anchored regex or empty>"
  }}
}}
"""

USER_PROMPT_NO_REGEX = """
**Error detection must be strictly data-driven**. Please carefully understand the meaning of the fields to formulate detection rules and make sure that reasonable free text is not regarded as an error. For example: 'Example Corp.' and 'General Item Name' are both legal free-text values.     They are not marked as errors unless the values are obviously junk (such as pure symbols, random control characters). For example, 12345 and 42 are all legal ids and do not require a fixed length. Mark only non-integer strings.
Do not treat rare source names, uncommon abbreviations, short labels, or
free-text names as errors solely because they are uncommon. Rarity can only
prioritize review; the final label needs missingness, a clear structural
violation, a distributional extreme, or a typo relative to a well-supported
canonical value in this column.
Each error type is defined as follows:
- **missing_errors**: actual nulls or semantic placeholders like "N/A", "nul", "nan", "empty".
- **format_errors**: clear structural violations relative to `{regex}`, for instance, unexpected characters, wrong delimiters, or completely different patterns.   Minor deviations that still follow the general pattern (but might be flagged as spelling errors later) should not be marked as `format_errors`. Explain exactly how the structure violates the expected format.
- **outliers**: values matching the correct pattern but lying far outside the normal distribution.
- **spelling_errors**: values matching the majority pattern but containing minor character-level typos.

Column: "{col}"
Data type: {dtype}
Semantic type: {semantic_type}
Description: {description}
No expected regex provided.
Format explanation: {fmt_expl}
Column profile evidence from data_summary.json:
{profile_json}

Values in this batch (row index → value):
{samples_json}
Some entries may include `profile_warnings`; these are high-priority generic candidate signals from data_summary/profile evidence.

Task:
1. Label any values that are true nulls or semantically missing (empty, `"N/A"`, `"nul"`, `"nan"`) as **missing_errors**, and explain why.
2. From the remaining non-missing values, label only those with a clear structural or pattern violation relative to typical column values as **format_errors**, and explain how they violate the structure (e.g., letters in numeric, wrong date delimiters).
3. From the remaining values that passed steps 1–2, label any entries that match the general pattern but lie far outside the normal distribution (numeric extremes, extremely long/short text) as **outliers**, and explain why.
4. From the remaining values that passed steps 1–3, label any that match the majority pattern but contain minor character-level spellings as **spelling_errors**, and explain the specific typo.
For date columns, use the profile evidence to check component-order consistency. Values that look like a cyclic month/day/year or month/year/day transposition can be format_errors when the column profile supports a different component order.
If a value has `date_component_order_candidate`, label it as a format_errors unless the profile evidence clearly shows that the same component order is the dominant valid order for this column. Do not exempt ambiguous compact dates just because two components are equal; use the warning and the column-level component profile.
If a value has `date_granularity_candidate`, label it as format_errors when it is visibly much coarser than the dominant date granularity in the profile.
For free-text columns, replacement characters, embedded null tokens, control characters, and abnormal repeated whitespace are candidate evidence of format_errors, but only label them when the value itself shows such corruption.
For list-like text fields, exact duplicate list items inside one cell are candidate evidence of format_errors because they are redundant structural corruption, but only label them when the value visibly repeats the same item.
If a value has `missing_like_candidate` or `placeholder_text_candidate`, label it as missing_errors. If it has `text_noise_candidate:*`, label it as format_errors when the visible value contains the cited corruption marker.
If a value has `list_duplicate_candidate`, label it as format_errors when the visible comma/semicolon-separated list repeats the same normalized item.
If a category/list-like value has `category_variant_candidate`, label it only when the column profile supports a different canonical category vocabulary and the value visibly uses an alternate taxonomy or compound category label.
5. Construct an `"error_regex_by_type"` dictionary mapping each errorType (`format_errors`, `spelling_errors`, `outliers`, `missing_errors`) to an anchored regex that matches exactly the values flagged for that errorType in this batch. If no values of a certain type were flagged, set that type’s regex to `""`. For example:
6. If multiple rows share the exact same `errorType`, `description`, and `correctFormat`, group their row indices into a list rather than repeating entries.
** Please note that all errors must have reasons, and these reasons should be analyzed based on the data set.  Do not imagine them out of thin air.  There should be data support.**
Return exactly a JSON object:
{{
  "errors": [
    {{
      "row": <row index>,
      "fieldName": "{col}",
      "errorType": "<format_errors|outliers|spelling_errors|missing_errors>",
      "description": "<why it deviates>",
      "correctFormat": "<brief description or corrected example>"
    }},
    …
  ],
  "error_regex_by_type": {{
    "format_errors": "<anchored regex or empty>",
    "spelling_errors": "<anchored regex or empty>",
    "outliers": "<anchored regex or empty>",
    "missing_errors": "<anchored regex or empty>"
  }}
}}

"""

def count_cells(errs: list) -> int:
    """计算错误对象列表对应的单元格总数"""
    cells = set()
    fallback_total = 0
    for e in errs:
        r = e.get("row")
        rows = r if isinstance(r, list) else [r]
        fallback_total += len(rows)
        field = e.get("fieldName")
        for row in rows:
            try:
                row_idx = int(row)
            except (TypeError, ValueError):
                continue
            cells.add((row_idx, field) if field else row_idx)
    return len(cells) if cells else fallback_total


def expand_representative_errors(
        errs: list,
        representative_rows: dict[int, list[int]],
        max_expanded_rows_per_error: int | None = None,
) -> list:
    """
    Expand LLM decisions made for one representative value back to every row that
    has the same value. This is a generic batching optimization: the LLM still
    decides whether each distinct suspicious value is erroneous, while duplicate
    values avoid repeated API calls.
    """
    expanded = []
    for err in errs:
        row_spec = err.get("row")
        rows = row_spec if isinstance(row_spec, list) else [row_spec]
        expanded_rows = []
        for row in rows:
            try:
                row_key = int(row)
            except (TypeError, ValueError):
                continue
            expanded_rows.extend(representative_rows.get(row_key, [row_key]))

        if not expanded_rows:
            continue

        unique_rows = sorted({int(r) for r in expanded_rows})
        if max_expanded_rows_per_error is not None and len(unique_rows) > max_expanded_rows_per_error:
            unique_rows = [int(row) for row in rows if str(row).lstrip("-").isdigit()]
            if not unique_rows:
                continue

        new_err = dict(err)
        new_err["row"] = unique_rows if len(unique_rows) > 1 else unique_rows[0]
        expanded.append(new_err)
    return expanded


def restrict_errors_to_allowed_rows(errs: list, allowed_rows: set[int]) -> tuple[list, int]:
    """
    Keep LLM decisions inside the suspicious row scope produced by initial
    screening. Representative-value and regex expansions may only add duplicate
    suspicious cells; they may not promote a cell that initial screening marked
    correct into final_errors.
    """
    allowed = {int(row) for row in allowed_rows}
    restricted = []
    dropped = 0
    for err in errs:
        row_spec = err.get("row")
        rows = row_spec if isinstance(row_spec, list) else [row_spec]
        kept_rows = []
        for row in rows:
            try:
                row_idx = int(row)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if row_idx in allowed:
                kept_rows.append(row_idx)
            else:
                dropped += 1
        kept_rows = sorted(set(kept_rows))
        if not kept_rows:
            continue
        new_err = dict(err)
        new_err["row"] = kept_rows if len(kept_rows) > 1 else kept_rows[0]
        restricted.append(new_err)
    return restricted, dropped


def has_nonempty_error_regex(regex_list: list) -> bool:
    for regex_by_type in regex_list:
        if not isinstance(regex_by_type, dict):
            continue
        for value in regex_by_type.values():
            if str(value or "").strip() and str(value).strip() != "Correct":
                return True
    return False


def is_transient_request_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    return any(token in message for token in TRANSIENT_REQUEST_ERROR_TOKENS)


def _extract_json_object(text: str) -> str:
    """Return the outermost JSON-like object from an LLM response."""
    text = str(text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def _escape_invalid_json_backslashes(text: str) -> str:
    """
    LLMs often return regex strings such as "\\d+" inside JSON. Valid JSON needs
    those regex backslashes escaped, so repair only backslashes that are not
    legal JSON escape prefixes.
    """
    repaired = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt and nxt not in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                repaired.append("\\\\")
            else:
                repaired.append(ch)
            i += 1
            continue
        repaired.append(ch)
        i += 1
    return "".join(repaired)


def parse_llm_json_response(response: str) -> dict:
    """
    Parse a JSON object returned by the LLM while tolerating generic formatting
    defects. This repairs the transport/serialization layer only; it does not
    infer or add any detected errors.
    """
    candidate = _extract_json_object(clean_code_snippet(response))
    attempts = [
        candidate,
        _escape_invalid_json_backslashes(candidate),
    ]
    last_error = None
    for text in attempts:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc

    pythonish = re.sub(r"\btrue\b", "True", attempts[-1])
    pythonish = re.sub(r"\bfalse\b", "False", pythonish)
    pythonish = re.sub(r"\bnull\b", "None", pythonish)
    try:
        parsed = ast.literal_eval(pythonish)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        last_error = exc

    raise ValueError(f"Unable to parse LLM JSON response: {last_error}")


def relationship_columns(relationships: dict, columns) -> set:
    df_cols = set(columns)
    cols = set()
    for rel_type, rel_data in (relationships or {}).items():
        if rel_type in {"hierarchical", "associative"} and isinstance(rel_data, dict):
            for key, values in rel_data.items():
                if key in df_cols:
                    cols.add(key)
                if isinstance(values, list):
                    cols.update(v for v in values if v in df_cols)
        elif rel_type == "temporal" and isinstance(rel_data, list):
            for seq in rel_data:
                if isinstance(seq, list):
                    cols.update(v for v in seq if v in df_cols)
        elif rel_type == "mathematical" and isinstance(rel_data, dict):
            for key, info in rel_data.items():
                if key in df_cols:
                    cols.add(key)
                if isinstance(info, dict):
                    cols.update(v for v in info.get("fields", []) if v in df_cols)
                elif isinstance(info, list):
                    cols.update(v for v in info if v in df_cols)
    return cols


def relationship_error_ratio_limit(col: str, props: dict | None = None) -> float:
    props = props or {}
    field_text = (
        f"{col} {props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if any(token in field_text for token in ("date", "time", "duration", "timestamp", "number", "numeric", "amount", "score")):
        return MAX_STRUCTURED_RELATIONSHIP_ERROR_RATIO
    if any(token in field_text for token in ("boolean", "bool", "flag", "indicator", "category", "label", "class")):
        return MAX_RELATIONSHIP_ERROR_RATIO
    return 0.15


def suppress_low_confidence_relationship_errors(
    rel_errors: set,
    rel_cols: set,
    row_count: int,
    label: str,
    field_props: dict | None = None,
) -> set:
    """
    Relationship errors originate from LLM-generated summary validators. A
    validator that marks a broad fraction of one field is weak evidence, so keep
    only field-level relationship signals with a narrow enough error footprint.
    """
    if not rel_errors or row_count <= 0:
        return rel_errors

    scoped_cols = {col for col in rel_cols} | {col for _, col in rel_errors}
    filtered_errors = {
        (rid, col)
        for rid, col in rel_errors
        if col in scoped_cols
    }
    if not filtered_errors:
        return set()

    if len(filtered_errors) / row_count > MAX_RELATIONSHIP_CANDIDATE_CELLS_PER_ROW:
        print(
            f"{label} produced {len(filtered_errors)} candidate cells "
            f"({len(filtered_errors) / row_count:.2f} per row); "
            "skipping broad low-confidence relationship candidates."
        )
        return set()

    counts_by_col = {}
    for _, col in filtered_errors:
        counts_by_col[col] = counts_by_col.get(col, 0) + 1

    field_props = field_props or {}
    broad_cols = {
        col
        for col, count in counts_by_col.items()
        if (
            count / row_count > relationship_error_ratio_limit(col, field_props.get(col, {}))
            and len(filtered_errors) / row_count > MAX_RELATIONSHIP_CANDIDATE_CELLS_PER_ROW
        )
    }
    for col in sorted(broad_cols):
        col_ratio = counts_by_col[col] / row_count
        print(
            f"{label} marked {col_ratio:.2%} of '{col}' cells; "
            "skipping that low-confidence field-level relationship signal."
        )

    filtered_errors = {
        (rid, col)
        for rid, col in filtered_errors
        if col not in broad_cols
    }

    rel_ratio = len(filtered_errors) / (row_count * len(scoped_cols))
    if rel_ratio > MAX_OVERALL_RELATIONSHIP_ERROR_RATIO:
        print(
            f"{label} marked {rel_ratio:.2%} of relationship cells; "
            "skipping low-confidence logical error map."
        )
        return set()

    return filtered_errors


MISSING_LIKE_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "nil",
    "missing",
    "empty",
    "?",
}


def normalize_missing_like_token(value) -> str:
    text = "" if value is None else str(value).strip().lower()
    return re.sub(r"^[\s\{\}\[\]\(\)\"']+|[\s\{\}\[\]\(\)\"']+$", "", text)


def is_missing_like_value(value) -> bool:
    text = "" if value is None else str(value).strip().lower()
    normalized = normalize_missing_like_token(value)
    return text in MISSING_LIKE_VALUES or normalized in MISSING_LIKE_VALUES


def text_noise_reasons(value) -> list[str]:
    text = "" if value is None else str(value)
    reasons = []
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", text):
        reasons.append("control_character")
    if re.search(r"\s{2,}", text):
        reasons.append("repeated_whitespace")
    if re.search(r"(?:\.{3,}|\u2026)\s*(?:$|[)\]\u00bb]|see\s+full)", text.strip(), re.IGNORECASE):
        reasons.append("truncation_ellipsis")
    if re.search(r"(?i)(^|[\s\{\[\(\",;:_-])null($|[\s\}\]\)\",;:_-])", text):
        reasons.append("embedded_null_token")
    if re.search(r"[\ufffd]{2,}|[\ufffd]_+", text):
        reasons.append("mojibake_marker")
    return reasons


def has_duplicate_list_items(value) -> bool:
    text = "" if value is None else str(value)
    if "," not in text and ";" not in text and "|" not in text:
        return False
    parts = [
        re.sub(r"\s+", " ", part.strip().lower())
        for part in re.split(r"[,;|]", text)
        if part and part.strip()
    ]
    if len(parts) < 2:
        return False
    return len(set(parts)) < len(parts)


def is_free_text_placeholder(value, props: dict | None = None) -> bool:
    props = props or {}
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text):
        return False
    lowered = re.sub(r"\s+", " ", text.lower()).strip(" .;:-_")
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    looks_free_text = any(
        token in field_text
        for token in (
            "text",
            "description",
            "summary",
            "comment",
            "review",
            "note",
            "content",
            "free-form",
        )
    )
    if not looks_free_text:
        return False

    if re.fullmatch(r"(?:unknown|not known|not available|unavailable|not provided|none provided|tbd|tba|pending)", lowered):
        return True
    if re.fullmatch(r"(?:the\s+)?(?:plot|description|summary|content|details?)\s+(?:is\s+)?unknown(?:\s+at\s+this\s+time)?", lowered):
        return True
    if re.fullmatch(r"(?:add|enter|insert|provide)\s+(?:a|an|the)?\s*[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3}", lowered):
        return True
    return False


def is_category_variant_candidate(value, props: dict | None = None) -> bool:
    props = props or {}
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if not any(token in field_text for token in ("category", "categorical", "label", "class", "type", "list")):
        return False
    parts = [part.strip() for part in re.split(r"[,;|]", text) if part and part.strip()]
    if len(parts) < 1:
        return False
    return any(re.search(r"\s(?:&|and|or)\s|/", part, re.IGNORECASE) for part in parts)


def normalized_text_key(value) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def profile_spelling_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text):
        return False
    if is_numeric_measure_field(props):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if field_text_has_token(field_text, ("number", "amount", "score", "date", "time", "timestamp")):
        return False
    skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
    value_skeleton = Summarizer._normalized_numeric_text_skeleton(text)
    if (
            field_text_has_token(field_text, ("id", "identifier", "code", "key"))
            and "<num>" in value_skeleton
    ):
        try:
            skeleton_count = int(skeleton_counts.get(value_skeleton) or 0)
        except (TypeError, ValueError):
            skeleton_count = 0
        if skeleton_count >= 8:
            return False
    top_counts = props.get("top_value_counts") or {}
    rare_values = props.get("rare_value_examples") or {}
    if not isinstance(top_counts, dict) or not top_counts:
        return False
    counts = {}
    total = 0
    for raw_value, raw_count in {**top_counts, **rare_values}.items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        key = normalized_text_key(raw_value)
        if not key or key in MISSING_LIKE_VALUES:
            continue
        counts[key] = max(counts.get(key, 0), count)
        total += count
    value_key = normalized_text_key(text)
    if not value_key:
        return False
    candidate_count = counts.get(value_key, 0)
    if candidate_count > max(10, int(max(total, 1) * 0.02)):
        return False
    frequent_threshold = max(8, int(max(total, 1) * 0.03))
    frequent_values = [
        key for key, count in counts.items()
        if count >= frequent_threshold and key != value_key and len(key) >= 3
    ][:100]
    for canonical in frequent_values:
        length = max(len(value_key), len(canonical))
        threshold = 0.76 if length <= 12 else 0.86
        if SequenceMatcher(None, value_key, canonical).ratio() >= threshold:
            return True
    return False


def code_skeleton_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text) or not re.search(r"\d", text):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if not field_text_has_token(field_text, ("id", "identifier", "code", "key")):
        return False
    skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
    if not isinstance(skeleton_counts, dict) or not skeleton_counts:
        return False
    value_skeleton = Summarizer._normalized_numeric_text_skeleton(text)
    if "<num>" not in value_skeleton:
        return False
    total = 0
    parsed_counts = {}
    for skeleton, raw_count in skeleton_counts.items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        parsed_counts[str(skeleton)] = count
        total += count
    current_count = parsed_counts.get(value_skeleton, 0)
    if current_count >= max(8, int(max(total, 1) * 0.03)):
        return False
    frequent_threshold = max(20, int(max(total, 1) * 0.05))
    normalized_value = value_skeleton.replace("<num>", "9")
    for skeleton, count in parsed_counts.items():
        if skeleton == value_skeleton or count < frequent_threshold or "<num>" not in skeleton:
            continue
        normalized_candidate = skeleton.replace("<num>", "9")
        threshold = 0.78 if max(len(normalized_value), len(normalized_candidate)) <= 12 else 0.86
        if SequenceMatcher(None, normalized_value, normalized_candidate).ratio() >= threshold:
            return True
    return False


def category_code_label_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text):
        return False
    top_counts = props.get("top_value_counts") or {}
    if not isinstance(top_counts, dict) or not top_counts:
        return False
    short_code_count = 0
    total = 0
    for raw_value, raw_count in top_counts.items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        raw_text = str(raw_value).strip()
        if is_missing_like_value(raw_text):
            continue
        total += count
        if re.fullmatch(r"[A-Za-z]{2,4}", raw_text):
            short_code_count += count
    if total <= 0 or short_code_count / total < 0.55:
        return False
    if re.fullmatch(r"[A-Za-z]{2,4}", text):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .;,_-]{4,}", text))


def _clean_relationship_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_missing_like_relationship_value(value) -> bool:
    return is_missing_like_value(_clean_relationship_value(value))


def profile_normalized_numeric_skeleton(props: dict, value: str) -> str:
    """
    Map a value to normalized skeleton evidence already stored in data_summary.
    If the summary has no grouping for this raw skeleton, keep the raw skeleton.
    """
    raw_skeleton = Summarizer._normalized_numeric_text_skeleton(value)
    for normalized_skeleton, group in (props.get("normalized_numeric_text_skeleton_groups") or {}).items():
        if isinstance(group, dict) and raw_skeleton in group:
            return str(normalized_skeleton)
    return raw_skeleton


def is_numeric_measure_field(props: dict) -> bool:
    """
    Return True only when summary profile evidence describes a numeric or
    numeric-with-unit surface. This stays dataset-neutral: field text is used
    only as a generic guard for identifier-like columns, while the decision is
    driven by shape and skeleton distributions from data_summary.json.
    """
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if str(props.get("dtype", "")).lower() in {"number", "float", "int", "integer"}:
        return True
    family_counts = props.get("shape_family_counts") or {}
    numeric_count = sum(
        int(family_counts.get(key, 0) or 0)
        for key in ("plain_numeric", "percent_numeric", "unit_or_text_numeric")
    )
    total_count = sum(int(value or 0) for value in family_counts.values())
    if total_count and numeric_count / total_count < 0.50:
        return False
    dominant_skeleton = str(props.get("dominant_normalized_numeric_text_skeleton") or "").lower()
    if "<num>" not in dominant_skeleton:
        return False
    if re.search(r"\b(id|identifier|code|key|name|label|type)\b", field_text):
        return numeric_count / max(total_count, 1) >= 0.90 and dominant_skeleton in {"<num>", "<num>.<num>"}
    return True


def is_integer_grouping_skeleton(skeleton: str) -> bool:
    return bool(re.fullmatch(r"<num>(?:,<num>)*", str(skeleton or "")))


def field_text_has_token(field_text: str, tokens: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", field_text)
        for token in tokens
    )


def has_noncanonical_numeric_skeleton(value: str, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text) or not re.search(r"\d", text):
        return False
    if not is_numeric_measure_field(props):
        return False
    field_text = f"{props.get('dtype', '')} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
    if field_text_has_token(field_text, ("date", "time", "timestamp")):
        return False
    dominant_skeleton = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
    if not dominant_skeleton:
        return False
    value_skeleton = profile_normalized_numeric_skeleton(props, text)
    if is_integer_grouping_skeleton(dominant_skeleton) and is_integer_grouping_skeleton(value_skeleton):
        return False
    if value_skeleton and value_skeleton == dominant_skeleton:
        return False
    if value_skeleton and value_skeleton != dominant_skeleton:
        return True

    raw_skeleton = Summarizer._numeric_text_skeleton(text)
    dominant_raw = str(props.get("dominant_numeric_text_skeleton") or "")
    if raw_skeleton and dominant_raw and raw_skeleton != dominant_raw:
        numeric_profiles = props.get("numeric_value_profiles") or {}
        if raw_skeleton in numeric_profiles or raw_skeleton in (props.get("normalized_numeric_text_skeleton_counts") or {}):
            return True
    return False


def text_value_is_summary_supported_canonical(props: dict, value) -> bool:
    """
    Return True for high-support textual/category surfaces in the summary
    profile. This is a precision guard for LLM-returned regex expansion: a
    regex that also matches the dominant observed surface should not expand to
    that surface unless the LLM explicitly diagnosed it.
    """
    if is_numeric_measure_field(props):
        return False
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text):
        return False
    top_counts = props.get("top_value_counts") or {}
    if not isinstance(top_counts, dict) or not top_counts:
        return False
    normalized_counts = {}
    total = 0
    for raw_value, raw_count in top_counts.items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        key = re.sub(r"\s+", " ", str(raw_value).strip().lower())
        normalized_counts[key] = normalized_counts.get(key, 0) + count
        total += count
    key = re.sub(r"\s+", " ", text.lower())
    count = normalized_counts.get(key, 0)
    if count <= 0:
        return False
    threshold = max(5, int(max(total, 1) * 0.05))
    return count >= threshold


def canonical_numeric_surface_skeleton(props: dict) -> str:
    if not is_numeric_measure_field(props):
        return ""
    dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
    groups = props.get("normalized_numeric_text_skeleton_groups") or {}
    group = groups.get(dominant_normalized)
    if not dominant_normalized or not isinstance(group, dict) or len(group) <= 1:
        return ""

    def score(raw_skeleton: str, count) -> tuple:
        text = str(raw_skeleton)
        try:
            observed_count = int(count or 0)
        except (TypeError, ValueError):
            observed_count = 0
        surface = text.replace("<num>", "").strip()
        trailing_punctuation = 1 if re.search(r"[A-Za-z]\.$", surface) else 0
        punctuation_count = len(re.findall(r"[^\w\s<>]", surface))
        alpha_len = len(re.sub(r"[^A-Za-z]", "", surface))
        return (trailing_punctuation, punctuation_count, alpha_len, len(surface), -observed_count, text)

    return min(group.items(), key=lambda item: score(item[0], item[1]))[0]


def numeric_surface_variant_warning(value: str, props: dict) -> str:
    text = "" if value is None else str(value).strip()
    if not text or is_missing_like_value(text) or not re.search(r"\d", text):
        return ""
    if not is_numeric_measure_field(props):
        return ""
    field_text = f"{props.get('dtype', '')} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
    if field_text_has_token(field_text, ("date", "time", "timestamp")):
        return ""
    canonical = canonical_numeric_surface_skeleton(props)
    if not canonical:
        return ""
    raw_skeleton = Summarizer._numeric_text_skeleton(text)
    normalized = profile_normalized_numeric_skeleton(props, text)
    dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
    if normalized == dominant_normalized and raw_skeleton != canonical:
        return f"numeric_surface_variant_candidate:{raw_skeleton}"
    return ""


def noncanonical_numeric_surface_skeletons(props: dict) -> set[str]:
    """
    Return raw numeric/unit skeletons that share the dominant normalized profile
    but are not the canonical surface spelling. This is distribution-derived
    evidence from data_summary.json, not a domain/unit whitelist.
    """
    if not is_numeric_measure_field(props):
        return set()
    canonical = canonical_numeric_surface_skeleton(props)
    dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
    if not canonical or not dominant_normalized:
        return set()
    groups = props.get("normalized_numeric_text_skeleton_groups") or {}
    group = groups.get(dominant_normalized)
    if not isinstance(group, dict):
        return set()
    return {
        str(raw_skeleton)
        for raw_skeleton in group.keys()
        if str(raw_skeleton) != canonical
    }


def numeric_variant_error_type(value: str, props: dict, fmt: dict) -> str:
    text = "" if value is None else str(value).strip()
    regex = str((fmt or {}).get("regex", "") or "").strip()
    if regex:
        try:
            if re.fullmatch(regex, text) is None:
                return "format_errors"
        except re.error:
            pass
    return "spelling_errors"


def relationship_requires_local_anomaly_gate(col: str, props: dict) -> bool:
    text = f"{col} {props.get('dtype', '')} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
    if any(token in text for token in ("time", "date", "duration")):
        return False
    return True


def relationship_requires_generated_validator_gate(col: str, props: dict) -> bool:
    """
    LLM-generated relationship code can over-mark low-information boolean/flag
    dependents via majority voting. For these fields, require independent local
    anomaly evidence before accepting a relationship-only error.
    """
    return relationship_requires_local_anomaly_gate(col, props)


def relationship_structured_field(field: str, props: dict | None = None) -> bool:
    props = props or {}
    field_text = (
        f"{field} {props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    return field_text_has_token(
        field_text,
        (
            "id", "identifier", "code", "key", "number", "num", "issn", "isbn",
            "doi", "date", "time", "datetime", "timestamp", "duration", "year",
            "numeric", "amount", "rate", "percent", "percentage", "score",
        ),
    )


def relationship_pair_is_actionable(
        key: str,
        dep: str,
        evidence: dict,
        field_props: dict,
) -> bool:
    """
    Relationship evidence can describe stable field logic without always being
    safe enough to generate final-detection candidates. Weak non-identifier
    text/category majority mappings are retained in summary metadata, but they
    need near-deterministic evidence before entering LLM final confirmation.
    """
    if int(evidence.get("conflicting_cells") or 0) <= 0:
        return False
    if bool(evidence.get("key_is_identifier_like")):
        return True
    key_props = field_props.get(key, {}) or {}
    dep_props = field_props.get(dep, {}) or {}
    if relationship_structured_field(key, key_props) or relationship_structured_field(dep, dep_props):
        return True
    return float(evidence.get("consistency_ratio") or 0.0) >= STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO


def relationship_value_has_local_anomaly(col: str, value, props: dict, format_rules: dict) -> bool:
    text = _clean_relationship_value(value)
    lower = text.lower()
    if is_missing_like_value(text):
        return True
    fmt = format_rules.get(col, {}) or {}
    regex = str(fmt.get("regex", "") or "")
    if regex and summary_regex_is_trustworthy(props, fmt):
        try:
            if re.fullmatch(regex, text) is None:
                return True
        except re.error:
            pass
    top_counts = {
        str(v).strip().lower(): count
        for v, count in (props.get("top_value_counts") or {}).items()
    }
    value_counts = {
        str(v).strip().lower(): count
        for v, count in (props.get("value_counts") or {}).items()
    }
    rare_counts = {
        str(v).strip().lower(): count
        for v, count in (props.get("rare_value_examples") or {}).items()
    }
    if lower in rare_counts and lower not in top_counts and lower not in value_counts:
        return True
    if text_noise_reasons(text):
        return True
    try:
        shape_count = int((props.get("shape_counts") or {}).get(Summarizer._value_shape(text)) or 0)
    except (TypeError, ValueError):
        shape_count = 0
    return 0 < shape_count <= 2


def summary_regex_shape_coverage_is_trustworthy(props: dict, regex: str) -> bool:
    shape_counts = props.get("shape_counts") or {}
    observed_examples = {}
    for source_key in ("top_value_counts", "rare_value_examples"):
        for value, raw_count in (props.get(source_key) or {}).items():
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                count = 1
            observed_examples[str(value)] = max(observed_examples.get(str(value), 0), count)
    if not shape_counts or not observed_examples:
        return True

    shape_match = {}
    example_total = 0
    example_matched = 0
    for value in observed_examples:
        text = str(value).strip()
        if is_missing_like_value(text):
            continue
        try:
            count = int(observed_examples.get(value) or 0)
        except (TypeError, ValueError):
            count = 1
        shape = Summarizer._value_shape(text)
        try:
            matched = re.fullmatch(regex, text) is not None
        except re.error:
            return False
        example_total += max(count, 1)
        if matched:
            example_matched += max(count, 1)
        current = shape_match.get(shape)
        shape_match[shape] = matched if current is None else (current or matched)

    total = 0
    represented = 0
    for shape, raw_count in shape_counts.items():
        shape_text = str(shape)
        if shape_text == "<missing-like>":
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        total += count
        if shape_match.get(shape_text):
            represented += count

    if total <= 0:
        return True
    coverage = represented / total
    if coverage >= 0.80:
        return True
    # Exact shape strings over-penalize structured text regexes with variable
    # alphabetic tokens, such as month names or country labels. In that case,
    # trust the rule when the observed examples themselves overwhelmingly
    # match, and let profile warnings keep rare noncanonical values suspicious.
    variable_alpha_regex = bool(re.search(r"[A-Za-z]{3,}", regex))
    if variable_alpha_regex and example_total > 0:
        return example_matched / example_total >= 0.75
    return False


def summary_regex_is_trustworthy(props: dict, fmt: dict) -> bool:
    regex = str((fmt or {}).get("regex", "") or "").strip()
    if not regex:
        return False
    if not summary_regex_shape_coverage_is_trustworthy(props, regex):
        return False
    top_counts = props.get("top_value_counts") or {}
    if not top_counts:
        return True
    matched = 0
    unmatched = 0
    for value, raw_count in top_counts.items():
        text = str(value).strip()
        if is_missing_like_value(text):
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            count = 1
        try:
            if re.fullmatch(regex, text):
                matched += count
            else:
                unmatched += count
        except re.error:
            return False
    total = matched + unmatched
    if total <= 0:
        return True
    return unmatched / total <= 0.15


def generic_associative_relationship_errors(summary: dict, df: pd.DataFrame) -> set:
    """
    Execute a generic key -> dependent majority-mapping check from the summary.
    This is a fallback/parallel check for LLM-generated relationship code: the
    relationship candidates still come from data_summary.json, not from
    dataset-specific patches.
    """
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    if not isinstance(associative, dict):
        associative = {}
    fields = {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }
    format_rules = summary.get("format_rules", {}) or {}

    evidence = summary.get("relationship_evidence", {}) or {}
    evidence_map = summary_associative_evidence_map(summary)
    evidence_dependents = set()
    field_props_for_pairs = summary_field_props(summary)
    for candidate in evidence.get("candidate_associative_dependencies", []) or []:
        key = candidate.get("key")
        deps = [
            dep.get("field")
            for dep in candidate.get("dependents", [])
            if dep.get("field")
            and float(dep.get("consistency_ratio") or 0) >= 0.5
            and int(dep.get("conflicting_cells") or 0) >= 5
            and relationship_pair_is_actionable(str(key), str(dep.get("field")), dep, field_props_for_pairs)
        ]
        if key and deps:
            evidence_dependents.update((key, dep) for dep in deps)
            existing = associative.get(key, [])
            if not isinstance(existing, list):
                existing = []
            associative[key] = sorted(set(existing) | set(deps))

    errors = set()
    for key, dependents in associative.items():
        if key not in df.columns or not isinstance(dependents, list):
            continue
        for dep in dependents:
            if dep not in df.columns or dep == key:
                continue
            if (key, dep) not in evidence_dependents:
                continue
            grouped = {}
            for idx, row in df[[key, dep]].iterrows():
                key_value = _clean_relationship_value(row[key])
                dep_value = _clean_relationship_value(row[dep])
                if _is_missing_like_relationship_value(key_value) or _is_missing_like_relationship_value(dep_value):
                    continue
                grouped.setdefault(key_value, []).append((int(idx), dep_value))

            for entries in grouped.values():
                if len(entries) < 2:
                    continue
                counts = {}
                for _, value in entries:
                    counts[value] = counts.get(value, 0) + 1
                if len(counts) <= 1:
                    continue
                canonical_value, canonical_count = sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[0]
                majority_ratio = canonical_count / len(entries)
                pair_evidence = evidence_map.get((key, dep), {}) or {}
                consistency_ratio = float(pair_evidence.get("consistency_ratio") or majority_ratio)
                if majority_ratio < 0.55 or consistency_ratio < LOW_CONSISTENCY_RELATIONSHIP_RATIO:
                    for idx, _ in entries:
                        errors.add((idx, dep))
                else:
                    for idx, value in entries:
                        if value != canonical_value:
                            errors.add((idx, dep))
    return errors


def summary_associative_evidence_pairs(summary: dict) -> set[tuple[str, str]]:
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    if not isinstance(associative, dict):
        associative = {}
    evidence = summary.get("relationship_evidence", {}) or {}
    pairs = set()
    for candidate in evidence.get("candidate_associative_dependencies", []) or []:
        key = candidate.get("key")
        if not key or key not in associative:
            continue
        existing = associative.get(key, [])
        if not isinstance(existing, list):
            existing = []
        for dep in candidate.get("dependents", []) or []:
            field = dep.get("field")
            if (
                field
                and field in existing
                and float(dep.get("consistency_ratio") or 0) >= 0.5
                and bool(dep.get("actionable_conflicts", int(dep.get("conflicting_cells") or 0) >= 5))
                and int(dep.get("conflicting_cells") or 0) > 0
                and relationship_pair_is_actionable(str(key), str(field), dep, summary_field_props(summary))
            ):
                pairs.add((str(key), str(field)))
    return pairs


def summary_associative_evidence_map(summary: dict) -> dict[tuple[str, str], dict]:
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    if not isinstance(associative, dict):
        associative = {}
    evidence = summary.get("relationship_evidence", {}) or {}
    evidence_map = {}
    for candidate in evidence.get("candidate_associative_dependencies", []) or []:
        key = candidate.get("key")
        if not key or key not in associative:
            continue
        existing = associative.get(key, [])
        if not isinstance(existing, list):
            existing = []
        for dep in candidate.get("dependents", []) or []:
            field = dep.get("field")
            if field and field in existing:
                evidence_map[(str(key), str(field))] = dep
    return evidence_map


def summary_field_props(summary: dict) -> dict:
    return {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }


def relationship_tuple_consensus_candidates(
        summary: dict,
        df: pd.DataFrame,
        field_props: dict,
        candidate_scope: set[tuple[int, str]] | None = None,
) -> tuple[set[tuple[int, str]], list[dict]]:
    """
    Build candidate cells from a generic same-key tuple consensus signal. This
    is candidate generation only. A caller must obtain LLM confirmation before
    adding these cells to final_errors.
    """
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    if not isinstance(associative, dict):
        return set(), []

    evidence_map = summary_associative_evidence_map(summary)
    candidates = set()
    group_summaries = []
    candidate_scope = candidate_scope or set()

    def structured_relationship_field(field: str) -> bool:
        props = field_props.get(field, {}) or {}
        field_text = (
            f"{field} {props.get('dtype', '')} {props.get('semantic_type', '')} "
            f"{props.get('description', '')}"
        ).lower()
        return field_text_has_token(
            field_text,
            (
                "time", "date", "datetime", "timestamp", "duration",
                "number", "numeric", "code", "score", "amount", "rate",
                "percent", "percentage", "boolean", "flag",
            ),
        )

    def tuple_consensus_expandable_field(field: str) -> bool:
        props = field_props.get(field, {}) or {}
        field_text = (
            f"{field} {props.get('dtype', '')} {props.get('semantic_type', '')} "
            f"{props.get('description', '')}"
        ).lower()
        return field_text_has_token(
            field_text,
            ("actual", "observed", "measured", "reported", "result", "score", "value"),
        )

    for key, dependents in associative.items():
        if key not in df.columns or not isinstance(dependents, list):
            continue
        scoped_dependents = [
            str(dep)
            for dep in dependents
            if dep in df.columns
            and dep != key
            and structured_relationship_field(str(dep))
        ]
        if len(scoped_dependents) < 2:
            continue

        low_consistency = [
            dep for dep in scoped_dependents
            if float((evidence_map.get((str(key), dep), {}) or {}).get("consistency_ratio") or 1.0)
            < LOW_CONSISTENCY_RELATIONSHIP_RATIO
            and int((evidence_map.get((str(key), dep), {}) or {}).get("conflicting_cells") or 0) >= 20
        ]
        if len(low_consistency) < 2:
            continue

        for key_value, group_df in df[[key, *scoped_dependents]].groupby(key, sort=False):
            group_rows = [int(idx) for idx in group_df.index]
            tuple_counts = {}
            tuple_completeness = {}
            for row_idx, row in group_df.iterrows():
                values = tuple(_clean_relationship_value(row[dep]) for dep in scoped_dependents)
                completeness = sum(
                    1 for value in values
                    if not _is_missing_like_relationship_value(value)
                )
                if completeness < max(2, min(4, len(scoped_dependents))):
                    continue
                tuple_counts[values] = tuple_counts.get(values, 0) + 1
                tuple_completeness[values] = completeness
            if not tuple_counts or len(tuple_counts) <= 1:
                continue

            canonical_tuple, canonical_count = sorted(
                tuple_counts.items(),
                key=lambda item: (-tuple_completeness.get(item[0], 0), -item[1], item[0]),
            )[0]
            if int(canonical_count) < 2:
                continue
            field_candidates = []
            for row_idx in group_rows:
                for dep_idx, dep in enumerate(scoped_dependents):
                    if not tuple_consensus_expandable_field(dep):
                        continue
                    value = _clean_relationship_value(df.at[row_idx, dep])
                    expected = canonical_tuple[dep_idx]
                    if (
                            _is_missing_like_relationship_value(value)
                            or _is_missing_like_relationship_value(expected)
                            or value == expected
                    ):
                        continue
                    cell = (int(row_idx), dep)
                    if candidate_scope and cell not in candidate_scope:
                        continue
                    candidates.add(cell)
                    field_candidates.append({
                        "row": int(row_idx),
                        "field": dep,
                        "value": value,
                        "expected_tuple_value": expected,
                    })

            if field_candidates:
                group_summaries.append({
                    "key_field": str(key),
                    "key_value": str(key_value),
                    "dependent_fields": scoped_dependents,
                    "group_size": len(group_rows),
                    "canonical_tuple_count": int(canonical_count),
                    "canonical_tuple": {
                        dep: canonical_tuple[idx]
                        for idx, dep in enumerate(scoped_dependents)
                    },
                    "candidate_cell_count": len(field_candidates),
                    "candidate_examples": field_candidates[:8],
                })

    return candidates, group_summaries


class DetectionExplorer:
    def __init__(self, experience_file='ErrorDetection_Experience_file.json'):
        self.experience_file = experience_file
        if not os.path.exists(self.experience_file):
            with open(self.experience_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) * AVG_TOKEN_PER_CHAR) + 1

    def column_profile_for_prompt(self, props: dict) -> str:
        profile = {}
        for key in (
            "top_value_counts",
            "rare_value_examples",
            "shape_counts",
            "shape_family_counts",
            "numeric_text_skeleton_counts",
            "normalized_numeric_text_skeleton_counts",
            "dominant_normalized_numeric_text_skeleton",
            "case_punctuation_variant_groups",
            "missing_like_values",
            "date_component_profile",
            "text_noise_examples",
            "top_value_coverage",
            "num_unique_observed_values",
        ):
            value = props.get(key)
            if value not in (None, {}, [], ""):
                profile[key] = value
        return json.dumps(profile, ensure_ascii=False, indent=2)

    def date_component_order_warning(self, value: str, props: dict) -> str:
        dtype = str(props.get("dtype", "")).lower()
        semantic_type = str(props.get("semantic_type", "")).lower()
        if dtype != "date" and "date" not in semantic_type and "time" not in semantic_type:
            return ""
        profile = props.get("date_component_profile") or {}
        matched = int(profile.get("matched_count") or 0)
        positions = profile.get("positions") or []
        if matched < 30 or len(positions) != 3:
            return ""
        match = re.fullmatch(r"(\d{1,4})([\/\-.])(\d{1,4})\2(\d{1,4})", str(value).strip())
        if not match:
            return ""
        parts = [int(match.group(1)), int(match.group(3)), int(match.group(4))]
        stats_by_pos = []
        for stats in positions:
            top_values = stats.get("top_values") or {}
            modal_value = None
            modal_count = 0
            for raw_value, raw_count in top_values.items():
                if not str(raw_value).lstrip("-").isdigit():
                    continue
                try:
                    value_int = int(raw_value)
                    count_int = int(raw_count or 0)
                except (TypeError, ValueError):
                    continue
                if count_int > modal_count:
                    modal_value = value_int
                    modal_count = count_int
            stats_by_pos.append(
                {
                    "modal_value": modal_value,
                    "modal_share": modal_count / matched if matched else 0.0,
                    "gt_12_share": int(stats.get("gt_12_count") or 0) / matched if matched else 0.0,
                    "gt_31_share": int(stats.get("gt_31_count") or 0) / matched if matched else 0.0,
                }
            )

        constant_positions = [
            idx
            for idx, stats in enumerate(stats_by_pos)
            if stats["modal_value"] is not None and stats["modal_share"] >= 0.55
        ]
        variable_year_positions = [
            idx
            for idx, stats in enumerate(stats_by_pos)
            if stats["gt_12_share"] >= 0.20 or stats["gt_31_share"] >= 0.03
        ]
        for const_idx in constant_positions:
            modal_value = stats_by_pos[const_idx]["modal_value"]
            if parts[const_idx] == modal_value:
                continue
            for year_idx in variable_year_positions:
                if year_idx == const_idx:
                    continue
                if parts[year_idx] == modal_value and parts[const_idx] <= 31:
                    return (
                        "date_component_order_candidate: a normally constant "
                        "date component appears in a different position, which "
                        "suggests cyclic date component transposition"
                    )

        for pos, stats in enumerate(positions):
            top_values = stats.get("top_values") or {}
            if not top_values:
                continue
            modal_value, modal_count = max(
                ((int(k), int(v)) for k, v in top_values.items() if str(k).lstrip("-").isdigit()),
                key=lambda item: item[1],
                default=(None, 0),
            )
            if modal_value is None or modal_count / matched < 0.7 or parts[pos] != modal_value:
                continue
            other_year_like = any(
                i != pos and (
                    int((positions[i] or {}).get("gt_12_count") or 0) / matched >= 0.2
                    or int((positions[i] or {}).get("gt_31_count") or 0) / matched >= 0.03
                )
                for i in range(3)
            )
            if other_year_like:
                return (
                    "date_component_order_candidate: one component position is near-constant "
                    "while another position is year-like; check for cyclic date component transposition"
                )
        return ""

    def date_granularity_warning(self, value: str, props: dict) -> str:
        dtype = str(props.get("dtype", "")).lower()
        semantic_type = str(props.get("semantic_type", "")).lower()
        description = str(props.get("description", "")).lower()
        if dtype != "date" and "date" not in semantic_type and "date" not in description:
            return ""
        text = str(value or "").strip()
        if is_missing_like_value(text):
            return ""
        if not re.fullmatch(r"\d{4}\s*(?:\([^)]*\))?", text):
            return ""
        shape_counts = props.get("shape_counts") or {}
        full_date_count = 0
        year_only_count = 0
        for shape, raw_count in shape_counts.items():
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            shape_text = str(shape)
            if re.search(r"9{1,2}\s+A{3,}\s+9{4}", shape_text):
                full_date_count += count
            elif re.fullmatch(r"9{4}(?:\s+\([^)]*\))?", shape_text):
                year_only_count += count
        if full_date_count >= max(30, year_only_count * 3):
            return (
                "date_granularity_candidate: year-only value is much coarser "
                "than the dominant full-date profile"
            )
        return ""

    def generic_value_warnings(self, value: str, props: dict, fmt: dict) -> list[str]:
        warnings = []
        if is_missing_like_value(value):
            warnings.append("missing_like_candidate")
        if is_free_text_placeholder(value, props):
            warnings.append("placeholder_text_candidate")
        warnings.extend(f"text_noise_candidate:{reason}" for reason in text_noise_reasons(value))
        if has_duplicate_list_items(value):
            warnings.append("list_duplicate_candidate")
        if is_category_variant_candidate(value, props):
            warnings.append("category_variant_candidate")
        if profile_spelling_variant_warning(value, props):
            warnings.append("profile_spelling_variant_candidate")
        if code_skeleton_variant_warning(value, props):
            warnings.append("code_skeleton_variant_candidate")
        if category_code_label_variant_warning(value, props):
            warnings.append("category_code_label_variant_candidate")
        surface_warning = numeric_surface_variant_warning(value, props)
        if surface_warning:
            warnings.append(surface_warning)
        if has_noncanonical_numeric_skeleton(value, props):
            warnings.append("numeric_representation_variant_candidate")
        date_warning = self.date_component_order_warning(value, props)
        if date_warning:
            warnings.append(date_warning)
        granularity_warning = self.date_granularity_warning(value, props)
        if granularity_warning:
            warnings.append(granularity_warning)
        regex = str((fmt or {}).get("regex", "") or "")
        if regex:
            try:
                if (
                        self.summary_regex_is_trustworthy_for_expansion(props, fmt)
                        and re.fullmatch(regex, str(value).strip()) is None
                ):
                    warnings.append("regex_mismatch_candidate")
            except re.error:
                pass
        return warnings

    def make_sample_entry(self, row: int, value: str, props: dict, fmt: dict) -> dict:
        entry = {"row": row, "value": value}
        warnings = self.generic_value_warnings(value, props, fmt)
        if warnings:
            entry["profile_warnings"] = warnings
        return entry

    @staticmethod
    def warning_family(warning: str) -> str:
        if warning.startswith("text_noise_candidate:"):
            return "text_noise_candidate"
        if warning.startswith("numeric_surface_variant_candidate:"):
            return warning
        if warning.startswith("date_component_order_candidate"):
            return "date_component_order_candidate"
        if warning.startswith("date_granularity_candidate"):
            return "date_granularity_candidate"
        return warning

    @staticmethod
    def warning_family_rank(priority: dict, family: str) -> int:
        if family in priority:
            return priority[family]
        if family.startswith("numeric_surface_variant_candidate:"):
            return priority.get("numeric_surface_variant_candidate", 99)
        return 99

    @staticmethod
    def is_confirmable_warning_family(family: str, candidate_families: set[str]) -> bool:
        return family in candidate_families or family.startswith("numeric_surface_variant_candidate:")

    @staticmethod
    def warning_family_confirmed(family: str, confirmed: set[str]) -> bool:
        if family in confirmed:
            return True
        return (
            family.startswith("numeric_surface_variant_candidate:")
            and "numeric_surface_variant_candidate" in confirmed
        )

    @staticmethod
    def warning_error_type(warning_family: str) -> str | None:
        if warning_family == "missing_like_candidate":
            return "missing_errors"
        if warning_family == "placeholder_text_candidate":
            return "missing_errors"
        if warning_family in {
            "text_noise_candidate",
            "date_component_order_candidate",
            "date_granularity_candidate",
            "list_duplicate_candidate",
            "numeric_surface_variant_candidate",
            "numeric_representation_variant_candidate",
            "profile_spelling_variant_candidate",
            "code_skeleton_variant_candidate",
            "category_code_label_variant_candidate",
        }:
            return "format_errors"
        return None

    def allow_regex_mismatch_profile_expansion(self, col: str, props: dict, fmt: dict) -> bool:
        regex = str((fmt or {}).get("regex", "") or "").strip()
        if not regex or self.is_identifier_like_field(col, props):
            return False
        if not self.summary_regex_is_trustworthy_for_expansion(props, fmt):
            return False
        text = f"{col} {props.get('dtype', '')} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if any(token in text for token in ("date", "time", "duration", "timestamp")):
            return False
        structured_tokens = (
            "year", "count", "rating",
            "score", "number", "numeric", "amount", "price", "percent",
            "percentage", "currency",
        )
        return any(token in text for token in structured_tokens)

    @staticmethod
    def summary_regex_is_trustworthy_for_expansion(props: dict, fmt: dict) -> bool:
        """
        Old LLM summaries can contain regexes that reject frequent canonical
        values. Such regex mismatches may still be useful candidates for an LLM
        batch, but they are too weak for automatic expansion.
        """
        return summary_regex_is_trustworthy(props, fmt)

    def warning_error_type_for_column(self, warning_family: str, col: str, props: dict, fmt: dict) -> str | None:
        if warning_family == "regex_mismatch_candidate":
            if self.allow_regex_mismatch_profile_expansion(col, props, fmt):
                return "format_errors"
            return None
        if warning_family == "numeric_representation_variant_candidate":
            return numeric_variant_error_type("", props, fmt)
        if warning_family.startswith("numeric_surface_variant_candidate:"):
            return "format_errors"
        return self.warning_error_type(warning_family)

    def compatible_error_types_for_warning(
            self,
            warning_family: str,
            col: str,
            props: dict,
            fmt: dict,
            value: str,
    ) -> set[str]:
        base_type = self.warning_error_type_for_column(warning_family, col, props, fmt)
        if warning_family == "regex_mismatch_candidate" and not self.summary_regex_is_trustworthy_for_expansion(props, fmt):
            return set()
        if warning_family in {"missing_like_candidate", "placeholder_text_candidate"}:
            return {"missing_errors"}
        if warning_family == "regex_mismatch_candidate" and not base_type:
            return set()
        if warning_family.startswith("numeric_surface_variant_candidate:"):
            warning_family = "numeric_surface_variant_candidate"
        if warning_family in {
            "regex_mismatch_candidate",
            "numeric_representation_variant_candidate",
            "numeric_surface_variant_candidate",
        }:
            compatible = {"format_errors", "spelling_errors", "outliers"}
            if base_type:
                compatible.add(base_type)
            return compatible
        if warning_family in {
            "text_noise_candidate",
            "list_duplicate_candidate",
            "category_variant_candidate",
            "date_component_order_candidate",
            "date_granularity_candidate",
            "profile_spelling_variant_candidate",
            "code_skeleton_variant_candidate",
            "category_code_label_variant_candidate",
        }:
            compatible = {"format_errors", "spelling_errors"}
            if base_type:
                compatible.add(base_type)
            return compatible
        return {base_type} if base_type else set()

    @staticmethod
    def numeric_correct_format_skeletons(errors: list) -> set[str]:
        """
        Extract numeric/unit skeletons from LLM-provided correctFormat examples.
        These skeletons are treated as LLM-indicated safe canonical surfaces, so
        broad numeric-variant expansion should not mark matching candidate cells.
        """
        skeletons = set()
        for err in errors:
            text = str(err.get("correctFormat") or "")
            if not text:
                continue
            quoted = re.findall(r"['\"]([^'\"]*\d[^'\"]*)['\"]", text)
            candidates = quoted or re.findall(
                r"[-+]?\d+(?:\.\d+)?\s*[A-Za-z%][A-Za-z0-9%.\-/ ]{0,30}",
                text,
            )
            for candidate in candidates:
                candidate = re.sub(r"\b(?:or|and|e\.g\.|example|such as)\b.*$", "", str(candidate), flags=re.I)
                candidate = candidate.strip(" .,:;()[]{}\"'")
                if not candidate or not re.search(r"\d", candidate):
                    continue
                skeleton = Summarizer._numeric_text_skeleton(candidate)
                if skeleton and "<num>" in skeleton:
                    skeletons.add(skeleton)
        return skeletons

    def expand_llm_confirmed_profile_warnings(
            self,
            col: str,
            df: pd.DataFrame,
            props: dict,
            fmt: dict,
            indices_to_check: list[int],
            errors: list,
    ) -> list:
        """
        Expand generic profile warnings only after the LLM has confirmed the
        same warning family for this column. This keeps the decision anchored in
        the summary/profile evidence and an LLM diagnosis, while avoiding
        dataset-specific value rules.
        """
        candidate_values = {}
        for idx in indices_to_check:
            if idx not in df.index:
                continue
            value_key = str(df.at[idx, col])
            warnings = self.generic_value_warnings(df.at[idx, col], props, fmt)
            for warning in warnings:
                family = self.warning_family(warning)
                candidate_values.setdefault(family, set()).add(value_key)

        accepted_values = {}
        accepted_type_counts = {}
        accepted_numeric_skeletons = set()
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            error_types = {
                et.strip()
                for et in str(err.get("errorType", "")).split(",")
                if et.strip()
            }
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                if row_idx not in df.index:
                    continue
                value = df.at[row_idx, col]
                raw_numeric_skeleton = Summarizer._numeric_text_skeleton(str(value))
                warnings = self.generic_value_warnings(value, props, fmt)
                for warning in warnings:
                    family = self.warning_family(warning)
                    canonical_type = self.warning_error_type_for_column(family, col, props, fmt)
                    compatible_types = self.compatible_error_types_for_warning(
                        family,
                        col,
                        props,
                        fmt,
                        str(value),
                    )
                    matched_types = sorted(error_types & compatible_types)
                    if not matched_types and canonical_type and error_types and family in {
                        "missing_like_candidate",
                        "placeholder_text_candidate",
                        "text_noise_candidate",
                        "list_duplicate_candidate",
                        "date_component_order_candidate",
                        "date_granularity_candidate",
                        "profile_spelling_variant_candidate",
                        "code_skeleton_variant_candidate",
                        "category_code_label_variant_candidate",
                    }:
                        matched_types = [canonical_type]
                    if matched_types:
                        accepted_values.setdefault(family, set()).add(str(value))
                        type_counts = accepted_type_counts.setdefault(family, {})
                        for etype in matched_types:
                            type_counts[etype] = type_counts.get(etype, 0) + 1
                        if canonical_type:
                            type_counts[canonical_type] = type_counts.get(canonical_type, 0) + 1
                        if (
                                family == "numeric_representation_variant_candidate"
                                or family.startswith("numeric_surface_variant_candidate:")
                        ):
                            accepted_numeric_skeletons.add(raw_numeric_skeleton)

        accepted_families = set()
        for family, values in accepted_values.items():
            count = len(values)
            candidate_total = max(1, len(candidate_values.get(family, set())))
            confirmed_ratio = count / candidate_total
            min_confirmed = 1 if family in {"missing_like_candidate", "placeholder_text_candidate"} else (
                3 if family in BROAD_LLM_CONFIRMED_WARNING_FAMILIES else 1
            )
            if count < min_confirmed:
                continue
            if (
                    family in BROAD_LLM_CONFIRMED_WARNING_FAMILIES
                    and candidate_total > 25
                    and confirmed_ratio < 0.10
            ):
                if family == "date_component_order_candidate" and count >= 1:
                    accepted_families.add(family)
                    continue
                print(
                    f"[Warn] profile warning family '{family}' for '{col}' "
                    f"has low LLM confirmation ratio {confirmed_ratio:.2%}; expansion disabled"
                )
                continue
            accepted_families.add(family)
        if not accepted_families:
            return errors

        existing = set()
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            error_types = {
                et.strip()
                for et in str(err.get("errorType", "")).split(",")
                if et.strip()
            }
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                for etype in error_types:
                    existing.add((row_idx, etype))

        additions = []
        warning_rows_by_family = {family: [] for family in accepted_families}
        protected_numeric_skeletons = self.numeric_correct_format_skeletons(errors)
        for idx in indices_to_check:
            if idx not in df.index:
                continue
            warnings = self.generic_value_warnings(df.at[idx, col], props, fmt)
            for warning in warnings:
                family = self.warning_family(warning)
                if family not in accepted_families:
                    continue
                if (
                        family == "numeric_representation_variant_candidate"
                        or family.startswith("numeric_surface_variant_candidate:")
                ):
                    raw_skeleton = Summarizer._numeric_text_skeleton(str(df.at[idx, col]))
                    if family == "numeric_representation_variant_candidate":
                        normalized_skeleton = profile_normalized_numeric_skeleton(props, str(df.at[idx, col]))
                        dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
                        if normalized_skeleton and normalized_skeleton == dominant_normalized:
                            continue
                    if raw_skeleton not in accepted_numeric_skeletons:
                        continue
                if family in accepted_families:
                    warning_rows_by_family[family].append(int(idx))

        for family, rows in warning_rows_by_family.items():
            max_expansion = 25
            if family == "date_component_order_candidate":
                max_expansion = max(25, int(len(df) * 0.95))
            broad_family = (
                family in BROAD_LLM_CONFIRMED_WARNING_FAMILIES
                or family.startswith("numeric_surface_variant_candidate:")
            )
            if len(set(rows)) > max_expansion and not broad_family:
                print(
                    f"[Warn] profile warning expansion for '{col}'/{family} "
                    f"is too broad ({len(set(rows))} cells); skipped"
                )
                continue
            type_counts = accepted_type_counts.get(family, {})
            if type_counts:
                etype = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            else:
                etype = self.warning_error_type_for_column(family, col, props, fmt)
            if not etype:
                continue
            for idx in sorted(set(rows)):
                if (
                        family == "numeric_representation_variant_candidate"
                        or family.startswith("numeric_surface_variant_candidate:")
                ):
                    raw_skeleton = Summarizer._numeric_text_skeleton(str(df.at[idx, col]))
                    if raw_skeleton in protected_numeric_skeletons:
                        continue
                if (idx, etype) in existing:
                    continue
                additions.append({
                    "row": idx,
                    "fieldName": col,
                    "errorType": etype,
                    "description": f"LLM-confirmed generic profile warning: {family}",
                    "correctFormat": None,
                })
                existing.add((idx, etype))

        if additions:
            print(
                f"Expanded {len(additions)} LLM-confirmed profile-warning cells "
                f"for '{col}'"
            )
        return errors + additions

    def salvage_malformed_minimal_batch(self, response: str, batch: list, col: str) -> list:
        """
        If a one-row LLM response is malformed but visibly contains an allowed
        errorType, keep that LLM decision instead of silently dropping it.
        This repairs serialization failure only and never invents a label.
        """
        if len(batch) != 1:
            return []
        raw = clean_code_snippet(response)
        match = re.search(
            r'"errorType"\s*:\s*"(format_errors|spelling_errors|outliers|missing_errors)"',
            raw,
        )
        if not match:
            return []
        row = batch[0].get("row")
        try:
            row = int(row)
        except (TypeError, ValueError):
            return []
        desc_match = re.search(r'"description"\s*:\s*"([^"\n\r]{0,240})', raw)
        corr_match = re.search(r'"correctFormat"\s*:\s*"([^"\n\r]{0,160})', raw)
        return [{
            "row": row,
            "fieldName": col,
            "errorType": match.group(1),
            "description": desc_match.group(1) if desc_match else "LLM indicated this error type in a malformed JSON response.",
            "correctFormat": corr_match.group(1) if corr_match else None,
        }]

    def should_probe_free_text(self, props, fmt, distinct_count, suspicious_count) -> bool:
        regex = fmt.get("regex", "") if isinstance(fmt, dict) else ""
        if regex:
            return False
        dtype = str(props.get("dtype", "")).lower()
        semantic_type = str(props.get("semantic_type", "")).lower()
        uniqueness = float(props.get("uniqueness_ratio") or 0)
        if dtype in {"number", "int", "float", "date", "boolean", "category"}:
            return False
        if any(token in semantic_type for token in ("id", "identifier", "code", "category", "date", "time")):
            return False
        return suspicious_count >= 500 and distinct_count >= 300 and uniqueness >= 0.5

    def is_identifier_like_field(self, col: str, props: dict) -> bool:
        text = f"{col} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
        tokens = ("id", "identifier", "code", "source", "src", "key", "issn", "isbn", "doi")
        return any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) for token in tokens)

    def effective_format_rule_for_detection(self, props: dict, fmt: dict) -> dict:
        """
        Use a summary regex as an LLM prompt constraint only when the compact
        distribution and shape evidence supports it. Weak regexes still leave
        the cell for explicit LLM review, but they are not presented as a hard
        canonical format.
        """
        if not isinstance(fmt, dict):
            return {"regex": "", "explanation": ""}
        regex = str(fmt.get("regex", "") or "").strip()
        if not regex or self.summary_regex_is_trustworthy_for_expansion(props, fmt):
            return fmt
        effective = dict(fmt)
        effective["regex"] = ""
        explanation = str(effective.get("explanation", "") or "")
        suffix = " Regex omitted from the detection prompt because compact distribution evidence does not support it as a stable canonical rule."
        effective["explanation"] = (explanation + suffix).strip()
        return effective

    def initial_batch_limit_for_field(self, col: str, props: dict, samples: list) -> int | None:
        """
        Short identifier/code-like values often cause the LLM to emit long,
        malformed JSON when too many distinct candidates are sent together.
        Start those columns with smaller batches; this is a generic API
        stability guard and does not alter the detection criteria.
        """
        if not samples:
            return None
        text = f"{col} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if self.is_identifier_like_field(col, props):
            return 10
        if any(token in text for token in ("pagination", "page", "pages")):
            return 20
        if field_text_has_token(text, ("time", "date", "datetime", "timestamp")) and len(samples) > 30:
            return 15
        avg_len = sum(len(str(item.get("value", ""))) for item in samples) / max(len(samples), 1)
        if len(samples) > 100:
            return 40
        if avg_len <= 18 and len(samples) > 60:
            return 30
        return None

    def should_suppress_low_evidence_identifier_error(self, col, df, props, fmt, err, row) -> bool:
        """
        For identifier/code/source-like fields, do not accept an LLM label that is
        supported only by rarity when the value follows a well-supported column
        shape. This is a generic precision guard; missing values and clear
        low-support structural anomalies still pass through.
        """
        if not self.is_identifier_like_field(col, props):
            return False
        error_types = {
            et.strip()
            for et in str(err.get("errorType", "")).split(",")
            if et.strip()
        }
        if not error_types or error_types - {"format_errors", "spelling_errors", "outliers"}:
            return False
        try:
            value = df.at[int(row), col]
        except Exception:
            return False
        text = "" if pd.isna(value) else str(value).strip()
        if text.lower() in MISSING_LIKE_VALUES:
            return False
        regex = str((fmt or {}).get("regex", "") or "")
        if regex:
            try:
                if re.fullmatch(regex, text) is None:
                    return False
            except re.error:
                pass
        shape = Summarizer._value_shape(text)
        try:
            shape_count = int((props.get("shape_counts") or {}).get(shape) or 0)
        except (TypeError, ValueError):
            shape_count = 0
        support_threshold = max(8, min(50, int(max(len(df), 1) * 0.01)))
        return shape_count >= support_threshold

    def filter_low_evidence_identifier_errors(self, col, df, props, fmt, errors):
        filtered = []
        suppressed = 0
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            kept_rows = []
            for row in rows:
                if self.should_suppress_low_evidence_identifier_error(col, df, props, fmt, err, row):
                    suppressed += 1
                else:
                    kept_rows.append(row)
            if not kept_rows:
                continue
            new_err = dict(err)
            new_err["row"] = kept_rows if len(kept_rows) > 1 else kept_rows[0]
            filtered.append(new_err)
        if suppressed:
            print(f"Suppressed {suppressed} low-evidence identifier/code-like errors in '{col}'")
        return filtered

    @staticmethod
    def _extract_numeric_parts(value) -> list[float]:
        text = "" if pd.isna(value) else str(value)
        nums = []
        for match in re.findall(r"-?\d+(?:\.\d+)?", text):
            try:
                nums.append(float(match))
            except ValueError:
                continue
        return nums

    def should_suppress_summary_unsupported_outlier(self, col, df, props, err, row) -> bool:
        """
        Keep LLM outlier labels only when the LLM claim is supported by the
        summary's numeric distribution profile. This prevents normal values near
        the center of a numeric/unit distribution from being accepted merely
        because the LLM called them "extreme".
        """
        error_types = {
            et.strip()
            for et in str(err.get("errorType", "")).split(",")
            if et.strip()
        }
        if error_types != {"outliers"}:
            return False

        profiles = props.get("numeric_value_profiles") or {}
        if not isinstance(profiles, dict) or not profiles:
            return False

        try:
            value = df.at[int(row), col]
        except Exception:
            return False
        text = "" if pd.isna(value) else str(value).strip()
        if not text or text.lower() in MISSING_LIKE_VALUES:
            return False

        skeletons = [
            profile_normalized_numeric_skeleton(props, text),
            Summarizer._normalized_numeric_text_skeleton(text),
            Summarizer._numeric_text_skeleton(text),
        ]
        profile = None
        for skeleton in skeletons:
            if skeleton and skeleton in profiles:
                profile = profiles.get(skeleton)
                break
        if not isinstance(profile, dict):
            return False

        nums = self._extract_numeric_parts(text)
        if not nums:
            return False

        try:
            low = float(profile.get("p05", profile.get("min")))
            high = float(profile.get("p95", profile.get("max")))
        except (TypeError, ValueError):
            return False
        if low > high:
            low, high = high, low

        return all(low <= num <= high for num in nums)

    def filter_summary_unsupported_outliers(self, col, df, props, errors):
        filtered = []
        suppressed = 0
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            kept_rows = []
            for row in rows:
                if self.should_suppress_summary_unsupported_outlier(col, df, props, err, row):
                    suppressed += 1
                else:
                    kept_rows.append(row)
            if not kept_rows:
                continue
            new_err = dict(err)
            new_err["row"] = kept_rows if len(kept_rows) > 1 else kept_rows[0]
            filtered.append(new_err)
        if suppressed:
            print(f"Suppressed {suppressed} summary-unsupported outlier labels in '{col}'")
        return filtered

    def normalize_single_column_error_types(self, col, errors):
        normalized = []
        dropped = 0
        stripped = 0
        for err in errors:
            error_types = [
                et.strip()
                for et in str(err.get("errorType", "")).split(",")
                if et.strip()
            ]
            allowed = [et for et in error_types if et in SINGLE_COLUMN_ERROR_TYPES]
            if not allowed:
                dropped += count_cells([err])
                continue
            if allowed != error_types:
                stripped += count_cells([err])
            new_err = dict(err)
            new_err["errorType"] = ",".join(allowed)
            normalized.append(new_err)
        if dropped:
            print(f"Dropped {dropped} single-column labels with unsupported error types in '{col}'")
        if stripped:
            print(f"Stripped unsupported single-column error type tags from {stripped} cells in '{col}'")
        return normalized

    def filter_llm_indicated_numeric_canonical_errors(self, col, df, props, errors):
        """
        If the LLM supplies a numeric/unit correctFormat example, treat matching
        raw skeletons as LLM-indicated canonical surfaces. Do not keep
        format/spelling labels for those surfaces unless the same cell is also
        missing or a numeric outlier.
        """
        if not is_numeric_measure_field(props):
            return errors
        protected_skeletons = self.numeric_correct_format_skeletons(errors)
        dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        if dominant_normalized:
            protected_skeletons = {
                skeleton
                for skeleton in protected_skeletons
                if profile_normalized_numeric_skeleton(props, skeleton) == dominant_normalized
            }
        if not protected_skeletons:
            return errors

        filtered = []
        suppressed = 0
        for err in errors:
            error_types = {
                et.strip()
                for et in str(err.get("errorType", "")).split(",")
                if et.strip()
            }
            if not error_types or error_types & {"missing_errors", "outliers"}:
                filtered.append(err)
                continue
            if not error_types <= {"format_errors", "spelling_errors"}:
                filtered.append(err)
                continue

            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            kept_rows = []
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    kept_rows.append(row)
                    continue
                if row_idx not in df.index:
                    kept_rows.append(row)
                    continue
                raw_skeleton = Summarizer._numeric_text_skeleton(str(df.at[row_idx, col]))
                if raw_skeleton in protected_skeletons:
                    suppressed += 1
                    continue
                kept_rows.append(row)

            if not kept_rows:
                continue
            new_err = dict(err)
            new_err["row"] = kept_rows if len(kept_rows) > 1 else kept_rows[0]
            filtered.append(new_err)

        if suppressed:
            print(
                f"Suppressed {suppressed} LLM-indicated canonical numeric/unit "
                f"surface labels in '{col}'"
            )
        return filtered

    def filter_summary_supported_text_canonical_errors(self, col, df, props, errors):
        """
        Suppress LLM format/spelling labels on high-support textual/category
        surfaces already represented as canonical by the summary profile. This
        does not protect rare variants; it only prevents broad LLM wording from
        swallowing the dominant observed surface in the same column.
        """
        filtered = []
        suppressed = 0
        for err in errors:
            error_types = {
                et.strip()
                for et in str(err.get("errorType", "")).split(",")
                if et.strip()
            }
            if not error_types or not error_types <= {"format_errors", "spelling_errors", "outliers"}:
                filtered.append(err)
                continue
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            kept_rows = []
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    kept_rows.append(row)
                    continue
                if row_idx not in df.index:
                    kept_rows.append(row)
                    continue
                if text_value_is_summary_supported_canonical(props, df.at[row_idx, col]):
                    suppressed += 1
                    continue
                kept_rows.append(row)
            if not kept_rows:
                continue
            new_err = dict(err)
            new_err["row"] = kept_rows if len(kept_rows) > 1 else kept_rows[0]
            filtered.append(new_err)
        if suppressed:
            print(f"Suppressed {suppressed} summary-supported canonical text labels in '{col}'")
        return filtered

    @staticmethod
    def numeric_skeleton_is_summary_supported(props: dict, value) -> bool:
        if not is_numeric_measure_field(props):
            return False
        text = "" if value is None else str(value).strip()
        if not text or is_missing_like_value(text) or not re.search(r"\d", text):
            return False
        raw_skeleton = Summarizer._numeric_text_skeleton(text)
        canonical = canonical_numeric_surface_skeleton(props)
        if canonical and raw_skeleton != canonical:
            return False
        normalized = profile_normalized_numeric_skeleton(props, text)
        dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        if dominant_normalized and normalized == dominant_normalized:
            return True
        if dominant_normalized and normalized and normalized != dominant_normalized:
            return False
        skeleton_counts = props.get("numeric_text_skeleton_counts") or {}
        normalized_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        return raw_skeleton in skeleton_counts or normalized in normalized_counts

    @staticmethod
    def is_textual_near_duplicate_field(props: dict) -> bool:
        dtype = str(props.get("dtype", "")).lower()
        semantic_type = str(props.get("semantic_type", "")).lower()
        description = str(props.get("description", "")).lower()
        text = f"{semantic_type} {description}"
        if dtype in {"number", "int", "float", "date", "boolean"}:
            return False
        blocked = ("id", "identifier", "code", "score", "amount", "sample")
        if any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) for token in blocked):
            return False
        return True

    @staticmethod
    def _near_duplicate_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def near_duplicate_text_values(self, series: pd.Series, props: dict) -> set[str]:
        """
        Find low-support textual values that are very close to well-supported
        values in the same dirty column. This is only candidate selection for
        the LLM; it is not a final error rule.
        """
        if not self.is_textual_near_duplicate_field(props):
            return set()

        normalized = series.astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip().lower()))
        counts = normalized.value_counts(dropna=False)
        n = max(int(counts.sum()), 1)
        frequent_threshold = max(4, int(n * 0.01))
        rare_threshold = max(2, int(n * 0.006))

        frequent_values = [
            value for value, count in counts.items()
            if count >= frequent_threshold
            and value not in MISSING_LIKE_VALUES
            and len(value) >= 4
        ][:80]
        if not frequent_values:
            return set()

        candidates = set()
        for value, count in counts.items():
            if count > rare_threshold or value in MISSING_LIKE_VALUES or len(value) < 4:
                continue
            for canonical in frequent_values:
                if value == canonical:
                    continue
                length = max(len(value), len(canonical))
                ratio = self._near_duplicate_ratio(value, canonical)
                threshold = 0.78 if length <= 12 else 0.88
                if ratio >= threshold:
                    candidates.add(value)
                    break
        return candidates

    def select_regex_profile_probe_indices(self, col, df, props, fmt, suspicious_idx, max_values: int = 160):
        series = df.loc[suspicious_idx, col].astype(str)
        value_to_rows = OrderedDict()
        family_rank = {}
        priority = {
            "missing_like_candidate": 0,
            "placeholder_text_candidate": 1,
            "date_granularity_candidate": 2,
            "date_component_order_candidate": 3,
            "text_noise_candidate": 4,
            "list_duplicate_candidate": 5,
            "profile_spelling_variant_candidate": 6,
            "code_skeleton_variant_candidate": 7,
            "category_code_label_variant_candidate": 8,
            "category_variant_candidate": 9,
            "numeric_surface_variant_candidate": 10,
            "numeric_representation_variant_candidate": 11,
            "regex_mismatch_candidate": 12,
        }
        useful_families = set(priority)
        for row_idx, value in series.items():
            warnings = self.generic_value_warnings(value, props, fmt)
            if not warnings:
                continue
            families = {self.warning_family(warning) for warning in warnings}
            if not any(self.warning_family_rank(priority, family) < 99 for family in families):
                continue
            value_to_rows.setdefault(str(value), []).append(int(row_idx))
            family_rank[str(value)] = min(self.warning_family_rank(priority, family) for family in families)

        if len(value_to_rows) <= max_values:
            return [rows[0] for rows in value_to_rows.values() if rows]

        ranked = sorted(
            value_to_rows.items(),
            key=lambda item: (family_rank.get(item[0], 99), -len(item[1]), item[1][0], item[0]),
        )
        selected = [rows[0] for _, rows in ranked[:max_values] if rows]
        print(
            f"Structured regex-profile probe kept {len(selected)}/{len(value_to_rows)} "
            f"distinct regex-mismatch values for '{col}'"
        )
        return selected

    def select_profile_warning_probe_indices(self, col, df, props, fmt, candidate_idx, max_values: int = 160):
        value_to_rows = OrderedDict()
        family_rank = {}
        priority = {
            "missing_like_candidate": 0,
            "text_noise_candidate": 1,
            "list_duplicate_candidate": 2,
            "placeholder_text_candidate": 3,
            "date_granularity_candidate": 4,
            "date_component_order_candidate": 5,
            "profile_spelling_variant_candidate": 6,
            "code_skeleton_variant_candidate": 7,
            "category_code_label_variant_candidate": 8,
            "category_variant_candidate": 9,
            "numeric_surface_variant_candidate": 10,
            "numeric_representation_variant_candidate": 11,
            "regex_mismatch_candidate": 12,
        }
        for row_idx in candidate_idx:
            if row_idx not in df.index:
                continue
            value = str(df.at[row_idx, col])
            warnings = self.generic_value_warnings(value, props, fmt)
            if not warnings:
                continue
            families = {self.warning_family(warning) for warning in warnings}
            value_to_rows.setdefault(value, []).append(int(row_idx))
            family_rank[value] = min(self.warning_family_rank(priority, family) for family in families)

        if len(value_to_rows) <= max_values:
            return [rows[0] for rows in value_to_rows.values() if rows]

        ranked = sorted(
            value_to_rows.items(),
            key=lambda item: (family_rank.get(item[0], 99), -len(item[1]), item[1][0], item[0]),
        )
        selected = [rows[0] for _, rows in ranked[:max_values] if rows]
        print(
            f"Generic profile-warning probe kept {len(selected)}/{len(value_to_rows)} "
            f"distinct warning values for '{col}'"
        )
        return selected

    def confirm_profile_warning_errors_with_llm(self, col, props, fmt, samples, representative_rows):
        """
        Confirm generic profile-warning families with a compact LLM request that
        avoids sending long free-text values verbatim. This is still an LLM
        diagnosis step; it only changes the evidence representation for values
        whose visible issue is already captured by data_summary/profile warning
        metadata.
        """
        candidate_families = {
            "missing_like_candidate",
            "placeholder_text_candidate",
            "text_noise_candidate",
            "list_duplicate_candidate",
            "date_component_order_candidate",
            "date_granularity_candidate",
            "category_variant_candidate",
            "numeric_surface_variant_candidate",
            "numeric_representation_variant_candidate",
            "profile_spelling_variant_candidate",
            "code_skeleton_variant_candidate",
            "category_code_label_variant_candidate",
            "regex_mismatch_candidate",
        }
        examples_by_family = OrderedDict()
        for sample in samples:
            warnings = sample.get("profile_warnings") or []
            families = {
                self.warning_family(warning)
                for warning in warnings
                if self.is_confirmable_warning_family(self.warning_family(warning), candidate_families)
            }
            if not families:
                continue
            text = str(sample.get("value", ""))
            compact_example = {
                "row": sample.get("row"),
                "value_length": len(text),
                "starts_with_missing_like": is_missing_like_value(text),
                "warnings": list(warnings),
                "suffix": text[-24:] if len(text) <= 80 else text[-24:],
            }
            if has_duplicate_list_items(text):
                parts = [
                    re.sub(r"\s+", " ", part.strip().lower())
                    for part in re.split(r"[,;|]", text)
                    if part and part.strip()
                ]
                compact_example["list_item_count"] = len(parts)
                compact_example["duplicate_item_count"] = len(parts) - len(set(parts))
            for family in families:
                examples_by_family.setdefault(family, []).append(compact_example)

        if not examples_by_family:
            return {"errors": [], "error_regex_by_type_list": []}

        total_examples = sum(len(items) for items in examples_by_family.values())
        compact_confirmable_small_families = {
            "list_duplicate_candidate",
            "numeric_surface_variant_candidate",
            "numeric_representation_variant_candidate",
            "regex_mismatch_candidate",
            "missing_like_candidate",
            "placeholder_text_candidate",
        }
        if (
                total_examples < 10
                and not any(family in examples_by_family for family in compact_confirmable_small_families)
                and not any(
                    str(family).startswith("numeric_surface_variant_candidate:")
                    for family in examples_by_family
                )
        ):
            return {"errors": [], "error_regex_by_type_list": []}

        compact_examples = {
            family: items[:12]
            for family, items in examples_by_family.items()
        }
        prompt = f"""
You are confirming generic data-quality warning families for one table column.
The candidate cells were selected from data_summary/profile evidence. Long raw
free text is not shown; only structural warning metadata is shown. Decide which
warning families should be treated as real single-column errors for this column.

Column: {col}
dtype: {props.get("dtype", "")}
semantic_type: {props.get("semantic_type", "")}
description: {props.get("description", "")}
format_rule: {json.dumps(fmt or {}, ensure_ascii=False)}
column_profile: {json.dumps(self.column_profile_for_prompt(props), ensure_ascii=False)}

Warning family definitions:
- missing_like_candidate: the visible value is a null or semantic missing placeholder.
- placeholder_text_candidate: a short free-text value is a placeholder asking for content or saying content is unknown.
- text_noise_candidate: the visible value has structural corruption such as replacement characters, control characters, embedded null tokens, repeated whitespace, or truncation ellipsis.
- list_duplicate_candidate: a comma/semicolon/pipe-separated list repeats the same normalized item inside the same cell.
- date_component_order_candidate: compact date components conflict with the column-level dominant component profile.
- date_granularity_candidate: the value has much coarser date granularity than the dominant date profile.
- category_variant_candidate: a categorical/list-like value uses an alternate surface taxonomy or compound label relative to the column profile.
- profile_spelling_variant_candidate: a low-support value is character-level similar to a high-support value in the same column.
- code_skeleton_variant_candidate: an identifier/code value has a low-support skeleton that is close to a high-support skeleton.
- category_code_label_variant_candidate: most observed values are short category codes, while this candidate uses a long label surface.
- numeric_surface_variant_candidate:<raw_skeleton>: the value shares the dominant normalized numeric/unit skeleton but uses a non-canonical surface spelling, punctuation, case, or unit word.
- numeric_representation_variant_candidate: the value has a numeric/unit/text skeleton that differs from the dominant summary profile, such as an extra unit marker, percent sign, alternate unit spelling, or appended descriptor.
- regex_mismatch_candidate: the value fails the LLM-generated summary regex for a structured numeric/date/time/code field.

Compact warning examples:
{json.dumps(compact_examples, ensure_ascii=False, indent=2)}

Return exactly this JSON object:
{{
  "confirmed_warning_families": ["family_name"],
  "rationale": "brief evidence-based reason"
}}
"""
        try:
            resp = text_gen.send_message(
                [{"role": "system", "content": SYSTEM_INSTRUCTIONS}, {"role": "user", "content": prompt}],
                max_tokens=text_gen.max_tokens,
                retries=DETECTION_REQUEST_RETRIES,
                request_timeout=DETECTION_REQUEST_TIMEOUT,
            )
            parsed = parse_llm_json_response(resp)
        except Exception as exc:
            print(f"[Warn] compact profile-warning LLM confirmation failed for '{col}': {exc}")
            return {"errors": [], "error_regex_by_type_list": []}

        confirmed = parsed.get("confirmed_warning_families", [])
        if isinstance(confirmed, str):
            confirmed = [confirmed]
        confirmed = {
            self.warning_family(str(family))
            for family in confirmed
            if self.is_confirmable_warning_family(self.warning_family(str(family)), candidate_families)
        }
        if not confirmed:
            return {"errors": [], "error_regex_by_type_list": []}

        errors = []
        seen = set()
        for sample in samples:
            row = sample.get("row")
            warnings = sample.get("profile_warnings") or []
            families = {self.warning_family(warning) for warning in warnings}
            matched = sorted(family for family in families if self.warning_family_confirmed(family, confirmed))
            for family in matched:
                etype = self.warning_error_type_for_column(family, col, props, fmt)
                if (
                        family == "numeric_representation_variant_candidate"
                        or family.startswith("numeric_surface_variant_candidate:")
                ):
                    etype = numeric_variant_error_type(str(sample.get("value", "")), props, fmt)
                if not etype:
                    continue
                key = (row, etype, family)
                if key in seen:
                    continue
                seen.add(key)
                errors.append({
                    "row": row,
                    "fieldName": col,
                    "errorType": etype,
                    "description": f"LLM-confirmed compact profile warning: {family}",
                    "correctFormat": None,
                })

        if errors:
            print(
                f"Compact LLM profile-warning confirmation accepted "
                f"{len(errors)} representative warnings for '{col}'"
            )
        errors = self.suppress_broad_compact_warning_expansion(
            col=col,
            errors=errors,
            samples=samples,
            representative_rows=representative_rows,
        )
        confirmed_representative_rows = set()
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            for row in rows:
                try:
                    confirmed_representative_rows.add(int(row))
                except (TypeError, ValueError):
                    continue
        return {
            "errors": errors,
            "error_regex_by_type_list": [],
            "confirmed_representative_rows": sorted(confirmed_representative_rows),
        }

    def suppress_broad_compact_warning_expansion(
            self,
            col: str,
            errors: list,
            samples: list,
            representative_rows: dict[int, list[int]],
    ) -> list:
        """
        Compact warning confirmation asks the LLM about representative values.
        Only expand those representatives to duplicate rows when the expansion
        footprint is narrow. A broad footprint means the warning is a candidate
        family, not enough evidence to mark many cells without deeper LLM review.
        """
        candidate_values = {}
        for sample in samples:
            value_key = str(sample.get("value", ""))
            for warning in sample.get("profile_warnings") or []:
                family = self.warning_family(warning)
                candidate_values.setdefault(family, set()).add(value_key)

        confirmed_values = {}
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                sample = next((item for item in samples if item.get("row") == row_idx), {})
                value_key = str(sample.get("value", ""))
                for warning in sample.get("profile_warnings") or []:
                    family = self.warning_family(warning)
                    confirmed_values.setdefault(family, set()).add(value_key)

        filtered = []
        suppressed = 0
        for err in errors:
            row_spec = err.get("row")
            rows = row_spec if isinstance(row_spec, list) else [row_spec]
            expanded_count = 0
            families = set()
            for row in rows:
                try:
                    row_idx = int(row)
                except (TypeError, ValueError):
                    continue
                expanded_count += len(representative_rows.get(row_idx, [row_idx]))
                sample = next((item for item in samples if item.get("row") == row_idx), {})
                families.update(self.warning_family(warning) for warning in sample.get("profile_warnings") or [])
            broad_confirmed = False
            for family in families:
                if family not in BROAD_LLM_CONFIRMED_WARNING_FAMILIES:
                    continue
                total = max(1, len(candidate_values.get(family, set())))
                confirmed = len(confirmed_values.get(family, set()))
                if family in {"missing_like_candidate", "placeholder_text_candidate"} and confirmed >= 1:
                    broad_confirmed = True
                    break
                min_confirmed = 1 if family in {"missing_like_candidate", "placeholder_text_candidate"} else 3
                if confirmed >= min_confirmed and confirmed / total >= 0.10:
                    broad_confirmed = True
                    break
            numeric_variant_family = any(
                family == "numeric_representation_variant_candidate"
                or family.startswith("numeric_surface_variant_candidate")
                for family in families
            )
            if expanded_count > 25 and (numeric_variant_family or not broad_confirmed):
                suppressed += expanded_count
                continue
            filtered.append(err)
        if suppressed:
            print(
                f"Suppressed broad compact profile-warning expansion for '{col}' "
                f"({suppressed} cells); leaving those candidates for explicit LLM batches."
            )
        return filtered

    @staticmethod
    def compact_prompt_value(value, max_chars: int = 160) -> str:
        text = "" if pd.isna(value) else str(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...[truncated]"

    def relationship_candidate_entries(
            self,
            candidates: set[tuple[int, str]],
            summary: dict,
            df: pd.DataFrame,
            field_props: dict,
            candidate_sources: dict[tuple[int, str], set[str]],
    ) -> list[dict]:
        rel_cols = sorted(relationship_columns(summary.get("field_relationships", {}), df.columns))
        if not rel_cols:
            rel_cols = list(df.columns[:12])

        entries = []
        for row_idx, col in sorted(candidates):
            if row_idx not in df.index or col not in df.columns:
                continue
            context_cols = list(dict.fromkeys([*rel_cols, col]))
            row_context = {
                context_col: self.compact_prompt_value(df.at[row_idx, context_col])
                for context_col in context_cols
                if context_col in df.columns
            }
            props = field_props.get(col, {}) or {}
            entries.append({
                "row": int(row_idx),
                "fieldName": str(col),
                "value": self.compact_prompt_value(df.at[row_idx, col]),
                "candidate_sources": sorted(candidate_sources.get((int(row_idx), str(col)), set())),
                "field_semantics": {
                    "dtype": props.get("dtype", ""),
                    "semantic_type": props.get("semantic_type", ""),
                    "description": props.get("description", ""),
                },
                "row_relationship_context": row_context,
            })
        return entries

    def relationship_representative_entries(
            self,
            candidates: set[tuple[int, str]],
            summary: dict,
            df: pd.DataFrame,
            field_props: dict,
            candidate_sources: dict[tuple[int, str], set[str]],
    ) -> tuple[list[dict], dict[tuple, list[tuple[int, str]]]]:
        relationships = summary.get("field_relationships", {}) or {}
        rel_cols = sorted(relationship_columns(relationships, df.columns))
        pairs = summary_associative_evidence_pairs(summary)
        evidence_map = summary_associative_evidence_map(summary)
        relationship_fields_by_key = {}
        associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
        if isinstance(associative, dict):
            for key, deps in associative.items():
                if isinstance(deps, list):
                    relationship_fields_by_key[str(key)] = [str(dep) for dep in deps if str(dep) in df.columns]
        grouped_entries = OrderedDict()
        representative_cells = {}
        for key_col, dep_col in sorted(pairs):
            if key_col not in df.columns or dep_col not in df.columns:
                continue
            grouped = {}
            for idx, row in df[[key_col, dep_col]].iterrows():
                key_value = _clean_relationship_value(row[key_col])
                dep_value = _clean_relationship_value(row[dep_col])
                if _is_missing_like_relationship_value(key_value) or _is_missing_like_relationship_value(dep_value):
                    continue
                grouped.setdefault(key_value, []).append((int(idx), dep_value))
            for key_value, rows in grouped.items():
                counts = {}
                for _, dep_value in rows:
                    counts[dep_value] = counts.get(dep_value, 0) + 1
                if len(counts) <= 1:
                    continue
                majority_value, majority_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
                pair_evidence = evidence_map.get((key_col, dep_col), {}) or {}
                consistency_ratio = float(pair_evidence.get("consistency_ratio") or (majority_count / len(rows)))
                top_alternatives = [
                    {
                        "value": self.compact_prompt_value(value, max_chars=80),
                        "count": int(value_count),
                    }
                    for value, value_count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
                ]
                for dep_value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                    cell_rows = [
                        row_idx
                        for row_idx, value in rows
                        if value == dep_value and (int(row_idx), dep_col) in candidates
                    ]
                    if not cell_rows:
                        continue
                    rep_key = (key_col, dep_col, key_value, dep_value)
                    representative_cells[rep_key] = [(int(row_idx), dep_col) for row_idx in sorted(cell_rows)]
                    sample_rows = sorted(cell_rows)[:2]
                    context_cols = list(dict.fromkeys([key_col, dep_col, *rel_cols[:6]]))
                    group_df = df.loc[[row_idx for row_idx, _ in rows]]
                    context_fields = [
                        field for field in relationship_fields_by_key.get(key_col, [])
                        if field in group_df.columns and field != dep_col
                    ][:6]
                    context_modes = {}
                    candidate_context_support = {}
                    for field in context_fields:
                        values = [
                            _clean_relationship_value(value)
                            for value in group_df[field].tolist()
                            if not _is_missing_like_relationship_value(_clean_relationship_value(value))
                        ]
                        field_counts = {}
                        for value in values:
                            field_counts[value] = field_counts.get(value, 0) + 1
                        if field_counts:
                            mode_value, mode_count = sorted(field_counts.items(), key=lambda item: (-item[1], item[0]))[0]
                            context_modes[field] = {
                                "mode": self.compact_prompt_value(mode_value, max_chars=80),
                                "count": int(mode_count),
                            }
                    for value, _value_count in counts.items():
                        support = 0
                        rows_for_value = [
                            row_idx for row_idx, row_value in rows
                            if row_value == value and int(row_idx) in df.index
                        ]
                        for row_idx in rows_for_value:
                            for field, mode_info in context_modes.items():
                                if _clean_relationship_value(df.at[row_idx, field]) == mode_info.get("mode"):
                                    support += 1
                        candidate_context_support[value] = {
                            "total": int(support),
                            "rows": int(len(rows_for_value)),
                            "average": round(support / max(1, len(rows_for_value)), 4),
                        }
                    best_context_value = ""
                    if candidate_context_support:
                        best_context_value = sorted(
                            candidate_context_support.items(),
                            key=lambda item: (
                                -float((item[1] or {}).get("average") or 0),
                                -int((item[1] or {}).get("total") or 0),
                                item[0],
                            ),
                        )[0][0]
                    grouped_entries[rep_key] = {
                        "representative_id": len(grouped_entries),
                        "key_field": key_col,
                        "key_value": self.compact_prompt_value(key_value, max_chars=80),
                        "candidate_field": dep_col,
                        "candidate_value": self.compact_prompt_value(dep_value, max_chars=80),
                        "candidate_count": int(count),
                        "majority_value": self.compact_prompt_value(majority_value, max_chars=80),
                        "majority_count": int(majority_count),
                        "same_key_value_distribution": top_alternatives,
                        "pair_consistency_ratio": round(consistency_ratio, 4),
                        "pair_conflicting_cells": int(pair_evidence.get("conflicting_cells") or 0),
                        "context_modes_for_same_key": context_modes,
                        "candidate_context_support": candidate_context_support.get(
                            dep_value,
                            {"total": 0, "rows": int(count), "average": 0.0},
                        ),
                        "best_context_supported_value": self.compact_prompt_value(best_context_value, max_chars=80),
                        "group_size": int(len(rows)),
                        "sources": sorted(
                            set().union(
                                *[
                                    candidate_sources.get((int(row_idx), dep_col), set())
                                    for row_idx in cell_rows[:20]
                                ]
                            )
                        ),
                        "candidate_field_profile": {
                            "dtype": (field_props.get(dep_col, {}) or {}).get("dtype", ""),
                            "semantic_type": (field_props.get(dep_col, {}) or {}).get("semantic_type", ""),
                            "description": self.compact_prompt_value(
                                (field_props.get(dep_col, {}) or {}).get("description", ""),
                                max_chars=120,
                            ),
                        },
                        "sample_context": [
                            {
                                context_col: self.compact_prompt_value(df.at[row_idx, context_col], max_chars=80)
                                for context_col in context_cols
                                if context_col in df.columns
                            }
                            for row_idx in sample_rows
                        ],
                    }
        return list(grouped_entries.values()), representative_cells

    def confirm_relationship_representatives_with_llm(
            self,
            representative_entries: list[dict],
            representative_cells: dict[tuple, list[tuple[int, str]]],
            summary: dict,
            field_props: dict,
            format_rules: dict,
    ) -> list[dict]:
        rep_id_to_key = {
            int(entry["representative_id"]): key
            for key, entry in zip(representative_cells.keys(), representative_entries)
        }
        relationships = summary.get("field_relationships", {}) or {}
        compact_relationships = {}
        pairs = {
            (entry.get("key_field"), entry.get("candidate_field"))
            for entry in representative_entries
        }
        associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
        if isinstance(associative, dict):
            compact_relationships["associative"] = {
                key: [
                    dep
                    for dep in deps
                    if (key, dep) in pairs
                ]
                for key, deps in associative.items()
                if isinstance(deps, list) and any((key, dep) in pairs for dep in deps)
            }
        temporal = relationships.get("temporal", []) if isinstance(relationships, dict) else []
        if isinstance(temporal, list):
            compact_relationships["temporal"] = [
                seq for seq in temporal
                if isinstance(seq, list)
                and any(field in {pair_item for pair in pairs for pair_item in pair} for field in seq)
            ][:5]
        compact_field_props = {
            col: {
                key: self.compact_prompt_value(value, max_chars=140) if isinstance(value, str) else value
                for key, value in (field_props.get(col, {}) or {}).items()
                if key in {"dtype", "semantic_type", "description"}
            }
            for col in field_props
            if col in {entry.get("candidate_field") for entry in representative_entries}
        }
        compact_format_rules = {}
        for col in compact_field_props:
            rule = format_rules.get(col, {})
            if not isinstance(rule, dict):
                continue
            compact_format_rules[col] = {
                key: self.compact_prompt_value(value, max_chars=180)
                for key, value in rule.items()
                if key in {"regex", "description", "format_description"}
            }
        system_prompt = (
            "You are the final LAED relationship-error judge. The input contains "
            "representative groups of relationship-candidate cells. Confirm a "
            "representative when the candidate value clearly contradicts the "
            "same-key relationship evidence, including the value distribution "
            "and row context. In low-consistency groups, do not trust a simple "
            "majority blindly; use the supplied multi-field context and "
            "plausible relationship-consistent alternatives. Omit ambiguous groups. "
            "Use only the supplied dirty-data evidence and summary metadata. "
            "Return compact JSON only."
        )
        base_prompt = f"""
Relevant summary relationships:
{json.dumps(compact_relationships, ensure_ascii=False, separators=(',', ':'))}

Candidate-field metadata:
{json.dumps(compact_field_props, ensure_ascii=False, separators=(',', ':'))}

Candidate-field format rules:
{json.dumps(compact_format_rules, ensure_ascii=False, separators=(',', ':'))}

For each representative group, decide whether all cells in that group should be
final logical_errors cells. A representative can be confirmed when its value is
incompatible with the same-key record pattern, even if that wrong value is a
large local cluster; use same_key_value_distribution, pair_consistency_ratio,
context_modes_for_same_key, candidate_context_support,
best_context_supported_value, and sample_context as evidence. When
best_context_supported_value is non-empty and differs from candidate_value,
prefer confirming the candidate if sample_context agrees with the context modes.
Return confirmed representative_id values only. The system will expand a
confirmed representative only to the same candidate value inside the same
summary-derived suspicious relationship group.

Return exactly this JSON object:
{{
  "confirmed_representatives": [
    {{
      "representative_id": <integer>,
      "description": "<brief relationship contradiction>",
      "correctFormat": "<expected relationship-consistent value or null>"
    }}
  ]
}}
"""
        confirmed = []
        idx = 0
        batch_limit = max(1, RELATIONSHIP_REPRESENTATIVE_BATCH_LIMIT)
        failures = 0
        print(
            f"Confirming {len(representative_entries)} relationship representative "
            f"groups with LLM (batch_limit={batch_limit})"
        )
        while idx < len(representative_entries):
            batch = representative_entries[idx: idx + batch_limit]
            print(
                f"→ Relationship representative LLM batch "
                f"{idx + 1}-{idx + len(batch)}/{len(representative_entries)}"
            )
            prompt = (
                base_prompt
                + "\nRepresentative relationship candidates:\n"
                + json.dumps(batch, ensure_ascii=False, separators=(',', ':'))
            )
            try:
                response = text_gen.send_message(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_RESPONSE_TOKENS,
                    retries=DETECTION_REQUEST_RETRIES,
                    request_timeout=DETECTION_REQUEST_TIMEOUT,
                )
                parsed = parse_llm_json_response(response)
                for item in parsed.get("confirmed_representatives", []):
                    try:
                        rep_id = int(item.get("representative_id"))
                    except (TypeError, ValueError):
                        continue
                    rep_key = rep_id_to_key.get(rep_id)
                    if rep_key is None:
                        continue
                    for row_idx, field_name in representative_cells.get(rep_key, []):
                        confirmed.append({
                            "row": int(row_idx),
                            "fieldName": str(field_name),
                            "errorType": "logical_errors",
                            "description": item.get("description") or "LLM-confirmed relationship representative contradiction.",
                            "correctFormat": item.get("correctFormat"),
                            "llmSource": "relationship_representative_llm_confirmation",
                        })
                idx += batch_limit
            except Exception as exc:
                failures += 1
                if batch_limit > 1:
                    batch_limit = max(1, batch_limit // 2)
                    print(
                        f"[Warn] relationship representative LLM confirmation failed ({exc}); "
                        f"retrying with batch_limit={batch_limit}"
                    )
                    continue
                print(f"[Warn] skipping one relationship representative after LLM failure: {exc}")
                idx += 1
                if failures > 20:
                    print("[Warn] relationship representative LLM confirmation failure budget reached")
                    break
        if confirmed:
            print(
                f"LLM confirmed {len(confirmed)} cells from "
                f"{len(representative_entries)} relationship representative groups"
            )
        return confirmed

    def confirm_relationship_errors_with_llm(
            self,
            candidates: set[tuple[int, str]],
            summary: dict,
            df: pd.DataFrame,
            field_props: dict,
            format_rules: dict,
            candidate_sources: dict[tuple[int, str], set[str]],
    ) -> list[dict]:
        """
        Relationship validators produce candidates only. A relationship-derived
        cell can enter final_errors only when this LLM call confirms it.
        """
        if not candidates:
            return []

        representative_entries, representative_cells = self.relationship_representative_entries(
            candidates=candidates,
            summary=summary,
            df=df,
            field_props=field_props,
            candidate_sources=candidate_sources,
        )
        if representative_entries:
            confirmed = self.confirm_relationship_representatives_with_llm(
                representative_entries=representative_entries,
                representative_cells=representative_cells,
                summary=summary,
                field_props=field_props,
                format_rules=format_rules,
            )
            if confirmed:
                return confirmed
            print("LLM confirmed 0 relationship representative groups")
            return []

        entries = self.relationship_candidate_entries(
            candidates=candidates,
            summary=summary,
            df=df,
            field_props=field_props,
            candidate_sources=candidate_sources,
        )
        if not entries:
            return []

        relationships = summary.get("field_relationships", {}) or {}
        rel_cols = sorted(relationship_columns(relationships, df.columns))
        compact_field_props = {
            col: {
                key: value
                for key, value in (field_props.get(col, {}) or {}).items()
                if key in {
                    "dtype",
                    "semantic_type",
                    "description",
                    "top_value_counts",
                    "shape_counts",
                    "missing_like_values",
                }
            }
            for col in rel_cols
        }
        compact_format_rules = {
            col: format_rules.get(col, {})
            for col in rel_cols
            if isinstance(format_rules.get(col, {}), dict)
        }

        system_prompt = (
            "You are the final LAED relationship-error judge. Relationship "
            "validators and profile checks have produced candidate cells, but "
            "they are not errors yet. Confirm a candidate only when the provided "
            "row context and summary relationships show a clear contradiction. "
            "Omit candidates that are plausible, ambiguous, or merely rare. "
            "Use only the supplied dirty-data evidence and summary metadata; "
            "do not assume access to clean ground truth."
        )
        base_prompt = f"""
Summary relationships:
{json.dumps(relationships, ensure_ascii=False, indent=2)}

Relationship-column field metadata:
{json.dumps(compact_field_props, ensure_ascii=False, indent=2)}

Relationship-column format rules:
{json.dumps(compact_format_rules, ensure_ascii=False, indent=2)}

For each candidate, decide whether that exact cell should be a final
logical_errors cell. Return only confirmed errors. If a row has several
relationship inconsistencies, return only the cells that are directly wrong.

Return exactly this JSON object:
{{
  "errors": [
    {{
      "row": <row index>,
      "fieldName": "<candidate fieldName>",
      "errorType": "logical_errors",
      "description": "<brief evidence-based relationship contradiction>",
      "correctFormat": "<expected relationship-consistent value or null>"
    }}
  ]
}}
"""

        confirmed = []
        candidate_keys = {(int(row), str(col)) for row, col in candidates}
        idx = 0
        batch_limit = 10
        parse_failures = 0
        while idx < len(entries):
            batch = entries[idx: idx + batch_limit]
            prompt = (
                base_prompt
                + "\nCandidate cells:\n"
                + json.dumps(batch, ensure_ascii=False, indent=2)
            )
            try:
                response = text_gen.send_message(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=MAX_RESPONSE_TOKENS,
                    retries=DETECTION_REQUEST_RETRIES,
                    request_timeout=DETECTION_REQUEST_TIMEOUT,
                )
                parsed = parse_llm_json_response(response)
                for item in parsed.get("errors", []):
                    try:
                        row_idx = int(item.get("row"))
                    except (TypeError, ValueError):
                        continue
                    field_name = str(item.get("fieldName", ""))
                    if (row_idx, field_name) not in candidate_keys:
                        continue
                    if str(item.get("errorType", "")) != "logical_errors":
                        continue
                    confirmed.append({
                        "row": row_idx,
                        "fieldName": field_name,
                        "errorType": "logical_errors",
                        "description": item.get("description") or "LLM-confirmed relationship contradiction.",
                        "correctFormat": item.get("correctFormat"),
                        "llmSource": "relationship_candidate_llm_confirmation",
                    })
                idx += batch_limit
            except Exception as exc:
                parse_failures += 1
                if batch_limit > 1:
                    batch_limit = max(1, batch_limit // 2)
                    print(
                        f"[Warn] relationship LLM confirmation failed ({exc}); "
                        f"retrying with batch_limit={batch_limit}"
                    )
                    continue
                print(f"[Warn] skipping one relationship candidate after LLM failure: {exc}")
                idx += 1
                if parse_failures > 20:
                    print("[Warn] relationship LLM confirmation failure budget reached")
                    break

        if confirmed:
            print(
                f"LLM confirmed {len(confirmed)}/{len(entries)} "
                "relationship-candidate cells"
            )
        elif entries:
            print(f"LLM confirmed 0/{len(entries)} relationship-candidate cells")
        return confirmed

    def confirm_tuple_consensus_relationship_rule_with_llm(
            self,
            tuple_candidates: set[tuple[int, str]],
            tuple_group_summaries: list[dict],
            summary: dict,
    ) -> list[dict]:
        """
        Ask the LLM to approve a generic same-key tuple-consensus expansion.
        The expansion is applied only to candidates already produced from
        summary relationships and initial-screening suspicious scope.
        """
        if not tuple_candidates or not tuple_group_summaries:
            return []
        relationships = summary.get("field_relationships", {}) or {}
        compact_groups = sorted(
            tuple_group_summaries,
            key=lambda item: (-int(item.get("candidate_cell_count") or 0), str(item.get("key_value", ""))),
        )[:24]
        prompt = f"""
You are the final LAED relationship-rule judge. The system has generated
candidate cells from a generic same-key tuple-consensus rule:
for an associative key, rows with the same key should share a coherent tuple
across the related dependent fields. A candidate cell is a value that differs
from the best-supported complete tuple for that same key.

This is not final detection yet. Decide whether this tuple-consensus rule is a
valid LLM-confirmed relationship-error expansion for this dataset summary. Use
only the dirty-data summary relationships and group evidence below. Confirm the
rule only if the examples show strong same-key record consistency and the
candidate values are directly inconsistent with that complete tuple pattern.

Summary relationships:
{json.dumps(relationships, ensure_ascii=False, separators=(',', ':'))}

Candidate tuple-consensus group examples:
{json.dumps(compact_groups, ensure_ascii=False, indent=2)}

Return exactly this JSON object:
{{
  "confirm_tuple_consensus_rule": true,
  "description": "<brief reason>",
  "correctFormat": "<expected same-key tuple consistency rule or null>"
}}
"""
        try:
            response = text_gen.send_message(
                [
                    {"role": "system", "content": "Return compact JSON only. You are a strict data-quality relationship judge."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_RESPONSE_TOKENS,
                retries=DETECTION_REQUEST_RETRIES,
                request_timeout=DETECTION_REQUEST_TIMEOUT,
            )
            parsed = parse_llm_json_response(response)
        except Exception as exc:
            print(f"[Warn] tuple-consensus relationship LLM rule confirmation failed: {exc}")
            return []

        if parsed.get("confirm_tuple_consensus_rule") is not True:
            print("LLM did not confirm tuple-consensus relationship expansion rule")
            return []

        description = parsed.get("description") or "LLM-confirmed same-key tuple-consensus relationship contradiction."
        correct_format = parsed.get("correctFormat")
        confirmed = [
            {
                "row": int(row_idx),
                "fieldName": str(field_name),
                "errorType": "logical_errors",
                "description": description,
                "correctFormat": correct_format,
                "llmSource": "relationship_tuple_consensus_llm_rule_confirmation",
            }
            for row_idx, field_name in sorted(tuple_candidates)
        ]
        print(
            f"LLM confirmed tuple-consensus relationship rule for "
            f"{len(confirmed)} candidate cells"
        )
        return confirmed

    def select_indices_for_llm(self, col, df, props, fmt, suspicious_idx):
        """
        Return every cell left suspicious by Initial_Screening_all.py.

        The paper-aligned boundary is intentional: initial screening can only
        shrink the search space. It cannot decide that a suspicious cell is
        harmless, so Error_Detection_update.py must pass the full suspicious
        scope into the LLM judgment path. Later batching may use one
        representative per duplicate value, but each final error still derives
        from an LLM decision on that representative or an LLM-returned rule.
        """
        del df, props, fmt
        return [int(idx) for idx in suspicious_idx]

    def score_probe_sample(self, sample, representative_rows, props) -> float:
        """
        Rank free-text representatives for a cheap LLM probe.
        The score uses only generic profile metadata from the LLM-generated
        summary and local distribution statistics; it does not encode any
        dataset/domain-specific values.
        """
        row = sample.get("row")
        value = str(sample.get("value", "")).strip()
        lower_value = value.lower()
        freq = len(representative_rows.get(row, []))
        score = 0.0

        missing_values = {
            str(v).strip().lower()
            for v in (props.get("missing_like_values") or {}).keys()
        }
        if (
            lower_value in missing_values
            or normalize_missing_like_token(value) in missing_values
            or is_missing_like_value(value)
        ):
            score += 100.0

        rare_values = {
            str(v).strip()
            for v in (props.get("rare_value_examples") or {}).keys()
        }
        if value in rare_values:
            score += 40.0

        shape_counts = props.get("shape_counts") or {}
        if shape_counts:
            shape = Summarizer._value_shape(value)
            shape_count = int(shape_counts.get(shape) or 0)
            if shape_count == 0:
                score += 30.0
            elif shape_count <= 2:
                score += 15.0

        dominant_skeleton = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        if dominant_skeleton and re.search(r"\d", value):
            value_skeleton = profile_normalized_numeric_skeleton(props, value)
            if value_skeleton and value_skeleton != dominant_skeleton:
                score += 25.0
        if Summarizer._numeric_text_skeleton(value) in noncanonical_numeric_surface_skeletons(props):
            score += 35.0

        variant_groups = props.get("case_punctuation_variant_groups") or {}
        normalized_value = Summarizer._case_punctuation_key(value)
        if normalized_value in variant_groups and value in variant_groups.get(normalized_value, {}):
            score += 10.0
        if text_noise_reasons(value):
            score += 25.0

        return score + min(freq, 10) * 0.1

    @staticmethod
    def collect_explicit_error_rows(col_errors: dict) -> dict:
        explicit = {}
        for col, errs in col_errors.items():
            for err in errs:
                row_spec = err.get("row")
                rows = row_spec if isinstance(row_spec, list) else [row_spec]
                error_types = [
                    et.strip()
                    for et in str(err.get("errorType", "")).split(",")
                    if et.strip()
                ]
                for et in error_types:
                    key = (col, et)
                    explicit.setdefault(key, set())
                    for row in rows:
                        try:
                            explicit[key].add(int(row))
                        except (TypeError, ValueError):
                            continue
        return explicit

    def regex_expansion_is_safe(
            self,
            col,
            et,
            pattern,
            matched_rows,
            explicit_rows,
            total_suspicious: int = 0,
            protected_matched_rows: set[int] | None = None,
    ) -> bool:
        """
        Guard against over-broad LLM-returned regexes. The detector still stores
        and executes LLM rules, but it will not let a rule add a large population
        that the LLM did not explicitly diagnose.
        """
        if not matched_rows:
            return True
        if not explicit_rows:
            print(f"[Warn] skipping LLM regex expansion for '{col}'/{et}: no explicit LLM rows")
            return False
        if et in {"format_errors", "spelling_errors"} and re.search(r"\.\*|\\w|\\s\[\^|\\s\[A-Z", str(pattern)):
            print(
                f"[Warn] skipping broad open-text LLM regex expansion for '{col}'/{et}: {pattern}"
            )
            return False
        if et in {"format_errors", "spelling_errors"} and protected_matched_rows:
            print(
                f"[Warn] skipping LLM regex expansion for '{col}'/{et}: "
                f"{len(protected_matched_rows)} matched rows have summary-supported "
                "canonical surfaces"
            )
            return False
        protected_matched_rows = protected_matched_rows or set()
        protected_extra = protected_matched_rows - set(explicit_rows)
        if protected_extra:
            print(
                f"[Warn] skipping LLM regex expansion for '{col}'/{et}: "
                f"{len(protected_extra)} matched rows have summary-supported numeric/unit surfaces "
                "that were not explicitly diagnosed by the LLM"
            )
            return False
        extra = set(matched_rows) - set(explicit_rows)
        denominator = max(int(total_suspicious or 0), len(matched_rows), 1)
        coverage = len(matched_rows) / denominator
        if len(explicit_rows) >= 1 and (coverage <= 0.75 or len(matched_rows) <= 1000):
            return True
        allowed_extra = max(10, int(len(explicit_rows) * 2.0))
        if len(extra) > allowed_extra:
            print(
                f"[Warn] skipping broad LLM regex expansion for '{col}'/{et}: "
                f"{len(extra)} extra rows beyond {len(explicit_rows)} explicit LLM rows"
            )
            return False
        return True

    def select_probe_samples_for_free_text(self, samples, representative_rows, props, probe_size):
        scored = [
            (self.score_probe_sample(sample, representative_rows, props), sample)
            for sample in samples
        ]
        high_risk = [
            sample
            for score, sample in sorted(scored, key=lambda item: (-item[0], item[1].get("row", 0)))
            if score > 0
        ]
        frequent = sorted(
            samples,
            key=lambda item: (
                -len(representative_rows.get(item["row"], [])),
                item["row"],
            ),
        )

        selected = []
        seen_rows = set()
        for sample in high_risk + frequent + samples:
            row = sample.get("row")
            if row in seen_rows:
                continue
            selected.append(sample)
            seen_rows.add(row)
            if len(selected) >= probe_size:
                break
        return selected

    def representative_sample_budget(self, props, fmt, distinct_count: int) -> int | None:
        """
        Bound expensive per-column LLM deep scans with a generic representative
        value budget. The LLM still reviews the selected suspicious values and
        its returned regexes are executed later with expansion guards. This keeps
        repeated runs stable when one column has many similar candidate values.
        """
        budget = max(0, MAX_LLM_REPRESENTATIVE_VALUES_PER_COLUMN)
        if budget <= 0 or distinct_count <= budget:
            return None
        if self.should_probe_free_text(props, fmt, distinct_count, distinct_count):
            return None
        return budget

    def select_representative_samples_for_budget(self, samples, representative_rows, props, budget: int):
        """
        Select a deterministic, diverse subset using only generic evidence:
        summary/profile warnings, shape and skeleton differences, text noise,
        missing-like values, rarity, and local duplicate frequency.
        """
        if len(samples) <= budget:
            return samples
        scored = []
        for sample in samples:
            row = sample.get("row")
            value = str(sample.get("value", ""))
            score = self.score_probe_sample(sample, representative_rows, props)
            score += 0.2 * len(sample.get("profile_warnings", []) or [])
            scored.append((score, sample))

        selected = []
        seen_rows = set()

        def add(sample):
            row = sample.get("row")
            if row in seen_rows:
                return
            selected.append(sample)
            seen_rows.add(row)

        for _, sample in sorted(scored, key=lambda item: (-item[0], item[1].get("row", 0))):
            if len(selected) >= budget:
                break
            if _ > 0 or sample.get("profile_warnings"):
                add(sample)

        shape_best = {}
        for score, sample in scored:
            value = str(sample.get("value", ""))
            key = (
                Summarizer._value_shape(value),
                profile_normalized_numeric_skeleton(props, value) or "",
            )
            current = shape_best.get(key)
            if current is None or score > current[0] or (
                score == current[0] and sample.get("row", 0) < current[1].get("row", 0)
            ):
                shape_best[key] = (score, sample)
        for _, sample in sorted(shape_best.values(), key=lambda item: (-item[0], item[1].get("row", 0))):
            if len(selected) >= budget:
                break
            add(sample)

        frequent = sorted(
            samples,
            key=lambda item: (
                -len(representative_rows.get(item.get("row"), [])),
                item.get("row", 0),
            ),
        )
        for sample in frequent + samples:
            if len(selected) >= budget:
                break
            add(sample)
        return selected

    def update_experience(self, errs, max_examples=5):
        exp = {}
        if os.path.exists(self.experience_file):
            with open(self.experience_file, "r", encoding="utf-8") as f:
                exp = json.load(f)
        for e in errs:
            t = e["errorType"]
            exp.setdefault(t, []).append(e)
            exp[t] = exp[t][-max_examples:]
        with open(self.experience_file, "w", encoding="utf-8") as f:
            json.dump(exp, f, ensure_ascii=False, indent=2)

    def _detect_column_error_batches(self, col, samples, props, fmt, representative_rows):
        total = len(samples)
        errors_all = []
        regexes_by_type_list = []
        idx = 0
        initial_batch_limit = self.initial_batch_limit_for_field(col, props, samples)
        batch_limit = initial_batch_limit
        parse_failure_budget = 12
        parse_failures = 0
        transient_failures = 0
        transient_failure_budget = 4

        while idx < total:
            batch = []
            tokens_used = PROMPT_OVERHEAD
            count = 0

            while idx < total:
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

            effective_fmt = self.effective_format_rule_for_detection(props, fmt)
            regex = effective_fmt.get("regex", "")
            fmt_expl = effective_fmt.get("explanation", "")
            prompt_t = USER_PROMPT_WITH_REGEX if regex else USER_PROMPT_NO_REGEX

            prompt = prompt_t.format(
                col=col,
                dtype=props.get("dtype", ""),
                semantic_type=props.get("semantic_type", ""),
                description=props.get("description", ""),
                regex=regex,
                fmt_expl=fmt_expl,
                profile_json=self.column_profile_for_prompt(props),
                samples_json=json.dumps(batch, ensure_ascii=False, indent=2)
            )

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ]
            print(f"→ Sending batch of {len(batch)} rows for '{col}' (~{tokens_used} tokens)")
            resp = None

            try:
                resp = text_gen.send_message(
                    messages,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    retries=SINGLE_COLUMN_DETECTION_REQUEST_RETRIES,
                    request_timeout=DETECTION_REQUEST_TIMEOUT,
                )
                parsed = parse_llm_json_response(resp)
                errs = parsed.get("errors", [])
                regex_by_type = parsed.get("error_regex_by_type", {
                    "format_errors": "",
                    "spelling_errors": "",
                    "outliers": "",
                    "missing_errors": ""
                })
                errors_all.extend(expand_representative_errors(errs, representative_rows))
                regexes_by_type_list.append(regex_by_type)
            except Exception as exc:
                transient = resp is None and is_transient_request_error(exc)
                if resp is not None:
                    parse_failures += 1
                    if parse_failures >= parse_failure_budget:
                        print(
                            f"[Warn] parse failure budget reached for '{col}' "
                            f"({parse_failures}); stopping remaining batches for this column"
                        )
                        break
                if batch_limit is None or batch_limit > 1:
                    reason = "request fail" if resp is None else "parse fail"
                    if transient:
                        transient_failures += 1
                        if transient_failures >= transient_failure_budget:
                            print(
                                f"[Warn] transient request failure budget reached for '{col}'; "
                                f"skipping batch of {len(batch)} rows"
                            )
                            transient_failures = 0
                            continue
                        print(f"[Warn] transient {reason} for '{col}' ({exc}), retrying same batch")
                    else:
                        transient_failures = 0
                        batch_limit = max(1, len(batch) // 2)
                        print(f"[Warn] {reason} for '{col}' ({exc}), new batch_limit={batch_limit}")
                    idx -= len(batch)
                else:
                    if transient:
                        transient_failures += 1
                        if transient_failures >= transient_failure_budget:
                            print(
                                f"[Warn] transient minimal-batch failure budget reached for '{col}'; "
                                "skipping this row"
                            )
                            transient_failures = 0
                            continue
                        print(f"[Warn] transient request fail for '{col}' ({exc}), retrying minimal batch")
                        idx -= len(batch)
                    else:
                        transient_failures = 0
                        salvaged = self.salvage_malformed_minimal_batch(resp, batch, col) if resp is not None else []
                        if salvaged:
                            print(f"[Warn] salvaged malformed minimal LLM batch for '{col}'")
                            errors_all.extend(expand_representative_errors(salvaged, representative_rows))
                        else:
                            detail = repr(clean_code_snippet(resp)) if resp is not None else repr(exc)
                            print(f"[Error] skip minimal batch for '{col}': {detail}")
                continue

        cleaned_list = []
        for item in regexes_by_type_list:
            if isinstance(item, dict):
                cleaned_list.append(item)
            else:
                print(f"[Warn] detect_column_errors got non-dict error_regex_by_type: {item!r}")

        return {"errors": errors_all, "error_regex_by_type_list": cleaned_list}

    def detect_column_errors(self, col, df, props, fmt,suspicious_idx):
        """
                只对 df[col] 中索引属于 suspicious_idx 那些行进行错误检测，
                并且把原本缺失值（NaN）也当做字符串 'nan' 一并发送给 LLM，由它来标记 missing_errors。
        """
        # ——先把需要检测的那些行取出来（包括原生 NaN）——
        # series_raw 会把 NaN 转为字符串 "nan"
        series = df.loc[suspicious_idx, col].astype(str).copy()
        print(f"Suspicious cells selected for '{col}': {len(series)}")

        # 如果是 number 类型，再把 “xx.0” 规范成 “xx”
        if props.get("dtype") == "number":
            series = series.map(lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x)

        if len(series) == 0:
            return {"errors": [], "error_regex_by_type_list": []}

        # 把 index、value 做成 LLM 样本格式
        value_to_rows = OrderedDict()
        for row_idx, value in series.items():
            value_to_rows.setdefault(str(value), []).append(int(row_idx))
        representative_rows = {
            rows[0]: rows
            for rows in value_to_rows.values()
            if rows
        }
        samples = [
            self.make_sample_entry(rows[0], value, props, fmt)
            for value, rows in value_to_rows.items()
            if rows
        ]
        total = len(samples)
        print(f"Prepared {total} distinct suspicious values for '{col}' from {len(series)} suspicious cells")
        if total == 0:
            return {"errors": [], "error_regex_by_type_list": []}
        compact_warning_result = self.confirm_profile_warning_errors_with_llm(
            col=col,
            props=props,
            fmt=fmt,
            samples=samples,
            representative_rows=representative_rows,
        )
        seed_errors = list(compact_warning_result["errors"])
        seed_regexes = list(compact_warning_result.get("error_regex_by_type_list", []))
        errors_all = seed_errors
        regexes_by_type_list = seed_regexes  # 本批次每个 LLM 调用返回的 error_regex_by_type 对象列表
        confirmed_representatives = {
            int(row)
            for row in compact_warning_result.get("confirmed_representative_rows", [])
            if row is not None
        }
        if confirmed_representatives:
            before = len(samples)
            samples = [
                sample for sample in samples
                if int(sample.get("row")) not in confirmed_representatives
            ]
            total = len(samples)
            skipped = before - total
            print(
                f"Skipping {skipped} representative values for '{col}' already "
                "confirmed by compact LLM profile-warning diagnosis"
            )
            if total == 0:
                return {
                    "errors": errors_all,
                    "error_regex_by_type_list": regexes_by_type_list,
                }
        if self.is_identifier_like_field(col, props) and total > 40 and not errors_all and not seed_regexes:
            has_trustworthy_regex_mismatch = any(
                "regex_mismatch_candidate" in (sample.get("profile_warnings") or [])
                for sample in samples
            )
            if not has_trustworthy_regex_mismatch:
                print(
                    f"Skipping broad identifier/code-like deep scan for '{col}' "
                    "because no generic profile warning was LLM-confirmed; "
                    "relationship candidates remain subject to LLM confirmation."
                )
                return {"errors": [], "error_regex_by_type_list": []}
        idx = 0
        initial_batch_limit = self.initial_batch_limit_for_field(col, props, samples)
        batch_limit = initial_batch_limit
        parse_failure_budget = 12
        parse_failures = 0
        transient_failures = 0
        transient_failure_budget = 4

        while idx < total:
            batch = []
            tokens_used = PROMPT_OVERHEAD
            count = 0

            while idx < total:
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

            # 如果一次都没放进 batch，至少放一个避免死循环
            if not batch:
                batch.append(samples[idx])
                idx += 1

            # 选提示词模板
            effective_fmt = self.effective_format_rule_for_detection(props, fmt)
            regex = effective_fmt.get("regex", "")
            fmt_expl = effective_fmt.get("explanation", "")
            prompt_t = USER_PROMPT_WITH_REGEX if regex else USER_PROMPT_NO_REGEX

            prompt = prompt_t.format(
                col=col,
                dtype=props.get("dtype", ""),
                semantic_type=props.get("semantic_type", ""),
                description=props.get("description", ""),
                regex=regex,
                fmt_expl=fmt_expl,
                profile_json=self.column_profile_for_prompt(props),
                samples_json=json.dumps(batch, ensure_ascii=False, indent=2)
            )

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ]
            print(f"→ Sending batch of {len(batch)} rows for '{col}' (~{tokens_used} tokens)")
            resp = None
            # time.sleep(RATE_LIMIT_SLEEP)

            try:
                resp = text_gen.send_message(
                    messages,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    retries=SINGLE_COLUMN_DETECTION_REQUEST_RETRIES,
                    request_timeout=DETECTION_REQUEST_TIMEOUT,
                )
                parsed = parse_llm_json_response(resp)
                errs = parsed.get("errors", [])
                rex = parsed.get("error_regex", "")
                regex_by_type = parsed.get("error_regex_by_type", {
                    "format_errors": "",
                    "spelling_errors": "",
                    "outliers": "",
                    "missing_errors": ""
                })
                # 收集错误对象和本次调用返回的 error_regex_by_type
                errors_all.extend(expand_representative_errors(errs, representative_rows))
                regexes_by_type_list.append(regex_by_type)
            except Exception as exc:
                # 解析失败则减小 batch 重试
                transient = resp is None and is_transient_request_error(exc)
                if resp is not None:
                    parse_failures += 1
                    if parse_failures >= parse_failure_budget:
                        print(
                            f"[Warn] parse failure budget reached for '{col}' "
                            f"({parse_failures}); stopping remaining batches for this column"
                        )
                        break
                if batch_limit is None or batch_limit > 1:
                    reason = "request fail" if resp is None else "parse fail"
                    if transient:
                        transient_failures += 1
                        if transient_failures >= transient_failure_budget:
                            print(
                                f"[Warn] transient request failure budget reached for '{col}'; "
                                f"skipping batch of {len(batch)} rows"
                            )
                            transient_failures = 0
                            continue
                        print(f"[Warn] transient {reason} for '{col}' ({exc}), retrying same batch")
                    else:
                        transient_failures = 0
                        batch_limit = max(1, len(batch) // 2)
                        print(f"[Warn] {reason} for '{col}' ({exc}), new batch_limit={batch_limit}")
                    idx -= len(batch)
                else:
                    if transient:
                        transient_failures += 1
                        if transient_failures >= transient_failure_budget:
                            print(
                                f"[Warn] transient minimal-batch failure budget reached for '{col}'; "
                                "skipping this row"
                            )
                            transient_failures = 0
                            continue
                        print(f"[Warn] transient request fail for '{col}' ({exc}), retrying minimal batch")
                        idx -= len(batch)
                    else:
                        transient_failures = 0
                        salvaged = self.salvage_malformed_minimal_batch(resp, batch, col) if resp is not None else []
                        if salvaged:
                            print(f"[Warn] salvaged malformed minimal LLM batch for '{col}'")
                            errors_all.extend(expand_representative_errors(salvaged, representative_rows))
                        else:
                            detail = repr(clean_code_snippet(resp)) if resp is not None else repr(exc)
                            print(f"[Error] skip minimal batch for '{col}': {detail}")
                continue
        cleaned_list = []
        for item in regexes_by_type_list:
            if isinstance(item, dict):
                cleaned_list.append(item)
            else:
                print(f"[Warn] detect_column_errors got non-dict error_regex_by_type: {item!r}")
        regexes_by_type_list = cleaned_list

        return {"errors": errors_all, "error_regex_by_type_list": regexes_by_type_list}




    def generate(self, summary, df, screening_result=None):
        """
        1. 初筛：拿到 df_original + correct_cells mask + pct
        2. 构造 suspicious_mask = ~correct_cells
        3. 逐列对“可疑位置”拆批调用 LLM，得到 col_errors[col] 和本列各批次的 error_regex_by_type 列表
        4. 合并各批次的 error_regex_by_type，按 errorType 分类生成全局正则 compiled_error_rules[col][errorType]
        5. 运行 compiled_error_rules 生成格式/拼写/离群/缺失的单列错误 map
        6. 运行字段间逻辑校验函数生成逻辑错误 map
        7. 整合 LLM 返回的错误对象（已包含 errorType、description、correctFormat）与逻辑错误，得到详细错误列表 detailed_errors
        8. 将 detailed_errors 写入 detailed_errors.json，并将完整结果写入 errors_with_context.json
        """
        fields       = {f["column"]: f["properties"] for f in summary["fields"]}
        format_rules = summary.get("format_rules", {})

        # 1. 数值列先转为字符串
        for col, props in fields.items():
            if props.get("dtype") in ("number", "int", "float"):
                df[col] = df[col].astype(str)

        # 2. 初筛（不修改 df，只拿回 correct_cells mask）
        if screening_result is None:
            df_original, correct_cells, pct = initial_screening(df, summary)
        else:
            df_original, correct_cells, pct = screening_result
        print(f"Initial screening done, {pct:.2f}% cells marked as correct")

        # 3. 计算 suspicious_mask = ~correct_cells
        suspicious_mask = ~correct_cells
        col_errors = {}
        col_regexes_by_type = {}  # { col: [ { format_errors: "...", spelling_errors: "...", ... }, ... ] }

        # 4. 按列调用 LLM 检测
        for col in df.columns:
            print(f"\nDetecting errors in '{col}'...")
            props = fields.get(col, {})
            fmt = format_rules.get(col, {"regex": "", "explanation": ""})
            all_suspicious_indices = suspicious_mask.index[suspicious_mask[col]].tolist()
            indices_to_check = self.select_indices_for_llm(col, df_original, props, fmt, all_suspicious_indices)
            if not indices_to_check:
                col_errors[col] = []
                col_regexes_by_type[col] = []
                print(f" → no suspicious cells in '{col}', skip LLM")
                continue

            res = self.detect_column_errors(
                col,
                df_original,
                props,
                fmt,
                indices_to_check
            )

            col_errors[col] = res["errors"]
            col_errors[col] = self.normalize_single_column_error_types(col, col_errors[col])
            col_errors[col] = self.filter_summary_unsupported_outliers(
                col,
                df_original,
                props,
                col_errors[col],
            )
            col_errors[col] = self.filter_llm_indicated_numeric_canonical_errors(
                col,
                df_original,
                props,
                col_errors[col],
            )
            col_errors[col] = self.filter_summary_supported_text_canonical_errors(
                col,
                df_original,
                props,
                col_errors[col],
            )
            col_errors[col] = self.filter_low_evidence_identifier_errors(
                col,
                df_original,
                props,
                fmt,
                col_errors[col],
            )
            col_errors[col] = self.expand_llm_confirmed_profile_warnings(
                col,
                df_original,
                props,
                fmt,
                all_suspicious_indices,
                col_errors[col],
            )
            allowed_rows_for_col = {int(idx) for idx in all_suspicious_indices}
            col_errors[col], dropped_scope_errors = restrict_errors_to_allowed_rows(
                col_errors[col],
                allowed_rows_for_col,
            )
            if dropped_scope_errors:
                print(
                    f"[Warn] dropped {dropped_scope_errors} '{col}' LLM-expanded rows "
                    "outside the initial-screening suspicious scope"
                )
            col_regexes_by_type[col] = res["error_regex_by_type_list"]

            cnt = count_cells(col_errors[col])
            print(f" → found {cnt} erroneous cells in '{col}'")
            self.update_experience(col_errors[col])

        # 5. 合并各列各批次返回的 error_regex_by_type
        compiled_error_rules = {}  # { col: {errorType: global_regex 或 "Correct", … } }
        for col, regex_list in col_regexes_by_type.items():
            # 初始化每种类型的集合，用于去重
            patterns_by_type = {
                "format_errors": set(),
                "spelling_errors": set(),
                "outliers": set(),
                "missing_errors": set()
            }
            for batch_regex in regex_list:
                for et, rex in batch_regex.items():
                    if not rex or rex == "Correct":
                        continue
                    # 提取并去除外层锚点，保留内部模式
                    pattern = str(rex).strip()
                    if not pattern.startswith("^"):
                        pattern = f"^(?:{pattern})"
                    if not pattern.endswith("$"):
                        pattern = f"{pattern}$"
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        print(f"[Warn] ignoring invalid LLM regex for column '{col}', type '{et}': {pattern} ({e})")
                        continue
                    m = re.match(r'^\^\(\?:(.*)\)\$$', pattern)
                    inner = m.group(1) if m else pattern.strip('^$')
                    # 分割后对每一段进行 escape，避免括号不匹配
                    inner = inner.strip()
                    if inner:
                        patterns_by_type[et].add(inner)

            # 构造全局正则或标记为 "Correct"
            compiled_error_rules[col] = {}
            for et, parts in patterns_by_type.items():
                if parts:
                    joined = "|".join(sorted(parts))
                    compiled_error_rules[col][et] = f"^(?:{joined})$"
                else:
                    compiled_error_rules[col][et] = "Correct"

        # 6. 打印汇总
        # print("\n=== Summary per column and errorType ===")
        total_cells = 0
        for col in df.columns:
            errs = col_errors[col]
            cnt = count_cells(errs)
            total_cells += cnt
            # print(f"Column '{col}': {cnt} cells (by LLM). Regexes:")
            # for et, rx in compiled_error_rules[col].items():
            #     print(f"  - {et}: {rx}")
        # print(f"\nTotal errors (single-column): {total_cells}")

        # 7. 构造单列错误映射：根据 compiled_error_rules 按类型匹配
        single_col_error_map = {}  # { row_idx: set([ (col, errorType), ... ]) }
        explicit_error_rows = self.collect_explicit_error_rows(col_errors)
        for col, rules in compiled_error_rules.items():
            for et, pattern in rules.items():
                if pattern == "Correct":
                    continue
                try:
                    regex = re.compile(pattern)
                except re.error as e:
                    print(f"[Error] Invalid regex for column '{col}', type '{et}': {pattern}\n  {e}")
                    continue
                suspect_indices = suspicious_mask.index[suspicious_mask[col]].tolist()
                suspect_indices = self.select_indices_for_llm(
                    col,
                    df_original,
                    fields.get(col, {}),
                    format_rules.get(col, {"regex": "", "explanation": ""}),
                    suspect_indices,
                )
                matched_rows = []
                protected_matched_rows = set()
                props_for_col = fields.get(col, {})
                for idx in suspect_indices:
                    val = df_original.at[idx, col]
                    if pd.isna(val):
                        continue
                    vstr = str(val)
                    if regex.match(vstr):
                        matched_rows.append(int(idx))
                        if (
                                et in {"format_errors", "spelling_errors"}
                                and (
                                    self.numeric_skeleton_is_summary_supported(props_for_col, vstr)
                                    or text_value_is_summary_supported_canonical(props_for_col, vstr)
                                )
                        ):
                            protected_matched_rows.add(int(idx))
                if not self.regex_expansion_is_safe(
                        col,
                        et,
                        pattern,
                        matched_rows,
                        explicit_error_rows.get((col, et), set()),
                        total_suspicious=len(suspect_indices),
                        protected_matched_rows=protected_matched_rows,
                ):
                    continue
                for idx in matched_rows:
                    single_col_error_map.setdefault(idx, set()).add((col, et))

        single_column_llm_confirmed_cells = set()
        for col, errs in col_errors.items():
            for err in errs:
                row_spec = err.get("row")
                rows = row_spec if isinstance(row_spec, list) else [row_spec]
                for rid in rows:
                    try:
                        single_column_llm_confirmed_cells.add((int(rid), str(col)))
                    except (TypeError, ValueError):
                        continue
        for rid, col_et_set in single_col_error_map.items():
            try:
                row_idx = int(rid)
            except (TypeError, ValueError):
                continue
            for col, _et in col_et_set:
                single_column_llm_confirmed_cells.add((row_idx, str(col)))

        # 8. 字段间逻辑校验：生成逻辑错误 map { row_idx: set([col1, col2, …]) }
        # Relationship validators generate candidates only. Candidates must be
        # confirmed by an LLM before becoming final logical_errors.
        relationship_candidate_cells = set()
        relationship_candidate_sources = {}
        field_props = {
            f.get("column"): f.get("properties", {})
            for f in summary.get("fields", [])
            if isinstance(f, dict)
        }
        raw_code = summary.get("relationship_validator_code", {})\
                          .get("validation", {})\
                          .get("code", "")
        if raw_code:
            code = "import pandas as pd\n" + raw_code
            exec_globals = {}
            relationship_df = df_original.where(pd.notna(df_original), "").astype(str)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, exec_globals)
                fn = exec_globals.get("validate_relationships")
                rel_errors = set()
                if fn:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rel_errors = fn(relationship_df) or set()
                    rel_cols = relationship_columns(summary.get("field_relationships", {}), df_original.columns)
                    if rel_cols:
                        rel_errors = {
                            (rid, c)
                            for rid, c in rel_errors
                            if c in df_original.columns
                        }
                        rel_errors = suppress_low_confidence_relationship_errors(
                            rel_errors,
                            rel_cols,
                            len(df_original),
                            "Relationship validator",
                            field_props,
                        )
                    for rid, c in rel_errors:
                        props = field_props.get(c, {})
                        if (
                            not relationship_requires_generated_validator_gate(c, props)
                            or relationship_value_has_local_anomaly(c, df_original.at[rid, c], props, format_rules)
                        ):
                            key = (int(rid), str(c))
                            relationship_candidate_cells.add(key)
                            relationship_candidate_sources.setdefault(key, set()).add("summary_relationship_validator")
            except Exception as exc:
                print(f"Relationship validator failed at detection stage; skipping generated relationship candidates: {exc}")

        generic_rel_errors = generic_associative_relationship_errors(summary, df_original)
        if generic_rel_errors:
            print(f"Generic summary associative validation found {len(generic_rel_errors)} cells")
            rel_cols = relationship_columns(summary.get("field_relationships", {}), df_original.columns)
            if rel_cols:
                generic_rel_errors = suppress_low_confidence_relationship_errors(
                    generic_rel_errors,
                    rel_cols,
                    len(df_original),
                    "Generic associative validation",
                    field_props,
                )
            for rid, c in generic_rel_errors:
                key = (int(rid), str(c))
                relationship_candidate_cells.add(key)
                relationship_candidate_sources.setdefault(key, set()).add("generic_associative_summary_check")

        repeated_relationship_candidates = relationship_candidate_cells & single_column_llm_confirmed_cells
        if repeated_relationship_candidates:
            relationship_candidate_cells = relationship_candidate_cells - repeated_relationship_candidates
            for key in repeated_relationship_candidates:
                relationship_candidate_sources.pop(key, None)
            print(
                f"Skipped {len(repeated_relationship_candidates)} relationship candidates "
                "already confirmed by single-column LLM decisions or LLM regex expansion; "
                f"{len(relationship_candidate_cells)} relationship-only candidates remain for LLM confirmation"
            )

        relationship_llm_errors = self.confirm_relationship_errors_with_llm(
            candidates=relationship_candidate_cells,
            summary=summary,
            df=df_original,
            field_props=field_props,
            format_rules=format_rules,
            candidate_sources=relationship_candidate_sources,
        )
        tuple_candidates, tuple_group_summaries = relationship_tuple_consensus_candidates(
            summary=summary,
            df=df_original,
            field_props=field_props,
            candidate_scope=relationship_candidate_cells,
        )
        if tuple_candidates:
            print(
                f"Tuple-consensus relationship candidates prepared: "
                f"{len(tuple_candidates)} cells across {len(tuple_group_summaries)} groups"
            )
        tuple_relationship_llm_errors = self.confirm_tuple_consensus_relationship_rule_with_llm(
            tuple_candidates=tuple_candidates,
            tuple_group_summaries=tuple_group_summaries,
            summary=summary,
        )

        # 9. Build detailed_errors from LLM-returned single-column decisions,
        #    guarded LLM regex expansion, and LLM-confirmed relationship errors.
        detailed_errors = []

        # 9.1 拆分 LLM 原始错误对象（这一步仅为了保留 description 和 correctFormat）
        for col, errs in col_errors.items():
            for e in errs:
                row_spec = e.get("row")
                etype = e.get("errorType")
                desc = e.get("description")
                corr = e.get("correctFormat")
                if isinstance(row_spec, list):
                    for rid in row_spec:
                        detailed_errors.append({
                            "row": int(rid),
                            "fieldName": col,
                            "errorType": etype,
                            "description": desc,
                            "correctFormat": corr
                        })
                else:
                    detailed_errors.append({
                        "row": int(row_spec),
                        "fieldName": col,
                        "errorType": etype,
                        "description": desc,
                        "correctFormat": corr
                    })

        # 9.2 补充 single_col_error_map 中但不在 LLM “errors” 内的条目（以防 LLM 忽略了某些类型或批次未返回）
        for rid, col_et_set in single_col_error_map.items():
            for (col, et) in col_et_set:
                # 检查是否已有相同 (row, col, et)
                exists = any(
                    (d["row"] == rid and d["fieldName"] == col and d["errorType"] == et)
                    for d in detailed_errors
                )
                if not exists:
                    detailed_errors.append({
                        "row": rid,
                        "fieldName": col,
                        "errorType": et,
                        "description": None,
                        "correctFormat": None
                    })

        # 9.3 添加逻辑错误（如果某些逻辑错误单元格 LLM 没标注过）
        for error in relationship_llm_errors:
            rid = int(error["row"])
            c = str(error["fieldName"])
            exists = any(
                (d["row"] == rid and d["fieldName"] == c and d["errorType"] == "logical_errors")
                for d in detailed_errors
            )
            if not exists:
                detailed_errors.append(error)
        for error in tuple_relationship_llm_errors:
            rid = int(error["row"])
            c = str(error["fieldName"])
            exists = any(
                (d["row"] == rid and d["fieldName"] == c and d["errorType"] == "logical_errors")
                for d in detailed_errors
            )
            if not exists:
                detailed_errors.append(error)

        filtered_detailed_errors = []
        dropped_out_of_scope = 0
        for entry in detailed_errors:
            try:
                rid = int(entry["row"])
            except (TypeError, ValueError):
                dropped_out_of_scope += 1
                continue
            col = entry.get("fieldName")
            if col in suspicious_mask.columns and rid in suspicious_mask.index and bool(suspicious_mask.at[rid, col]):
                entry = dict(entry)
                entry["row"] = rid
                filtered_detailed_errors.append(entry)
            else:
                dropped_out_of_scope += 1
        if dropped_out_of_scope:
            print(
                f"[Warn] dropped {dropped_out_of_scope} LLM-confirmed cells outside "
                "the initial-screening suspicious scope before final_errors"
            )
        detailed_errors = filtered_detailed_errors

        # 10.对detailed_errors进行去重/合并
        #合并相同(row,fieldName)但errorType不同的情况，并去掉完全相同的条目
        merged_map = {}  # key = (row, fieldName), value = dict with merged content
        for entry in detailed_errors:
            key = (entry["row"], entry["fieldName"])
            etype = entry["errorType"]
            desc = entry.get("description") or ""
            corr = entry.get("correctFormat") or ""

            if key not in merged_map:
                merged_map[key] = {
                    "row": entry["row"],
                    "fieldName": entry["fieldName"],
                    "errorTypes": {etype},
                    "descriptions": {desc} if desc else set(),
                    "correctFormats": {corr} if corr else set()
                }
            else:
                merged_map[key]["errorTypes"].add(etype)
                if desc:
                    merged_map[key]["descriptions"].add(desc)
                if corr:
                    merged_map[key]["correctFormats"].add(corr)

        # 将 merged_map 转换回列表形式
        deduped_errors = []
        for (rid, col), info in merged_map.items():
            merged_errorType = ",".join(sorted(info["errorTypes"]))
            # 如果有多个 description，则用分号分隔，否则可能为空字符串
            merged_desc = ";".join(sorted(info["descriptions"])) if info["descriptions"] else None
            merged_corr = ";".join(sorted(info["correctFormats"])) if info["correctFormats"] else None

            deduped_errors.append({
                "row": rid,
                "fieldName": col,
                "errorType": merged_errorType,
                "description": merged_desc,
                "correctFormat": merged_corr
            })

        detailed_errors = deduped_errors

        # 11. 构造 final_errors 仅保留 [row, col] 形式（如有需要）
        final_errors = [[d["row"], d["fieldName"]] for d in detailed_errors]

        # 12. 返回并写文件
        result = {
            "format_rules": format_rules,
            "compiled_error_rules": compiled_error_rules,
            "field_relationships": summary.get("field_relationships", {}),
            "relationship_candidates": [
                {
                    "row": row,
                    "fieldName": col,
                    "candidate_sources": sorted(relationship_candidate_sources.get((row, col), set())),
                }
                for row, col in sorted(relationship_candidate_cells)
            ],
            "tuple_consensus_relationship_candidates": [
                {"row": row, "fieldName": col}
                for row, col in sorted(tuple_candidates)
            ],
            "llm_confirmation_policy": {
                "initial_screening_output": "correct_cells mask and suspicious scope only",
                "final_error_requirement": (
                    "Every final error must come from an LLM batch decision, "
                    "an LLM-confirmed profile warning, an LLM-returned regex "
                    "applied with expansion guards, or LLM confirmation of a "
                    "relationship candidate."
                ),
                "relationship_candidates": (
                    "Summary relationship checks generate candidates only; "
                    "confirm_relationship_errors_with_llm decides whether they "
                    "enter final_errors."
                ),
                "duplicate_value_batching": (
                    "Duplicate suspicious values can be represented once in an "
                    "LLM batch and expanded only to identical values that remained "
                    "inside the initial-screening suspicious scope."
                ),
            },
            "errors": detailed_errors,
            "final_errors": final_errors
        }
        # 将 detailed_errors 写入 detailed_errors.json
        with open("detailed_errors.json", "w", encoding="utf-8") as f:
            json.dump(detailed_errors, f, ensure_ascii=False, indent=2)

        # 将完整结果写入 errors_with_context.json
        with open("errors_with_context.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nTotal detailed errors saved: {len(detailed_errors)}")
        return result

