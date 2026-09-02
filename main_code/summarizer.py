# summarizer.py

import contextlib
import io
import json
import os
import re
import sys
from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
from utils import clean_code_snippet, read_dataframe
import warnings
from pythonProject1.API_invocation.qwen_gen import shared_qwen_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

summarizer_generator = shared_qwen_client
relationship_validator_generator = shared_qwen_client

# 系统提示词
system_prompt = """
You are an experienced data analyst that can annotate datasets.  The dataset may contain some erroneous entries (a small minority);  for types and relationships, base your inference on the majority of correct data and ignore isolated anomalies. Your instructions are as follows:
i.)  ALWAYS generate the name of the dataset and dataset_description.
ii.)  ALWAYS generate a brief, **semantic** description for each field—explain what the field represents in the business context (e.g., "Scheduled departure time"), **not** how its values are formatted or any data-quality issues.
iii.)  Update "dtype" for each field based on the majority of valid values, ignoring anomalies,e.g.  number, string, char, boolean, date, category, etc.
- When distinguishing **string** vs. **category**, classify as **category** only if the field has a small, fixed set of distinct values;    otherwise **string**.
- For any field initially labeled "category": sample its values (ignoring obvious errors), verify the label, and if misclassified, pick from **[number, boolean, date, string]**.
- The `value_counts`/`top_value_counts` metadata is only a compact observation sample. Do not treat it as a complete list of legal values, and do not copy rare or missing-like values into the field's valid definition.
iv.) ALWAYS generate a data type (a single word) for each field given its values e.g. **number, string, char, boolean** etc.
v.)  Identify logical relationships BETWEEN FIELDS and generate a "field_relationships" section.  Only infer relationships among columns actually present in the dataset. Focus on
1.  **Hierarchical Consistency**
- Parent–child chains (e.g., City → Province → Country).
- Values must respect that hierarchy.
- Output example:
```json
"hierarchical": {"City": ["Province","Country"], ...}
```
2.  **Mathematical Dependency**
- Identify mathematical relationships BETWEEN DIFFERENT FIELDS, which can include:
- Equality relationships (e.g., Total = Price × Quantity)
- Inequality relationships (e.g., work_year≤ Current Year - Birth Year)
- Range constraints BETWEEN FIELDS (e.g., Start_Date ≤ End_Date)
- Functional dependencies BETWEEN FIELDS (e.g., Area = Length × Width)
- **CRITICAL**: Mathematical relationships must involve AT LEAST TWO DISTINCT FIELDS.  Do not create mathematical relationships for single-field constraints.
- **CRITICAL**: Do not include relationships where a field is only related to itself (e.g., "FieldA" >= 0 is NOT a valid mathematical relationship)
- For duration/experience fields (like work experience): relate work_experience to start_date and current_time
- For financial fields: identify relationships BETWEEN financial fields (e.g., Net = Gross - Tax), not single-field constraints
- Output example:
```json
"mathematical": {
"Total": {"operation": "=", "fields": ["Price", "Quantity"], "formula": "Price * Quantity"},
"工龄": {"operation": "<=", "fields": ["参加工作时间"], "formula": "current_year - work_start_year"}
}
```
3.  Temporal Sequence
- Find every set of two or more timestamp/date fields that form a strict chronological order (e.g., Start < End).
- Treat each independent sequence separately (do not merge them).
- Output under `"temporal"` as a list of sequences, e.g.:
```json
"temporal": [
["StartTime","EndTime"],
["PlannedStart","PlannedEnd"],
["ActualStart","ActualEnd"]
...
]
```
4.  Associative Dependency
To accurately detect "key→attributes" mappings, proceed as follows:
a) **Understand field semantics**: Based on field names and value patterns, Identify any column K that semantically serves as an identifier (e.g., an ID or code).
b) For each candidate key K, gather all other columns C, and **test** whether for every distinct value of K, the values of C remain **identical** across all rows.  Allow for occasional anomalies (errors) and focus on the predominant mapping.
c) Select exactly those columns C that satisfy this "constant mapping" property.
d) Output under "associative":
```json
"associative": {"key": ["attributes1", "attributes2", "attributes3", "act_arr_time4"], ...}
```

vi.) ALWAYS include a list called "fields", where each entry is an object:
{
"column": "<column name>",
"properties": {
"uniqueness_ratio": <float>,
"dtype": "<data type>",
"num_unique_values": <>,
"semantic_type": "<semantic type>",
"description": "<brief description>"
}
}

**IMPORTANT REMINDERS**:
- Focus ONLY on relationships BETWEEN DIFFERENT FIELDS
- Do not include single-field constraints in mathematical relationships
- Ensure all mathematical relationships involve at least two distinct fields
- Current time can be used as an implicit field in mathematical relationships (e.g., for age/experience calculations)

Return a JSON dictionary without any preamble or explanation.
"""


# 格式提示词
format_system_prompt = """
   {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": {
                       "column": "<column name>",
                       "properties": {
                            "uniqueness_ratio": <float>,
                            "dtype": "<data type>",
                            "value_counts": <Optional compact top-k counts for category-like fields; never a complete whitelist>
                            "top_value_counts": <Optional top-k observed values>
                            "rare_value_examples": <Optional low-frequency examples used as anomaly evidence>
                            "shape_counts": <Optional observed value-shape frequencies>
                            "shape_family_counts": <Optional grouped shape frequencies: plain_numeric, percent_numeric, unit_or_text_numeric, missing_like, other>
                            "numeric_text_skeleton_counts": <Optional values normalized by replacing numbers with <num>>
                            "normalized_numeric_text_skeleton_counts": <Optional skeleton counts after normalizing case, punctuation, and common unit spellings>
                            "normalized_numeric_text_skeleton_groups": <Optional raw skeleton variants grouped under the same normalized skeleton>
                            "numeric_value_profiles": <Optional numeric min/quantile/max statistics grouped by normalized skeleton>
                            "dominant_shape_family": <Optional dominant grouped shape family>
                            "dominant_numeric_text_skeleton": <Optional most frequent raw numeric skeleton>
                            "dominant_normalized_numeric_text_skeleton": <Optional most frequent normalized numeric skeleton>
                            "case_punctuation_variant_groups": <Optional groups that expose case/punctuation variants sharing the same normalized text>
                            "missing_like_values": <Optional missing-placeholder frequencies>
                            "std": <Optional, exists when the field type is numeric>,
                            "min": <Optional, exists when the field type is numeric>,
                            "max": <Optional, exists when the field type is numeric>,
                            "num_unique_values": <>,
                            "semantic_type": "<semantic type>",
                            "description": "<brief description>"
                            }
                       },
            "field_relationships": ""
    }


Return a JSON dictionary without any preamble or explanation.

Do not expand compact profiling fields into full observed-value lists. Preserve
`value_counts` as compact top-k evidence only. The summary must help later LLM
modules infer canonical valid formats from field semantics and observed patterns,
not whitelist every value that appeared in a dirty dataset.

"""


# format_prompt="""
# Return exactly the following JSON, with no extra keys or commentary:
#             {
#             "rules": {
#                 "ColumnA": {"format": "<description or regex>", "regex": "<anchored regex or empty>"},
#                 "ColumnB": {"format": "<description or regex>", "regex": "<anchored regex or empty>"},
#                 …
#               }
#             }
# """

def _normalize_obj(obj):
    """
    将 numpy 标量、Pandas Timestamp 等转换为纯 Python 类型，
    并递归处理 dict 和 list，确保可 JSON 序列化。
    """
    if isinstance(obj, dict):
        return {k: _normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_obj(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


_MISSING_LIKE_VALUES = {
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

_PROFILE_METADATA_KEYS = (
    "top_value_counts",
    "rare_value_examples",
    "shape_counts",
    "shape_family_counts",
    "date_component_profile",
    "text_noise_examples",
    "numeric_text_skeleton_counts",
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
)

FORMAT_RULE_REQUEST_TIMEOUT = int(os.getenv("LAED_FORMAT_RULE_REQUEST_TIMEOUT", "60"))
FORMAT_RULE_REPAIR_REQUEST_TIMEOUT = int(os.getenv("LAED_FORMAT_RULE_REPAIR_REQUEST_TIMEOUT", "30"))
FORMAT_RULE_REPAIR_ROUNDS = int(os.getenv("LAED_FORMAT_RULE_REPAIR_ROUNDS", "0"))
FORMAT_RULE_REVIEW_ROUNDS = int(os.getenv("LAED_FORMAT_RULE_REVIEW_ROUNDS", "0"))


class Summarizer():
    def __init__(self) -> None:
        self.summary = None

    @staticmethod
    def _stringify_profile_value(value) -> str:
        if pd.isna(value):
            return "nan"
        return str(value).strip()

    @staticmethod
    def _value_shape(value: str) -> str:
        value = "" if value is None else str(value).strip()
        if Summarizer._is_missing_like_text(value):
            return "<missing-like>"
        chars = []
        for ch in value:
            if ch.isdigit():
                chars.append("9")
            elif ch.isalpha():
                chars.append("A")
            elif ch.isspace():
                chars.append(" ")
            else:
                chars.append(ch)
        return "".join(chars)

    @staticmethod
    def _is_missing_like_text(value: str) -> bool:
        text = str(value or "").strip().lower()
        normalized = re.sub(r"^[\\s\\{\\}\\[\\]\\(\\)\"']+|[\\s\\{\\}\\[\\]\\(\\)\"']+$", "", text)
        return text in _MISSING_LIKE_VALUES or normalized in _MISSING_LIKE_VALUES

    @staticmethod
    def _text_noise_reasons(value: str) -> list[str]:
        text = str(value or "")
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

    @staticmethod
    def _date_component_profile(values: pd.Series) -> dict:
        component_counts = [Counter(), Counter(), Counter()]
        delimiter_counts = Counter()
        total = 0
        for value in values:
            text = str(value or "").strip()
            if Summarizer._is_missing_like_text(text):
                continue
            match = re.fullmatch(r"(\d{1,4})([\/\-.])(\d{1,4})\2(\d{1,4})", text)
            if not match:
                continue
            parts = [int(match.group(1)), int(match.group(3)), int(match.group(4))]
            delimiter_counts[match.group(2)] += 1
            total += 1
            for pos, part in enumerate(parts):
                component_counts[pos][part] += 1
        if not total:
            return {}
        positions = []
        for counts in component_counts:
            positions.append({
                "top_values": {str(k): int(v) for k, v in counts.most_common(8)},
                "gt_12_count": int(sum(v for k, v in counts.items() if k > 12)),
                "gt_31_count": int(sum(v for k, v in counts.items() if k > 31)),
                "unique_count": int(len(counts)),
            })
        return {
            "matched_count": int(total),
            "delimiter_counts": {str(k): int(v) for k, v in delimiter_counts.items()},
            "positions": positions,
        }

    @staticmethod
    def _tokenized_column_text(column: str) -> str:
        tokenized_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(column))
        tokenized_text = re.sub(r"[_\-/]+", " ", tokenized_text)
        # Some compact schema names use suffix abbreviations without a camel
        # boundary. Split the generic suffix for semantic gating; value
        # evidence still decides whether the field can act as a key.
        tokenized_text = re.sub(r"(?i)(?<=[a-z])(?=avg\b)", " ", tokenized_text)
        return tokenized_text

    @staticmethod
    def _is_identifier_like_column(column: str, props: dict | None = None) -> bool:
        props = props or {}
        text = f"{column} {props.get('semantic_type', '')} {props.get('description', '')}".lower()
        tokenized_text = Summarizer._tokenized_column_text(column).lower()
        search_text = f"{text} {tokenized_text}"
        concrete_identifier_tokens = (
            "id",
            "identifier",
            "code",
            "key",
            "no.",
            "record",
            "reference",
            "registry",
            "accession",
            "catalog",
            "issn",
            "isbn",
            "doi",
        )
        if any(token in search_text for token in ("time", "date", "duration", "timestamp", "year")):
            return False
        if Summarizer._field_text_has_token(
                search_text,
                ("address", "street", "phone", "telephone", "contact", "postal", "zip"),
        ):
            return False
        measurement_tokens = (
            "count",
            "rating",
            "score",
            "sample",
            "amount",
            "value",
            "rate",
            "percent",
            "percentage",
            "average",
            "avg",
            "metric",
        )
        column_text = tokenized_text
        has_concrete_identifier = any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", column_text)
            for token in concrete_identifier_tokens
        ) or Summarizer._field_text_has_token(search_text, ("identifier", "unique identifier"))
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", search_text)
            for token in measurement_tokens
        ) and not has_concrete_identifier:
            return False
        if has_concrete_identifier:
            return True
        if Summarizer._field_text_has_token(column_text, ("number", "num", "no")):
            return not Summarizer._is_open_text_column(column, props)
        if Summarizer._has_structured_identifier_profile(props):
            return True
        return False

    @staticmethod
    def _is_associative_key_candidate(column: str, props: dict | None = None) -> bool:
        """
        Decide whether a field may serve as the determinant side of an
        associative dependency. This is deliberately stricter than
        _is_identifier_like_column: phone numbers, addresses, postal codes,
        titles, and abbreviations are structured attributes, but they should not
        be promoted to keys that determine broad entity attributes.
        """
        props = props or {}
        field_text = Summarizer._field_text(column, props)
        column_text = Summarizer._tokenized_column_text(column).lower()

        if Summarizer._is_temporal_like_column(column, props):
            return False

        if Summarizer._field_text_has_token(
                field_text,
                ("address", "street", "phone", "telephone", "contact", "postal", "zip"),
        ):
            return False

        if Summarizer._field_text_has_token(field_text, ("title", "abbreviation")) and not Summarizer._field_text_has_token(
                field_text,
                ("id", "identifier", "code", "key", "issn", "isbn", "doi"),
        ):
            return False

        if Summarizer._field_text_has_token(
                field_text,
                ("count", "rating", "score", "sample", "amount", "value", "rate", "percent", "percentage", "average", "avg"),
        ):
            return False

        return Summarizer._is_identifier_like_column(column, props)

    @staticmethod
    def _entity_namespace_prefix(column: str) -> str:
        text = str(column or "").strip().lower()
        if "_" not in text:
            return ""
        first = re.split(r"[_\s\-]+", text, maxsplit=1)[0]
        if not re.fullmatch(r"[a-z]{3,}", first or ""):
            return ""
        return first

    @classmethod
    def _namespace_compatible(cls, key: str, target: str) -> bool:
        key_prefix = cls._entity_namespace_prefix(key)
        target_prefix = cls._entity_namespace_prefix(target)
        return not key_prefix or not target_prefix or key_prefix == target_prefix

    @classmethod
    def _is_contact_or_address_attribute(cls, column: str, props: dict | None = None) -> bool:
        return cls._field_text_has_token(
            cls._field_text(column, props),
            ("address", "street", "phone", "telephone", "contact", "postal", "zip"),
        )

    @staticmethod
    def _is_numeric_measure_like_key(props: dict | None = None) -> bool:
        props = props or {}
        field_text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        shape_family_counts = props.get("shape_family_counts") or {}
        total = sum(int(value or 0) for value in shape_family_counts.values())
        if not total:
            return False
        numeric_count = (
            int(shape_family_counts.get("plain_numeric") or 0)
            + int(shape_family_counts.get("percent_numeric") or 0)
        )
        if numeric_count / total < 0.80:
            return False
        skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or props.get("numeric_text_skeleton_counts") or {}
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", field_text)
            for token in ("count", "rating", "score", "sample", "amount", "value", "rate", "percent", "percentage")
        ):
            return True
        return len(skeleton_counts) <= 5

    @staticmethod
    def _field_text_has_token(field_text: str, tokens: tuple[str, ...]) -> bool:
        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", field_text)
            for token in tokens
        )

    @staticmethod
    def _field_text(column: str, props: dict | None = None) -> str:
        props = props or {}
        tokenized_column = Summarizer._tokenized_column_text(column)
        return f"{column} {tokenized_column} {props.get('semantic_type', '')} {props.get('description', '')}".lower()

    @staticmethod
    def _profile_family_totals(props: dict | None = None) -> tuple[int, int, int, int, int]:
        props = props or {}
        counts = props.get("shape_family_counts") or {}

        def count(key: str) -> int:
            try:
                return int(counts.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        plain = count("plain_numeric")
        percent = count("percent_numeric")
        unit = count("unit_or_text_numeric")
        missing = count("missing_like")
        total = sum(count(key) for key in ("plain_numeric", "percent_numeric", "unit_or_text_numeric", "missing_like", "other"))
        return total, max(0, total - missing), plain, percent, unit

    @staticmethod
    def _has_structured_identifier_profile(props: dict | None = None) -> bool:
        props = props or {}
        shape_counts = props.get("shape_counts") or {}
        if not shape_counts:
            return False
        total = 0
        structured = 0
        for shape, raw_count in shape_counts.items():
            shape_text = str(shape)
            if shape_text == "<missing-like>":
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            total += count
            if (
                re.fullmatch(r"9+", shape_text)
                or re.fullmatch(r"A+\d+(?:[-_/]A+|\d+)*", shape_text)
                or re.fullmatch(r"A+(?:[-_/]A+)*[-_/]\d+A?", shape_text)
                or re.fullmatch(r"\d+(?:[-_/]\d+)+A?", shape_text)
                or re.fullmatch(r"A{1,6}-\d{1,6}(?:-[A-Z]{2,6})*", shape_text)
            ):
                structured += count
        if not total:
            return False
        unique = int(props.get("num_unique_observed_values") or props.get("num_unique_values") or 0)
        return structured / total >= 0.70 and unique >= 2

    @staticmethod
    def _is_open_text_column(column: str, props: dict | None = None) -> bool:
        field_text = Summarizer._field_text(column, props)
        if Summarizer._field_text_has_token(field_text, ("zip", "postal", "phone", "telephone")):
            return False
        if Summarizer._field_text_has_token(field_text, ("address", "street")):
            return True
        if Summarizer._field_text_has_token(
                field_text,
                ("identifier", "id", "code", "issn", "isbn", "doi", "key"),
        ):
            return False
        if Summarizer._field_text_has_token(
                field_text,
                (
                    "number", "num", "time", "date", "duration", "year", "score", "rating",
                    "count", "percent", "percentage", "sample", "zip",
                    "postal", "phone", "telephone", "page", "pagination", "volume", "issue",
                ),
        ):
            return False
        if Summarizer._field_text_has_token(field_text, ("state", "province", "region")):
            return False
        return Summarizer._field_text_has_token(
            field_text,
            (
                "address", "street", "description", "title", "name", "creator",
                "author", "actor", "actors", "cast", "person", "location", "locations", "city", "county",
                "article", "free text", "textual",
            ),
        )

    @staticmethod
    def _is_list_like_column(column: str, props: dict | None = None) -> bool:
        field_text = Summarizer._field_text(column, props)
        if Summarizer._field_text_has_token(
                field_text,
                ("address", "street", "location", "locations", "description", "title"),
        ):
            return False
        return Summarizer._field_text_has_token(
            field_text,
            ("list", "language", "country", "genre", "category", "categories", "tags"),
        )

    @staticmethod
    def _top_values_alpha_case(props: dict | None = None) -> str:
        values = []
        for value in (props or {}).get("top_value_counts", {}) or {}:
            text = str(value).strip()
            if not text or Summarizer._is_missing_like_text(text):
                continue
            values.append(text)
        letters = "".join(ch for value in values for ch in value if ch.isalpha())
        if not letters:
            return "mixed"
        lower = sum(1 for ch in letters if ch.islower())
        upper = sum(1 for ch in letters if ch.isupper())
        if lower / max(1, lower + upper) >= 0.90:
            return "lower"
        if upper / max(1, lower + upper) >= 0.90:
            return "upper"
        return "mixed"

    @staticmethod
    def _alpha_class_from_profile(props: dict | None = None) -> str:
        case = Summarizer._top_values_alpha_case(props)
        if case == "lower":
            return "[a-z]"
        if case == "upper":
            return "[A-Z]"
        return "[A-Za-z]"

    @classmethod
    def _is_temporal_like_column(cls, column: str, props: dict | None = None) -> bool:
        return cls._field_text_has_token(
            cls._field_text(column, props),
            ("date", "time", "datetime", "timestamp", "duration", "year"),
        )

    @classmethod
    def _is_year_like_column(cls, column: str, props: dict | None = None) -> bool:
        return cls._field_text_has_token(cls._field_text(column, props), ("year",))

    @classmethod
    def _is_language_like_column(cls, column: str, props: dict | None = None) -> bool:
        return cls._field_text_has_token(cls._field_text(column, props), ("language",))

    @classmethod
    def _is_planned_temporal_like_column(cls, column: str, props: dict | None = None) -> bool:
        if not cls._is_temporal_like_column(column, props):
            return False
        return cls._field_text_has_token(
            cls._field_text(column, props),
            ("schedule", "scheduled", "sched", "planned", "plan", "expected"),
        )

    @classmethod
    def _is_geographic_hierarchy_pair(
            cls,
            key: str,
            key_props: dict | None,
            target: str,
            target_props: dict | None,
    ) -> bool:
        key_text = cls._field_text(key, key_props)
        target_text = cls._field_text(target, target_props)
        key_geo = cls._field_text_has_token(
            key_text,
            ("city", "town", "county", "state", "province", "country", "region"),
        )
        target_geo = cls._field_text_has_token(
            target_text,
            ("county", "state", "province", "country", "region"),
        )
        return key_geo and target_geo and key != target

    @classmethod
    def _is_low_cardinality_attribute(cls, column: str, props: dict | None, profile: dict, row_count: int) -> bool:
        props = props or {}
        unique_values = int(profile.get("unique_values") or props.get("num_unique_values") or 0)
        non_missing = int(profile.get("non_missing") or row_count or 0)
        if not non_missing:
            return False
        if unique_values <= max(12, int(non_missing * 0.08)):
            return True
        text = cls._field_text(column, props)
        return cls._field_text_has_token(
            text,
            ("language", "type", "service", "flag", "indicator", "category", "class", "label"),
        ) and unique_values <= max(60, int(non_missing * 0.12))

    @classmethod
    def _dependency_semantically_plausible(
            cls,
            key: str,
            key_props: dict | None,
            target: str,
            target_props: dict | None,
            target_profile: dict,
            row_count: int,
            key_is_identifier_like: bool,
    ) -> bool:
        key_text = cls._field_text(key, key_props)
        target_text = cls._field_text(target, target_props)

        if cls._is_language_like_column(target, target_props) and not cls._is_language_like_column(key, key_props):
            return False

        if cls._is_temporal_like_column(target, target_props) and not (
            cls._is_temporal_like_column(key, key_props)
            or cls._is_year_like_column(key, key_props)
            or key_is_identifier_like
        ):
            return False

        if key_is_identifier_like and not cls._namespace_compatible(key, target):
            return False

        if cls._is_contact_or_address_attribute(target, target_props):
            return cls._field_text_has_token(
                target_text,
                ("city", "county", "state", "province", "country", "region"),
            )

        if cls._field_text_has_token(key_text, ("abbreviation", "title")) and not cls._field_text_has_token(
                key_text,
                ("identifier", "id", "code", "key", "issn", "isbn", "doi"),
        ):
            return False

        if cls._field_text_has_token(key_text, ("avg", "average")):
            return False

        if cls._field_text_has_token(key_text, ("measure", "metric")):
            return key_is_identifier_like and cls._field_text_has_token(
                target_text,
                ("name", "description", "title", "label"),
            )

        if cls._field_text_has_token(
                target_text,
                ("name", "description", "title", "label", "type", "owner", "service", "city", "county", "state", "abbreviation"),
        ):
            return key_is_identifier_like or cls._is_geographic_hierarchy_pair(key, key_props, target, target_props)

        low_cardinality_target = cls._is_low_cardinality_attribute(target, target_props, target_profile, row_count)
        if low_cardinality_target and not key_is_identifier_like:
            return (
                cls._is_geographic_hierarchy_pair(key, key_props, target, target_props)
                or cls._is_temporal_like_column(key, key_props)
                or cls._is_year_like_column(target, target_props)
            )

        return True

    @staticmethod
    def _target_global_profile(values: pd.Series) -> dict:
        comparable = values.map(Summarizer._stringify_profile_value)
        comparable = comparable[~comparable.map(Summarizer._is_missing_like_text)]
        total = int(len(comparable))
        if not total:
            return {
                "non_missing": 0,
                "unique_values": 0,
                "dominant_ratio": 0.0,
            }
        counts = comparable.value_counts(dropna=False)
        dominant = int(counts.iloc[0]) if not counts.empty else 0
        return {
            "non_missing": total,
            "unique_values": int(counts.size),
            "dominant_ratio": round(dominant / total, 4) if total else 0.0,
        }

    @staticmethod
    def _is_near_constant_dependent(profile: dict, row_count: int) -> bool:
        unique_values = int(profile.get("unique_values") or 0)
        dominant_ratio = float(profile.get("dominant_ratio") or 0.0)
        non_missing = int(profile.get("non_missing") or 0)
        if non_missing < max(20, int(row_count * 0.05)):
            return True
        return dominant_ratio >= 0.90 and unique_values <= max(3, min(20, int(row_count * 0.05)))

    def _build_relationship_evidence(
            self,
            df: pd.DataFrame,
            data_summary: dict,
            max_keys: int = 12,
            max_dependents_per_key: int = 10
    ) -> dict:
        """
        Build compact, dataset-neutral evidence for functional dependencies.
        The evidence is not a hand-written rule: it summarizes whether repeated
        key values mostly map to stable dependent values. Identifier-like keys
        can be moderately noisy; non-identifier keys must show strong functional
        behavior so common-but-unrelated category co-occurrence is not promoted
        to a relationship.
        """
        field_props = {
            f.get("column"): f.get("properties", {})
            for f in data_summary.get("fields", [])
            if isinstance(f, dict)
        }
        row_count = len(df)
        candidates = []
        target_profiles = {
            col: self._target_global_profile(df[col])
            for col in df.columns
        }

        for key in df.columns:
            props = field_props.get(key, {})
            key_is_identifier_like = self._is_identifier_like_column(key, props)
            key_is_associative_candidate = self._is_associative_key_candidate(key, props)
            key_is_geo_hierarchy_source = self._field_text_has_token(
                self._field_text(key, props),
                ("city", "town", "county", "state", "province", "country", "region"),
            )

            key_values = df[key].map(self._stringify_profile_value)
            key_non_missing = key_values[~key_values.map(self._is_missing_like_text)]
            key_unique = int(key_non_missing.nunique(dropna=True))
            if key_unique <= 1 or key_unique > max(500, int(max(row_count * 0.8, 20))):
                continue
            key_unique_ratio = key_unique / max(1, int(len(key_non_missing)))

            dependents = []
            grouped = df.groupby(key_values, dropna=False)
            for target in df.columns:
                if target == key:
                    continue
                target_props = field_props.get(target, {})
                target_profile = target_profiles.get(target, {})
                if self._is_identifier_like_column(target, target_props) and not self._is_temporal_like_column(target, target_props):
                    continue
                key_is_temporal = self._is_temporal_like_column(key, props)
                target_is_temporal = self._is_temporal_like_column(target, target_props)
                is_geo_pair = self._is_geographic_hierarchy_pair(key, props, target, target_props)
                if key_is_temporal and not (
                    target_is_temporal
                    or self._is_year_like_column(target, target_props)
                ):
                    continue
                if target_is_temporal and not key_is_identifier_like and not key_is_temporal:
                    continue
                if (
                    not key_is_associative_candidate
                    and not is_geo_pair
                    and not (key_is_temporal and self._is_year_like_column(target, target_props))
                ):
                    continue
                if self._is_language_like_column(target, target_props) and not key_is_identifier_like:
                    continue
                if not self._dependency_semantically_plausible(
                    key,
                    props,
                    target,
                    target_props,
                    target_profile,
                    row_count,
                    key_is_identifier_like,
                ):
                    continue
                if (
                    not key_is_identifier_like
                    and self._is_numeric_measure_like_key(props)
                    and not self._is_identifier_like_column(target, target_props)
                ):
                    continue

                total = 0
                majority_total = 0
                conflicting_cells = 0
                groups_used = 0
                stable_groups = 0
                conflict_groups = 0
                example_conflicts = []

                for key_value, group in grouped:
                    if self._is_missing_like_text(key_value) or len(group) < 2:
                        continue
                    values = group[target].map(self._stringify_profile_value)
                    comparable = values[~values.map(self._is_missing_like_text)]
                    if len(comparable) < 2:
                        continue
                    counts = comparable.value_counts(dropna=False)
                    if counts.empty:
                        continue
                    majority_value = str(counts.index[0])
                    majority_count = int(counts.iloc[0])
                    group_total = int(counts.sum())
                    group_conflicts = group_total - majority_count

                    total += group_total
                    majority_total += majority_count
                    conflicting_cells += group_conflicts
                    groups_used += 1
                    if group_conflicts:
                        conflict_groups += 1
                    else:
                        stable_groups += 1

                    if group_conflicts and len(example_conflicts) < 4:
                        minority_values = [str(v) for v in counts.index[1:4]]
                        example_conflicts.append({
                            "key_value": str(key_value),
                            "canonical_value": majority_value,
                            "minority_values": minority_values,
                        })

                if total < max(20, int(row_count * 0.03)) or groups_used < 2:
                    continue

                consistency = majority_total / total if total else 0.0
                conflict_ratio = conflicting_cells / total if total else 0.0
                stable_group_ratio = stable_groups / groups_used if groups_used else 0.0
                strong_stable_dependency = (
                    consistency >= 0.98
                    and stable_group_ratio >= 0.90
                    and groups_used >= 5
                )
                actionable_conflicts = (
                    conflicting_cells >= 3
                    and conflict_ratio >= 0.001
                )
                if key_is_identifier_like:
                    keep_dependency = (
                        (consistency >= 0.50 and actionable_conflicts)
                        or strong_stable_dependency
                    )
                elif is_geo_pair and key_is_geo_hierarchy_source:
                    keep_dependency = (
                        consistency >= 0.97
                        and stable_group_ratio >= 0.90
                        and groups_used >= 10
                        and (
                            actionable_conflicts
                            or strong_stable_dependency
                        )
                    )
                elif self._is_temporal_like_column(key, props) and self._is_year_like_column(target, target_props):
                    keep_dependency = (
                        consistency >= 0.85
                        and stable_group_ratio >= 0.70
                        and groups_used >= 10
                        and actionable_conflicts
                    )
                elif self._is_temporal_like_column(key, props) or self._is_temporal_like_column(target, target_props):
                    keep_dependency = (
                        consistency >= 0.98
                        and stable_group_ratio >= 0.90
                        and (
                            actionable_conflicts
                            or strong_stable_dependency
                        )
                    )
                elif self._is_low_cardinality_attribute(target, target_props, target_profile, row_count):
                    keep_dependency = (
                        consistency >= 0.99
                        and stable_group_ratio >= 0.95
                        and strong_stable_dependency
                    )
                elif key_unique_ratio > 0.30:
                    keep_dependency = (
                        consistency >= 0.88
                        and stable_group_ratio >= 0.75
                        and groups_used >= 10
                        and (
                            actionable_conflicts
                            or strong_stable_dependency
                        )
                    )
                else:
                    keep_dependency = (
                        consistency >= 0.90
                        and stable_group_ratio >= 0.75
                        and (
                            actionable_conflicts
                            or strong_stable_dependency
                        )
                    )
                if not keep_dependency:
                    continue

                dependents.append({
                    "field": target,
                    "consistency_ratio": round(consistency, 4),
                    "conflicting_cells": int(conflicting_cells),
                    "conflict_ratio": round(conflict_ratio, 4),
                    "compared_cells": int(total),
                    "groups_used": int(groups_used),
                    "stable_groups": int(stable_groups),
                    "conflict_groups": int(conflict_groups),
                    "stable_group_ratio": round(stable_group_ratio, 4),
                    "key_is_identifier_like": bool(key_is_identifier_like),
                    "actionable_conflicts": bool(actionable_conflicts),
                    "stable_dependency": bool(strong_stable_dependency),
                    "target_global_profile": target_profile,
                    "examples": example_conflicts,
                })

            if dependents:
                dependents = sorted(
                    dependents,
                    key=lambda item: (
                        -item["consistency_ratio"],
                        -item["conflicting_cells"],
                        item["field"],
                    ),
                )[:max_dependents_per_key]
                candidates.append({
                    "key": key,
                    "key_unique_values": key_unique,
                    "dependents": dependents,
                })

        candidates = sorted(
            candidates,
            key=lambda item: (
                -max(dep["consistency_ratio"] for dep in item["dependents"]),
                -sum(dep["conflicting_cells"] for dep in item["dependents"]),
                item["key"],
            ),
        )[:max_keys]
        return {"candidate_associative_dependencies": candidates}

    def _refine_relationships_with_evidence(self, data_summary: dict, relationship_evidence: dict) -> dict:
        candidates = relationship_evidence.get("candidate_associative_dependencies") or []
        if not candidates:
            data_summary["relationship_evidence"] = relationship_evidence
            return data_summary

        compact_fields = []
        for field in data_summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            props = field.get("properties", {}) or {}
            compact_fields.append({
                "column": field.get("column"),
                "dtype": props.get("dtype", ""),
                "semantic_type": props.get("semantic_type", ""),
                "description": str(props.get("description", ""))[:160],
                "num_unique_values": props.get("num_unique_values"),
                "dominant_shape_family": props.get("dominant_shape_family", ""),
            })

        compact_evidence = {"candidate_associative_dependencies": []}
        for candidate in candidates[:12]:
            compact_deps = []
            for dep in (candidate.get("dependents", []) or [])[:12]:
                compact_deps.append({
                    "field": dep.get("field"),
                    "consistency_ratio": dep.get("consistency_ratio"),
                    "conflicting_cells": dep.get("conflicting_cells"),
                    "groups_used": dep.get("groups_used"),
                    "stable_dependency": dep.get("stable_dependency"),
                    "actionable_conflicts": dep.get("actionable_conflicts"),
                })
            compact_evidence["candidate_associative_dependencies"].append({
                "key": candidate.get("key"),
                "key_unique_values": candidate.get("key_unique_values"),
                "dependents": compact_deps,
            })

        prompt = f"""
You are refining the field_relationships section of an LLM-generated data summary.
Use only the current summary and the compact, data-derived relationship evidence.
Do not add domain-specific rules. Add an associative dependency only when an
identifier-like key mostly maps to stable dependent attributes in the evidence.

Current field_relationships:
{json.dumps(data_summary.get("field_relationships", {}), ensure_ascii=False, indent=2)}

Fields:
{json.dumps(compact_fields, ensure_ascii=False, indent=2)}

Relationship evidence:
{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}

Requirements:
- Preserve existing valid hierarchical, mathematical, temporal, and associative relationships.
- Remove or leave empty relationships that are only semantic guesses and are not
  supported by the current fields and evidence.
- For associative dependencies, the key is contextual evidence. The dependent
  fields are the cells that should be checked for majority-mapping conflicts.
- Prefer dependent fields with repeated observations and clear majority mappings.
- Do not add dependencies between unrelated descriptive fields only because a
  weak statistical association exists.

Return exactly this JSON:
{{
  "field_relationships": {{
    "hierarchical": {{}},
    "mathematical": {{}},
    "temporal": [],
    "associative": {{}}
  }}
}}
"""
        try:
            response = summarizer_generator.send_message(
                [{"role": "user", "content": prompt}],
                max_tokens=summarizer_generator.max_tokens,
                request_timeout=int(os.getenv("LAED_RELATIONSHIP_REQUEST_TIMEOUT", "90")),
                retries=1,
            )
            parsed = json.loads(clean_code_snippet(response))
            relationships = parsed.get("field_relationships", parsed)
            if isinstance(relationships, dict):
                data_summary["field_relationships"] = {
                    "hierarchical": relationships.get("hierarchical", {}),
                    "mathematical": relationships.get("mathematical", {}),
                    "temporal": relationships.get("temporal", []),
                    "associative": relationships.get("associative", {}),
                }
        except Exception as exc:
            print(f"[Warning] relationship evidence refinement failed: {exc}")

        relationships = data_summary.get("field_relationships", {}) or {}
        if not isinstance(relationships, dict):
            relationships = {}
        relationships.setdefault("hierarchical", {})
        relationships.setdefault("mathematical", {})
        relationships.setdefault("temporal", [])
        relationships.setdefault("associative", {})
        if not isinstance(relationships["associative"], dict):
            relationships["associative"] = {}

        evidence_pairs = {
            (str(candidate.get("key")), str(dep.get("field")))
            for candidate in candidates
            for dep in (candidate.get("dependents", []) or [])
            if candidate.get("key") and dep.get("field")
        }
        pruned_associative = {}
        for key, deps in relationships["associative"].items():
            if not isinstance(deps, list):
                continue
            kept_deps = [
                str(dep)
                for dep in deps
                if (str(key), str(dep)) in evidence_pairs
            ]
            if kept_deps:
                pruned_associative[str(key)] = sorted(set(kept_deps))
        relationships["associative"] = pruned_associative

        # Generic merge: if the LLM omits a strong identifier -> attribute
        # candidate that is present in compact evidence, keep it in the summary
        # so downstream validation can still execute the summary evidence.
        for candidate in candidates:
            key = candidate.get("key")
            deps = [
                dep.get("field")
                for dep in candidate.get("dependents", [])
                if dep.get("field")
                and float(dep.get("consistency_ratio") or 0) >= 0.5
                and (
                    (
                        bool(dep.get("key_is_identifier_like"))
                        and bool(dep.get("actionable_conflicts"))
                    )
                    or (
                        float(dep.get("stable_group_ratio") or 0) >= 0.50
                        and (
                            bool(dep.get("actionable_conflicts"))
                            or bool(dep.get("stable_dependency"))
                        )
                    )
                    or int(dep.get("conflicting_cells") or 0) >= 5
                )
            ]
            if key and deps:
                existing = relationships["associative"].get(key, [])
                if not isinstance(existing, list):
                    existing = []
                relationships["associative"][key] = sorted(set(existing) | set(deps))

        data_summary["field_relationships"] = relationships
        data_summary["relationship_evidence"] = relationship_evidence
        return data_summary

    @staticmethod
    def _case_punctuation_key(value: str) -> str:
        value = "" if value is None else str(value).strip().lower()
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"[\s\.,;:]+$", "", value)
        return value

    @staticmethod
    def _numeric_text_skeleton(value: str) -> str:
        value = "" if value is None else str(value).strip().lower()
        value = re.sub(r"\d+(?:\.\d+)?", "<num>", value)
        value = re.sub(r"\s+", " ", value)
        return value

    @staticmethod
    def _normalized_numeric_text_skeleton(value: str) -> str:
        skeleton = Summarizer._numeric_text_skeleton(value)
        if skeleton.lower() in _MISSING_LIKE_VALUES:
            return skeleton
        skeleton = re.sub(r"\s+", " ", skeleton).strip().lower()
        skeleton = re.sub(r"[\s\.,;:]+$", "", skeleton)
        skeleton = re.sub(r"(?<=<num> )([a-z]{3,})s\b", r"\1", skeleton)
        return skeleton

    @staticmethod
    def _shape_family_counts(shape_counts: dict) -> dict:
        families = {
            "plain_numeric": 0,
            "percent_numeric": 0,
            "unit_or_text_numeric": 0,
            "missing_like": 0,
            "other": 0,
        }
        for shape, count in (shape_counts or {}).items():
            shape = str(shape)
            count = int(count or 0)
            if shape == "<missing-like>":
                families["missing_like"] += count
            elif "9" in shape and "%" in shape:
                families["percent_numeric"] += count
            elif "9" in shape and "A" in shape:
                families["unit_or_text_numeric"] += count
            elif "9" in shape and "A" not in shape:
                families["plain_numeric"] += count
            else:
                families["other"] += count
        return families

    def _numeric_value_profiles_by_skeleton(self, values: pd.Series) -> dict:
        profiles = {}
        buckets = {}
        for value in values:
            text = "" if value is None else str(value).strip()
            if text.lower() in _MISSING_LIKE_VALUES:
                continue
            skeleton = self._normalized_numeric_text_skeleton(text)
            matches = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
            if not matches:
                continue
            try:
                number = float(matches[0])
            except ValueError:
                continue
            buckets.setdefault(skeleton, []).append(number)
        for skeleton, nums in buckets.items():
            if not nums:
                continue
            arr = pd.Series(nums, dtype="float64")
            profiles[skeleton] = {
                "count": int(arr.shape[0]),
                "min": float(arr.min()),
                "p05": float(arr.quantile(0.05)),
                "median": float(arr.median()),
                "p95": float(arr.quantile(0.95)),
                "max": float(arr.max()),
            }
        return profiles

    def _build_observed_value_profile(
            self,
            series: pd.Series,
            top_n: int = 12,
            rare_n: int = 12,
            shape_n: int = 12
    ) -> dict:
        values = series.map(self._stringify_profile_value)
        counts = values.value_counts(dropna=False)
        total = int(len(values))
        top_counts = counts.head(top_n)
        rare_counts = counts.sort_values(ascending=True, kind="mergesort").head(rare_n)
        shape_counts = values.map(self._value_shape).value_counts(dropna=False).head(shape_n)
        shape_family_counts = self._shape_family_counts(
            {str(k): int(v) for k, v in values.map(self._value_shape).value_counts(dropna=False).items()}
        )
        numeric_family_count = (
            shape_family_counts.get("plain_numeric", 0)
            + shape_family_counts.get("percent_numeric", 0)
            + shape_family_counts.get("unit_or_text_numeric", 0)
        )
        include_numeric_skeletons = total > 0 and numeric_family_count / total >= 0.2
        skeleton_counts = (
            values.map(self._numeric_text_skeleton).value_counts(dropna=False).head(shape_n)
            if include_numeric_skeletons
            else pd.Series(dtype=int)
        )
        normalized_skeleton_counts_all = (
            values.map(self._normalized_numeric_text_skeleton).value_counts(dropna=False)
            if include_numeric_skeletons
            else pd.Series(dtype=int)
        )
        normalized_skeleton_counts = normalized_skeleton_counts_all.head(shape_n)
        dominant_shape_family = max(
            shape_family_counts.items(),
            key=lambda item: item[1],
            default=("", 0),
        )[0]
        dominant_skeleton = skeleton_counts.index[0] if not skeleton_counts.empty else ""
        dominant_normalized_skeleton = (
            normalized_skeleton_counts.index[0] if not normalized_skeleton_counts.empty else ""
        )
        normalized_groups = {}
        raw_skeleton_counts_all = (
            values.map(self._numeric_text_skeleton).value_counts(dropna=False)
            if include_numeric_skeletons
            else pd.Series(dtype=int)
        )
        for raw_skeleton, count in raw_skeleton_counts_all.items():
            key = self._normalized_numeric_text_skeleton(raw_skeleton)
            if not key or key in _MISSING_LIKE_VALUES:
                continue
            group = normalized_groups.setdefault(key, {})
            group[str(raw_skeleton)] = int(count)
        normalized_groups = {
            key: dict(sorted(group.items(), key=lambda item: item[1], reverse=True)[:8])
            for key, group in normalized_groups.items()
            if len(group) > 1
        }
        normalized_groups = dict(
            sorted(
                normalized_groups.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True
            )[:10]
        )
        variant_groups = {}
        for raw_value, count in counts.items():
            key = self._case_punctuation_key(raw_value)
            if not key or key in _MISSING_LIKE_VALUES:
                continue
            group = variant_groups.setdefault(key, {})
            group[str(raw_value)] = int(count)
        variant_groups = {
            key: dict(sorted(group.items(), key=lambda item: item[1], reverse=True)[:8])
            for key, group in variant_groups.items()
            if len(group) > 1
        }
        variant_groups = dict(
            sorted(
                variant_groups.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True
            )[:10]
        )
        missing_like_counts = {
            str(value): int(count)
            for value, count in counts.items()
            if self._is_missing_like_text(str(value))
        }
        text_noise_examples = {}
        for value, count in counts.items():
            reasons = self._text_noise_reasons(str(value))
            if reasons:
                text_noise_examples[str(value)] = {
                    "count": int(count),
                    "reasons": reasons,
                }
            if len(text_noise_examples) >= rare_n:
                break
        top_total = int(top_counts.sum()) if total else 0

        return {
            "top_value_counts": {str(k): int(v) for k, v in top_counts.items()},
            "rare_value_examples": {str(k): int(v) for k, v in rare_counts.items()},
            "shape_counts": {str(k): int(v) for k, v in shape_counts.items()},
            "shape_family_counts": shape_family_counts,
            "date_component_profile": self._date_component_profile(values),
            "text_noise_examples": text_noise_examples,
            "numeric_text_skeleton_counts": {str(k): int(v) for k, v in skeleton_counts.items()},
            "normalized_numeric_text_skeleton_counts": {str(k): int(v) for k, v in normalized_skeleton_counts.items()},
            "normalized_numeric_text_skeleton_groups": normalized_groups,
            "numeric_value_profiles": self._numeric_value_profiles_by_skeleton(values) if include_numeric_skeletons else {},
            "dominant_shape_family": dominant_shape_family,
            "dominant_numeric_text_skeleton": str(dominant_skeleton),
            "dominant_normalized_numeric_text_skeleton": str(dominant_normalized_skeleton),
            "case_punctuation_variant_groups": variant_groups,
            "missing_like_values": missing_like_counts,
            "num_unique_observed_values": int(counts.shape[0]),
            "value_counts_truncated": bool(counts.shape[0] > top_n),
            "top_value_coverage": round(top_total / total, 4) if total else 0.0,
        }

    def _refresh_profile_metadata(self, summary: dict, df: pd.DataFrame) -> dict:
        if not isinstance(summary, dict):
            return summary
        for field in summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            col = field.get("column")
            props = field.get("properties", {})
            if col not in df.columns or not isinstance(props, dict):
                continue
            if props.get("dtype") in ["category", "number", "integer", "int", "float", "date", "string"]:
                for key in _PROFILE_METADATA_KEYS:
                    props.pop(key, None)
                props.update(self._build_observed_value_profile(df[col]))
                if props.get("dtype") == "category":
                    props["value_counts"] = props["top_value_counts"]
                else:
                    props.pop("value_counts", None)
        return summary

    def _normalize_field_dtypes_from_profiles(self, summary: dict) -> dict:
        """
        Keep dtype as a physical representation label. Low-cardinality columns
        with structured values should not stay "category" merely because they
        repeat; semantic_type carries the business meaning such as identifier,
        time, measurement, or percentage.
        """
        if not isinstance(summary, dict):
            return summary
        for field in summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            col = str(field.get("column") or "")
            props = field.get("properties", {})
            if not col or not isinstance(props, dict):
                continue

            total, non_missing, plain, percent, unit = self._profile_family_totals(props)
            if not total or not non_missing:
                continue
            field_text = self._field_text(col, props)
            current = str(props.get("dtype", "") or "").lower()
            numeric_ratio = (plain + percent + unit) / max(1, non_missing)
            plain_ratio = plain / max(1, non_missing)
            percent_ratio = percent / max(1, non_missing)
            unit_ratio = unit / max(1, non_missing)

            if self._field_text_has_token(field_text, ("date", "datetime", "timestamp")):
                props["dtype"] = "date"
            elif self._field_text_has_token(field_text, ("time", "duration")):
                props["dtype"] = "string"
            elif self._field_text_has_token(field_text, ("phone", "telephone", "zip", "postal")):
                props["dtype"] = "string"
            elif self._field_text_has_token(field_text, ("page", "pagination")):
                props["dtype"] = "string"
            elif self._field_text_has_token(field_text, ("identifier", "id", "code", "issn", "isbn", "doi", "key", "reference", "record")):
                props["dtype"] = "string"
            elif self._field_text_has_token(field_text, ("percent", "percentage", "score", "rating", "count", "sample", "amount", "value", "measure", "measurement")):
                props["dtype"] = "number" if plain_ratio >= 0.60 or percent_ratio >= 0.60 else "string"
            elif numeric_ratio >= 0.90 and current != "date" and not self._is_open_text_column(col, props):
                props["dtype"] = "number" if unit_ratio < 0.50 else "string"
            elif self._is_open_text_column(col, props):
                props["dtype"] = "string"
            elif current not in {"number", "date", "boolean", "string", "category"}:
                props["dtype"] = "string"

            if props.get("dtype") == "category":
                props["value_counts"] = props.get("top_value_counts", {})
            else:
                props.pop("value_counts", None)
        return summary

    @staticmethod
    def _extract_literal_alternatives(regex: str) -> list[str]:
        pattern = str(regex or "").strip()
        if not pattern.startswith("^") or not pattern.endswith("$") or "|" not in pattern:
            return []
        inner = pattern[1:-1]
        if inner.startswith("(?:") and inner.endswith(")"):
            inner = inner[3:-1]
        elif inner.startswith("(") and inner.endswith(")"):
            inner = inner[1:-1]
        parts = re.split(r"(?<!\\)\|", inner)
        if len(parts) <= 1:
            return []
        literals = []
        for part in parts:
            if re.search(r"(?<!\\)[\[\]\+\*\?\{\}]", part):
                return []
            literals.append(part.replace("\\.", ".").replace("\\(", "(").replace("\\)", ")").replace("\\|", "|"))
        return literals

    def _needs_llm_rule_review(self, col: str, rule: dict, meta: dict, sample_values: list[str]) -> tuple[bool, list[str]]:
        regex = str(rule.get("regex", "") or "")
        reasons = []
        if not regex:
            return False, reasons
        literals = self._extract_literal_alternatives(regex)
        if literals and (
                len(literals) > 8
                or meta.get("value_counts_truncated")
                or len(literals) >= max(4, len(meta.get("top_value_counts", {})) // 2)
        ):
            reasons.append("regex looks like an observed-value whitelist rather than a canonical pattern")
        missing_values = list((meta.get("missing_like_values") or {}).keys())
        for value in missing_values:
            try:
                if re.fullmatch(regex, str(value).strip()):
                    reasons.append(f"regex matches missing-like observed value: {value!r}")
                    break
            except re.error:
                break
        for value in (meta.get("rare_value_examples") or {}).keys():
            try:
                if re.fullmatch(regex, str(value).strip()) and self._rare_match_requires_review(value, meta):
                    reasons.append(f"regex matches rare value that conflicts with the dominant profile: {value!r}")
                    break
            except re.error:
                break
        dtype = str(meta.get("dtype", "") or "").lower()
        shape_family_counts = meta.get("shape_family_counts") or {}
        plain_count = int(shape_family_counts.get("plain_numeric") or 0)
        percent_count = int(shape_family_counts.get("percent_numeric") or 0)
        unit_count = int(shape_family_counts.get("unit_or_text_numeric") or 0)
        if dtype in {"number", "float", "int", "integer"} or (plain_count + percent_count + unit_count > 0):
            top_values = list((meta.get("top_value_counts") or {}).keys())
            percent_top = sum(1 for value in top_values if "%" in str(value))
            plain_numeric_top = sum(
                1
                for value in top_values
                if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(value).strip())
            )
            if percent_top and plain_numeric_top or len([c for c in [plain_count, percent_count, unit_count] if c > 0]) > 1:
                reasons.append("column has mixed plain, percent-marked, or unit-marked numeric forms; LLM must choose the dominant canonical representation")
        try:
            samples = [str(v).strip() for v in sample_values[:120]]
            matched = sum(1 for value in samples if re.fullmatch(regex, value))
            if samples and matched / len(samples) < 0.35:
                reasons.append("regex matches very few sampled values, suggesting it selected a minority format or wrong casing")
        except re.error:
            pass
        top_counts = meta.get("top_value_counts") or {}
        if top_counts:
            try:
                unmatched_top = 0
                matched_top = 0
                mismatched_same_normalized = 0
                dominant_skeleton = str(meta.get("dominant_normalized_numeric_text_skeleton") or "")
                for value, count in top_counts.items():
                    text = str(value).strip()
                    if self._is_missing_like_text(text):
                        continue
                    value_skeleton = self._normalized_numeric_text_skeleton(text)
                    if re.fullmatch(regex, text):
                        matched_top += int(count or 0)
                    else:
                        unmatched_top += int(count or 0)
                        if dominant_skeleton and value_skeleton == dominant_skeleton:
                            mismatched_same_normalized += int(count or 0)
                if (
                        unmatched_top >= max(10, int((matched_top + unmatched_top) * 0.15))
                        and not mismatched_same_normalized
                ):
                    reasons.append("regex rejects a recurring top-count value; LLM should include all recurring canonical surface forms")
            except (re.error, TypeError, ValueError):
                pass
        example_values = list((meta.get("top_value_counts") or {}).keys()) + list((meta.get("rare_value_examples") or {}).keys())
        if re.search(r"\[A-Za-z\]\+", regex):
            for value in example_values:
                parts = [part.strip() for part in str(value).split(",")]
                if any(re.search(r"[ \-'\.\(\)]", part) for part in parts if part):
                    reasons.append("regex uses bare alphabetic list items although observed valid-looking items contain spaces or punctuation")
                    break
        if re.search(r"\\\(\\w\+\\\)", regex):
            for value in example_values:
                match = re.search(r"\(([^)]*[\s-][^)]*)\)", str(value))
                if match:
                    reasons.append("regex allows only single-word parenthesized qualifiers although observed qualifiers can contain spaces or hyphens")
                    break
        if "date" in str(meta.get("semantic_type", "")).lower() or "date" in str(meta.get("description", "")).lower():
            date_like_values = [str(value).strip() for value in example_values if not self._is_missing_like_text(str(value))]
            has_year_only = any(re.fullmatch(r"\d{4}\s*\([^)]*\)", value) for value in date_like_values)
            has_month_year = any(re.fullmatch(r"[A-Za-z]{3,9}\s+\d{4}\s*\([^)]*\)", value) for value in date_like_values)
            if (has_year_only or has_month_year) and re.search(r"\\d\{1,2\}.*\\d\{4\}", regex):
                reasons.append("date regex forces day-month-year although recurring valid-looking values use coarser date granularity")
            date_profile = meta.get("date_component_profile") or {}
            if int(date_profile.get("matched_count") or 0) >= 20:
                date_top_values = [
                    str(value).strip()
                    for value in example_values
                    if re.search(r"\d{1,4}[\/\-.]\d{1,4}[\/\-.]\d{1,4}", str(value))
                ]
                if date_top_values:
                    try:
                        matched_dates = sum(1 for value in date_top_values if re.fullmatch(regex, value))
                        if matched_dates / max(1, len(date_top_values)) < 0.50:
                            reasons.append("date regex misses recurring date-delimited values")
                    except re.error:
                        pass
        field_text = f"{meta.get('semantic_type', '')} {meta.get('description', '')}".lower()
        if self._field_text_has_token(field_text, ("page", "pagination")):
            page_range_values = [
                str(value).strip()
                for value in example_values
                if re.fullmatch(r"\d+-\d+", str(value).strip())
            ]
            if page_range_values:
                try:
                    matched_ranges = sum(1 for value in page_range_values if re.fullmatch(regex, value))
                    if matched_ranges / max(1, len(page_range_values)) < 0.50:
                        reasons.append("pagination regex misses recurring page-range values")
                except re.error:
                    pass
        shape_counts = meta.get("shape_counts") or {}
        if shape_counts:
            try:
                represented_shapes = set()
                for value in (meta.get("top_value_counts") or {}).keys():
                    text = str(value).strip()
                    if self._is_missing_like_text(text):
                        continue
                    if re.fullmatch(regex, text):
                        represented_shapes.add(self._value_shape(text))
                total_shape = 0
                represented_shape = 0
                for shape, count in shape_counts.items():
                    if str(shape) == "<missing-like>":
                        continue
                    count = int(count or 0)
                    total_shape += count
                    if str(shape) in represented_shapes:
                        represented_shape += count
                if total_shape and represented_shape / total_shape < 0.80:
                    reasons.append("regex covers too little observed shape support")
            except (re.error, TypeError, ValueError):
                pass
        if shape_counts and re.search(r"\\d\{\d+(?:,\d*)?\}", regex):
            digit_shapes = [
                str(shape)
                for shape, count in shape_counts.items()
                if int(count or 0) > 0 and re.search(r"9", str(shape))
            ]
            normalized_shapes = {re.sub(r"9+", "9+", shape) for shape in digit_shapes}
            if len(digit_shapes) > 1 and len(normalized_shapes) == 1:
                reasons.append("regex fixes a digit width although observed canonical values share the same structure with variable digit lengths")
        top_values = list((meta.get("top_value_counts") or {}).keys())
        if top_values:
            numbers_in_regex = set(re.findall(r"(?<!\\)\d+(?:\\\.\d+|(?:\.\d+)?)", regex))
            numeric_top_values = {
                token.replace(".", r"\.")
                for value in top_values
                for token in re.findall(r"\d+(?:\.\d+)?", str(value))
            }
            if numbers_in_regex and len(numbers_in_regex & numeric_top_values) >= 2:
                reasons.append("regex hard-codes observed numeric magnitudes instead of generalizing the canonical pattern")
        return bool(reasons), reasons

    def _rare_match_requires_review(self, value: str, meta: dict) -> bool:
        text = str(value).strip()
        if text.lower() in _MISSING_LIKE_VALUES:
            return True
        dominant_skeleton = str(meta.get("dominant_normalized_numeric_text_skeleton") or "")
        value_skeleton = self._normalized_numeric_text_skeleton(text)
        if dominant_skeleton and value_skeleton and value_skeleton != dominant_skeleton:
            return True

        matches = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        profiles = meta.get("numeric_value_profiles") or {}
        profile = profiles.get(dominant_skeleton) or profiles.get(value_skeleton) or {}
        if not matches or not profile:
            return False
        try:
            number = float(matches[0])
            p05 = float(profile.get("p05"))
            p95 = float(profile.get("p95"))
        except (TypeError, ValueError):
            return False
        if p95 > 0 and number > p95 * 3:
            return True
        if p05 > 0 and number < p05 / 3:
            return True
        return False

    def _review_format_rule_with_llm(
            self,
            col: str,
            rule: dict,
            sample_values: list[str],
            meta: dict,
            reasons: list[str]
    ) -> dict:
        review_prompt = f"""
You are reviewing one LLM-generated data-summary format rule. The existing rule may have copied observed dirty values into a whitelist. Regenerate the rule using only the compact evidence below.

Column: {col}
Current rule: {json.dumps(rule, ensure_ascii=False)}
Review reasons: {json.dumps(reasons, ensure_ascii=False)}
Column meta evidence: {json.dumps(meta, ensure_ascii=False)}
Sample values: {json.dumps(sample_values[:120], ensure_ascii=False)}

Requirements:
- Infer the canonical valid representation from the majority pattern, field semantics, shape_counts, top_value_counts, rare_value_examples, and missing_like_values.
- Use dominant_shape_family, dominant_normalized_numeric_text_skeleton,
  shape_family_counts, numeric_text_skeleton_counts, and
  normalized_numeric_text_skeleton_groups to choose the dominant representation
  family and to generalize numeric magnitudes instead of hard-coding only
  observed numbers.
- If dominant_normalized_numeric_text_skeleton is present, use it as the default
  canonical surface skeleton by replacing <num> with a numeric pattern. Raw
  skeleton variants grouped under it are variant evidence, not extra valid
  alternatives, unless the column evidence clearly says otherwise.
- Treat field semantics as meaning, not syntax. If the observed dominant
  representation is plain numeric for a percentage-like concept, keep the regex
  plain numeric rather than adding a percent sign.
- Use numeric_value_profiles as range evidence. Values far outside the central
  numeric profile should remain suspicious for downstream LLM detection, rather
  than being silently treated as canonical by an over-broad regex.
- Do not enumerate observed values unless the field is a genuinely finite stable label/code set.
- Reject non-canonical variants such as alternate punctuation/case/unit spelling, appended qualifiers, missing placeholders, and values that belong to a different semantic field.
- Use case_punctuation_variant_groups to identify competing spellings of the same underlying value. Choose one canonical representation; do not accept all variants just because they appear in the sample.
- For numeric semantic columns, choose the dominant canonical representation: plain numeric, percent-marked, unit-marked, or another fixed text form. Do not accept conflicting variants.
- Return an anchored Python regex (`^...$`) only when there is a stable canonical format; otherwise return an empty regex.

Return exactly this JSON:
{{
  "rule": {{"format": "<description or unknown>", "regex": "<anchored Python regex or empty>", "explanation": "<brief reason>"}}
}}
"""
        response = summarizer_generator.send_message(
            [{"role": "user", "content": review_prompt}],
            max_tokens=summarizer_generator.max_tokens,
            request_timeout=FORMAT_RULE_REPAIR_REQUEST_TIMEOUT,
            retries=1,
        )
        try:
            reviewed = json.loads(clean_code_snippet(response)).get("rule", {})
            if isinstance(reviewed, dict):
                return reviewed
        except Exception as exc:
            print(f"[Warning] LLM format-rule review failed for '{col}': {exc}")
        return rule

    @staticmethod
    def _plain_numeric_regex_from_profile(props: dict, allow_grouping: bool = False) -> str:
        profiles = props.get("numeric_value_profiles") or {}
        dominant = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        profile = profiles.get(dominant) or profiles.get("<num>") or {}
        allow_decimal = False
        top_values = list((props.get("top_value_counts") or {}).keys())
        for value in top_values:
            text = str(value).strip()
            if Summarizer._is_missing_like_text(text) or "%" in text or re.search(r"[A-Za-z]", text):
                continue
            if re.fullmatch(r"[+-]?\d+\.\d+", text):
                allow_decimal = True
                break
        max_value = None
        try:
            max_value = float(profile.get("p95", profile.get("max")))
        except (TypeError, ValueError):
            pass
        if max_value is not None and 0 < max_value <= 1.5:
            return r"^(?:0(?:\.\d+)?|1(?:\.0+)?)$"
        if allow_grouping:
            grouped_num = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
            return rf"^[+-]?{grouped_num}(?:\.\d+)?$" if allow_decimal else rf"^[+-]?{grouped_num}$"
        return r"^[+-]?\d+(?:\.\d+)?$" if allow_decimal else r"^[+-]?\d+$"

    @staticmethod
    def _numeric_value_regex_from_profile(props: dict) -> str:
        text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if Summarizer._field_text_has_token(text, ("year",)):
            fixed = Summarizer._fixed_width_digit_regex_from_profile(props)
            if fixed:
                return fixed
        counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        allow_grouping = any("," in str(key) for key in counts) or any(
            "," in str(value)
            for value in (props.get("top_value_counts") or {})
        )
        return Summarizer._plain_numeric_regex_from_profile(props, allow_grouping=allow_grouping)

    @staticmethod
    def _numeric_token_regex_for_skeleton(skeleton: str, props: dict | None = None) -> str:
        props = props or {}
        counts = props.get("normalized_numeric_text_skeleton_counts") or props.get("numeric_text_skeleton_counts") or {}
        has_grouped = any("," in str(item) for item in counts)
        has_decimal = False
        for value in (props.get("top_value_counts") or {}):
            text = str(value)
            if re.search(r"\d+\.\d+", text):
                has_decimal = True
                break
        if "," in skeleton or has_grouped:
            base = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        else:
            base = r"\d+"
        return rf"{base}(?:\.\d+)?" if has_decimal else base

    @staticmethod
    def _regex_from_numeric_skeleton(skeleton: str, props: dict | None = None) -> str:
        skeleton = str(skeleton or "").strip()
        if not skeleton or "<num>" not in skeleton:
            return ""
        num = Summarizer._numeric_token_regex_for_skeleton(skeleton, props)
        escaped = re.escape(skeleton)
        escaped = escaped.replace(re.escape("<num>"), num)
        escaped = escaped.replace(r"\ ", r"\s+")
        return f"^{escaped}$"

    @staticmethod
    def _skeleton_alpha_tokens(skeleton: str) -> set[str]:
        tokens = set()
        for token in re.findall(r"[a-z]+", str(skeleton or "").lower()):
            if token == "num":
                continue
            if len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            tokens.add(token)
        return tokens

    @staticmethod
    def _collapse_grouped_number_skeleton(skeleton: str) -> str:
        text = str(skeleton or "")
        return re.sub(r"<num>(?:,<num>)+", "<num>", text)

    @staticmethod
    def _dominant_regex_from_numeric_skeletons(props: dict, min_ratio: float = 0.08, max_parts: int = 4) -> str:
        counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        if not counts:
            return ""
        total, non_missing, _plain, _percent, _unit = Summarizer._profile_family_totals(props)
        del total
        parts = []
        support = 0
        for skeleton, raw_count in counts.items():
            skeleton_text = str(skeleton)
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            if not skeleton_text or skeleton_text in _MISSING_LIKE_VALUES or "<num>" not in skeleton_text:
                continue
            if count < max(3, int(non_missing * min_ratio)):
                continue
            regex = Summarizer._regex_from_numeric_skeleton(skeleton_text, props)
            if not regex:
                continue
            parts.append(regex[1:-1])
            support += count
            if len(parts) >= max_parts:
                break
        if not parts or support < max(10, int(non_missing * min_ratio)):
            return ""
        if len(parts) == 1:
            return f"^{parts[0]}$"
        return f"^(?:{'|'.join(parts)})$"

    @staticmethod
    def _dominant_raw_skeleton_for_normalized(props: dict) -> str:
        dominant = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        groups = props.get("normalized_numeric_text_skeleton_groups") or {}
        group = groups.get(dominant)
        if isinstance(group, dict) and group:
            return str(max(group.items(), key=lambda item: int(item[1] or 0))[0])
        return str(props.get("dominant_numeric_text_skeleton") or dominant)

    @staticmethod
    def _compatible_numeric_text_regex_from_profile(props: dict, min_ratio: float = 0.01, max_parts: int = 5) -> str:
        counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        if not counts:
            return ""
        dominant = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        dominant_words = Summarizer._skeleton_alpha_tokens(dominant)
        if not dominant_words:
            return ""
        _total, non_missing, _plain, _percent, unit = Summarizer._profile_family_totals(props)
        if not non_missing or unit / max(1, non_missing) < 0.50:
            return ""

        parts = []
        seen = set()
        support = 0
        for skeleton, raw_count in sorted(counts.items(), key=lambda item: -int(item[1] or 0)):
            skeleton_text = Summarizer._collapse_grouped_number_skeleton(str(skeleton or "").strip())
            if not skeleton_text or skeleton_text in _MISSING_LIKE_VALUES or "<num>" not in skeleton_text:
                continue
            if skeleton_text in seen:
                continue
            words = Summarizer._skeleton_alpha_tokens(skeleton_text)
            if not words or not words.issubset(dominant_words):
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            if count < max(3, int(non_missing * min_ratio)):
                continue
            regex = Summarizer._regex_from_numeric_skeleton(skeleton_text, props)
            if not regex:
                continue
            parts.append(regex[1:-1])
            seen.add(skeleton_text)
            support += count
            if len(parts) >= max_parts:
                break
        if len(parts) <= 1 or support < max(20, int(non_missing * 0.50)):
            return ""
        return f"^(?:{'|'.join(parts)})$"

    @staticmethod
    def _plural_count_regex_from_skeleton(raw_skeleton: str, normalized_skeleton: str, props: dict) -> str:
        raw = str(raw_skeleton or "").strip().lower()
        normalized = str(normalized_skeleton or "").strip().lower()
        match = re.fullmatch(r"<num>\s+([a-z]{3,})s", raw)
        if not match:
            return ""
        singular = match.group(1)
        if normalized and normalized != f"<num> {singular}":
            return ""
        num = Summarizer._numeric_token_regex_for_skeleton(raw_skeleton, props)
        plural_num = num if "," in num else r"(?:0|[2-9]|\d{2,})"
        return rf"^(?:1\s+{singular}|{plural_num}\s+{singular}s)$"

    @staticmethod
    def _canonical_numeric_text_regex_from_profile(props: dict) -> str:
        normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "").strip()
        raw = Summarizer._dominant_raw_skeleton_for_normalized(props)
        if not normalized and not raw:
            return ""

        plural_regex = Summarizer._plural_count_regex_from_skeleton(raw, normalized, props)
        if plural_regex:
            return plural_regex

        compatible_regex = Summarizer._compatible_numeric_text_regex_from_profile(props)
        if compatible_regex:
            return compatible_regex

        field_text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if Summarizer._field_text_has_token(field_text, ("review", "rating count", "count with units")):
            multi_shape = Summarizer._dominant_regex_from_numeric_skeletons(props, min_ratio=0.01, max_parts=5)
            if multi_shape:
                return multi_shape

        skeleton = normalized if "<num>" in normalized else raw
        return Summarizer._regex_from_numeric_skeleton(skeleton, props)

    @staticmethod
    def _structured_shape_regex_from_profile(props: dict) -> str:
        shape_counts = props.get("shape_counts") or {}
        if not shape_counts:
            return ""
        supported_shapes = []
        total = 0
        for shape, raw_count in shape_counts.items():
            shape_text = str(shape)
            if shape_text == "<missing-like>":
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            total += count
            if count >= 3:
                supported_shapes.append((shape_text, count))
        if not supported_shapes or total <= 0:
            return ""

        text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if not Summarizer._field_text_has_token(
                text,
                ("identifier", "id", "code", "issn", "isbn", "doi", "key", "record", "reference", "number"),
        ):
            return ""
        if Summarizer._field_text_has_token(text, ("address", "street", "phone", "telephone", "postal", "zip")):
            return ""

        if any(re.fullmatch(r"AA-\d{1,4}-AAA-AAA", shape) for shape, _ in supported_shapes):
            return r"^[A-Z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3}$"

        if any(re.fullmatch(r"9{4}-9{4}", shape) for shape, _ in supported_shapes):
            return r"^\d{4}-\d{3}[\dX]$"

        code_shapes = [
            shape
            for shape, _ in supported_shapes
            if re.fullmatch(r"A{2,6}(?:-A{2,6})*-\d+A?", shape)
            or re.fullmatch(r"A{2,6}-\d+A?", shape)
        ]
        code_support = sum(count for shape, count in supported_shapes if shape in code_shapes)
        if code_shapes and code_support / max(total, 1) >= 0.50:
            return r"^[a-z]+(?:-[a-z]+)*-\d+[a-z]?$"

        return ""

    @staticmethod
    def _structured_numeric_text_code_regex_from_profile(props: dict) -> str:
        skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        if not skeleton_counts:
            return ""
        _total, non_missing, _plain, _percent, unit = Summarizer._profile_family_totals(props)
        if not non_missing or unit / max(1, non_missing) < 0.50:
            return ""

        supported = []
        support = 0
        for skeleton, raw_count in skeleton_counts.items():
            text = str(skeleton or "").strip()
            if not text or "<num>" not in text or text in _MISSING_LIKE_VALUES:
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            if count < max(3, int(non_missing * 0.02)):
                continue
            supported.append(text)
            support += count
        if support < max(20, int(non_missing * 0.50)):
            return ""

        if any(
                re.fullmatch(r"[a-z]{1,4}_[a-z]+(?:-[a-z]+)*-<num>[a-z]?", item)
                for item in supported[:12]
        ):
            return r"^[a-z]{1,4}_[a-z]+(?:-[a-z]+)*-\d+[a-z]?$"
        joined = " ".join(supported).lower()
        if "_" in joined and "-" in joined and re.search(r"(?:_<num>|-<num>)", joined):
            return r"^[a-z]+(?:_[a-z]+)+(?:-[a-z]+)*-\d+[a-z]?$"
        return ""

    @staticmethod
    def _pagination_regex_from_profile(props: dict) -> str:
        field_text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if not Summarizer._field_text_has_token(field_text, ("page", "pagination")):
            return ""
        skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
        non_missing = max(1, Summarizer._profile_family_totals(props)[1])
        supported = 0
        for skeleton, count in skeleton_counts.items():
            text = str(skeleton or "")
            if text in _MISSING_LIKE_VALUES:
                continue
            if re.fullmatch(r"<num>(?:-<num>)?", text):
                supported += int(count or 0)
        if supported >= max(20, int(non_missing * 0.50)):
            return r"^\d+(?:-\d+)?$"
        return ""

    @staticmethod
    def _fixed_width_digit_regex_from_profile(props: dict) -> str:
        shape_counts = props.get("shape_counts") or {}
        numeric_shapes = []
        total = 0
        for shape, raw_count in shape_counts.items():
            shape_text = str(shape)
            if shape_text == "<missing-like>":
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                continue
            total += count
            if re.fullmatch(r"9+", shape_text):
                numeric_shapes.append((shape_text, count))
        if not numeric_shapes or not total:
            return ""
        shape, count = max(numeric_shapes, key=lambda item: item[1])
        if count / max(1, total) >= 0.80 and len(set(shape)) == 1:
            return rf"^\d{{{len(shape)}}}$"
        return ""

    @staticmethod
    def _time_regex_from_profile(props: dict) -> str:
        skeletons = props.get("normalized_numeric_text_skeleton_counts") or {}
        time_support = 0
        for skeleton, count in skeletons.items():
            text = str(skeleton).lower()
            if re.fullmatch(r"<num>:<num>\s+(?:a\.?m|p\.?m)", text):
                time_support += int(count or 0)
        total = sum(int(value or 0) for value in (props.get("shape_family_counts") or {}).values())
        if time_support >= max(20, int(total * 0.25)):
            return r"^(?:[1-9]|1[0-2]):[0-5]\d\s+(?:a\.m\.|p\.m\.)$"
        return ""

    @staticmethod
    def _date_text_regex_from_profile(props: dict) -> str:
        text = f"{props.get('semantic_type', '')} {props.get('description', '')}".lower()
        if "date" not in text:
            return ""
        skeletons = props.get("normalized_numeric_text_skeleton_counts") or {}
        month_names = (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        )
        has_month_date = any(
            re.fullmatch(rf"<num>\s+(?:{'|'.join(month_names)})\s+<num>\s+\([^)]*\)", str(skeleton).lower())
            for skeleton in skeletons
        )
        if has_month_date:
            month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            return rf"^(?:(?:[1-9]|[12]\d|3[01])\s+{month}\s+\d{{4}}|{month}\s+\d{{4}}|\d{{4}})\s+\([^)]*\)$"
        date_profile = props.get("date_component_profile") or {}
        if int(date_profile.get("matched_count") or 0) >= 20:
            return r"^\d{1,2}/\d{1,2}/\d{2,4}$"
        return ""

    @staticmethod
    def _list_regex_from_profile(props: dict) -> str:
        alpha = Summarizer._alpha_class_from_profile(props)
        item = rf"{alpha}+(?:[ .'\-/()]|{alpha})*"
        return rf"^{item}(?:,{item})*$"

    def _fallback_format_rule_from_profile(self, col: str, props: dict, existing_rule: dict | None = None) -> dict:
        existing_rule = existing_rule or {}
        existing_regex = str(existing_rule.get("regex", "") or "").strip()
        if existing_regex.startswith("^") and existing_regex.endswith("$") and "|" in existing_regex and not existing_regex.startswith("^(?:"):
            inner = existing_regex[1:-1]
            if not (inner.startswith("(") and inner.endswith(")")):
                existing_rule = {**existing_rule, "regex": f"^(?:{inner})$"}

        total, non_missing_total, plain, percent, unit = self._profile_family_totals(props)
        if not total:
            return existing_rule
        non_missing_total = max(1, non_missing_total)
        dominant_skeleton = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        field_text = self._field_text(col, props)
        open_text_field = self._is_open_text_column(col, props)
        existing_needs_review = False
        existing_review_reasons = []
        if existing_regex:
            existing_needs_review, existing_review_reasons = self._needs_llm_rule_review(
                col,
                {**existing_rule, "regex": existing_regex},
                props,
                list((props.get("top_value_counts") or {}).keys()),
            )

        if open_text_field:
            if existing_regex and (
                    existing_needs_review
                    or self._field_text_has_token(
                        field_text,
                        ("address", "street", "description", "title", "location", "locations"),
                    )
            ):
                return {
                    **existing_rule,
                    "format": "unknown",
                    "regex": "",
                    "explanation": (
                        "Open-text field; removed unsafe regex because generic validation flagged it: "
                        + "; ".join(existing_review_reasons[:3])
                    ),
                    "source": "generic_open_text_profile_guard",
                }
            return existing_rule

        regex = ""
        reason = ""
        if self._field_text_has_token(field_text, ("state", "province", "region")) and not open_text_field:
            fixed = self._fixed_width_digit_regex_from_profile(props)
            alpha = self._alpha_class_from_profile(props)
            shape_counts = props.get("shape_counts") or {}
            top_shape = max(shape_counts.items(), key=lambda item: int(item[1] or 0), default=("", 0))[0]
            if str(top_shape) == "AA":
                regex = rf"^{alpha}{{2}}$"
                reason = "profile-derived fixed-width alphabetic region code"
            elif fixed:
                regex = fixed
                reason = "profile-derived fixed-width numeric region code"

        if not regex and self._field_text_has_token(field_text, ("yes", "no", "boolean", "availability", "flag", "indicator")):
            top_values = {
                str(value).strip().lower(): int(count or 0)
                for value, count in (props.get("top_value_counts") or {}).items()
                if not self._is_missing_like_text(str(value))
            }
            yes_no_support = top_values.get("yes", 0) + top_values.get("no", 0)
            if yes_no_support >= max(20, int(non_missing_total * 0.70)):
                regex = r"^(?:yes|no)$"
                reason = "profile-derived finite boolean label set"

        if not regex:
            regex = self._time_regex_from_profile(props) or self._date_text_regex_from_profile(props)
            if regex:
                reason = "profile-derived temporal/date canonical format"

        if not regex and self._is_list_like_column(col, props) and not open_text_field:
            regex = self._list_regex_from_profile(props)
            reason = "profile-derived comma-separated label list"

        if not regex and not open_text_field:
            regex = self._structured_shape_regex_from_profile(props)
            if regex:
                reason = "profile-derived structured identifier/code canonical shape"

        if not regex and not open_text_field and self._field_text_has_token(field_text, ("zip", "postal", "phone", "telephone")):
            regex = self._fixed_width_digit_regex_from_profile(props)
            if regex:
                reason = "profile-derived fixed-width numeric contact/postal format"

        if not regex and not open_text_field:
            regex = self._pagination_regex_from_profile(props)
            if regex:
                reason = "profile-derived pagination/range canonical format"

        if not regex and not open_text_field and plain / non_missing_total >= 0.60:
            regex = self._numeric_value_regex_from_profile(props)
            reason = "profile-derived dominant plain numeric canonical format"
        elif not regex and not open_text_field and unit / non_missing_total >= 0.60 and "<num>" in dominant_skeleton:
            regex = (
                self._pagination_regex_from_profile(props)
                or self._structured_numeric_text_code_regex_from_profile(props)
                or self._canonical_numeric_text_regex_from_profile(props)
            )
            reason = "profile-derived dominant numeric-text skeleton canonical format"
        elif not regex and not open_text_field and percent / non_missing_total >= 0.60:
            regex = r"^\d+(?:\.\d+)?%$"
            reason = "profile-derived dominant percent canonical format"

        if regex and self._field_text_has_token(field_text, ("year",)):
            fixed = self._fixed_width_digit_regex_from_profile(props)
            if fixed:
                regex = fixed
                reason = "profile-derived fixed-width year format"

        if not regex:
            return existing_rule
        try:
            re.compile(regex)
        except re.error:
            return existing_rule
        if existing_regex:
            severe_replacement_reasons = (
                "observed-value whitelist",
                "matches missing-like",
                "hard-codes observed numeric magnitudes",
                "mixed plain, percent-marked, or unit-marked",
                "matches very few sampled values",
                "rejects a recurring top-count value",
                "date regex misses recurring date-delimited values",
                "pagination regex misses recurring page-range values",
            )
            coverage_miss = any(
                "covers too little observed shape support" in reason
                for reason in existing_review_reasons
            )
            grouped_numeric_miss = (
                coverage_miss
                and plain / non_missing_total >= 0.60
                and "," in regex
                and "," not in existing_regex
                and any("," in str(shape) for shape in (props.get("shape_counts") or {}))
            )
            normalized_numeric_text_miss = (
                coverage_miss
                and unit / non_missing_total >= 0.60
                and "<num>" in dominant_skeleton
                and regex
                and regex != existing_regex
                and regex == self._canonical_numeric_text_regex_from_profile(props)
            )
            plural_count_miss = (
                coverage_miss
                and regex
                and regex != existing_regex
                and regex.startswith(r"^(?:1\s+")
            )
            if not existing_needs_review or not (
                    grouped_numeric_miss
                    or normalized_numeric_text_miss
                    or plural_count_miss
                    or any(
                        any(marker in reason for marker in severe_replacement_reasons)
                        for reason in existing_review_reasons
                    )
            ):
                return existing_rule
        return {
            "format": existing_rule.get("format") or reason,
            "regex": regex,
            "explanation": f"Added by generic summary profile fallback: {reason}.",
            "source": existing_rule.get("source") or "llm_summary_profile_fallback",
        }

    def _fill_missing_format_rules_from_profiles(self, rules: dict, data_summary: dict) -> dict:
        filled = dict(rules or {})
        for field in data_summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            col = field.get("column")
            if not col:
                continue
            props = field.get("properties", {}) or {}
            current = filled.get(col, {})
            if not isinstance(current, dict):
                current = {}
            filled[col] = self._fallback_format_rule_from_profile(str(col), props, current)
        return filled

    def compact_for_data_summary(self, runtime_summary: dict) -> dict:
        """
        Public data_summary.json should be readable and aligned with the paper:
        field semantics, LLM/profile-derived format rules, field relationships,
        and relationship validator scope. Heavy profile evidence remains an
        internal summarizer artifact and is saved separately by LAED_Demo.py.
        """
        fields = []
        for field in runtime_summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            props = field.get("properties", {}) or {}
            fields.append({
                "column": field.get("column"),
                "semantic_type": props.get("semantic_type", ""),
                "description": props.get("description", ""),
                "dtype": props.get("dtype", ""),
            })
        compact = {
            "dataset_description": runtime_summary.get("dataset_description", ""),
            "fields": fields,
            "format_rules": runtime_summary.get("format_rules", {}) or {},
            "field_relationships": runtime_summary.get("field_relationships", {}) or {},
            "relationship_validator_code": runtime_summary.get("relationship_validator_code", {}) or {},
            "relationship_validator_scope": runtime_summary.get(
                "relationship_validator_scope",
                "Relationship validators generate candidate contradictions from summary field relationships; final error labels require LLM confirmation in Error_Detection_update.py.",
            ),
        }
        return compact

    def check_type(self, dtype: str, value):
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame) -> list[dict]:
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            nunique = df[column].nunique()
            total = len(df[column])
            uniqueness_ratio = round(nunique / total, 2) if total > 0 else 0
            properties["uniqueness_ratio"] = uniqueness_ratio

            # 1. 数值类型
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                # 先判断：如果是浮点列且所有非空值都是整数
                series = df[column]
                if pd.api.types.is_float_dtype(series):
                    nonnull = series.dropna()
                    if nonnull.map(float.is_integer).all():
                        # 全部都形如 X.0，转成整型再统计
                        series_int = nonnull.astype(int)
                        properties["std"] = series_int.std()
                        properties["min"] = series_int.min()
                        properties["max"] = series_int.max()
                    else:
                        # 真正浮点，按原值统计
                        properties["std"] = series.std()
                        properties["min"] = series.min()
                        properties["max"] = series.max()
                else:
                    # 原本就是整型或 complex
                    properties["std"] = series.std()
                    properties["min"] = series.min()
                    properties["max"] = series.max()

            # 2. 布尔类型
            elif dtype == bool:
                properties["dtype"] = "boolean"

            # 3. 显示分类类型
            elif isinstance(df[column].dtype, pd.CategoricalDtype) or uniqueness_ratio < 0.2:
                properties["dtype"] = "category"
                # 计算并存入 value_counts
                properties.update(self._build_observed_value_profile(df[column]))
                # 将 numpy 类型转换为原生 Python
                properties["value_counts"] = properties["top_value_counts"]
                # value_counts is top-k only; full enumeration can whitelist dirty values.


            # 4. 日期判断
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            elif dtype == object:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "date"
                except ValueError:
                    properties["dtype"] = "string"
            else:
                properties["dtype"] = str(dtype)

            # 日期字段的最小最大值
            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()

            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            observed_profile = self._build_observed_value_profile(df[column])
            for key, value in observed_profile.items():
                properties.setdefault(key, value)
            properties_list.append({"column": column, "properties": properties})
        return properties_list

    # 按照列提取每列的格式
    def extract_format_rules(
            self,
            df: pd.DataFrame,
            data_summary: dict,
            sample_size: int = 100,
            group_size: int = 6
    ) -> dict:
        """
        仅对dtype属于category、number、date或string（能归纳出固定格式）的字段，从混合了正确和错误值的样本中提取”正确格式“。
        格式表达为清晰描述或regex:^...$，无法确定时返回"unknown"。
        对于 category，还会根据 value_counts 确认低频值是否为合法类别。
        """
        # 1. 先筛选出有资格提取格式的列
        eligible_cols = [
            f["column"]
            for f in data_summary["fields"]
            if f["properties"]["dtype"] in ["category", "float", "int", "number", "date", "string", "integer"]
        ]
        # print("***eligible_cols***")
        # print(eligible_cols)

        # 2. 抽样并全部转成字符串
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)[eligible_cols]
        sample_df = sample_df.astype(str)

        # 3. 对于标记为 number 的列，将 “xx.0” 这种浮点整数格式归一为 “xx”
        for f in data_summary["fields"]:
            col = f["column"]
            dtype = f["properties"].get("dtype")
            if col in sample_df.columns and dtype == "number":
                sample_df[col] = sample_df[col].apply(
                    lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x
                )

        def compact_rule_value(value, max_chars: int = 120) -> str:
            text = re.sub(r"\s+", " ", str(value)).strip()
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "...[truncated]"

        def compact_rule_samples(series: pd.Series, limit: int = 30) -> list[str]:
            values = pd.Series(series.dropna().map(lambda value: compact_rule_value(value)).drop_duplicates().tolist())
            if len(values) > limit:
                values = values.sample(n=limit, random_state=42)
            return values.tolist()

        def compact_rule_meta(properties: dict, max_items: int = 8, max_chars: int = 120):
            keep_keys = {
                "dtype",
                "semantic_type",
                "description",
                "top_value_counts",
                "value_counts",
                "rare_value_examples",
                "shape_counts",
                "shape_family_counts",
                "numeric_text_skeleton_counts",
                "normalized_numeric_text_skeleton_counts",
                "normalized_numeric_text_skeleton_groups",
                "numeric_value_profiles",
                "dominant_shape_family",
                "dominant_numeric_text_skeleton",
                "dominant_normalized_numeric_text_skeleton",
                "case_punctuation_variant_groups",
                "missing_like_values",
                "date_component_profile",
            }

            def compact(value):
                if isinstance(value, dict):
                    compacted = {}
                    for key, item in list(value.items())[:max_items]:
                        compacted[compact_rule_value(key, max_chars)] = compact(item)
                    return compacted
                if isinstance(value, list):
                    return [compact(item) for item in value[:max_items]]
                if isinstance(value, str):
                    return compact_rule_value(value, max_chars)
                return value

            return {
                key: compact(value)
                for key, value in (properties or {}).items()
                if key in keep_keys
            }

        all_rules = []
        total_groups = (len(eligible_cols) + group_size - 1) // group_size if eligible_cols else 0

        # 4. 分组调用 LLM
        for i in range(0, len(eligible_cols), group_size):
            subset = eligible_cols[i: i + group_size]
            group_no = i // group_size + 1
            print(f"LLM format-rule batch {group_no}/{total_groups}: {subset}")
            sample_data = {col: compact_rule_samples(sample_df[col]) for col in subset}
            meta = {col: compact_rule_meta(next((f["properties"] for f in data_summary["fields"] if f["column"] == col), {}))
                    for col in subset}

            # 如果这一列是 category，带上 value_counts
            category_counts = {
                col: meta[col].get("value_counts", {})
                for col in subset
                if meta[col].get("dtype") == "category"
            }

            prompt = f"""
            You are a professional data quality auditor. The sample data below contains both correct and incorrect values; extract the single, absolute correct format for each eligible column (type category, number, date, or string with a fixed pattern) based on the majority of valid examples, ignoring erroneous entries. Downstream code will use only the `regex` field to filter out correct-format values—ensure each regex is an anchored pattern (`^...$`) that matches all and only valid examples.

            Error types may include:
            - spelling_errors: typos or misspellings  
            - format_errors: inconsistent date/number/text formats  
            - logical_errors: illogical values or contradictions  
            - missing_values: NULL, 'N/A', 'nul', etc.  
            - outliers: values far outside reasonable ranges  

            For each column:
              • If number: provide `"regex": "<anchored regex>"` (e.g. `^\\d+(?:\\.\\d+)?$`)  
              • If date: provide `"regex": "<anchored regex>"` matching the exact date format (e.g. `^\\d{4}-\\d{2}-\\d{2}$`)  
              • If **string with a fixed, repeatable pattern** (e.g. identifiers like "AB-1234", measured values like "93 min", timestamps like "2026-05-11 08:30"):  
                   – Derive the common structure as an anchored regex (`^...$`).  
                   – Only output a regex if that pattern strictly matches *all* valid samples and excludes anomalies.  
              • If **string without a strict, repeatable pattern** (e.g. actor names, free-text descriptions):  
                   – Do **not** invent a regex—return `"regex": ""` and `"format": "unknown"`.  
              • If category: treat compact counts as evidence, not as a legal-value list. Enumerate categories only for a truly finite, stable label/code set; otherwise infer a structural canonical regex or return an empty regex.

            Return exactly the following JSON, with no extra keys or commentary:
            {{
            "rules": {{
                "ColumnA": {{"format": "<description or regex>", "regex": "<anchored regex or empty>", "explanation": "<a brief rationale, referencing the observed correct examples and incorrect examples.>"}},
                "ColumnB": {{"format": "<description or regex>", "regex": "<anchored regex or empty>", "explanation": "<a brief rationale, referencing the observed correct examples and incorrect examples.>"}},
                …
              }}
            }}

            # Sample Data:
            {json.dumps(sample_data, ensure_ascii=False)}

            # Column Meta Info:
            {json.dumps(meta, ensure_ascii=False)}
            """
            prompt += """

Canonical-format requirement:
- A value can be semantically understandable but still be a data-quality error if
  it is a non-canonical variant of the dominant format in this column.
- These requirements override any earlier category instruction. Do not build a
  whitelist by joining every observed category/value. Enumeration is allowed
  only for a genuinely finite, stable label/code set whose canonical labels are
  clear from the compact evidence.
- Prefer the dominant normalized representation over enumerating every observed
  variant. Exclude punctuation/case/unit variants, appended qualifiers, missing
  placeholders, values that belong to another column, and percentage/unit markers
  when the dominant canonical representation is a plain numeric value.
- For measurement-like columns, infer the canonical unit spelling, punctuation,
  and case from the dominant valid pattern. Reject alternate spellings, uppercase
  variants, trailing punctuation, appended qualifiers/material notes, and missing
  placeholders unless they are clearly the canonical representation.
- Use `case_punctuation_variant_groups` to notice values that differ only by
  case, final punctuation, or spacing. Pick one canonical representation from
  the column semantics and dominant clean-looking pattern; do not accept all
  variants just because they are frequent.
- Use `dominant_shape_family`, `dominant_normalized_numeric_text_skeleton`,
  `shape_family_counts`, `numeric_text_skeleton_counts`, and
  `normalized_numeric_text_skeleton_groups` to choose the dominant representation
  family. When plain numeric values are more common than percent-marked values,
  the canonical regex should be plain numeric even if the field semantics mention
  percentages. When a numeric value plus unit skeleton dominates, generalize the
  numeric magnitude instead of hard-coding only the observed magnitudes.
- If `dominant_normalized_numeric_text_skeleton` is present, use it as the
  default canonical surface skeleton by replacing `<num>` with a numeric pattern.
  Raw skeleton variants grouped under it are evidence of variants, not additional
  valid alternatives, unless the evidence clearly says otherwise.
- If top counts show multiple recurring numeric-text skeletons for the same
  field (for example one component and two component count strings), build an
  anchored alternation that accepts the recurring canonical skeletons instead of
  requiring only the single most common skeleton.
- If observed canonical numeric text contains thousands separators, make the
  numeric subpattern accept both grouped and ungrouped numbers when both appear.
- For date fields, do not force a single granularity if the profile shows
  recurring year-only, month-year, and full-date forms. Use an alternation for
  supported canonical granularities, and allow multi-word parenthesized
  locations when they appear in valid-looking repeated examples.
- For comma-separated label/name/list fields, allow spaces, hyphens,
  apostrophes, periods, and parenthetical qualifiers inside an item when the
  profile shows such items. Do not use `[A-Za-z]+` for list items if examples
  contain multi-word labels.
- Field semantics describe meaning, not surface syntax. If the data says the
  canonical representation is a plain decimal for a percentage-like concept, do
  not add a percent sign to the regex.
- Use `numeric_value_profiles` as range evidence for numeric-looking fields. If
  a value is orders of magnitude outside the central numeric profile, the format
  rule should not silently accept it as canonical; explain that it is outlier
  evidence for the downstream LLM detector.
- For numeric semantic columns, distinguish plain decimals/integers from percent
  strings and unit-marked strings. Reject percent signs or unit suffixes when the
  canonical representation is plain numeric; reject bare numbers when a unit or
  percent marker is required by the dominant valid pattern.
- The `value_counts` field is compact top-k evidence only. `rare_value_examples`
  and `missing_like_values` are anomaly evidence, not extra valid categories.
- The regex should represent the canonical summary of the column, not a whitelist
  of all values seen in the dirty sample.
"""
            prompt = self._sanitize_prompt_examples(prompt)
            messages = [{"role": "user", "content": prompt}]
            response = summarizer_generator.send_message(
                messages,
                max_tokens=summarizer_generator.max_tokens,
                request_timeout=FORMAT_RULE_REQUEST_TIMEOUT,
                retries=1,
            )
            print(f"LLM format-rule batch {group_no}/{total_groups} completed.")
            try:
                cleaned = clean_code_snippet(response)
                parsed = json.loads(cleaned)
                group_rules = parsed.get("rules", {})
                group_rules = self.validate_format_rules(
                    rules=group_rules,
                    columns=subset,
                    sample_data=sample_data,
                    meta=meta,
                )
                for col, rule in group_rules.items():
                    all_rules.append({col: rule})
            except Exception as e:
                print(f"[Warning] rules extraction failed for {subset}: {e}")
        # 4.合并所有分组规则
        merged = {}
        for item in all_rules:
            merged.update(item)
        return merged

    def validate_format_rules(
            self,
            rules: dict,
            columns: list[str],
            sample_data: dict,
            meta: dict,
            max_retries: int = 3,
            allow_llm_review: bool = True
    ) -> dict:
        """
        Validate LLM-generated regex rules before initial screening uses them.
        This method never adds domain knowledge; it only checks Python regex syntax
        and asks the LLM to repair invalid syntax using the same column samples and
        summary metadata. If repair still fails, the regex is disabled so the later
        LLM diagnosis sees the cells instead of relying on an unsafe local rule.
        """
        validated = {}
        for col in columns:
            rule = rules.get(col, {}) if isinstance(rules, dict) else {}
            if not isinstance(rule, dict):
                rule = {"format": "unknown", "regex": "", "explanation": "LLM returned a non-object rule."}

            regex = str(rule.get("regex", "") or "").strip()
            if regex and not regex.startswith("^"):
                regex = f"^(?:{regex})"
            if regex and not regex.endswith("$"):
                regex = f"{regex}$"
            if regex.startswith("^") and regex.endswith("$") and "|" in regex and not regex.startswith("^(?:"):
                inner = regex[1:-1]
                if not (inner.startswith("(") and inner.endswith(")")):
                    regex = f"^(?:{inner})$"
            rule["regex"] = regex

            if not regex:
                validated[col] = rule
                continue

            last_error = ""
            effective_repair_rounds = min(max_retries, max(0, FORMAT_RULE_REPAIR_ROUNDS))
            for _ in range(effective_repair_rounds + 1):
                try:
                    re.compile(regex)
                    rule["regex"] = regex
                    validated[col] = rule
                    break
                except re.error as exc:
                    last_error = str(exc)
                    if effective_repair_rounds <= 0:
                        regex = ""
                        break
                    repair_prompt = f"""
You generated an invalid Python regular expression for one table column.
Repair only the regex syntax. Do not add domain-specific assumptions beyond the
provided column samples and metadata.

Column: {col}
Column meta: {json.dumps(meta.get(col, {}), ensure_ascii=False)}
Sample values: {json.dumps(sample_data.get(col, [])[:100], ensure_ascii=False)}
Invalid regex: {regex}
Python regex error: {last_error}

Return exactly this JSON:
{{
  "rule": {{"format": "<description or unknown>", "regex": "<valid anchored Python regex or empty>", "explanation": "<brief reason>"}}
}}
"""
                    try:
                        response = summarizer_generator.send_message(
                            [{"role": "user", "content": repair_prompt}],
                            max_tokens=summarizer_generator.max_tokens,
                            request_timeout=FORMAT_RULE_REPAIR_REQUEST_TIMEOUT,
                            retries=1,
                        )
                        repaired = json.loads(clean_code_snippet(response)).get("rule", {})
                    except Exception as exc:
                        print(f"[Warning] LLM regex repair failed for '{col}': {exc}")
                        repaired = {}
                    if not isinstance(repaired, dict):
                        regex = ""
                        break
                    rule.update(repaired)
                    regex = str(rule.get("regex", "") or "").strip()
                    if regex and not regex.startswith("^"):
                        regex = f"^(?:{regex})"
                    if regex and not regex.endswith("$"):
                        regex = f"{regex}$"
            else:
                print(f"[Warning] regex for column '{col}' is invalid after LLM repair; leaving it empty: {last_error}")
                rule["regex"] = ""
                rule["format"] = rule.get("format") or "unknown"
                rule["explanation"] = (
                    str(rule.get("explanation", "") or "")
                    + f" Regex disabled after validation failure: {last_error}"
                ).strip()
                validated[col] = rule
                continue

            needs_review, reasons = self._needs_llm_rule_review(
                col=col,
                rule=validated[col],
                meta=meta.get(col, {}),
                sample_values=sample_data.get(col, []),
            )
            if allow_llm_review and needs_review:
                review_reasons = list(reasons)
                reviewed_rule = validated[col]
                needs_more_review = needs_review
                next_reasons = []
                for review_round in range(max(0, FORMAT_RULE_REVIEW_ROUNDS)):
                    try:
                        reviewed_rule = self._review_format_rule_with_llm(
                            col=col,
                            rule=reviewed_rule,
                            sample_values=sample_data.get(col, []),
                            meta=meta.get(col, {}),
                            reasons=review_reasons,
                        )
                    except Exception as exc:
                        print(f"[Warning] LLM format-rule review request failed for '{col}': {exc}")
                        break
                    reviewed_rule = self.validate_format_rules(
                        rules={col: reviewed_rule},
                        columns=[col],
                        sample_data={col: sample_data.get(col, [])},
                        meta={col: meta.get(col, {})},
                        max_retries=max_retries,
                        allow_llm_review=False,
                    ).get(col, reviewed_rule)
                    needs_more_review, next_reasons = self._needs_llm_rule_review(
                        col=col,
                        rule=reviewed_rule,
                        meta=meta.get(col, {}),
                        sample_values=sample_data.get(col, []),
                    )
                    review_reasons.extend(reason for reason in next_reasons if reason not in review_reasons)
                    if not needs_more_review:
                        break
                disable_reasons = next_reasons or review_reasons
                if needs_more_review and any(
                        "observed-value whitelist" in reason
                        or "very few sampled values" in reason
                        or "wrong casing" in reason
                        or "fixes a digit width" in reason
                        or "rejects a recurring top-count value" in reason
                        or "covers too little observed shape support" in reason
                        or "mixed plain, percent-marked, or unit-marked numeric forms" in reason
                        or "hard-codes observed numeric magnitudes" in reason
                        for reason in disable_reasons
                ):
                    reviewed_rule["regex"] = ""
                    reviewed_rule["format"] = reviewed_rule.get("format") or "unknown"
                    reviewed_rule["explanation"] = (
                        str(reviewed_rule.get("explanation", "") or "")
                        + " Regex disabled because generic validation found an unsafe whitelist, low sample coverage, recurring-value rejection, mixed numeric/unit forms, hard-coded numeric magnitudes, or over-fixed digit width."
                    ).strip()
                reviewed_rule["llm_review_reasons"] = review_reasons
                validated[col] = reviewed_rule

        return validated

    def _sanitize_prompt_examples(self, prompt: str) -> str:
        """
        Keep prompts dataset-neutral. These replacements only remove illustrative
        domain examples from prompt text; they do not add detection rules.
        """
        replacements = [
            (
                r"For hierarchical relationships like .*? build a canonical mapping",
                "For each hierarchical relationship in the current summary, build a canonical mapping",
            ),
            (r"- Example:.*?cells\n", ""),
            (
                r"Special case for .*? ensure consistent mapping across all rows",
                "Apply the same majority-mapping strategy to every child-parent pair inferred from the current summary",
            ),
        ]
        cleaned = prompt
        for pattern, repl in replacements:
            cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned

    def _parse_relationship_validation_response(self, response: str) -> tuple[str, str]:
        cleaned = clean_code_snippet(response)
        if not cleaned:
            raise ValueError("Empty LLM response after cleaning")

        try:
            parsed = json.loads(cleaned)
            validation = parsed.get("validation", {})
            return str(validation.get("code", "") or ""), str(validation.get("explanation", "") or "")
        except json.JSONDecodeError:
            pass

        code_match = re.search(
            r'"code"\s*:\s*"(?P<code>.*?)(?<!\\)"\s*,\s*"explanation"',
            cleaned,
            flags=re.DOTALL,
        )
        explanation_match = re.search(
            r'"explanation"\s*:\s*"(?P<explanation>.*?)(?<!\\)"',
            cleaned,
            flags=re.DOTALL,
        )
        if code_match:
            code = code_match.group("code")
            try:
                code = bytes(code, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                code = code.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
            explanation = explanation_match.group("explanation") if explanation_match else ""
            return code, explanation

        fenced = re.search(r"```(?:python)?\s*(?P<code>.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group("code").strip(), "Extracted Python code block from LLM response."

        if "def validate_relationships" in cleaned:
            return cleaned, "Extracted raw Python code from LLM response."

        raise ValueError("Could not parse relationship validation code from LLM response.")

    ####### 看是否需要增加核验代码模块  （时间：2025/10/13）
    # summarizer.py

    # summarizer.py

    # summarizer.py

    def generate_relationship_validator(self, df: pd.DataFrame, data_summary: dict, sample_size: int = 200,
                                        max_retries: int | None = None) -> dict:
        """
        基于识别到的字段关系和样本数据，生成通用验证函数代码。
        专注于仅定位违反定义关系的特定单元格。
        """
        rel = data_summary.get("field_relationships", {}) or {}
        # print("字段关系:", rel)

        # 分析实际存在的字段关系
        existing_relationships = {}
        empty_relationships = []

        for rel_type, rel_data in rel.items():
            if rel_data and len(rel_data) > 0:
                existing_relationships[rel_type] = rel_data
            else:
                empty_relationships.append(rel_type)

        # print(f"实际存在的关系: {list(existing_relationships.keys())}")
        # print(f"空的关系类型: {empty_relationships}")

        # 收集所有关系涉及的有效列名
        cols = set()
        for rel_type, rel_data in existing_relationships.items():
            if rel_type == "mathematical":
                for derived, math_info in rel_data.items():
                    if isinstance(math_info, dict) and "fields" in math_info:
                        # 只包含实际存在的字段，确保至少有两个不同字段
                        related_fields = [f for f in math_info["fields"] if f in df.columns and f != derived]
                        if len(related_fields) >= 1:  # 至少有一个其他字段
                            cols.update([derived] + related_fields)
                    elif isinstance(math_info, list):
                        related_fields = [f for f in math_info if f in df.columns and f != derived]
                        if len(related_fields) >= 1:  # 至少有一个其他字段
                            cols.update([derived] + related_fields)
            elif rel_type == "associative":
                for key, vals in rel_data.items():
                    cols.add(key)
                    if isinstance(vals, list):
                        cols.update([v for v in vals if v in df.columns])
            elif rel_type == "hierarchical":
                for child, parents in rel_data.items():
                    cols.add(child)
                    if isinstance(parents, list):
                        cols.update([p for p in parents if p in df.columns])
            elif rel_type == "temporal":
                for sequence in rel_data:
                    if isinstance(sequence, list):
                        cols.update([s for s in sequence if s in df.columns])

        # 过滤实际存在的列
        selected_cols = [c for c in df.columns if c in cols]
        if selected_cols:
            sample_rows = df[selected_cols].dropna().head(sample_size).to_dict(orient="records")
        else:
            sample_rows = df.head(min(sample_size, len(df))).to_dict(orient="records")

        # 获取格式规则
        format_rules = data_summary.get('format_rules', {})
        relationship_evidence = data_summary.get("relationship_evidence", {})

        # 改进的提示词，特别加强层次关系验证
        base_prompt = f"""
    You are an expert data quality auditor. The dataset may itself contain both correct and incorrect values. Your task is to analyze the **field relationship definitions** and the **sample data** below, and generate Python code for validating relational dependencies among fields.

    **PRIMARY OBJECTIVE**: 
    Generate Python code that **ONLY locates the specific cells** that violate the defined field relationships. The code should return a set of (row_index, column_name) tuples identifying exactly which cells break the relationships.

    **CRITICAL REQUIREMENTS**:
    1. **Use format rules as pre-filter**: Before relationship validation, use the provided regex patterns to identify and skip cells with abnormal formats
    2. **Focus on logical violations**: Only check logical consistency between properly formatted fields
    3. **Report exact conflicting cells**: When a logical conflict is detected, report only the cells whose values directly deviate from the majority/canonical mapping
    4. **Skip unparseable data**: If data cannot be parsed for validation, skip it - do not report format errors
    5. **Focus on INTER-FIELD relationships**: Only validate relationships between different fields, not single-field constraints

    **IMPORTANT: WHEN LOGICAL CONFLICTS OCCUR, REPORT ONLY CELLS WITH DIRECT EVIDENCE**
    - For temporal relationships: report the field(s) whose parsed value violates the sequence, and avoid marking stable context fields
    - For mathematical relationships: report the derived field and only base fields that are directly unparseable or inconsistent
    - For associative relationships: the key is contextual evidence; if key-attribute mapping is inconsistent, report dependent attribute cells that differ from the majority mapping, not the key cell
    - For hierarchical relationships: report the child/parent cell that differs from the majority mapping; do not mark all related cells by default

    **HIERARCHICAL RELATIONSHIPS**:
    - For each hierarchical relationship in the current summary, build a canonical mapping from child to parent values
    - Check that each child value consistently maps to the same parent value across all rows
    - If a child value appears with different parent values, report only the parent cells that differ from the majority mapping unless the child value itself is malformed

    **MATHEMATICAL RELATIONSHIP HANDLING**:
    - Support various mathematical operations: =, <, >, <=, >=
    - For duration/experience fields: ensure non-negative and ≤ (current_time - start_time)
    - Use dynamic current date for time-based calculations
    - Allow reasonable tolerance for numeric calculations
    - **ONLY validate relationships between different fields**

    **Important Instructions**:
    - Your final output must be valid multi-line Python code (no escaped `\\n`)
    - DO NOT return code in strings or quotes - only raw code inside the JSON key `"code"`
    - Please ensure that all necessary modules are imported at the beginning of the code. Do not import modules inside the function.
    - Output JSON must be exactly:
    {{
        "validation": {{
            "code": "<actual Python code>",
            "explanation": "<brief explanation>"
        }}
    }}

    **Available Format Rules** (use these regex patterns to filter out abnormal data BEFORE relationship validation):
    {json.dumps(format_rules, indent=2, ensure_ascii=False)}

    **Field Relationship Definitions to Validate**:
    {json.dumps(existing_relationships, indent=2, ensure_ascii=False)}

    **Compact Relationship Evidence from the Summary Stage**:
    {json.dumps(relationship_evidence, indent=2, ensure_ascii=False)}

    **IMPLEMENTATION STRATEGY**:

    For **Hierarchical Relationships** (child → parents):
    1. Build a canonical mapping of child values to parent values using ONLY properly formatted data
    2. For each child value, determine the most frequent parent value as the canonical mapping
    3. For each row where both child and parent fields are properly formatted:
       - Check if the child-parent combination matches the canonical mapping
       - If not, report the parent cell that differs from the majority mapping as (row_index, column_name)
    4. Apply the same majority-mapping strategy to every child-parent pair inferred from the current summary.

    For **Associative Relationships** (key → [dependent_fields]):
    1. Build a frequency map of dependent tuples for each key value, using ONLY properly formatted data
    2. Select the most frequent tuple as the canonical mapping for each key
    3. For each row, if the key and all dependent fields are properly formatted, check if the dependent tuple matches the canonical mapping
    4. If not, report only the dependent fields that don't match as (row_index, column_name)

    For **Mathematical Relationships**:
    1. For each mathematical relationship definition, extract the operation and fields
    2. For each row where all fields are properly formatted and parseable
    3. Apply the specific mathematical operation:
       - For "=": check if values are approximately equal (with tolerance)
       - For "<", ">", "<=", ">=": check the inequality
       - For duration/experience fields: ensure non-negative and ≤ (current_time - start_time)
    4. If violated, report the derived field and only directly inconsistent/unparseable base fields as (row_index, column_name)

    For **Temporal Relationships** (sequence of time fields):
    1. For each row where all time fields are properly formatted and parseable as dates/times
    2. Check if the sequence maintains chronological order
    3. If violated, report only the time field(s) whose value breaks the local chronological order

    **CODE REQUIREMENTS**:
    - Implement a main function: `validate_relationships(df) -> set(tuple)`
    - Use the format rules to skip cells with abnormal formats before any relationship check
    - For hierarchical relationships, build canonical mappings from the ENTIRE dataset
    - Return a set of (row_index, column_name) for each violating cell
    - Handle data parsing errors gracefully by skipping unparseable rows
    - Use dynamic current date (`datetime.now().year`) for time-based calculations
    - When a logical conflict is detected between multiple fields, report only cells with direct evidence of being inconsistent
    - Include debugging prints to track validation progress
    - Support various mathematical operations based on relationship definitions
    - **IGNORE single-field constraints** - focus only on inter-field relationships

    *Sample Data (first {min(10, len(sample_rows))} rows shown)*:
    {json.dumps(sample_rows[:10], indent=2, ensure_ascii=False)}

    Return exactly this JSON, with no extra text.
    """

        if max_retries is None:
            max_retries = int(os.getenv("LAED_RELATIONSHIP_MAX_RETRIES", "0"))

        # 简化生成逻辑
        current_code = ""
        all_generated_codes = []
        retry_count = 0
        request_timeout = int(os.getenv("LAED_RELATIONSHIP_REQUEST_TIMEOUT", "45"))

        while retry_count <= max_retries:
            if retry_count == 0:
                # print(f"第{retry_count + 1}次生成验证代码...")
                prompt = base_prompt
            else:
                # print(f"第{retry_count + 1}次重新生成验证代码...")
                previous_code = all_generated_codes[-1]["code"]
                previous_errors = all_generated_codes[-1]["errors"]

                prompt = base_prompt + f"""

                **Previous code had issues**:
                {previous_errors}

                **Please fix these issues and regenerate the validation code.**
                """

            prompt = self._sanitize_prompt_examples(prompt)
            try:
                response = relationship_validator_generator.send_message(
                    [{"role": "user", "content": prompt}],
                    max_tokens=relationship_validator_generator.max_tokens,
                    request_timeout=request_timeout,
                    retries=1,
                )
            except Exception as e:
                error_msg = f"第{retry_count + 1}次生成失败: LLM relationship validator request failed: {e}"
                print(f"[ERROR] {error_msg}")
                code_record = {
                    "version": retry_count + 1,
                    "code": current_code if current_code else "生成失败",
                    "explanation": "",
                    "errors": error_msg,
                    "status": "failed"
                }
                all_generated_codes.append(code_record)
                retry_count += 1
                break
                continue

            # 处理API响应
            if response is None:
                print(f"第{retry_count + 1}次生成失败: API返回None")
                current_code = "# API返回None，无法生成验证代码"
                break

            try:
                current_code, current_explanation = self._parse_relationship_validation_response(response)
                validation_error = self.validate_relationship_code(current_code, df, selected_cols, sample_rows)
                if validation_error:
                    raise ValueError(validation_error)

                code_record = {
                    "version": retry_count + 1,
                    "code": current_code,
                    "explanation": current_explanation,
                    "errors": "",
                    "status": "generated"
                }

                all_generated_codes.append(code_record)
                break

            except Exception as e:
                error_msg = f"第{retry_count + 1}次生成失败: {str(e)}"
                print(f"[ERROR] {error_msg}")

                code_record = {
                    "version": retry_count + 1,
                    "code": current_code if current_code else "生成失败",
                    "explanation": "",
                    "errors": error_msg,
                    "status": "failed"
                }
                all_generated_codes.append(code_record)
                retry_count += 1
                break

        # 如果所有尝试都失败或没有生成代码，使用通用关系兜底代码。
        validation_error = self.validate_relationship_code(current_code, df, selected_cols, sample_rows) if current_code else "empty code"
        fallback_used = False
        if validation_error:
            fallback_code = self.build_generic_relationship_validator_code(data_summary)
            fallback_error = self.validate_relationship_code(fallback_code, df, selected_cols, sample_rows)
            if fallback_error:
                current_code = "# 无法生成验证代码"
                print(f"无法生成验证代码: {fallback_error}")
            else:
                current_code = fallback_code
                fallback_used = True
                print("LLM relationship code failed validation; using generic summary-derived associative validator.")

        final_result = {
            "validation": {
                "code": current_code,
                "explanation": "关系验证代码 - 专注于定位违反关系的特定单元格",
                "generation_history": all_generated_codes,
                "final_version": retry_count + 1,
                "existing_relationships": list(existing_relationships.keys()),
                "empty_relationships": empty_relationships,
                "status": "generic_fallback" if fallback_used else "completed"
            }
        }

        # print("*********生成的验证代码********")
        # print(current_code)
        # print("*********生成的验证代码********")
        return final_result

    def validate_relationship_code(
            self,
            code: str,
            df: pd.DataFrame,
            selected_cols: list[str],
            sample_rows: list[dict]
    ) -> str:
        """
        Compile and smoke-test LLM-generated relationship code. This is a generic
        safety check for the summary-derived code path, not an additional rule set.
        """
        if not code or "validate_relationships" not in code:
            return "Generated code must define validate_relationships(df)."

        exec_globals = {"pd": pd, "re": re, "np": np}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec("import pandas as pd\nimport re\nimport numpy as np\n" + code, exec_globals)
            validate_fn = exec_globals.get("validate_relationships")
            if not callable(validate_fn):
                return "validate_relationships is missing or not callable."
            if selected_cols:
                smoke_df = df[selected_cols].head(min(20, len(df))).copy()
            elif sample_rows:
                smoke_df = pd.DataFrame(sample_rows[:20])
            else:
                smoke_df = df.head(min(20, len(df))).copy()
            with contextlib.redirect_stdout(io.StringIO()):
                result = validate_fn(smoke_df)
            if result is None:
                return "validate_relationships returned None; it must return a set of (row_index, column_name)."
            for item in list(result)[:10]:
                if not (isinstance(item, tuple) and len(item) == 2):
                    return "validate_relationships must return tuples shaped as (row_index, column_name)."
        except Exception as exc:
            return f"Generated relationship code failed validation: {exc}"
        return ""

    def build_generic_relationship_validator_code(self, data_summary: dict) -> str:
        """
        Generic fallback for situations where LLM-generated code fails validation.
        It executes only associative key -> dependent mappings already present
        in the LLM/refined summary, using majority mapping within each key group.
        """
        relationships = data_summary.get("field_relationships", {}) or {}
        associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
        format_rules = data_summary.get("format_rules", {}) or {}
        if not associative:
            return "# No associative relationships available for generic fallback."

        field_props = {}
        for field in data_summary.get("fields", []):
            if not isinstance(field, dict):
                continue
            col = field.get("column")
            props = field.get("properties", {}) if isinstance(field.get("properties", {}), dict) else {}
            if not col:
                continue
            field_props[col] = {
                "dtype": props.get("dtype", field.get("dtype", "")),
                "semantic_type": props.get("semantic_type", field.get("semantic_type", "")),
                "description": props.get("description", field.get("description", "")),
            }

        relationships_literal = repr({"associative": associative})
        format_rules_literal = repr(format_rules)
        field_props_literal = repr(field_props)
        missing_like_literal = repr(sorted(_MISSING_LIKE_VALUES))

        return f'''
import re
from collections import Counter

RELATIONSHIPS = {relationships_literal}
FORMAT_RULES = {format_rules_literal}
FIELD_PROPS = {field_props_literal}
MISSING_LIKE = {missing_like_literal}


def _clean_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _is_missing_like(value):
    return _clean_value(value).lower() in MISSING_LIKE


def _matches_format(column, value):
    rule = FORMAT_RULES.get(column, {{}}) or {{}}
    regex = str(rule.get("regex", "") or "")
    if not regex or _is_missing_like(value):
        return not _is_missing_like(value)
    try:
        return re.fullmatch(regex, _clean_value(value)) is not None
    except re.error:
        return not _is_missing_like(value)


def _value_shape(value):
    value = _clean_value(value)
    if value.lower() in MISSING_LIKE:
        return "<missing-like>"
    chars = []
    for ch in value:
        if ch.isdigit():
            chars.append("9")
        elif ch.isalpha():
            chars.append("A")
        elif ch.isspace():
            chars.append(" ")
        else:
            chars.append(ch)
    return "".join(chars)


def _requires_local_anomaly_gate(column):
    props = FIELD_PROPS.get(column, {{}}) or {{}}
    text = f"{{props.get('dtype', '')}} {{props.get('semantic_type', '')}} {{props.get('description', '')}}".lower()
    return not any(token in text for token in ("time", "date", "duration", "number", "score", "amount"))


def validate_relationships(df):
    errors = set()
    associative = RELATIONSHIPS.get("associative", {{}}) or {{}}
    for key, dependents in associative.items():
        if key not in df.columns or not isinstance(dependents, list):
            continue
        for dep in dependents:
            if dep not in df.columns or dep == key:
                continue
            grouped = {{}}
            for idx, row in df[[key, dep]].iterrows():
                key_value = _clean_value(row[key])
                dep_value = _clean_value(row[dep])
                if _is_missing_like(key_value) or _is_missing_like(dep_value):
                    continue
                if not _matches_format(key, key_value) or not _matches_format(dep, dep_value):
                    continue
                grouped.setdefault(key_value, []).append((idx, dep_value))

            for entries in grouped.values():
                if len(entries) < 2:
                    continue
                counts = Counter(value for _, value in entries)
                if len(counts) <= 1:
                    continue
                canonical_value, canonical_count = counts.most_common(1)[0]
                majority_ratio = canonical_count / len(entries)
                if majority_ratio < 0.55:
                    for idx, _ in entries:
                        errors.add((idx, dep))
                else:
                    for idx, value in entries:
                        if value != canonical_value:
                            errors.add((idx, dep))
    return errors
'''.strip()

    def enrich(self, base_summary: dict, df: pd.DataFrame, sample_size: int = 100) -> dict:
        """
        使用 LLM 丰富字段关系，并兼容可能的 list 返回格式。
        """
        # 1. 按行抽样，保证能捕获列间关系
        def truncate_prompt_value(value, max_chars: int = 120) -> str:
            if pd.isna(value):
                return ""
            text = re.sub(r"\s+", " ", str(value)).strip()
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "...[truncated]"

        def compact_column_evidence(max_examples: int = 8, max_chars: int = 120) -> list[dict]:
            evidence = []
            for field in base_summary.get("fields", []):
                col = field.get("column")
                if col not in df.columns:
                    continue
                series = df[col]
                as_text = series.map(lambda value: "" if pd.isna(value) else str(value))
                unique_values = pd.Series(as_text.drop_duplicates().tolist())
                if len(unique_values) > max_examples:
                    sample_values = unique_values.sample(n=max_examples, random_state=42).tolist()
                else:
                    sample_values = unique_values.tolist()

                top_counts = as_text.value_counts(dropna=False).head(2)
                lengths = as_text.map(len)
                evidence.append({
                    "column": col,
                    "non_null_count": int(series.notna().sum()),
                    "empty_string_count": int((as_text.str.strip() == "").sum()),
                    "num_unique_values": int(series.nunique(dropna=False)),
                    "top_observed_values": {
                        truncate_prompt_value(index, max_chars): int(count)
                        for index, count in top_counts.items()
                    },
                    "sample_observed_values": [
                        truncate_prompt_value(value, max_chars)
                        for value in sample_values
                    ],
                    "length_profile": {
                        "min": int(lengths.min()) if len(lengths) else 0,
                        "median": float(lengths.median()) if len(lengths) else 0.0,
                        "max": int(lengths.max()) if len(lengths) else 0,
                    },
                })
            return evidence

        def compact_fields_for_prompt(max_dict_items: int = 5, max_chars: int = 80) -> list[dict]:
            keep_keys = {
                "uniqueness_ratio",
                "dtype",
                "num_unique_values",
                "semantic_type",
                "description",
                "top_value_counts",
                "shape_family_counts",
                "dominant_shape_family",
                "missing_like_values",
                "date_component_profile",
            }

            def compact_value(value):
                if isinstance(value, dict):
                    compacted = {}
                    for key, item in list(value.items())[:max_dict_items]:
                        compacted[truncate_prompt_value(key, max_chars)] = compact_value(item)
                    return compacted
                if isinstance(value, list):
                    return [compact_value(item) for item in value[:max_dict_items]]
                if isinstance(value, str):
                    return truncate_prompt_value(value, max_chars)
                return value

            compact_fields = []
            for field in base_summary.get("fields", []):
                props = field.get("properties", {}) or {}
                compact_props = {
                    key: compact_value(value)
                    for key, value in props.items()
                    if key in keep_keys
                }
                compact_fields.append({
                    "column": field.get("column"),
                    "properties": compact_props,
                })
            return compact_fields

        # Use compact, truncated row samples. Long free-text cells can make the
        # LLM summary prompt too large without adding useful schema signal.
        prompt_row_count = min(sample_size, len(df), 2)
        sample_df = df.sample(n=prompt_row_count, random_state=42).copy()

        # 对数值型列进行"xx.0"->"xx"规范化
        number_cols = [
            f["column"]
            for f in base_summary["fields"]
            if f["properties"].get("dtype") == "number"
        ]
        for col in number_cols:
            if col in sample_df.columns:
                sample_df[col] = sample_df[col].astype(str).map(
                    lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x
                )

        # 3. to_dict 后递归 normalize
        sample_df = sample_df.apply(lambda col: col.map(lambda value: truncate_prompt_value(value, 40)))
        raw_rows = sample_df.to_dict(orient="records")
        serializable_rows = _normalize_obj(raw_rows)

        # 4. 构造 payload，并 normalize
        payload = {
            **base_summary,
            "fields": compact_fields_for_prompt(),
            "row_samples": serializable_rows,
            "column_evidence": compact_column_evidence(max_examples=2, max_chars=40),
            "evidence_note": (
                "All evidence is sampled or summarized from the dirty input table. "
                "Use it to infer semantics and majority patterns; do not treat "
                "observed values as a complete whitelist."
            ),
        }
        payload = _normalize_obj(payload)
        # print(payload)

        enriched_summary = {
            **base_summary,
            "fields": _normalize_obj(base_summary.get("fields", [])),
            "field_relationships": {
                "hierarchical": {},
                "mathematical": {},
                "temporal": [],
                "associative": {},
            },
        }
        field_index = {
            field.get("column"): field
            for field in enriched_summary.get("fields", [])
            if isinstance(field, dict)
        }
        compact_fields = payload.get("fields", [])
        evidence_by_col = {
            item.get("column"): item
            for item in payload.get("column_evidence", [])
            if isinstance(item, dict)
        }

        batch_system_prompt = (
            "You annotate dirty tabular dataset schemas. Infer each field's "
            "majority dtype, semantic_type, and semantic description from the "
            "field name and compact dirty-data evidence. Ignore isolated "
            "anomalies and do not whitelist observed values. Return JSON only."
        )
        dataset_description = ""
        batch_size = 6
        total_batches = (len(compact_fields) + batch_size - 1) // batch_size if compact_fields else 0
        for start in range(0, len(compact_fields), batch_size):
            field_batch = compact_fields[start:start + batch_size]
            batch_columns = [field.get("column") for field in field_batch]
            batch_no = start // batch_size + 1
            print(f"LLM summary enrichment batch {batch_no}/{total_batches}: {batch_columns}")
            row_batch = [
                {col: row.get(col, "") for col in batch_columns if col in row}
                for row in serializable_rows
            ]
            batch_payload = {
                "dataset_name": base_summary.get("name", ""),
                "fields": field_batch,
                "column_evidence": [
                    evidence_by_col.get(col, {})
                    for col in batch_columns
                    if col in evidence_by_col
                ],
                "row_samples": row_batch,
                "required_output": {
                    "dataset_description": "short table-level description",
                    "fields": [
                        {
                            "column": "existing column name",
                            "properties": {
                                "dtype": "number|string|date|category|boolean",
                                "semantic_type": "short semantic type",
                                "description": "brief semantic description",
                            },
                        }
                    ],
                },
            }
            messages = [
                {"role": "system", "content": batch_system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Enrich this field batch using only the compact evidence. "
                        f"{json.dumps(batch_payload, ensure_ascii=False)}"
                    ),
                },
            ]
            response = summarizer_generator.send_message(messages, max_tokens=summarizer_generator.max_tokens)
            print(f"LLM summary enrichment batch {batch_no}/{total_batches} completed.")
            try:
                parsed = json.loads(clean_code_snippet(response))
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM returned invalid JSON: {e}\\nRaw response: {response}")

            if not dataset_description and isinstance(parsed.get("dataset_description"), str):
                dataset_description = parsed.get("dataset_description", "").strip()

            returned_fields = parsed.get("fields", [])
            if isinstance(returned_fields, dict):
                returned_fields = [returned_fields]
            for item in returned_fields:
                if not isinstance(item, dict):
                    continue
                col = item.get("column")
                if col not in field_index:
                    continue
                props = item.get("properties", {}) or {}
                target_props = field_index[col].setdefault("properties", {})
                for key in ("dtype", "semantic_type", "description"):
                    value = props.get(key)
                    if value not in (None, ""):
                        target_props[key] = value

        if dataset_description:
            enriched_summary["dataset_description"] = dataset_description
        return enriched_summary

        # 5. 调用 LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                "Enriches the following dataset summary JSON based on the provided content: "
                f"{json.dumps(payload, ensure_ascii=False)} "
                "and returns the following JSON without any additional text:" + format_system_prompt
             }
        ]
        response = summarizer_generator.send_message(messages, max_tokens=summarizer_generator.max_tokens)
        enriched_summary = base_summary

        try:
            json_str = clean_code_snippet(response)
            enriched_summary = json.loads(json_str)
            relationships = enriched_summary.get("field_relationships", {})

            # 如果是 list，取第一个 dict，否则重置为 {}
            if isinstance(relationships, list):
                if relationships and isinstance(relationships[0], dict):
                    relationships = relationships[0]
                else:
                    relationships = {}

            detailed = {
                "hierarchical": relationships.get(
                    "hierarchical_consistency",
                    relationships.get("hierarchical", {})
                ),
                "mathematical": relationships.get(
                    "mathematical_dependency",
                    relationships.get("mathematical", {})
                ),
                "temporal": relationships.get(
                    "temporal_sequence",
                    relationships.get("temporal", [])
                ),
                # 直接取 LLM 返回的 associative_dependency 或 associative
                "associative": relationships.get(
                    "associative_dependency",
                    relationships.get("associative", {})
                )
            }
            enriched_summary["field_relationships"] = detailed

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\\nRaw response: {response}")

        return enriched_summary

    # def get_user_feedback(self, summary: dict) -> dict:
    #     print("Generated Summary:")
    #     print(json.dumps(summary, indent=4, ensure_ascii=False))
    #     feedback = input("Do you want to modify the summary? (yes/no): ").strip().lower()
    #     if feedback == "yes":
    #         try:
    #             with open("repair.jsonl", "r", encoding="utf-8") as f:
    #                 corrected_summary = json.load(f)
    #             return corrected_summary
    #         except Exception:
    #             print("[Warning] Failed to load repair.jsonl. Using original summary.")
    #     return summary

    def summarize(
            self,
            data: Union[pd.DataFrame, str],
            file_name: str = "",
            summary_method: str = "llm",
            encoding: str = "utf-8"
    ) -> dict:
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data, encoding=encoding)

        data_properties = self.get_column_properties(data)

        # print(data_properties)

        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
            "field_relationships": ""
        }

        data_summary = self.enrich(base_summary, data, sample_size=50) if summary_method == "llm" else base_summary
        data_summary = self._refresh_profile_metadata(data_summary, data)
        data_summary = self._normalize_field_dtypes_from_profiles(data_summary)
        relationship_evidence = self._build_relationship_evidence(data, data_summary)
        data_summary = self._refine_relationships_with_evidence(data_summary, relationship_evidence)
        # print("**************data_summary**************")
        # print(data_summary)
        # print("**************data_summary**************")
        # 🆕 提取并累积每列正确格式
        format_rules = self.extract_format_rules(data, data_summary)
        format_rules = self._fill_missing_format_rules_from_profiles(format_rules, data_summary)
        data_summary["format_rules"] = format_rules
        # print("*********format_data_summary********")
        # print(data_summary)
        # print("*********format_data_summary********")
        # 生成校验字段关系的函数
        data_summary["relationship_validator_code"] = self.generate_relationship_validator(data, data_summary)
        data_summary["relationship_validator_scope"] = (
            "Relationship validators generate candidate contradictions from summary field relationships; "
            "format, missingness, and distribution anomalies remain candidate evidence only, and final errors require LLM confirmation."
        )

        # print("final summary")
        # print(json.dumps(data_summary, indent=2, ensure_ascii=False))

        # data_summary = self.get_user_feedback(data_summary)
        if isinstance(data_summary, dict):
            data_summary["field_names"] = data.columns.tolist()
            data_summary["file_name"] = file_name
        else:
            data_summary = base_summary

        return data_summary


