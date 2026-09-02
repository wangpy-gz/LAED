# Initial_Screening.py

import re
import os
import pandas as pd
import sys
import io
import contextlib
from difflib import SequenceMatcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _count_regex_alternatives(regex: str) -> int:
    if not regex:
        return 0
    return max(1, regex.count("|") + 1)


def _is_low_confidence_format_rule(rule: dict) -> bool:
    """Return True for over-broad summary rules that enumerate many observed categories."""
    regex = str(rule.get("regex", "") or "")
    fmt = str(rule.get("format", "") or "").lower()
    if not regex:
        return True
    if "category" in fmt and _count_regex_alternatives(regex) > 8:
        return True
    if len(regex) > 600 and _count_regex_alternatives(regex) > 12:
        return True
    return False


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
MAX_RELATIONSHIP_ERROR_RATIO = 0.05
MAX_STRUCTURED_RELATIONSHIP_ERROR_RATIO = 0.65
MAX_OVERALL_RELATIONSHIP_ERROR_RATIO = 0.50
MAX_RELATIONSHIP_CANDIDATE_CELLS_PER_ROW = 2.00
LOW_CONSISTENCY_RELATIONSHIP_RATIO = float(os.getenv("LAED_LOW_CONSISTENCY_RELATIONSHIP_RATIO", "0.70"))
STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO = float(os.getenv("LAED_STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO", "0.99"))


def _normalize_missing_like_token(value) -> str:
    text = "" if value is None else str(value).strip().lower()
    return re.sub(r"^[\s\{\}\[\]\(\)\"']+|[\s\{\}\[\]\(\)\"']+$", "", text)


def _is_missing_like_value(value) -> bool:
    text = "" if value is None else str(value).strip().lower()
    normalized = _normalize_missing_like_token(value)
    return text in _MISSING_LIKE_VALUES or normalized in _MISSING_LIKE_VALUES


def _text_noise_reasons(value) -> list[str]:
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


def _has_duplicate_list_items(value) -> bool:
    text = "" if value is None else str(value)
    if "," not in text and ";" not in text and "|" not in text:
        return False
    parts = [
        re.sub(r"\s+", " ", part.strip().lower())
        for part in re.split(r"[,;|]", text)
        if part and part.strip()
    ]
    return len(parts) >= 2 and len(set(parts)) < len(parts)


def _is_free_text_placeholder(value, props: dict | None = None) -> bool:
    props = props or {}
    text = "" if value is None else str(value).strip()
    if not text or _is_missing_like_value(text):
        return False
    lowered = re.sub(r"\s+", " ", text.lower()).strip(" .;:-_")
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if not any(
        token in field_text
        for token in ("text", "description", "summary", "comment", "review", "note", "content", "free-form")
    ):
        return False
    if re.fullmatch(r"(?:unknown|not known|not available|unavailable|not provided|none provided|tbd|tba|pending)", lowered):
        return True
    if re.fullmatch(r"(?:the\s+)?(?:plot|description|summary|content|details?)\s+(?:is\s+)?unknown(?:\s+at\s+this\s+time)?", lowered):
        return True
    if re.fullmatch(r"(?:add|enter|insert|provide)\s+(?:a|an|the)?\s*[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,3}", lowered):
        return True
    return False


def _date_granularity_warning(value, props: dict) -> bool:
    dtype = str(props.get("dtype", "")).lower()
    semantic_type = str(props.get("semantic_type", "")).lower()
    description = str(props.get("description", "")).lower()
    if dtype != "date" and "date" not in semantic_type and "date" not in description:
        return False
    text = str(value or "").strip()
    if _is_missing_like_value(text):
        return False
    if not re.fullmatch(r"\d{4}\s*(?:\([^)]*\))?", text):
        return False
    full_date_count = 0
    year_only_count = 0
    for shape, raw_count in (props.get("shape_counts") or {}).items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        shape_text = str(shape)
        if re.search(r"9{1,2}\s+A{3,}\s+9{4}", shape_text):
            full_date_count += count
        elif re.fullmatch(r"9{4}(?:\s+\([^)]*\))?", shape_text):
            year_only_count += count
    return full_date_count >= max(30, year_only_count * 3)


def _date_component_order_warning(value, props: dict) -> bool:
    dtype = str(props.get("dtype", "")).lower()
    semantic_type = str(props.get("semantic_type", "")).lower()
    description = str(props.get("description", "")).lower()
    if dtype != "date" and "date" not in semantic_type and "date" not in description:
        return False
    profile = props.get("date_component_profile") or {}
    try:
        matched = int(profile.get("matched_count") or 0)
    except (TypeError, ValueError):
        matched = 0
    positions = profile.get("positions") or []
    if matched < 30 or len(positions) != 3:
        return False
    match = re.fullmatch(r"(\d{1,4})([\/\-.])(\d{1,4})\2(\d{1,4})", str(value).strip())
    if not match:
        return False
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
        stats_by_pos.append({
            "modal_value": modal_value,
            "modal_share": modal_count / matched if matched else 0.0,
            "gt_12_share": int(stats.get("gt_12_count") or 0) / matched if matched else 0.0,
            "gt_31_share": int(stats.get("gt_31_count") or 0) / matched if matched else 0.0,
        })
    constant_positions = [
        idx for idx, stats in enumerate(stats_by_pos)
        if stats["modal_value"] is not None and stats["modal_share"] >= 0.55
    ]
    variable_year_positions = [
        idx for idx, stats in enumerate(stats_by_pos)
        if stats["gt_12_share"] >= 0.20 or stats["gt_31_share"] >= 0.03
    ]
    for const_idx in constant_positions:
        modal_value = stats_by_pos[const_idx]["modal_value"]
        if parts[const_idx] == modal_value:
            continue
        for year_idx in variable_year_positions:
            if year_idx != const_idx and parts[year_idx] == modal_value and parts[const_idx] <= 31:
                return True
    for pos, stats in enumerate(positions):
        top_values = stats.get("top_values") or {}
        if not top_values:
            continue
        modal_value, modal_count = max(
            (
                (int(k), int(v))
                for k, v in top_values.items()
                if str(k).lstrip("-").isdigit()
            ),
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
            return True
    return False


def _normalized_text_key(value) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _profile_spelling_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or _is_missing_like_value(text):
        return False
    if _is_numeric_measure_field(props):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if _field_text_has_token(field_text, ("number", "amount", "score", "date", "time", "timestamp")):
        return False
    skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
    value_skeleton = _normalized_numeric_text_skeleton(text)
    if (
            _field_text_has_token(field_text, ("id", "identifier", "code", "key"))
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
        key = _normalized_text_key(raw_value)
        if not key or key in _MISSING_LIKE_VALUES:
            continue
        counts[key] = max(counts.get(key, 0), count)
        total += count
    value_key = _normalized_text_key(text)
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
    if not frequent_values:
        return False
    for canonical in frequent_values:
        length = max(len(value_key), len(canonical))
        threshold = 0.76 if length <= 12 else 0.86
        if SequenceMatcher(None, value_key, canonical).ratio() >= threshold:
            return True
    return False


def _code_skeleton_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or _is_missing_like_value(text) or not re.search(r"\d", text):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if not _field_text_has_token(field_text, ("id", "identifier", "code", "key")):
        return False
    skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
    if not isinstance(skeleton_counts, dict) or not skeleton_counts:
        return False
    value_skeleton = _normalized_numeric_text_skeleton(text)
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


def _category_code_label_variant_warning(value, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or _is_missing_like_value(text):
        return False
    top_counts = props.get("top_value_counts") or {}
    if not isinstance(top_counts, dict) or not top_counts:
        return False
    short_code_count = 0
    long_label_count = 0
    total = 0
    for raw_value, raw_count in top_counts.items():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        raw_text = str(raw_value).strip()
        if _is_missing_like_value(raw_text):
            continue
        total += count
        if re.fullmatch(r"[A-Za-z]{2,4}", raw_text):
            short_code_count += count
        elif re.fullmatch(r"[A-Za-z][A-Za-z .;,_-]{4,}", raw_text):
            long_label_count += count
    if total <= 0 or short_code_count / total < 0.55:
        return False
    if re.fullmatch(r"[A-Za-z]{2,4}", text):
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z .;,_-]{4,}", text):
        return True
    return False


def _numeric_text_skeleton(value: str) -> str:
    value = "" if value is None else str(value).strip().lower()
    value = re.sub(r"\d+(?:\.\d+)?", "<num>", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _value_shape(value: str) -> str:
    value = "" if value is None else str(value).strip()
    if _is_missing_like_value(value):
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


def _normalized_numeric_text_skeleton(value: str) -> str:
    skeleton = _numeric_text_skeleton(value)
    if skeleton.lower() in _MISSING_LIKE_VALUES:
        return skeleton
    skeleton = re.sub(r"\s+", " ", skeleton).strip().lower()
    skeleton = re.sub(r"[\s\.,;:]+$", "", skeleton)
    skeleton = re.sub(r"(?<=<num> )([a-z]{3,})s\b", r"\1", skeleton)
    return skeleton


def _profile_normalized_numeric_skeleton(props: dict, value: str) -> str:
    raw_skeleton = _normalized_numeric_text_skeleton(value)
    for normalized_skeleton, group in (props.get("normalized_numeric_text_skeleton_groups") or {}).items():
        if isinstance(group, dict) and raw_skeleton in group:
            return str(normalized_skeleton)
    return raw_skeleton


def _is_numeric_measure_field(props: dict) -> bool:
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


def _is_integer_grouping_skeleton(skeleton: str) -> bool:
    return bool(re.fullmatch(r"<num>(?:,<num>)*", str(skeleton or "")))


def _field_text_has_token(field_text: str, tokens: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", field_text)
        for token in tokens
    )


def _has_noncanonical_numeric_skeleton(value: str, props: dict) -> bool:
    text = "" if value is None else str(value).strip()
    if not text or _is_missing_like_value(text) or not re.search(r"\d", text):
        return False
    if not _is_numeric_measure_field(props):
        return False
    field_text = (
        f"{props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    if _field_text_has_token(field_text, ("date", "time", "timestamp")):
        return False
    dominant_normalized = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
    dominant_raw = str(props.get("dominant_numeric_text_skeleton") or "")
    normalized = _profile_normalized_numeric_skeleton(props, text)
    raw = _numeric_text_skeleton(text)
    if _is_integer_grouping_skeleton(dominant_normalized) and _is_integer_grouping_skeleton(normalized):
        return False
    if dominant_normalized and normalized and normalized == dominant_normalized:
        return False
    if dominant_normalized and normalized and normalized != dominant_normalized:
        return True
    numeric_profiles = props.get("numeric_value_profiles") or {}
    skeleton_counts = props.get("normalized_numeric_text_skeleton_counts") or {}
    return bool(dominant_raw and raw and raw != dominant_raw and (raw in numeric_profiles or raw in skeleton_counts))


def _summary_regex_is_trustworthy(props: dict, rule: dict) -> bool:
    regex = str((rule or {}).get("regex", "") or "").strip()
    if not regex:
        return False
    if not _summary_regex_shape_coverage_is_trustworthy(props, regex):
        return False
    top_counts = props.get("top_value_counts") or {}
    if not top_counts:
        return True
    matched = 0
    unmatched = 0
    for value, raw_count in top_counts.items():
        text = str(value).strip()
        if _is_missing_like_value(text):
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


def _summary_regex_shape_coverage_is_trustworthy(props: dict, regex: str) -> bool:
    """
    Validate a summary regex against the compact shape profile. A regex can
    match the top few observed values and still miss a large canonical shape
    family, so high-support shapes must be represented before the rule is used
    as a confident screening signal.
    """
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
        if _is_missing_like_value(text):
            continue
        try:
            count = int(observed_examples.get(value) or 0)
        except (TypeError, ValueError):
            count = 1
        shape = _value_shape(text)
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


def _is_broad_text_regex(regex: str) -> bool:
    pattern = str(regex or "")
    if not pattern:
        return False
    has_digit_constraint = bool(re.search(r"\\d|\[0-9\]|9", pattern))
    has_text_wildcard = bool(re.search(r"\\w\+|\[a-zA-Z|\[A-Za-z|[.][*+]", pattern))
    has_structure = any(token in pattern for token in (r"\d", "-", "_", "/", ":", "%", r"\s"))
    return has_text_wildcard and not has_digit_constraint and not has_structure


def filter_by_profile_evidence(df: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """
    Return True for cells that should remain suspicious even when a broad regex
    matches. This is generic distribution evidence from data_summary.json, not a
    dataset-specific rule list.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    fields = {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }
    format_rules = summary.get("format_rules", {})

    for col, props in fields.items():
        if col not in df.columns:
            continue

        risk_values = set(_MISSING_LIKE_VALUES)
        risk_values.update(
            str(v).strip().lower()
            for v in (props.get("missing_like_values") or {}).keys()
        )
        risk_values.update(
            _normalize_missing_like_token(v)
            for v in (props.get("missing_like_values") or {}).keys()
        )

        regex = str((format_rules.get(col, {}) or {}).get("regex", "") or "")
        if _is_broad_text_regex(regex):
            risk_values.update(
                str(v).strip().lower()
                for v in (props.get("rare_value_examples") or {}).keys()
            )

        variant_groups = props.get("case_punctuation_variant_groups") or {}
        for group in variant_groups.values():
            if not isinstance(group, dict) or len(group) <= 1:
                continue
            canonical = max(group.items(), key=lambda item: int(item[1] or 0))[0]
            for value in group.keys():
                if value != canonical:
                    risk_values.add(str(value).strip().lower())

        for value in {
            **(props.get("top_value_counts") or {}),
            **(props.get("rare_value_examples") or {}),
        }.keys():
            text = str(value)
            if _text_noise_reasons(text) or _has_duplicate_list_items(text) or _is_free_text_placeholder(text, props):
                risk_values.add(text.strip().lower())
        for value in (props.get("text_noise_examples") or {}).keys():
            risk_values.add(str(value).strip().lower())

        numeric_variant_skeletons = set()
        for group in (props.get("normalized_numeric_text_skeleton_groups") or {}).values():
            if not isinstance(group, dict) or len(group) <= 1:
                continue
            numeric_variant_skeletons.update(str(raw_skeleton) for raw_skeleton in group.keys())
        if numeric_variant_skeletons and _is_numeric_measure_field(props):
            for value in {
                **(props.get("top_value_counts") or {}),
                **(props.get("rare_value_examples") or {}),
            }.keys():
                if _numeric_text_skeleton(value) in numeric_variant_skeletons:
                    risk_values.add(str(value).strip().lower())

        dominant_normalized_skeleton = str(props.get("dominant_normalized_numeric_text_skeleton") or "")
        if dominant_normalized_skeleton and _is_numeric_measure_field(props):
            for value in {
                **(props.get("top_value_counts") or {}),
                **(props.get("rare_value_examples") or {}),
            }.keys():
                text = str(value).strip()
                if _is_missing_like_value(text):
                    continue
                value_skeleton = _profile_normalized_numeric_skeleton(props, text)
                if value_skeleton and value_skeleton != dominant_normalized_skeleton:
                    risk_values.add(text.lower())

        if _is_numeric_measure_field(props):
            for value in {
                **(props.get("top_value_counts") or {}),
                **(props.get("rare_value_examples") or {}),
            }.keys():
                text = str(value).strip()
                if _has_noncanonical_numeric_skeleton(text, props):
                    risk_values.add(text.lower())

        if risk_values:
            lowered = df[col].astype(str).str.strip().str.lower()
            normalized = df[col].astype(str).map(_normalize_missing_like_token)
            profile_warning = df[col].astype(str).map(
                lambda value: (
                    bool(_text_noise_reasons(value))
                    or _has_duplicate_list_items(value)
                    or _is_free_text_placeholder(value, props)
                    or _date_granularity_warning(value, props)
                    or _date_component_order_warning(value, props)
                    or _has_noncanonical_numeric_skeleton(value, props)
                    or _profile_spelling_variant_warning(value, props)
                    or _code_skeleton_variant_warning(value, props)
                    or _category_code_label_variant_warning(value, props)
                )
            )
            mask[col] = lowered.isin(risk_values) | normalized.isin(risk_values) | profile_warning

    return mask


def _relationship_columns(relationships: dict, df: pd.DataFrame) -> set:
    cols = set()
    for rel_type, rel_data in (relationships or {}).items():
        if rel_type in {"hierarchical", "associative"} and isinstance(rel_data, dict):
            for key, values in rel_data.items():
                if key in df.columns:
                    cols.add(key)
                if isinstance(values, list):
                    cols.update(v for v in values if v in df.columns)
        elif rel_type == "temporal" and isinstance(rel_data, list):
            for seq in rel_data:
                if isinstance(seq, list):
                    cols.update(v for v in seq if v in df.columns)
        elif rel_type == "mathematical" and isinstance(rel_data, dict):
            for key, info in rel_data.items():
                if key in df.columns:
                    cols.add(key)
                if isinstance(info, dict):
                    cols.update(v for v in info.get("fields", []) if v in df.columns)
                elif isinstance(info, list):
                    cols.update(v for v in info if v in df.columns)
    return cols


def _summary_associative_evidence_pairs(summary: dict, relationships: dict) -> set[tuple[str, str]]:
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
                and _relationship_pair_is_actionable(str(key), str(field), dep, _summary_field_props(summary))
            ):
                pairs.add((str(key), str(field)))
    return pairs


def _field_text_has_token(field_text: str, tokens: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", field_text)
        for token in tokens
    )


def _relationship_structured_field(field: str, props: dict | None = None) -> bool:
    props = props or {}
    field_text = (
        f"{field} {props.get('dtype', '')} {props.get('semantic_type', '')} "
        f"{props.get('description', '')}"
    ).lower()
    return _field_text_has_token(
        field_text,
        (
            "id", "identifier", "code", "key", "number", "num", "issn", "isbn",
            "doi", "date", "time", "datetime", "timestamp", "duration", "year",
            "numeric", "amount", "rate", "percent", "percentage", "score",
        ),
    )


def _summary_field_props(summary: dict) -> dict:
    return {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }


def _relationship_pair_is_actionable(
        key: str,
        dep: str,
        evidence: dict,
        field_props: dict,
) -> bool:
    """
    Decide whether a summary associative dependency may generate suspicious
    cells. The relationship remains in data_summary.json even when this returns
    False; this only prevents weak non-identifier majority mappings from
    becoming final-detection candidates.
    """
    if int(evidence.get("conflicting_cells") or 0) <= 0:
        return False
    if bool(evidence.get("key_is_identifier_like")):
        return True
    key_props = field_props.get(key, {}) or {}
    dep_props = field_props.get(dep, {}) or {}
    if _relationship_structured_field(key, key_props) or _relationship_structured_field(dep, dep_props):
        return True
    return float(evidence.get("consistency_ratio") or 0.0) >= STRICT_NON_IDENTIFIER_RELATIONSHIP_RATIO


def _associative_dependent_columns(summary: dict) -> set[str]:
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    if not isinstance(associative, dict):
        return set()
    deps = set()
    for values in associative.values():
        if isinstance(values, list):
            deps.update(str(value) for value in values)
    return deps


def _actionable_associative_dependent_columns(summary: dict) -> set[str]:
    field_props = _summary_field_props(summary)
    deps = set()
    for candidate in (summary.get("relationship_evidence", {}) or {}).get("candidate_associative_dependencies", []) or []:
        key = str(candidate.get("key") or "")
        for dep in candidate.get("dependents", []) or []:
            field = str(dep.get("field") or "")
            if field and _relationship_pair_is_actionable(key, field, dep, field_props):
                deps.add(field)
    return deps


def _suppress_non_actionable_associative_relationship_mask(
        rel_err_mask: pd.DataFrame,
        summary: dict,
) -> pd.DataFrame:
    assoc_deps = _associative_dependent_columns(summary)
    if not assoc_deps:
        return rel_err_mask
    actionable_deps = _actionable_associative_dependent_columns(summary)
    suppressed = sorted(col for col in assoc_deps - actionable_deps if col in rel_err_mask.columns)
    for col in suppressed:
        if int(rel_err_mask[col].sum()):
            print(
                f"Relationship candidates for '{col}' come only from weak "
                "non-identifier associative evidence; keeping the relationship "
                "in the summary but suppressing it for initial screening."
            )
        rel_err_mask.loc[:, col] = False
    return rel_err_mask


def _summary_associative_evidence_map(summary: dict, relationships: dict) -> dict[tuple[str, str], dict]:
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


def _generic_associative_relationship_mask(
    summary: dict,
    df: pd.DataFrame,
    field_props: dict,
    format_rules: dict,
) -> pd.DataFrame:
    """
    Build a relationship-candidate mask from summary associative evidence. This
    is initial-screening scope only: marked cells still require LLM confirmation
    in Error_Detection_update.py before becoming final errors.
    """
    relationships = summary.get("field_relationships", {}) or {}
    pairs = _summary_associative_evidence_pairs(summary, relationships)
    evidence_map = _summary_associative_evidence_map(summary, relationships)
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for key, dep in sorted(pairs):
        if key not in df.columns or dep not in df.columns or key == dep:
            continue
        grouped = {}
        for idx, row in df[[key, dep]].iterrows():
            key_value = str(row[key]).strip()
            dep_value = str(row[dep]).strip()
            if _is_missing_like_value(key_value) or _is_missing_like_value(dep_value):
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
                    mask.at[idx, dep] = True
            else:
                for idx, value in entries:
                    if value != canonical_value:
                        mask.at[idx, dep] = True
    return mask


def _relationship_error_ratio_limit(col: str, props: dict | None = None) -> float:
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


def _suppress_low_confidence_relationship_mask(
    rel_err_mask: pd.DataFrame,
    rel_cols: set,
    row_count: int,
    field_props: dict | None = None,
) -> pd.DataFrame:
    """
    LLM-generated relationship validators are summary evidence. If a generated
    validator marks a broad fraction of a single field, treat that field-level
    relationship signal as low-confidence instead of sending thousands of likely
    false candidates into the LLM detection stage.
    """
    if row_count <= 0:
        return rel_err_mask

    error_cols = {
        col
        for col in rel_err_mask.columns
        if int(rel_err_mask[col].sum()) > 0
    }
    scoped_cols = [col for col in sorted(set(rel_cols) | error_cols) if col in rel_err_mask.columns]
    if not scoped_cols:
        return rel_err_mask

    broad_cols = []
    field_props = field_props or {}
    for col in scoped_cols:
        col_ratio = int(rel_err_mask[col].sum()) / row_count
        limit = _relationship_error_ratio_limit(col, field_props.get(col, {}))
        if col_ratio > limit:
            broad_cols.append((col, col_ratio))

    rel_cell_count = int(rel_err_mask[scoped_cols].values.sum())
    if row_count and rel_cell_count / row_count <= MAX_RELATIONSHIP_CANDIDATE_CELLS_PER_ROW:
        return rel_err_mask

    if broad_cols:
        for col, col_ratio in broad_cols:
            print(
                f"Relationship validator marked {col_ratio:.2%} of '{col}' cells; "
                "treating this field-level relationship signal as low-confidence for screening."
            )
            rel_err_mask.loc[:, col] = False

    rel_cell_count = int(rel_err_mask[scoped_cols].values.sum())
    rel_scope = row_count * len(scoped_cols)
    rel_ratio = rel_cell_count / rel_scope if rel_scope else 0
    if rel_ratio > MAX_OVERALL_RELATIONSHIP_ERROR_RATIO:
        print(
            f"Relationship validation marked {rel_ratio:.2%} of relationship cells; "
            "treating these summary-derived relationships as low-confidence for screening."
        )
        rel_err_mask.loc[:, scoped_cols] = False

    return rel_err_mask


def filter_by_format(df: pd.DataFrame, rules: dict, field_props: dict | None = None) -> pd.DataFrame:
    """
    Return a DataFrame mask (same shape as df) of booleans:
      True  = this cell matches its column regex rule (after stripping whitespace).
      False = otherwise.
    Columns without regex are all False.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    field_props = field_props or {}
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        regex = rule.get("regex", "")
        if _is_low_confidence_format_rule(rule) or not _summary_regex_is_trustworthy(field_props.get(col, {}), rule):
            mask[col] = False
            continue
        if regex:
            # strip whitespace then full-string match
            mask[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.fullmatch(regex, na=False)
            )
        else:
            mask[col] = False

    # print("Format mask true counts per column:")
    # print(mask.sum())
    # print("Format mask sample (first 5 rows):")
    # print(mask.head())
    return mask


def _is_low_risk_unstructured_text_field(props: dict | None, rule: dict | None) -> bool:
    """
    Identify columns where the summary/profile provides no reliable local format
    rule and arbitrary textual values are expected. Initial screening should not
    send every unique free-text/name value to the final LLM detector solely
    because there is no regex that can prove it correct.
    """
    props = props or {}
    rule = rule or {}
    dtype = str(props.get("dtype", "")).lower()
    semantic_type = str(props.get("semantic_type", "")).lower()
    description = str(props.get("description", "")).lower()
    field_text = f"{dtype} {semantic_type} {description}"

    open_set_text_tokens = (
        "text", "name", "title", "description", "summary", "comment",
        "review", "note", "content", "free-form", "free form",
        "city", "town", "place", "location", "state", "province", "country",
        "region", "address",
    )
    num_unique = int(props.get("num_unique_values") or 0)
    uniqueness = float(props.get("uniqueness_ratio") or 0.0)
    top_counts = props.get("top_value_counts") or {}
    top_total = 0
    top_max = 0
    for raw_count in top_counts.values():
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            continue
        top_total += count
        top_max = max(top_max, count)
    dominant_top_share = (top_max / top_total) if top_total else 0.0
    high_cardinality_text = num_unique >= 200 or uniqueness >= 0.50
    semantically_open_set = any(token in field_text for token in open_set_text_tokens)
    low_dominance_label_set = (
        dtype in {"string", "str", "text", "category"}
        and num_unique >= 20
        and dominant_top_share <= 0.35
    )
    open_set_category = dtype == "category" and (semantically_open_set or low_dominance_label_set)

    if not high_cardinality_text and not semantically_open_set and not low_dominance_label_set:
        return False

    if dtype in {"number", "int", "integer", "float", "date", "datetime", "boolean", "bool"}:
        return False

    hard_structured_tokens = (
        "numeric", "integer", "float", "amount", "price", "rate",
        "percent", "percentage", "date", "time", "timestamp", "boolean",
        "bool", "flag", "indicator",
    )
    if any(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", field_text) for token in hard_structured_tokens):
        return False
    if (
        re.search(r"(?<![A-Za-z0-9])number(?![A-Za-z0-9])", field_text)
        and not any(token in field_text for token in ("address", "street", "location", "place"))
    ):
        return False

    identifier_tokens = ("code", "id", "identifier", "key")
    if not semantically_open_set and any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", field_text)
        for token in identifier_tokens
    ):
        return False

    if dtype == "category" and not open_set_category:
        return False

    regex = str(rule.get("regex", "") or "").strip()
    reliable_format = bool(regex) and not _is_low_confidence_format_rule(rule) and _summary_regex_is_trustworthy(props, rule)
    return not reliable_format


def low_risk_unstructured_text_mask(df: pd.DataFrame, summary: dict, field_props: dict, format_rules: dict) -> pd.DataFrame:
    """
    True means a cell may be treated as correct during initial screening unless
    generic profile evidence or relationship evidence marks that same cell as a
    candidate. The evidence is summary/profile based and dataset-agnostic.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, props in field_props.items():
        if col not in df.columns:
            continue
        if _is_low_risk_unstructured_text_field(props, format_rules.get(col, {})):
            mask[col] = True
    return mask


def filter_by_relationships(df: pd.DataFrame, code: str, relationships: dict) -> pd.DataFrame:
    """
    Execute summary-generated relationship validation code and return a mask
    where True means the cell is a relationship-suspicious candidate.
    Initial screening never turns these candidates into final errors; they
    still require LLM confirmation in Error_Detection_update.py.
    """
    # 1.动态加载验证函数
    relationship_df = df.where(pd.notna(df), "").astype(str)
    exec_globals = {}
    try:
        # 预导入必要的库
        exec_globals.update({
            'pd': pd,
            're': re,
            'datetime': __import__('datetime'),
            'defaultdict': __import__('collections').defaultdict
        })

        # 执行验证代码；LLM 代码可能附带示例 print，执行阶段统一静默。
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, exec_globals)
        validate_fn = exec_globals.get("validate_relationships")
        if validate_fn is None:
            print("Warning: validate_relationships function not found in generated code")
            # 返回空的候选mask
            return pd.DataFrame(False, index=df.index, columns=df.columns)
    except Exception as e:
        print(f"Error executing validation code: {e}")
        import traceback
        traceback.print_exc()
        # 返回空的候选mask
        return pd.DataFrame(False, index=df.index, columns=df.columns)

    # 2.获取所有关系可疑候选(row_idx, col_name)
    try:
        print("开始关系验证...")
        # 临时重定向标准输出，屏蔽验证过程中的详细输出
        with contextlib.redirect_stdout(io.StringIO()):
            base_errors = validate_fn(relationship_df)  # set of (row_idx, col_name)
        print(f"字段间关系候选：发现 {len(base_errors)} 个可疑单元格")
        if base_errors:
            print("前10个关系可疑位置:", list(base_errors)[:10])

            # 统计每个字段的候选数量
            error_by_type = {}
            for idx, col in base_errors:
                error_by_type[col] = error_by_type.get(col, 0) + 1
            print("各字段关系候选统计:", error_by_type)
        else:
            print("没有发现关系可疑候选")
    except Exception as e:
        print(f"运行validate_relationships时出错: {e}")
        import traceback
        traceback.print_exc()
        base_errors = set()

    # 3.构造输出mask - 只标记关系可疑候选的具体单元格
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # 直接标记base_errors中报告的候选单元格
    for idx, col in base_errors:
        if idx not in df.index:
            continue

        # 确保列名匹配
        if col in df.columns:
            mask.at[idx, col] = True
        else:
            # 如果没有完全匹配，尝试找到最接近的列名
            matching_cols = [c for c in df.columns if c.lower() == col.lower()]
            if matching_cols:
                mask.at[idx, matching_cols[0]] = True
            else:
                print(f"警告：在行 {idx} 找不到列 '{col}'")

    # print("Relationship-suspicious mask counts per column:")
    # print(mask.sum())

    # 显示候选详情
    total_errors = mask.sum().sum()
    if total_errors > 0:
        # print(f"关系候选总计: {total_errors} 个单元格")
        # print("关系候选详情:")
        for col in mask.columns:
            col_errors = mask[col].sum()
            if col_errors > 0:
                error_indices = mask.index[mask[col]].tolist()
                # print(
                #     f"  {col}: {col_errors} 个候选 (行: {error_indices[:10]}{'...' if len(error_indices) > 10 else ''})")
    else:
        print("没有发现关系可疑候选")

    return mask


def initial_screening(df: pd.DataFrame, summary: dict):
    # 创建副本以避免修改原始数据
    df_processed = df.copy()

    # 对数值列进行预处理
    number_cols = [
        f["column"]
        for f in summary["fields"]
        if f["properties"].get("dtype") == "number"
    ]
    for col in number_cols:
        if col in df_processed.columns:
            df_processed[col] = (
                df_processed[col]
                .astype(str)
                .map(lambda x: str(int(float(x))) if re.fullmatch(r"\d+\.0", x) else x)
            )

    format_rules = summary["format_rules"]
    rel_code = summary["relationship_validator_code"]["validation"]["code"]
    relationships = summary["field_relationships"]
    field_props = {
        f.get("column"): f.get("properties", {})
        for f in summary.get("fields", [])
        if isinstance(f, dict)
    }

    # print("=== filter_by_format ===")
    fmt_mask = filter_by_format(df_processed, format_rules, field_props)
    profile_suspicious_mask = filter_by_profile_evidence(df_processed, summary)
    unstructured_text_mask = low_risk_unstructured_text_mask(df_processed, summary, field_props, format_rules)

    # print("\n=== filter_by_relationships ===")
    rel_err_mask = filter_by_relationships(df_processed, rel_code, relationships)
    rel_cols = _relationship_columns(relationships, df_processed)
    rel_err_mask = _suppress_low_confidence_relationship_mask(
        rel_err_mask,
        rel_cols,
        len(df_processed),
        field_props,
    )
    rel_err_mask = _suppress_non_actionable_associative_relationship_mask(
        rel_err_mask,
        summary,
    )
    generic_rel_mask = _generic_associative_relationship_mask(
        summary,
        df_processed,
        field_props,
        format_rules,
    )
    if int(generic_rel_mask.values.sum()):
        print(
            f"Generic associative relationship candidates kept for LLM detection: "
            f"{int(generic_rel_mask.values.sum())} cells"
        )
        rel_err_mask = rel_err_mask | generic_rel_mask

    # 即满足格式、或属于无可靠本地格式规则的低风险自由文本；同时不违反关系且无通用 profile 警告
    correct_cells = (fmt_mask | unstructured_text_mask) & (~rel_err_mask) & (~profile_suspicious_mask)
    correct_count = correct_cells.values.sum()
    total_cells = df_processed.size
    pct = correct_count / total_cells * 100
    print(f"\nAbsolutely correct cells: {correct_count}/{total_cells} ({pct:.2f}%)")

    # 保存文件供错误检测模块调试
    correct_cells.to_csv("correct_cells_mask.csv", index=False, encoding="utf-8-sig")
    # 把"可疑行"存成 CSV 以备查看：可疑 == not correct_cells
    df_suspicious = df.mask(correct_cells)  # 对应位置的 True 会被置为 NaN，仅保留可疑
    df_suspicious.to_csv("suspicious_cells_for_error_detection.csv", index=False, encoding="utf-8-sig")

    print(f"已保存正确单元格掩码和可疑单元格子集，检测范围减少 {100 - pct:.2f}%")

    return df_processed, correct_cells, pct






















