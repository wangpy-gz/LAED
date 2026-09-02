from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from LAED_Demo import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_ROOT, load_csv, resolve_dataset_paths
from summarizer import Summarizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def regex_alternatives(regex: str) -> list[str]:
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
        literals.append(
            part.replace("\\.", ".")
            .replace("\\(", "(")
            .replace("\\)", ")")
            .replace("\\|", "|")
        )
    return literals


def missing_like(value) -> bool:
    text = "" if value is None else str(value).strip().lower()
    return text in {"", "nan", "none", "null", "n/a", "na", "nil", "missing", "empty", "?"}


def field_props(summary: dict) -> dict[str, dict]:
    return {
        str(field.get("column")): (
            field.get("properties", {}) if isinstance(field.get("properties", {}), dict)
            else {
                "dtype": field.get("dtype", ""),
                "semantic_type": field.get("semantic_type", ""),
                "description": field.get("description", ""),
            }
        ) or {}
        for field in summary.get("fields", [])
        if isinstance(field, dict) and field.get("column") is not None
    }


def audit_format_rules(summary: dict, profile_summary: dict | None = None) -> list[dict]:
    issues = []
    props_by_col = field_props(profile_summary or summary)
    for col, rule in (summary.get("format_rules") or {}).items():
        if not isinstance(rule, dict):
            issues.append({"column": col, "severity": "error", "issue": "format rule is not an object"})
            continue
        regex = str(rule.get("regex", "") or "").strip()
        props = props_by_col.get(col, {})
        dtype = str(props.get("dtype", "") or "").lower()
        top_counts = props.get("top_value_counts") or {}
        shape_counts = props.get("shape_counts") or {}
        if not regex:
            continue
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            issues.append({"column": col, "severity": "error", "issue": f"invalid regex: {exc}", "regex": regex})
            continue

        alternatives = regex_alternatives(regex)
        if alternatives and (
            props.get("value_counts_truncated")
            or len(alternatives) >= 4
            or (top_counts and len(alternatives) >= max(4, len(top_counts) // 2))
        ):
            issues.append({
                "column": col,
                "severity": "warning",
                "issue": "regex looks like a top-value whitelist rather than a canonical format",
                "regex": regex,
                "alternative_count": len(alternatives),
            })

        if top_counts:
            matched = 0
            unmatched = 0
            unmatched_examples = []
            for value, raw_count in top_counts.items():
                text = str(value).strip()
                if missing_like(text):
                    continue
                count = int(raw_count or 0)
                if compiled.fullmatch(text):
                    matched += count
                else:
                    unmatched += count
                    if len(unmatched_examples) < 5:
                        unmatched_examples.append(text)
            total = matched + unmatched
            if total and unmatched / total > 0.15:
                issues.append({
                    "column": col,
                    "severity": "warning",
                    "issue": "regex rejects many recurring top-count values",
                    "regex": regex,
                    "unmatched_top_ratio": round(unmatched / total, 4),
                    "unmatched_examples": unmatched_examples,
                })

        if shape_counts and top_counts:
            represented = set()
            for value in top_counts:
                text = str(value).strip()
                if missing_like(text):
                    continue
                shape = re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", text))
                if compiled.fullmatch(text):
                    represented.add(shape)
            total_shape = 0
            represented_shape = 0
            for shape, raw_count in shape_counts.items():
                if str(shape) == "<missing-like>":
                    continue
                count = int(raw_count or 0)
                total_shape += count
                if str(shape) in represented:
                    represented_shape += count
            if total_shape and represented_shape / total_shape < 0.80:
                issues.append({
                    "column": col,
                    "severity": "warning",
                    "issue": "regex covers too little observed shape support",
                    "regex": regex,
                    "represented_shape_ratio": round(represented_shape / total_shape, 4),
                })

        if dtype in {"category", "string"} and re.search(r"\.\*\[[^]]*[xuXU][^]]*]|\.\*\[xX]|\.\*[xuXU]", regex):
            issues.append({
                "column": col,
                "severity": "warning",
                "issue": "regex is an overly broad text/noise pattern",
                "regex": regex,
            })
    return issues


def audit_required_summary_contract(dataset: str, summary: dict) -> list[dict]:
    issues = []
    rules = summary.get("format_rules") or {}
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}

    def require_regex(col: str):
        regex = str((rules.get(col, {}) or {}).get("regex", "") or "")
        if not regex:
            issues.append({"severity": "error", "issue": "required canonical regex is missing", "column": col})

    def require_assoc(key: str, deps: set[str]):
        actual = set(associative.get(key, []) if isinstance(associative.get(key, []), list) else [])
        missing = sorted(deps - actual)
        if missing:
            issues.append({
                "severity": "error",
                "issue": "required associative dependents are missing",
                "key": key,
                "missing": missing,
                "actual": sorted(actual),
            })

    def forbid_assoc_key(key: str):
        if key in associative:
            issues.append({
                "severity": "error",
                "issue": "forbidden associative key present",
                "key": key,
                "actual": associative.get(key),
            })

    if dataset == "beers":
        require_regex("ounces")
        require_regex("abv")
        require_assoc("brewery_id", {"brewery_name", "city", "state"})
        require_assoc("city", {"state"})
    elif dataset == "flights":
        require_assoc("flight", {"sched_arr_time", "sched_dep_time", "act_arr_time", "act_dep_time"})
    elif dataset == "hospital":
        require_assoc("ProviderNumber", {
            "City", "CountyName", "EmergencyService", "HospitalName",
            "HospitalOwner", "HospitalType", "State",
        })
        require_assoc("MeasureCode", {"MeasureName"})
        forbid_assoc_key("Address")
        forbid_assoc_key("ZipCode")
    elif dataset == "rayyan":
        require_assoc("journal_issn", {"journal_abbreviation", "journal_title"})
        forbid_assoc_key("journal_abbreviation")
        forbid_assoc_key("journal_title")
    elif dataset == "movies":
        require_assoc("Release Date", {"First Shown Year"})
    return issues


def audit_relationships(summary: dict, relationship_evidence: dict | None = None) -> dict:
    relationships = summary.get("field_relationships", {}) or {}
    associative = relationships.get("associative", {}) if isinstance(relationships, dict) else {}
    evidence = relationship_evidence if relationship_evidence is not None else summary.get("relationship_evidence", {}) or {}
    evidence_pairs = {}
    missing_from_summary = []
    weak_in_summary = []
    for candidate in evidence.get("candidate_associative_dependencies", []) or []:
        key = str(candidate.get("key") or "")
        for dep in candidate.get("dependents", []) or []:
            field = str(dep.get("field") or "")
            if not key or not field:
                continue
            evidence_pairs[(key, field)] = dep
            if dep.get("actionable_conflicts") or dep.get("stable_dependency"):
                listed = field in (associative.get(key, []) if isinstance(associative.get(key, []), list) else [])
                if not listed:
                    missing_from_summary.append({
                        "key": key,
                        "dependent": field,
                        "consistency_ratio": dep.get("consistency_ratio"),
                        "conflicting_cells": dep.get("conflicting_cells"),
                        "stable_dependency": dep.get("stable_dependency"),
                        "actionable_conflicts": dep.get("actionable_conflicts"),
                    })
    for key, deps in associative.items():
        if not isinstance(deps, list):
            weak_in_summary.append({"key": key, "issue": "associative dependents are not a list"})
            continue
        for dep in deps:
            pair = (str(key), str(dep))
            ev = evidence_pairs.get(pair)
            if not ev:
                weak_in_summary.append({"key": key, "dependent": dep, "issue": "no supporting evidence"})
            elif not (ev.get("actionable_conflicts") or ev.get("stable_dependency")):
                weak_in_summary.append({
                    "key": key,
                    "dependent": dep,
                    "issue": "weak evidence",
                    "consistency_ratio": ev.get("consistency_ratio"),
                    "conflicting_cells": ev.get("conflicting_cells"),
                })
    return {
        "associative": associative,
        "missing_evidence_supported_pairs_from_summary": missing_from_summary,
        "weak_or_unsupported_pairs_in_summary": weak_in_summary,
    }


def audit_summary(dataset: str, summary_path: Path, data_dir: Path) -> dict:
    dataset_name, dirty_path, _clean_path = resolve_dataset_paths(dataset, data_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    df = load_csv(dirty_path)
    profile_summary = json.loads(json.dumps(summary, ensure_ascii=False))
    if profile_summary.get("fields") and not isinstance(profile_summary["fields"][0].get("properties", None), dict):
        profile_summary["fields"] = [
            {
                "column": field.get("column"),
                "properties": {
                    "dtype": field.get("dtype", ""),
                    "semantic_type": field.get("semantic_type", ""),
                    "description": field.get("description", ""),
                },
            }
            for field in profile_summary.get("fields", [])
            if isinstance(field, dict)
        ]
    profile_summary = Summarizer()._refresh_profile_metadata(profile_summary, df)
    recomputed_evidence = Summarizer()._build_relationship_evidence(
        df,
        profile_summary,
        max_keys=20,
        max_dependents_per_key=20,
    )
    return {
        "dataset": dataset_name,
        "summary_path": str(summary_path),
        "field_count": len(summary.get("fields", [])),
        "format_rule_count": len(summary.get("format_rules", {}) or {}),
        "format_rule_issues": audit_format_rules(summary, profile_summary),
        "contract_issues": audit_required_summary_contract(dataset_name, summary),
        "relationship_audit": audit_relationships(summary, recomputed_evidence),
        "embedded_relationship_audit": audit_relationships(summary),
        "recomputed_relationship_evidence": recomputed_evidence,
    }


def write_markdown(audits: list[dict], path: Path) -> None:
    lines = ["# LAED Summary Audit", ""]
    for audit in audits:
        lines.extend([
            f"## {audit['dataset']}",
            "",
            f"- Summary: `{audit['summary_path']}`",
            f"- Fields: {audit['field_count']}",
            f"- Format rules: {audit['format_rule_count']}",
            f"- Format issues: {len(audit['format_rule_issues'])}",
            f"- Contract issues: {len(audit.get('contract_issues', []))}",
            f"- Missing evidence-supported relationship pairs: {len(audit['relationship_audit']['missing_evidence_supported_pairs_from_summary'])}",
            f"- Weak/unsupported relationship pairs: {len(audit['relationship_audit']['weak_or_unsupported_pairs_in_summary'])}",
            "",
        ])
        for issue in audit["format_rule_issues"][:20]:
            lines.append(f"- `{issue.get('column')}`: {issue.get('issue')} ({issue.get('severity')})")
        for issue in audit.get("contract_issues", [])[:20]:
            if issue.get("column"):
                lines.append(f"- Contract `{issue.get('column')}`: {issue.get('issue')} ({issue.get('severity')})")
            else:
                lines.append(f"- Contract `{issue.get('key')}`: {issue.get('issue')} ({issue.get('severity')})")
        for pair in audit["relationship_audit"]["missing_evidence_supported_pairs_from_summary"][:20]:
            lines.append(
                f"- Missing relationship `{pair.get('key')} -> {pair.get('dependent')}` "
                f"cons={pair.get('consistency_ratio')} conflicts={pair.get('conflicting_cells')}"
            )
        for pair in audit["relationship_audit"]["weak_or_unsupported_pairs_in_summary"][:20]:
            lines.append(
                f"- Weak relationship `{pair.get('key')} -> {pair.get('dependent')}`: {pair.get('issue')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LAED data_summary.json files without using clean data.")
    parser.add_argument("items", nargs="+", help="Dataset=summary_path pairs or plain summary paths.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = []
    for item in args.items:
        if "=" in item:
            dataset, raw_path = item.split("=", 1)
            summary_path = Path(raw_path)
        else:
            summary_path = Path(item)
            dataset = summary_path.parent.parent.name
        audits.append(audit_summary(dataset, summary_path.resolve(), args.data_dir))
    out_dir = args.output_root / "summary_audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest_summary_audit.json"
    md_path = out_dir / "latest_summary_audit.md"
    json_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(audits, md_path)
    print(f"Saved summary audit JSON: {json_path}")
    print(f"Saved summary audit Markdown: {md_path}")
    for audit in audits:
        print(
            f"{audit['dataset']}: format_issues={len(audit['format_rule_issues'])}, "
            f"contract_issues={len(audit.get('contract_issues', []))}, "
            f"missing_relationships={len(audit['relationship_audit']['missing_evidence_supported_pairs_from_summary'])}, "
            f"weak_relationships={len(audit['relationship_audit']['weak_or_unsupported_pairs_in_summary'])}"
        )


if __name__ == "__main__":
    main()
