from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from LAED_Demo import QWEN_MAX_TOKENS, QWEN_MODEL, QWEN_TEMPERATURE


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Run_Results_Ablation_Strict"
DEFAULT_SMOKE_OUTPUT_ROOT = PROJECT_DIR / "Run_Results_Ablation_Strict_SmokeFull"

DATASETS = ["hospital", "flights", "beers", "rayyan", "movies"]
CONFIGS = ["no_screening", "no_summary", "no_summary_screening"]
ABLATION_SCRIPTS = {
    "no_screening": "ablation_strict_no_screening.py",
    "no_summary": "ablation_strict_no_summary.py",
    "no_summary_screening": "ablation_strict_no_summary_screening.py",
}
DISPLAY_NAMES = {
    "no_screening": "无初筛",
    "no_summary": "无summary",
    "no_summary_screening": "无初筛和summary",
}
METRIC_KEYS = ["precision", "recall", "f1_score"]
COUNT_KEYS = [
    "ground_truth_count",
    "detected_count",
    "true_positives",
    "false_positives",
    "false_negatives",
]
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
REQUIRED_RUN_FILES = {
    "run.log",
    "data_summary.json",
    "correct_cells_mask.csv",
    "suspicious_cells_for_error_detection.csv",
    "errors_with_context.json",
    "detected_errors.json",
    "ground_truth_errors.json",
    "evaluation_metrics.json",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def load_current_laed_baselines() -> dict[str, dict[str, float]]:
    """Load baselines only from actual completed full-LAED run reports."""
    baselines: dict[str, dict[str, float]] = {}
    fixed_path = PROJECT_DIR / "Run_Results" / "latest_fixed_one_run_metrics.json"
    if fixed_path.exists():
        try:
            payload = read_json(fixed_path)
            for run in payload.get("runs", []) or []:
                dataset = str(run.get("dataset", "")).lower()
                if dataset not in DATASETS:
                    continue
                values = {
                    key: float(run[key])
                    for key in METRIC_KEYS
                    if isinstance(run.get(key), (int, float))
                }
                if len(values) == len(METRIC_KEYS):
                    baselines[dataset] = values
        except Exception as exc:
            print(f"[Warn] failed to read current LAED baseline {fixed_path}: {exc}")

    for dataset in DATASETS:
        if dataset in baselines:
            continue
        latest_path = PROJECT_DIR / "Run_Results" / dataset / "latest_metrics_tokens.json"
        if not latest_path.exists():
            continue
        try:
            payload = read_json(latest_path)
            metrics = payload.get("metrics") or {}
            values = {
                key: float(metrics[key])
                for key in METRIC_KEYS
                if isinstance(metrics.get(key), (int, float))
            }
            if len(values) == len(METRIC_KEYS):
                baselines[dataset] = values
        except Exception as exc:
            print(f"[Warn] failed to read latest LAED baseline {latest_path}: {exc}")

    return baselines


def baseline_check(dataset: str, metrics: dict, baselines: dict[str, dict[str, float]]) -> dict:
    baseline = baselines.get(dataset)
    if not baseline:
        return {
            "available": False,
            "baseline": None,
            "metric_below_laed": None,
            "all_metrics_below_laed": None,
        }

    checks = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        base = baseline.get(key)
        checks[key] = (
            isinstance(value, (int, float))
            and isinstance(base, (int, float))
            and float(value) < float(base)
        )
    return {
        "available": True,
        "baseline": baseline,
        "metric_below_laed": checks,
        "all_metrics_below_laed": all(checks.values()) if checks else False,
    }


def field_semantics_blank(summary: dict) -> bool:
    if str(summary.get("dataset_description", "") or "").strip():
        return False
    for field in summary.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        props = field.get("properties", {})
        if not isinstance(props, dict):
            props = field
        if str(props.get("semantic_type", "") or "").strip():
            return False
        if str(props.get("description", "") or "").strip():
            return False
    return True


def field_semantics_present(summary: dict) -> bool:
    if str(summary.get("dataset_description", "") or "").strip():
        return True
    for field in summary.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        props = field.get("properties", {})
        if not isinstance(props, dict):
            props = field
        if str(props.get("semantic_type", "") or "").strip():
            return True
        if str(props.get("description", "") or "").strip():
            return True
    return False


def profile_metadata_absent(summary: dict) -> bool:
    for field in summary.get("fields", []) or []:
        if not isinstance(field, dict):
            continue
        props = field.get("properties", {})
        if not isinstance(props, dict):
            props = field
        if any(key in props for key in PROFILE_METADATA_KEYS):
            return False
    return True


def relationships_empty(summary: dict) -> bool:
    rels = summary.get("field_relationships") or {}
    if not isinstance(rels, dict):
        return not bool(rels)
    return not any(bool(rels.get(key)) for key in ("hierarchical", "mathematical", "temporal", "associative"))


def rules_empty(summary: dict) -> bool:
    return not bool(summary.get("format_rules") or {})


def validator_disabled(summary: dict) -> bool:
    validation = (summary.get("relationship_validator_code") or {}).get("validation") or {}
    status = str(validation.get("status", "") or "").lower()
    code = str(validation.get("code", "") or "")
    return status == "disabled" or "return set()" in code


def validate_run_artifacts(item: dict) -> dict:
    errors: list[str] = []
    config = item.get("config", "")
    dataset = item.get("dataset", "")
    output_dir_text = item.get("output_dir", "")

    if item.get("returncode") != 0:
        errors.append(f"returncode={item.get('returncode')}")
    if not output_dir_text:
        errors.append("missing output_dir in ablation report")
        return {"ok": False, "errors": errors}

    output_dir = Path(output_dir_text)
    if not output_dir.exists():
        errors.append(f"output_dir does not exist: {output_dir}")
        return {"ok": False, "errors": errors}

    for filename in sorted(REQUIRED_RUN_FILES):
        path = output_dir / filename
        if not path.exists():
            errors.append(f"missing required file: {filename}")

    summary_path = output_dir / "data_summary.json"
    detection_summary_path = output_dir / "data_summary_for_detection.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    detection_summary = read_json(detection_summary_path) if detection_summary_path.exists() else {}

    if config == "no_screening":
        if not detection_summary_path.exists():
            errors.append("no_screening missing data_summary_for_detection.json")
        if summary and not field_semantics_present(summary):
            errors.append("no_screening original data_summary.json does not contain semantic summary text")
        if detection_summary:
            if not field_semantics_present(detection_summary):
                errors.append("no_screening detection summary lost semantic text")
            if not rules_empty(detection_summary):
                errors.append("no_screening detection summary still contains format_rules")
            if not relationships_empty(detection_summary):
                errors.append("no_screening detection summary still contains field_relationships")
            if not validator_disabled(detection_summary):
                errors.append("no_screening detection summary still contains enabled validator")
            if not profile_metadata_absent(detection_summary):
                errors.append("no_screening detection summary still contains profile metadata")

    elif config == "no_summary":
        if not detection_summary_path.exists():
            errors.append("no_summary missing data_summary_for_detection.json")
        for label, payload in (("summary", summary), ("detection_summary", detection_summary)):
            if payload and not field_semantics_blank(payload):
                errors.append(f"no_summary {label} contains semantic summary text")
        if detection_summary and not profile_metadata_absent(detection_summary):
            errors.append("no_summary detection summary still contains profile metadata")

    elif config == "no_summary_screening":
        if not detection_summary_path.exists():
            errors.append("no_summary_screening missing data_summary_for_detection.json")
        for label, payload in (("summary", summary), ("detection_summary", detection_summary)):
            if payload and not field_semantics_blank(payload):
                errors.append(f"no_summary_screening {label} contains semantic summary text")
            if payload and not rules_empty(payload):
                errors.append(f"no_summary_screening {label} still contains format_rules")
            if payload and not relationships_empty(payload):
                errors.append(f"no_summary_screening {label} still contains field_relationships")
            if payload and not validator_disabled(payload):
                errors.append(f"no_summary_screening {label} still contains enabled validator")
            if payload and not profile_metadata_absent(payload):
                errors.append(f"no_summary_screening {label} still contains profile metadata")

    metrics_path = output_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        try:
            metrics = read_json(metrics_path)
            for key in METRIC_KEYS:
                if not isinstance(metrics.get(key), (int, float)):
                    errors.append(f"{dataset}/{config} evaluation_metrics missing numeric {key}")
        except Exception as exc:
            errors.append(f"failed to parse evaluation_metrics.json: {exc}")

    return {"ok": not errors, "errors": errors}


def run_one(config: str, dataset: str, output_root: Path, baselines: dict[str, dict[str, float]]) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    script = CURRENT_DIR / ABLATION_SCRIPTS[config]
    stdout_path = logs_dir / f"{config}_{dataset}_{command_id}.stdout.log"
    stderr_path = logs_dir / f"{config}_{dataset}_{command_id}.stderr.log"
    log_path = logs_dir / f"{config}_{dataset}_{command_id}.log"
    latest_report = output_root / config / dataset / "latest_ablation_metrics_tokens.json"

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--dataset",
        dataset,
        "--data-dir",
        str(DEFAULT_DATA_DIR),
        "--output-root",
        str(output_root),
        "--mode",
        "full",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = "0"
    env["LAED_DETECTION_REQUEST_TIMEOUT"] = env.get("LAED_DETECTION_REQUEST_TIMEOUT", "120")
    env["LAED_RELATIONSHIP_REQUEST_TIMEOUT"] = env.get("LAED_RELATIONSHIP_REQUEST_TIMEOUT", "120")
    env["LAED_DETECTION_RETRIES"] = env.get("LAED_DETECTION_RETRIES", "1")
    env["LAED_RELATIONSHIP_MAX_RETRIES"] = env.get("LAED_RELATIONSHIP_MAX_RETRIES", "0")

    started_perf = time.perf_counter()
    started_wall = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open("w", encoding="utf-8") as stderr_log:
        proc = subprocess.run(
            cmd,
            cwd=str(CURRENT_DIR),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=stdout_log,
            stderr=stderr_log,
        )

    with log_path.open("w", encoding="utf-8") as merged:
        merged.write("Command: " + " ".join(cmd) + "\n")
        merged.write("Model: " + str(QWEN_MODEL) + "\n")
        merged.write(f"Temperature: {QWEN_TEMPERATURE}\n")
        merged.write(f"Max tokens: {QWEN_MAX_TOKENS}\n")
        merged.write(f"Return code: {proc.returncode}\n")
        merged.write("=" * 80 + "\n")
        merged.write(stdout_path.read_text(encoding="utf-8", errors="replace"))
        if stderr_path.exists() and stderr_path.stat().st_size:
            merged.write("\n" + "=" * 80 + "\nSTDERR\n" + "=" * 80 + "\n")
            merged.write(stderr_path.read_text(encoding="utf-8", errors="replace"))

    report: dict = {}
    if proc.returncode == 0 and latest_report.exists() and latest_report.stat().st_mtime >= started_wall - 2:
        report = read_json(latest_report)
        output_dir = report.get("output_dir")
        if output_dir:
            run_log_path = Path(output_dir) / "run.log"
            try:
                run_log_path.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except OSError as exc:
                print(f"[Warn] failed to write per-run run.log for {output_dir}: {exc}")
    elif proc.returncode == 0:
        print(f"[Warn] {config}/{dataset} returned 0 but did not refresh latest report: {latest_report}")

    metrics = report.get("metrics", {})
    result = {
        "config": config,
        "module": DISPLAY_NAMES.get(config, config),
        "dataset": dataset,
        "returncode": proc.returncode,
        "runtime_seconds": round(time.perf_counter() - started_perf, 2),
        "metrics": metrics,
        "baseline_check": baseline_check(dataset, metrics, baselines),
        "ablation_report": report,
        "latest_report": str(latest_report),
        "output_dir": report.get("output_dir", ""),
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    result["artifact_validation"] = validate_run_artifacts(result)
    return result


def avg(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def build_table_rows(results: list[dict], baselines: dict[str, dict[str, float]]) -> list[dict]:
    by_config_dataset = {
        (item["config"], item["dataset"]): item
        for item in results
        if item.get("returncode") == 0
    }
    rows = []
    for config in CONFIGS:
        row = {"消融模块": DISPLAY_NAMES[config]}
        for dataset in DATASETS:
            item = by_config_dataset.get((config, dataset), {})
            metrics = item.get("metrics") or {}
            check = item.get("baseline_check") or baseline_check(dataset, metrics, baselines)
            row[f"{dataset}_P"] = metrics.get("precision")
            row[f"{dataset}_R"] = metrics.get("recall")
            row[f"{dataset}_F1"] = metrics.get("f1_score")
            row[f"{dataset}_below_LAED"] = check.get("all_metrics_below_laed")
        rows.append(row)
    return rows


def build_average_rows(round_results: list[list[dict]], baselines: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for config in CONFIGS:
        row = {"消融模块": DISPLAY_NAMES[config]}
        for dataset in DATASETS:
            items = [
                item
                for round_items in round_results
                for item in round_items
                if item.get("returncode") == 0
                and item.get("config") == config
                and item.get("dataset") == dataset
            ]
            averaged = {}
            for key, suffix in (("precision", "P"), ("recall", "R"), ("f1_score", "F1")):
                averaged[key] = avg([(item.get("metrics") or {}).get(key) for item in items])
                row[f"{dataset}_{suffix}"] = averaged[key]
            row[f"{dataset}_below_LAED"] = baseline_check(dataset, averaged, baselines).get("all_metrics_below_laed")
        rows.append(row)
    return rows


def markdown_table(rows: list[dict]) -> str:
    headers = ["消融模块"]
    for dataset in DATASETS:
        headers.extend([f"{dataset}-P", f"{dataset}-R", f"{dataset}-F1", f"{dataset}<LAED"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get("消融模块", ""))]
        for dataset in DATASETS:
            for suffix in ("P", "R", "F1"):
                value = row.get(f"{dataset}_{suffix}")
                values.append("-" if value is None else f"{float(value):.4f}")
            below_laed = row.get(f"{dataset}_below_LAED")
            values.append("-" if below_laed is None else ("yes" if below_laed else "no"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_rows_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["消融模块"]
    for dataset in DATASETS:
        fieldnames.extend([f"{dataset}_P", f"{dataset}_R", f"{dataset}_F1", f"{dataset}_below_LAED"])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_long_csv(round_results: list[list[dict]], output_root: Path, created_at: str) -> Path:
    path = output_root / f"strict_ablation_long_metrics_{created_at}.csv"
    fieldnames = [
        "round",
        "config",
        "module",
        "dataset",
        *METRIC_KEYS,
        *COUNT_KEYS,
        "all_metrics_below_laed",
        "artifact_validation_ok",
        "runtime_seconds",
        "output_dir",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for round_index, items in enumerate(round_results, start=1):
            for item in items:
                metrics = item.get("metrics") or {}
                writer.writerow({
                    "round": round_index,
                    "config": item.get("config"),
                    "module": item.get("module") or DISPLAY_NAMES.get(item.get("config"), item.get("config")),
                    "dataset": item.get("dataset"),
                    **{key: metrics.get(key) for key in METRIC_KEYS + COUNT_KEYS},
                    "all_metrics_below_laed": (item.get("baseline_check") or {}).get("all_metrics_below_laed"),
                    "artifact_validation_ok": (item.get("artifact_validation") or {}).get("ok"),
                    "runtime_seconds": item.get("runtime_seconds"),
                    "output_dir": item.get("output_dir"),
                    "log_path": item.get("log_path"),
                })
    latest = output_root / "latest_strict_ablation_long_metrics.csv"
    latest.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    return path


def write_summary(
    round_results: list[list[dict]],
    output_root: Path,
    baselines: dict[str, dict[str, float]],
    label: str,
) -> dict:
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    tables = []
    for index, results in enumerate(round_results, start=1):
        rows = build_table_rows(results, baselines)
        tables.append({
            "name": f"第{index}遍结果",
            "round": index,
            "rows": rows,
            "markdown": markdown_table(rows),
        })
        write_rows_csv(rows, output_root / f"strict_ablation_round_{index}_{created_at}.csv")

    average_rows = build_average_rows(round_results, baselines)
    tables.append({
        "name": f"{len(round_results)}遍平均结果",
        "round": "average_all_runs",
        "rows": average_rows,
        "markdown": markdown_table(average_rows),
    })
    write_rows_csv(average_rows, output_root / f"strict_ablation_average_{len(round_results)}_runs_{created_at}.csv")

    long_csv = write_long_csv(round_results, output_root, created_at)
    failed_validations = [
        item
        for round_items in round_results
        for item in round_items
        if not (item.get("artifact_validation") or {}).get("ok")
    ]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "model_configuration": {
            "model": QWEN_MODEL,
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": QWEN_MAX_TOKENS,
        },
        "average_policy": f"Run {len(round_results)} complete round(s); average all successful results.",
        "baseline_comparison_available": bool(baselines),
        "laed_baselines_source": "actual completed full-LAED run reports" if baselines else None,
        "laed_baselines": baselines,
        "datasets": DATASETS,
        "configs": CONFIGS,
        "display_names": DISPLAY_NAMES,
        "round_results": round_results,
        "failed_artifact_validations": failed_validations,
        "tables": tables,
        "long_metrics_csv": str(long_csv),
    }
    json_path = output_root / f"strict_ablation_{label}_tables_{created_at}.json"
    latest_json = output_root / f"latest_strict_ablation_{label}_tables.json"
    for path in (json_path, latest_json):
        dump_json(payload, path)

    md_lines = [
        "# LAED Strict Ablation Results",
        "",
        f"Label: {label}",
        f"Created at: {payload['created_at']}",
        f"Model: {QWEN_MODEL}; temperature: {QWEN_TEMPERATURE}; max_tokens: {QWEN_MAX_TOKENS}",
        "",
        payload["average_policy"],
    ]
    for table in tables:
        md_lines.extend(["", f"## {table['name']}", "", table["markdown"]])
    if failed_validations:
        md_lines.extend(["", "## Artifact Validation Failures", ""])
        for item in failed_validations:
            errors = "; ".join((item.get("artifact_validation") or {}).get("errors") or [])
            md_lines.append(f"- {item.get('config')}/{item.get('dataset')}: {errors}")
    md_path = output_root / f"strict_ablation_{label}_tables_{created_at}.md"
    latest_md = output_root / f"latest_strict_ablation_{label}_tables.md"
    md_text = "\n".join(md_lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")
    return payload


def run_rounds(
    rounds: int,
    configs: list[str],
    datasets: list[str],
    output_root: Path,
    baselines: dict[str, dict[str, float]],
    label: str,
) -> tuple[list[list[dict]], dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    round_results = []
    rounds = max(1, int(rounds))
    for round_index in range(1, rounds + 1):
        current = []
        for config in configs:
            for dataset in datasets:
                result = run_one(config, dataset, output_root, baselines)
                current.append(result)
                validation = result.get("artifact_validation") or {}
                print(
                    f"{label} round={round_index}/{rounds} {DISPLAY_NAMES.get(config, config)}/{dataset}: "
                    f"returncode={result['returncode']} metrics={result['metrics']} "
                    f"below_laed={(result.get('baseline_check') or {}).get('all_metrics_below_laed')} "
                    f"artifacts_ok={validation.get('ok')} log={result['log_path']}"
                )
                if not validation.get("ok"):
                    print(f"  artifact/errors: {'; '.join(validation.get('errors') or [])}")
        round_results.append(current)

    payload = write_summary(round_results, output_root, baselines, label)
    print(f"Saved strict ablation tables: {output_root / f'latest_strict_ablation_{label}_tables.md'}")
    print(f"Saved strict ablation JSON: {output_root / f'latest_strict_ablation_{label}_tables.json'}")
    return round_results, payload


def smoke_passed(round_results: list[list[dict]]) -> bool:
    expected = len(CONFIGS) * len(DATASETS)
    items = [item for round_items in round_results for item in round_items]
    if len(items) != expected:
        return False
    return all(item.get("returncode") == 0 and (item.get("artifact_validation") or {}).get("ok") for item in items)


def write_smoke_gate_report(round_results: list[list[dict]], output_root: Path, passed: bool) -> Path:
    failures = [
        {
            "config": item.get("config"),
            "dataset": item.get("dataset"),
            "returncode": item.get("returncode"),
            "artifact_errors": (item.get("artifact_validation") or {}).get("errors") or [],
            "log_path": item.get("log_path"),
            "stdout_path": item.get("stdout_path"),
            "stderr_path": item.get("stderr_path"),
        }
        for round_items in round_results
        for item in round_items
        if item.get("returncode") != 0 or not (item.get("artifact_validation") or {}).get("ok")
    ]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "required_child_tasks": len(CONFIGS) * len(DATASETS),
        "failures": failures,
        "policy": "Formal two-round experiment may run only after this full smoke gate passes.",
    }
    path = output_root / "latest_smoke_gate_report.json"
    dump_json(report, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict LAED ablation experiments.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-smoke-gate", action="store_true", help="Run one full smoke round before formal rounds.")
    parser.add_argument("--smoke-output-root", type=Path, default=DEFAULT_SMOKE_OUTPUT_ROOT)
    parser.add_argument("--smoke-only", action="store_true", help="Run only the full smoke gate and skip formal rounds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = load_current_laed_baselines()
    if baselines:
        print("Loaded current LAED baselines for ablation checks:")
        print(json.dumps(baselines, ensure_ascii=False, indent=2))
    else:
        print("No completed full-LAED baseline reports found; baseline comparison will be skipped.")

    if args.run_smoke_gate or args.smoke_only:
        smoke_root = args.smoke_output_root.resolve()
        smoke_results, _ = run_rounds(
            rounds=1,
            configs=args.configs,
            datasets=args.datasets,
            output_root=smoke_root,
            baselines=baselines,
            label="smoke_gate",
        )
        passed = smoke_passed(smoke_results)
        report_path = write_smoke_gate_report(smoke_results, smoke_root, passed)
        print(f"Smoke gate report: {report_path}")
        if not passed:
            print("Smoke gate failed; formal two-round experiment will not run.")
            sys.exit(1)
        print("Smoke gate passed.")
        if args.smoke_only:
            return

    output_root = args.output_root.resolve()
    formal_results, payload = run_rounds(
        rounds=args.rounds,
        configs=args.configs,
        datasets=args.datasets,
        output_root=output_root,
        baselines=baselines,
        label="formal",
    )
    formal_ok = all(
        item.get("returncode") == 0 and (item.get("artifact_validation") or {}).get("ok")
        for round_items in formal_results
        for item in round_items
    )
    print(f"Formal run artifact status: {'ok' if formal_ok else 'has failures'}")
    print(f"Tables generated: {len(payload['tables'])}")


if __name__ == "__main__":
    main()
