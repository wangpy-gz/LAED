# LAED_Demo.py
"""
Run LAED on a specified dataset, evaluate the detected error cells, and record
Qwen token usage for the whole demo_LAED run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from summarizer import Summarizer
from Error_Detection_update import DetectionExplorer
from Initial_Screening_all import initial_screening
from pythonProject1.API_invocation.qwen_gen import (
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    get_shared_usage,
    reset_shared_usage,
    shared_qwen_client,
)


QWEN_MODEL = "qwen2.5-72b-instruct"
QWEN_MAX_TOKENS = 8192
QWEN_TEMPERATURE = 0.0
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Run_Results"
ERROR_SUFFIXES = ("_error", "_dirty")

ErrorCell = Tuple[int, str]


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
    """Force all shared API calls in this run to use deterministic Qwen settings."""
    shared_qwen_client.model = QWEN_MODEL
    shared_qwen_client.max_tokens = QWEN_MAX_TOKENS
    shared_qwen_client.temperature = QWEN_TEMPERATURE
    if DEFAULT_MODEL != QWEN_MODEL:
        print(f"[Warning] qwen_gen.DEFAULT_MODEL is {DEFAULT_MODEL}, forcing {QWEN_MODEL}.")
    if DEFAULT_MAX_TOKENS != QWEN_MAX_TOKENS:
        print(f"[Warning] qwen_gen.DEFAULT_MAX_TOKENS is {DEFAULT_MAX_TOKENS}, forcing {QWEN_MAX_TOKENS}.")
    if float(DEFAULT_TEMPERATURE) != QWEN_TEMPERATURE:
        print(f"[Warning] qwen_gen.DEFAULT_TEMPERATURE is {DEFAULT_TEMPERATURE}, forcing {QWEN_TEMPERATURE}.")


def list_available_datasets(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    datasets = set()
    if not data_dir.exists():
        return []

    for csv_path in data_dir.glob("*.csv"):
        stem = csv_path.stem.lower()
        for suffix in ("_clean", *ERROR_SUFFIXES):
            if stem.endswith(suffix):
                datasets.add(stem[: -len(suffix)])
                break
    return sorted(datasets)


def _find_file_case_insensitive(data_dir: Path, filename: str) -> Optional[Path]:
    target = filename.lower()
    for csv_path in data_dir.glob("*.csv"):
        if csv_path.name.lower() == target:
            return csv_path
    return None


def _strip_dataset_suffix(name: str) -> str:
    lower_name = name.lower()
    for suffix in ("_clean", *ERROR_SUFFIXES):
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_dataset_paths(dataset: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[str, Path, Path]:
    """
    Resolve a dataset name such as "beers" to:
      Data/beers_error.csv and Data/beers_clean.csv.

    A direct CSV path is also accepted; its matching clean file is inferred from
    the same dataset prefix.
    """
    data_dir = data_dir.resolve()
    dataset_arg = dataset.strip()
    if not dataset_arg:
        raise ValueError("Dataset name cannot be empty.")

    raw_path = Path(dataset_arg)
    if raw_path.suffix.lower() == ".csv" or raw_path.exists():
        dirty_path = raw_path if raw_path.is_absolute() else data_dir / raw_path
        dirty_path = dirty_path.resolve()
        if not dirty_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {dirty_path}")
        dataset_name = _strip_dataset_suffix(dirty_path.stem).lower()
    else:
        dataset_name = _strip_dataset_suffix(dataset_arg).lower()
        dirty_path = None
        for suffix in ERROR_SUFFIXES:
            candidate = _find_file_case_insensitive(data_dir, f"{dataset_name}{suffix}.csv")
            if candidate is not None:
                dirty_path = candidate.resolve()
                break

        if dirty_path is None:
            available = ", ".join(list_available_datasets(data_dir)) or "none"
            raise FileNotFoundError(
                f"Could not find an error/dirty CSV for dataset '{dataset_arg}' in {data_dir}. "
                f"Available datasets: {available}"
            )

    clean_path = _find_file_case_insensitive(data_dir, f"{dataset_name}_clean.csv")
    if clean_path is None:
        raise FileNotFoundError(
            f"Could not find clean ground-truth CSV for dataset '{dataset_name}' in {data_dir}."
        )

    return dataset_name, dirty_path, clean_path.resolve()


def load_csv(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    chunks = pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        header=0,
        encoding="utf-8-sig",
    )
    return pd.concat(chunks, ignore_index=True)


def load_ground_truth_errors(clean_path: Path, dirty_path: Path) -> Set[ErrorCell]:
    """
    Build the ground-truth error set by comparing clean and dirty CSV cells.
    String comparison is used so values such as "N/A" stay literal.
    """
    df_clean = pd.read_csv(clean_path, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    df_dirty = pd.read_csv(dirty_path, keep_default_na=False, dtype=str, encoding="utf-8-sig")

    if df_clean.shape != df_dirty.shape:
        raise ValueError(
            f"Clean and dirty files have different shapes: {df_clean.shape} vs {df_dirty.shape}."
        )
    if list(df_clean.columns) != list(df_dirty.columns):
        raise ValueError("Clean and dirty files have different columns.")

    ground_truth: Set[ErrorCell] = set()
    for row_idx in df_clean.index:
        for col in df_clean.columns:
            if str(df_clean.at[row_idx, col]) != str(df_dirty.at[row_idx, col]):
                ground_truth.add((int(row_idx), str(col)))
    return ground_truth


def save_error_cells(errors: Iterable[ErrorCell], output_path: Path) -> None:
    payload = [[row, col] for row, col in sorted(errors)]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def detected_errors_from_result(result: dict) -> Set[ErrorCell]:
    detected: Set[ErrorCell] = set()
    for entry in result.get("final_errors", []):
        if len(entry) < 2:
            continue
        row_idx, cols = entry[0], entry[1]
        if isinstance(cols, list):
            for col in cols:
                detected.add((int(row_idx), str(col)))
        else:
            detected.add((int(row_idx), str(cols)))
    return detected


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


def preprocess_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibility helper from evaluation_metrics.py style inputs.
    Expected columns: row and column. fieldName is accepted as column.
    """
    df = df.copy()
    if "column" not in df.columns and "fieldName" in df.columns:
        df = df.rename(columns={"fieldName": "column"})
    if "row" not in df.columns or "column" not in df.columns:
        raise ValueError("Error DataFrame must contain 'row' and 'column' columns.")

    df["column"] = df["column"].astype(str).str.replace(";", ",", regex=False).str.split(",")
    df = df.explode("column")
    df["column"] = df["column"].astype(str).str.strip()
    df = df[df["column"].ne("")]
    df["row"] = df["row"].astype(int)
    return df.drop_duplicates(subset=["row", "column"]).reset_index(drop=True)


def calculate_metrics(true_errors: pd.DataFrame, detected_errors: pd.DataFrame, total_records=None) -> dict:
    """
    Compatibility helper matching evaluation_metrics.py.
    total_records is kept for API compatibility.
    """
    del total_records
    true_errors = preprocess_errors(true_errors)
    detected_errors = preprocess_errors(detected_errors)
    true_set = set(zip(true_errors["row"], true_errors["column"]))
    detected_set = set(zip(detected_errors["row"], detected_errors["column"]))
    metrics = compute_metrics(true_set, detected_set)
    return {
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "precision": metrics["precision"],
        "tp": metrics["true_positives"],
        "fp": metrics["false_positives"],
        "fn": metrics["false_negatives"],
    }


def evaluate_detection(clean_path: Path, dirty_path: Path, result: dict, output_dir: Path) -> dict:
    ground_truth = load_ground_truth_errors(clean_path, dirty_path)
    detected = detected_errors_from_result(result)

    save_error_cells(ground_truth, output_dir / "ground_truth_errors.json")
    save_error_cells(detected, output_dir / "detected_errors.json")

    metrics = compute_metrics(ground_truth, detected)
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
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


def compact_report_for_aggregate(report: dict) -> dict:
    return {
        "run_id": report["run_id"],
        "dataset": report["dataset"],
        "model": report["model"],
        "metrics": report["metrics"],
        "token_usage": report["token_usage"],
        "runtime_seconds": report["runtime_seconds"],
        "output_dir": report["output_dir"],
    }


def update_aggregate_report(report: dict, output_root: Path) -> Path:
    aggregate_path = output_root / "all_metrics_tokens.json"
    if aggregate_path.exists():
        with aggregate_path.open("r", encoding="utf-8") as f:
            aggregate = json.load(f)
    else:
        aggregate = []

    aggregate.append(compact_report_for_aggregate(report))
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    return aggregate_path


def save_run_report(report: dict, output_dir: Path, output_root: Path, dataset_dir: Path) -> tuple[Path, Path, Path]:
    run_report_path = output_dir / f"metrics_tokens_{report['run_id']}.json"
    latest_report_path = output_dir / "latest_metrics_tokens.json"
    dataset_latest_report_path = dataset_dir / "latest_metrics_tokens.json"
    latest_run_pointer_path = dataset_dir / "latest_run_path.txt"
    aggregate_path = output_root / "all_metrics_tokens.json"

    report.setdefault("output_files", {})
    report["output_files"]["metrics_tokens"] = str(run_report_path)
    report["output_files"]["latest_metrics_tokens"] = str(latest_report_path)
    report["output_files"]["dataset_latest_metrics_tokens"] = str(dataset_latest_report_path)
    report["output_files"]["latest_run_pointer"] = str(latest_run_pointer_path)
    report["output_files"]["aggregate_metrics_tokens"] = str(aggregate_path)

    for path in (run_report_path, latest_report_path, dataset_latest_report_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    latest_run_pointer_path.write_text(str(output_dir), encoding="utf-8")
    aggregate_path = update_aggregate_report(report, output_root)
    return run_report_path, latest_report_path, aggregate_path


def print_metrics(metrics: dict) -> None:
    print("===== Evaluation Results =====")
    print(f"Ground-truth error cells: {metrics['ground_truth_count']}")
    print(f"Detected error cells:     {metrics['detected_count']}")
    print(f"TP / FP / FN:             {metrics['true_positives']} / {metrics['false_positives']} / {metrics['false_negatives']}")
    print(f"Precision:                {metrics['precision']:.4f}")
    print(f"Recall:                   {metrics['recall']:.4f}")
    print(f"F1-score:                 {metrics['f1_score']:.4f}")


def demo_LAED(
    dataset: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    chunksize: int = 50_000,
    reuse_summary: bool = False,
    summary_source: Optional[Path] = None,
    summary_only: bool = False,
) -> dict:
    """
    Run the complete LAED workflow on one specified dataset.
    """
    configure_qwen()
    dataset_name, dirty_path, clean_path = resolve_dataset_paths(dataset, data_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = output_root.resolve()
    dataset_dir = output_root / dataset_name
    output_dir = dataset_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    reused_existing_summary = False
    summary_source_path = summary_source.resolve() if summary_source else None

    print("=" * 80)
    print("LAED: table data error detection demo")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Dirty data: {dirty_path}")
    print(f"Ground-truth clean data (evaluation only): {clean_path}")
    print(f"API model: {shared_qwen_client.model}")
    print(f"API max tokens: {shared_qwen_client.max_tokens}")
    print(f"API temperature: {shared_qwen_client.temperature}")
    print(f"Output directory: {output_dir}")

    reset_shared_usage()
    start_time = time.perf_counter()

    with working_directory(output_dir):
        print("\n1. Loading data")
        df = load_csv(dirty_path, chunksize=chunksize)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(3).to_string())

        print("\n2. Generating data summary")
        summary_path = output_dir / "data_summary.json"
        runtime_summary_path = output_dir / "data_summary_runtime_profile.json"
        if summary_source_path is not None:
            if not summary_source_path.exists():
                raise FileNotFoundError(f"Summary source not found: {summary_source_path}")
            with summary_source_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            runtime_source_path = summary_source_path.with_name("data_summary_runtime_profile.json")
            if runtime_source_path.exists():
                with runtime_source_path.open("r", encoding="utf-8") as f:
                    runtime_summary = json.load(f)
                print(f"Loaded runtime summary profile source: {runtime_source_path}")
            else:
                runtime_summary = summary
            with runtime_summary_path.open("w", encoding="utf-8") as f:
                json.dump(runtime_summary, f, ensure_ascii=False, indent=2)
            reused_existing_summary = True
            print(f"Reused LLM-generated summary source: {summary_source_path}")
        elif reuse_summary and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if runtime_summary_path.exists():
                with runtime_summary_path.open("r", encoding="utf-8") as f:
                    runtime_summary = json.load(f)
            else:
                runtime_summary = summary
            reused_existing_summary = True
            print(f"Reused existing LLM-generated summary: {summary_path}")
        else:
            summarizer = Summarizer()
            runtime_summary = summarizer.summarize(df, file_name=dirty_path.name)
            summary = summarizer.compact_for_data_summary(runtime_summary)
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            with runtime_summary_path.open("w", encoding="utf-8") as f:
                json.dump(runtime_summary, f, ensure_ascii=False, indent=2)
            print("Summary generated.")
        print(f"Dataset description: {summary.get('dataset_description', '')}")

        if summary_only:
            runtime_seconds = round(time.perf_counter() - start_time, 2)
            token_usage = get_shared_usage()
            report = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": shared_qwen_client.model,
                "max_tokens": shared_qwen_client.max_tokens,
                "temperature": shared_qwen_client.temperature,
                "data": {
                    "dirty_path": str(dirty_path),
                    "shape": list(df.shape),
                    "columns": df.columns.tolist(),
                },
                "summary": {
                    "reused_existing_summary": reused_existing_summary,
                    "summary_path": str(summary_path),
                    "runtime_summary_profile_path": str(runtime_summary_path),
                    "summary_source_path": str(summary_source_path) if summary_source_path else "",
                    "source_is_llm_generated": bool(summary_source_path) or not reused_existing_summary,
                    "generation_mode": "reused_llm_generated_summary" if reused_existing_summary else "fresh_llm_generated",
                    "summary_only": True,
                    "public_summary_is_compact": True,
                },
                "initial_screening": {
                    "skipped": True,
                    "reason": "summary-only mode",
                },
                "detection": {
                    "skipped": True,
                    "reason": "summary-only mode",
                },
                "metrics": {},
                "ground_truth_usage": {
                    "clean_path": str(clean_path),
                    "used_only_for_evaluation": False,
                    "not_passed_to_modules": [
                        "summarizer.py",
                        "Initial_Screening_all.py",
                        "Error_Detection_update.py",
                    ],
                    "note": "Summary-only mode resolves the clean path but does not read the clean CSV.",
                },
                "token_usage": {
                    "input_tokens": token_usage.get("input_tokens", 0),
                    "output_tokens": token_usage.get("output_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                    "api_call_count": token_usage.get("api_call_count", 0),
                },
                "runtime_seconds": runtime_seconds,
                "output_dir": str(output_dir),
                "output_files": {
                    "summary": str(summary_path),
                    "runtime_summary_profile": str(runtime_summary_path),
                },
            }
            run_report_path, _latest_report_path, aggregate_path = save_run_report(report, output_dir, output_root, dataset_dir)
            print("\nSummary-only mode: skipped initial screening, detection, and evaluation.")
            print(f"Qwen input tokens:  {report['token_usage']['input_tokens']}")
            print(f"Qwen output tokens: {report['token_usage']['output_tokens']}")
            print(f"Qwen total tokens:  {report['token_usage']['total_tokens']}")
            print(f"API calls:          {report['token_usage']['api_call_count']}")
            print(f"Runtime:            {runtime_seconds}s")
            print(f"Saved report:       {run_report_path}")
            print(f"Updated aggregate:  {aggregate_path}")
            print("=" * 80)
            return report

        print("\n3. Initial screening")
        df_processed, correct_cells, pct = initial_screening(df.copy(), runtime_summary)
        correct_count = int(correct_cells.values.sum())
        suspicious_count = int(df_processed.size - correct_count)
        print(f"Correct cells: {correct_count}/{df_processed.size} ({pct:.2f}%)")
        print(f"Suspicious cells: {suspicious_count}")

        print("\n4. Error detection")
        detector = DetectionExplorer(experience_file=str(output_dir / "ErrorDetection_Experience_file.json"))
        result = detector.generate(
            runtime_summary,
            df.copy(),
            screening_result=(df_processed, correct_cells, pct),
        )
        print("Error detection completed.")

        detailed_errors = result.get("errors", [])
        final_errors = result.get("final_errors", [])
        error_types_count, field_errors = summarize_error_distribution(detailed_errors)

        print("\n5. Detection summary")
        print(f"Final error cells: {len(final_errors)}")
        print("Error types:")
        for error_type, count in sorted(error_types_count.items()):
            print(f"  {error_type}: {count}")
        print("Top fields:")
        for field, count in sorted(field_errors.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"  {field}: {count}")

        print("\n6. Evaluation")
        print("Clean data is loaded only here to compute metrics; it is not used by summary, screening, or detection.")
        metrics = evaluate_detection(clean_path, dirty_path, result, output_dir)
        print_metrics(metrics)

    runtime_seconds = round(time.perf_counter() - start_time, 2)
    token_usage = get_shared_usage()

    report = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": shared_qwen_client.model,
        "max_tokens": shared_qwen_client.max_tokens,
        "temperature": shared_qwen_client.temperature,
        "data": {
            "dirty_path": str(dirty_path),
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
        },
        "ground_truth_usage": {
            "clean_path": str(clean_path),
            "used_only_for_evaluation": True,
            "not_passed_to_modules": [
                "summarizer.py",
                "Initial_Screening_all.py",
                "Error_Detection_update.py",
            ],
            "note": "The clean CSV is read only after detection completes, inside evaluate_detection(), to compute ground-truth metrics.",
        },
        "initial_screening": {
            "correct_cells": correct_count,
            "total_cells": int(df_processed.size),
            "correct_pct": round(float(pct), 4),
            "suspicious_cells": suspicious_count,
        },
        "summary": {
            "reused_existing_summary": reused_existing_summary,
            "summary_path": str(output_dir / "data_summary.json"),
            "runtime_summary_profile_path": str(output_dir / "data_summary_runtime_profile.json"),
            "summary_source_path": str(summary_source_path) if summary_source_path else "",
            "source_is_llm_generated": bool(summary_source_path) or not reused_existing_summary,
            "generation_mode": "reused_llm_generated_summary" if reused_existing_summary else "fresh_llm_generated",
            "public_summary_is_compact": True,
            "final_experiment_policy": (
                "Final three-round results must be generated after code is frozen; "
                "if any code changes, rerun all five datasets for all three rounds."
            ),
        },
        "detection": {
            "detailed_error_count": len(detailed_errors),
            "final_error_count": len(final_errors),
            "error_types_count": error_types_count,
            "field_errors": field_errors,
            "llm_confirmation_policy": result.get("llm_confirmation_policy", {}),
            "relationship_candidate_count": len(result.get("relationship_candidates", [])),
        },
        "metrics": metrics,
        "token_usage": {
            "input_tokens": token_usage.get("input_tokens", 0),
            "output_tokens": token_usage.get("output_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
            "api_call_count": token_usage.get("api_call_count", 0),
        },
        "module_alignment": {
            "data_summary_generation": {
                "module": "summarizer.py",
                "llm_calls": [
                    "Summarizer.enrich",
                    "Summarizer._refine_relationships_with_evidence",
                    "Summarizer.extract_format_rules",
                    "Summarizer.validate_format_rules (regex repair when needed)",
                    "Summarizer._review_format_rule_with_llm (generic rule review when needed)",
                    "Summarizer.generate_relationship_validator",
                ],
                "evidence_file": str(output_dir / "data_summary.json"),
                "runtime_profile_file": str(output_dir / "data_summary_runtime_profile.json"),
                "note": "data_summary.json is a compact readable LLM summary. Runtime profile evidence is generated inside summarizer.py and saved separately for module execution; no dataset-specific patches are used.",
            },
            "initial_screening": {
                "module": "Initial_Screening_all.py",
                "llm_calls": [],
                "uses": [
                    "summary.format_rules",
                    "summary.relationship_validator_code",
                    "summary field profile evidence",
                ],
                "note": "This stage does not call the LLM; it executes summary-derived regex, relationship code, and generic profile-evidence candidate retention to reduce the LLM search space without turning candidates into final errors.",
            },
            "error_detection_rule_generation_and_execution": {
                "module": "Error_Detection_update.py",
                "llm_calls": [
                    "DetectionExplorer.confirm_profile_warning_errors_with_llm",
                    "DetectionExplorer.detect_column_errors",
                    "DetectionExplorer._detect_column_error_batches",
                    "DetectionExplorer.confirm_relationship_errors_with_llm",
                ],
                "note": "All initial-screening suspicious cells enter the LLM judgment path. Duplicate suspicious values may be represented once and expanded only from an LLM decision; relationship checks produce candidates that must be confirmed by the LLM before final_errors.",
            },
        },
        "quality_dependency_statement": {
            "primary_success_driver": "The dataset-level performance should be attributed to the accuracy of the LLM-generated data_summary.json, including field semantics, format_rules, profile metadata, and relationship_validator_code.",
            "not_used": "No hand-written dataset-specific fixes for beers, flights, hospital, rayyan, or movies are used by LAED_Demo.py, summarizer.py, Initial_Screening_all.py, or Error_Detection_update.py.",
            "allowed_generic_logic": [
                "missing-placeholder detection",
                "format and regex consistency checks",
                "distribution/profile warnings",
                "field relationship consistency from the summary",
                "LLM-confirmed per-column error diagnosis",
                "LLM-confirmed relationship-candidate diagnosis",
            ],
            "summary_reuse_note": (
                "Final reported experiments should use fresh LLM summary generation for each run. "
                "summary_source_path is kept only for debug/reproduction comparisons; when set, "
                "initial screening and error detection still execute normally, and error detection still calls the LLM on suspicious cells."
            ),
        },
        "runtime_seconds": runtime_seconds,
        "output_dir": str(output_dir),
        "output_files": {
            "summary": str(output_dir / "data_summary.json"),
            "runtime_summary_profile": str(output_dir / "data_summary_runtime_profile.json"),
            "detailed_errors": str(output_dir / "detailed_errors.json"),
            "errors_with_context": str(output_dir / "errors_with_context.json"),
            "ground_truth_errors": str(output_dir / "ground_truth_errors.json"),
            "detected_errors": str(output_dir / "detected_errors.json"),
            "evaluation_metrics": str(output_dir / "evaluation_metrics.json"),
        },
    }

    run_report_path, latest_report_path, aggregate_path = save_run_report(report, output_dir, output_root, dataset_dir)

    print("\n7. Tokens and saved report")
    print(f"Qwen input tokens:  {report['token_usage']['input_tokens']}")
    print(f"Qwen output tokens: {report['token_usage']['output_tokens']}")
    print(f"Qwen total tokens:  {report['token_usage']['total_tokens']}")
    print(f"API calls:          {report['token_usage']['api_call_count']}")
    print(f"Runtime:            {runtime_seconds}s")
    print(f"Saved report:       {run_report_path}")
    print(f"Updated aggregate:  {aggregate_path}")
    print("=" * 80)

    return report


def show_detailed_relationship_analysis(dataset: str, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    dataset_name, dirty_path, _ = resolve_dataset_paths(dataset, data_dir)
    print(f"\nDetailed relationship analysis: {dataset_name}")
    df = load_csv(dirty_path)
    summary = Summarizer().summarize(df, file_name=dirty_path.name)
    relationships = summary.get("field_relationships", {})

    for rel_type, rel_data in relationships.items():
        if not rel_data:
            continue
        print(f"\n{rel_type.upper()}:")
        if isinstance(rel_data, dict):
            for key, value in rel_data.items():
                print(f"  {key} -> {value}")
        elif isinstance(rel_data, list):
            for item in rel_data:
                print(f"  {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LAED on a specified dataset and save metrics plus Qwen token usage."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example: beers, flights, hospital, movies, rayyan. A CSV path is also accepted.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing *_error.csv and *_clean.csv files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory used for run outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="CSV read chunksize.",
    )
    parser.add_argument(
        "--show-relationships",
        action="store_true",
        help="After the main run, print a fresh detailed relationship analysis for the same dataset.",
    )
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="Reuse output_dir/data_summary.json if it already exists. The file must be an LLM-generated LAED summary.",
    )
    parser.add_argument(
        "--summary-source",
        type=Path,
        default=None,
        help="Path to an existing LLM-generated data_summary.json to copy into this run before screening and detection.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate/copy data_summary.json and stop before initial screening, detection, and evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = demo_LAED(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_root=args.output_root,
        chunksize=args.chunksize,
        reuse_summary=args.reuse_summary,
        summary_source=args.summary_source,
        summary_only=args.summary_only,
    )

    if args.show_relationships:
        show_detailed_relationship_analysis(args.dataset, args.data_dir)

    print(
        f"\nDone. Metrics and token usage for '{report['dataset']}' are in "
        f"{report['output_files']['metrics_tokens']}"
    )


if __name__ == "__main__":
    main()
