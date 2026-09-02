from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Run_Results"
DATASETS = ["beers", "flights", "hospital", "rayyan", "movies"]
METRIC_KEYS = [
    "precision",
    "recall",
    "f1_score",
    "ground_truth_count",
    "detected_count",
    "true_positives",
    "false_positives",
    "false_negatives",
]


class Tee:
    """Write the batch log to the terminal and to the invocation log."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def reconfigure(self, **kwargs):
        for stream in self.streams:
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure:
                reconfigure(**kwargs)


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_dataset(dataset: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict:
    """Run one independent LAED_Demo.py process and collect its result."""
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = logs_dir / f"{dataset}_{run_id}.log"
    stdout_path = logs_dir / f"{dataset}_{run_id}.stdout.log"
    stderr_path = logs_dir / f"{dataset}_{run_id}.stderr.log"

    cmd = [
        sys.executable,
        "-u",
        str(CURRENT_DIR / "LAED_Demo.py"),
        "--dataset",
        dataset,
        "--data-dir",
        str(DEFAULT_DATA_DIR),
        "--output-root",
        str(output_root),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("LAED_RELATIONSHIP_REQUEST_TIMEOUT", "45")
    env.setdefault("LAED_RELATIONSHIP_MAX_RETRIES", "0")
    env.setdefault("LAED_SINGLE_COLUMN_DETECTION_RETRIES", "1")
    env.setdefault("LAED_FORMAT_RULE_REQUEST_TIMEOUT", "60")
    env.setdefault("LAED_FORMAT_RULE_REPAIR_REQUEST_TIMEOUT", "30")
    env.setdefault("LAED_FORMAT_RULE_REPAIR_ROUNDS", "0")
    env.setdefault("LAED_FORMAT_RULE_REVIEW_ROUNDS", "0")

    started = time.perf_counter()
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
        merged.write("=" * 80 + "\n")
        merged.write(stdout_path.read_text(encoding="utf-8", errors="replace"))
        if stderr_path.exists() and stderr_path.stat().st_size:
            merged.write("\n" + "=" * 80 + "\nSTDERR\n" + "=" * 80 + "\n")
            merged.write(stderr_path.read_text(encoding="utf-8", errors="replace"))

    runtime_seconds = round(time.perf_counter() - started, 2)
    latest_metrics = output_root / dataset / "latest_metrics_tokens.json"
    run_report_path = ""
    run_output_dir = ""
    metrics = {}
    report = {}
    if proc.returncode == 0 and latest_metrics.exists():
        report = load_report(latest_metrics)
        metrics = report.get("metrics", {}) or {}
        run_report_path = str((report.get("output_files") or {}).get("metrics_tokens", ""))
        run_output_dir = str(report.get("output_dir", ""))

    return {
        "dataset": dataset,
        "command": cmd,
        "returncode": proc.returncode,
        "runtime_seconds": runtime_seconds,
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "latest_metrics": str(latest_metrics),
        "run_report_path": run_report_path,
        "run_output_dir": run_output_dir,
        "metrics": metrics,
        "model": report.get("model"),
    }


def compact_report(report: dict, source: str, command_log: str = "") -> dict:
    """Keep only reproducibility-relevant fields from one LAED run."""
    return {
        "source": source,
        "dataset": report.get("dataset"),
        "run_id": report.get("run_id"),
        "model": report.get("model"),
        "max_tokens": report.get("max_tokens"),
        "temperature": report.get("temperature"),
        "metrics": report.get("metrics", {}),
        "runtime_seconds": report.get("runtime_seconds"),
        "token_usage": report.get("token_usage", {}),
        "output_dir": report.get("output_dir"),
        "output_files": report.get("output_files", {}),
        "command_log": command_log,
    }


def average_numeric(values: list) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def metric_ranges(reports: list[dict]) -> dict:
    ranges = {}
    for key in ("precision", "recall", "f1_score"):
        values = [
            float((report.get("metrics") or {}).get(key))
            for report in reports
            if isinstance((report.get("metrics") or {}).get(key), (int, float))
        ]
        ranges[key] = {
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
            "spread": round(max(values) - min(values), 4) if values else None,
        }
    return ranges


def build_three_run_summary(datasets: list[str], batch_runs: list[dict]) -> dict:
    """Aggregate only successful reports from the current invocation."""
    by_dataset: dict[str, list[dict]] = {dataset: [] for dataset in datasets}
    for batch_item in batch_runs:
        if batch_item.get("returncode") != 0:
            continue
        report_path_text = str(batch_item.get("run_report_path", ""))
        if not report_path_text:
            continue
        report_path = Path(report_path_text)
        if report_path.exists():
            report = load_report(report_path)
            dataset = str(batch_item.get("dataset", ""))
            by_dataset.setdefault(dataset, []).append(
                compact_report(report, batch_item.get("source", ""), batch_item.get("log_path", ""))
            )

    dataset_summaries = {}
    for dataset, reports in by_dataset.items():
        metric_averages = {
            key: average_numeric([(report.get("metrics") or {}).get(key) for report in reports])
            for key in METRIC_KEYS
        }
        dataset_summaries[dataset] = {
            "run_count": len(reports),
            "runs": reports,
            "average_metrics": metric_averages,
            "metric_ranges_across_runs": metric_ranges(reports),
            "average_runtime_seconds": average_numeric([
                report.get("runtime_seconds") for report in reports
            ]),
            "average_token_usage": {
                key: average_numeric([
                    (report.get("token_usage") or {}).get(key) for report in reports
                ])
                for key in ("input_tokens", "output_tokens", "total_tokens", "api_call_count")
            },
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "run_count": len(batch_runs),
        "successful_run_count": sum(1 for item in batch_runs if item.get("returncode") == 0),
        "dataset_summaries": dataset_summaries,
    }


def write_average_markdown(summary: dict, md_path: Path) -> None:
    lines = [
        "# LAED Run Summary",
        "",
        f"Created at: {summary.get('created_at', '')}",
        "",
        "| Dataset | Successful runs | Avg Precision | Avg Recall | Avg F1 | Avg Runtime (s) | Latest Output Dir | Latest Command Log |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset, item in summary.get("dataset_summaries", {}).items():
        runs = item.get("runs", [])
        latest = runs[-1] if runs else {}
        metrics = item.get("average_metrics", {})
        lines.append(
            "| {dataset} | {runs} | {precision} | {recall} | {f1} | {runtime} | {out} | {log} |".format(
                dataset=dataset,
                runs=item.get("run_count", 0),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1=metrics.get("f1_score"),
                runtime=item.get("average_runtime_seconds"),
                out=latest.get("output_dir", ""),
                log=latest.get("command_log", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_average_csv(summary: dict, csv_path: Path) -> None:
    fieldnames = [
        "dataset",
        "successful_run_count",
        "avg_precision",
        "avg_recall",
        "avg_f1_score",
        "avg_runtime_seconds",
        "avg_api_call_count",
        "latest_output_dir",
        "latest_command_log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, item in summary.get("dataset_summaries", {}).items():
            runs = item.get("runs", [])
            latest = runs[-1] if runs else {}
            metrics = item.get("average_metrics", {})
            token_usage = item.get("average_token_usage", {})
            writer.writerow({
                "dataset": dataset,
                "successful_run_count": item.get("run_count", 0),
                "avg_precision": metrics.get("precision"),
                "avg_recall": metrics.get("recall"),
                "avg_f1_score": metrics.get("f1_score"),
                "avg_runtime_seconds": item.get("average_runtime_seconds"),
                "avg_api_call_count": token_usage.get("api_call_count"),
                "latest_output_dir": latest.get("output_dir", ""),
                "latest_command_log": latest.get("command_log", ""),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LAED_Demo.py independently for each dataset and save actual results."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DATASETS,
        help=f"Datasets to run. Default: {' '.join(DATASETS)}",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of independent rounds for each dataset. Default: 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = DEFAULT_OUTPUT_ROOT
    command_logs_dir = output_root / "command_logs"
    command_logs_dir.mkdir(parents=True, exist_ok=True)
    invocation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    invocation_log_path = command_logs_dir / f"run_laed_datasets_{invocation_id}.log"
    with invocation_log_path.open("w", encoding="utf-8") as invocation_log:
        invocation_log.write("Command: " + " ".join([sys.executable, *sys.argv]) + "\n")
        invocation_log.write("=" * 80 + "\n")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, invocation_log)
        sys.stderr = Tee(original_stderr, invocation_log)
        failed = False
        try:
            summary = []
            rounds = max(1, int(args.rounds))
            for round_no in range(1, rounds + 1):
                for dataset in args.datasets:
                    result = run_dataset(dataset, output_root)
                    result["source"] = f"round_{round_no}"
                    summary.append(result)
                    print(
                        f"round={round_no}/{rounds} {dataset}: returncode={result['returncode']} "
                        f"runtime={result['runtime_seconds']}s metrics={result['metrics']} "
                        f"log={result['log_path']}"
                    )

            summary_path = output_root / f"batch_run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            latest_path = output_root / "latest_batch_run_summary.json"
            for path in (summary_path, latest_path):
                with path.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"Saved batch summary: {summary_path}")

            aggregate = build_three_run_summary(list(args.datasets), summary)
            average_json_path = output_root / f"three_run_average_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            latest_average_json_path = output_root / "latest_three_run_average_metrics.json"
            average_csv_path = output_root / "latest_three_run_average_metrics.csv"
            average_md_path = output_root / "latest_three_run_average_metrics.md"
            aggregate["batch_invocation_log"] = str(invocation_log_path)
            for path in (average_json_path, latest_average_json_path):
                with path.open("w", encoding="utf-8") as f:
                    json.dump(aggregate, f, ensure_ascii=False, indent=2)
            write_average_csv(aggregate, average_csv_path)
            write_average_markdown(aggregate, average_md_path)
            print(f"Saved average summary: {average_json_path}")
            print(f"Saved average CSV: {average_csv_path}")
            print(f"Saved average Markdown: {average_md_path}")
            invocation_log.write("\n" + "=" * 80 + "\nBATCH SUMMARY JSON\n" + "=" * 80 + "\n")
            invocation_log.write(json.dumps(summary, ensure_ascii=False, indent=2))
            invocation_log.write("\nAverage JSON: " + str(latest_average_json_path) + "\n")
            invocation_log.write("Average CSV: " + str(average_csv_path) + "\n")
            invocation_log.write("Average Markdown: " + str(average_md_path) + "\n")
        except Exception:
            failed = True
            print("\n" + "=" * 80, file=sys.stderr)
            print("BATCH FAILED", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    print(f"Saved batch invocation log: {invocation_log_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
