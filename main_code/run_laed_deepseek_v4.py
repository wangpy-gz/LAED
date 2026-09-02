from __future__ import annotations

import csv
import json
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "Run_Results_DeepSeekV4"
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


for path in (str(CURRENT_DIR), str(WORKSPACE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def configure_deepseek_environment() -> tuple[str, str, str]:
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_API_BASE") or "https://api.deepseek.com"
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY before running this script.")
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return model, api_key, base_url


DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE = configure_deepseek_environment()

import LAED_Demo as laed  # noqa: E402
from openai import OpenAI  # noqa: E402
from pythonProject1.API_invocation.qwen_gen import shared_qwen_client  # noqa: E402


def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
        completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
        total_tokens = getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def _deepseek_process_worker(result_conn, api_key: str, base_url: str, call_kwargs: dict, timeout_seconds: int):
    try:
        endpoint_base = base_url.rstrip("/")
        if endpoint_base.endswith("/v1"):
            endpoint_base = endpoint_base[:-3]
        payload = {
            "model": call_kwargs["model"],
            "messages": call_kwargs["messages"],
            "max_tokens": call_kwargs["max_tokens"],
            "temperature": call_kwargs["temperature"],
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        response = requests.post(
            f"{endpoint_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(15, max(1, int(timeout_seconds))),
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek response does not contain choices: {data}")
        message = choices[0].get("message") or {}
        result_conn.send(
            (
                "ok",
                {
                    "content": message.get("content") or "",
                    "usage": _usage_to_dict(data.get("usage")),
                },
            )
        )
    except BaseException as exc:
        try:
            result_conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        try:
            result_conn.close()
        except BaseException:
            pass


def configure_deepseek_client() -> None:
    shared_qwen_client.api_key = DEEPSEEK_API_KEY
    shared_qwen_client.base_url = DEEPSEEK_API_BASE
    shared_qwen_client.openai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
    shared_qwen_client.model = DEEPSEEK_MODEL
    shared_qwen_client.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))
    shared_qwen_client.temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.0"))
    shared_qwen_client._uses_openai_chat = lambda: True
    laed.QWEN_MODEL = DEEPSEEK_MODEL
    laed.QWEN_MAX_TOKENS = shared_qwen_client.max_tokens
    laed.QWEN_TEMPERATURE = shared_qwen_client.temperature

    def deepseek_generation_call_with_timeout(timeout_seconds: int, **call_kwargs):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        process = multiprocessing.Process(
            target=_deepseek_process_worker,
            args=(child_conn, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, call_kwargs, timeout_seconds),
            daemon=True,
        )
        process.start()
        child_conn.close()
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        try:
            while time.monotonic() < deadline:
                if parent_conn.poll(0.2):
                    status, payload = parent_conn.recv()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    if status == "error":
                        raise RuntimeError(payload)
                    return payload
                if not process.is_alive():
                    process.join(timeout=1)
                    if parent_conn.poll(1):
                        status, payload = parent_conn.recv()
                        if status == "error":
                            raise RuntimeError(payload)
                        return payload
                    raise RuntimeError(
                        f"DeepSeek worker exited with code {process.exitcode} without a response"
                    )

            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
            raise TimeoutError(f"DeepSeek request exceeded {timeout_seconds}s")
        finally:
            try:
                parent_conn.close()
            except BaseException:
                pass

    shared_qwen_client._generation_call_with_timeout = deepseek_generation_call_with_timeout


def preflight(output_root: Path) -> Path:
    configure_deepseek_client()
    logs_dir = output_root / "command_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"deepseek_v4_preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": shared_qwen_client.model,
        "base_url": DEEPSEEK_API_BASE,
    }
    try:
        response = shared_qwen_client.send_message(
            [
                {"role": "system", "content": "Return compact JSON only."},
                {"role": "user", "content": "Return {\"ok\": true}."},
            ],
            max_tokens=64,
            retries=1,
            request_timeout=int(os.getenv("DEEPSEEK_PREFLIGHT_TIMEOUT", "300")),
        )
        payload["status"] = "ok"
        payload["response"] = response
        payload["usage"] = shared_qwen_client.get_usage_stats()
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        shared_qwen_client.reset_usage()
    return path


def metric_average(values: list[dict], key: str) -> float | None:
    numeric = [
        float(item["metrics"][key])
        for item in values
        if isinstance(item.get("metrics", {}).get(key), (int, float))
    ]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def compact_run(report: dict, round_index: int) -> dict:
    metrics = report.get("metrics", {})
    return {
        "round": round_index,
        "run_id": report.get("run_id", ""),
        "dataset": report.get("dataset", ""),
        "model": report.get("model", ""),
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "runtime_seconds": report.get("runtime_seconds", 0),
        "token_usage": report.get("token_usage", {}),
        "output_dir": report.get("output_dir", ""),
    }


def write_summary(output_root: Path, run_records: list[dict], failures: list[dict]) -> dict:
    created = datetime.now().strftime("%Y%m%d_%H%M%S")
    by_dataset: dict[str, dict] = {}
    for dataset in DATASETS:
        runs = [record for record in run_records if record["dataset"] == dataset]
        by_dataset[dataset] = {
            "run_count": len(runs),
            "average_metrics": {key: metric_average(runs, key) for key in ("precision", "recall", "f1_score")},
            "runs": runs,
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": DEEPSEEK_MODEL,
        "base_url": DEEPSEEK_API_BASE,
        "rounds_requested": 3,
        "datasets": by_dataset,
        "failures": failures,
    }
    json_path = output_root / f"deepseek_v4_three_run_metrics_{created}.json"
    latest_json_path = output_root / "latest_deepseek_v4_three_run_metrics.json"
    csv_path = output_root / f"deepseek_v4_three_run_metrics_{created}.csv"
    latest_csv_path = output_root / "latest_deepseek_v4_three_run_metrics.csv"
    md_path = output_root / f"deepseek_v4_three_run_metrics_{created}.md"
    latest_md_path = output_root / "latest_deepseek_v4_three_run_metrics.md"

    for path in (json_path, latest_json_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "round", "precision", "recall", "f1_score", "run_id", "output_dir"])
        for dataset in DATASETS:
            for record in by_dataset[dataset]["runs"]:
                metrics = record["metrics"]
                writer.writerow([
                    dataset,
                    record["round"],
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1_score"),
                    record.get("run_id", ""),
                    record.get("output_dir", ""),
                ])
            avg = by_dataset[dataset]["average_metrics"]
            writer.writerow([dataset, "average", avg["precision"], avg["recall"], avg["f1_score"], "", ""])
    latest_csv_path.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    lines = [
        f"# DeepSeek-V4 Error Detection Results",
        "",
        f"- Model: `{DEEPSEEK_MODEL}`",
        f"- Created: `{summary['created_at']}`",
        f"- Output root: `{output_root}`",
        "",
        "| Dataset | Round | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for record in by_dataset[dataset]["runs"]:
            metrics = record["metrics"]
            lines.append(
                f"| {dataset} | {record['round']} | "
                f"{metrics.get('precision')} | {metrics.get('recall')} | {metrics.get('f1_score')} |"
            )
        avg = by_dataset[dataset]["average_metrics"]
        lines.append(
            f"| {dataset} | average | {avg['precision']} | {avg['recall']} | {avg['f1_score']} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure['dataset']} round {failure['round']}: {failure['error']}")
    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md_path.write_text(md_text, encoding="utf-8")

    summary["summary_files"] = {
        "json": str(json_path),
        "latest_json": str(latest_json_path),
        "csv": str(csv_path),
        "latest_csv": str(latest_csv_path),
        "markdown": str(md_path),
        "latest_markdown": str(latest_md_path),
    }
    for path in (json_path, latest_json_path):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    output_root = Path(os.getenv("DEEPSEEK_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = preflight(output_root)
    print(f"DeepSeek preflight saved: {preflight_path}", flush=True)

    run_records: list[dict] = []
    failures: list[dict] = []
    max_attempts = int(os.getenv("DEEPSEEK_RUN_ATTEMPTS", "2"))
    for round_index in range(1, 4):
        for dataset in DATASETS:
            print(f"\n===== DeepSeek-V4 round {round_index}/3 dataset {dataset} =====", flush=True)
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                configure_deepseek_client()
                try:
                    report = laed.demo_LAED(
                        dataset=dataset,
                        data_dir=DEFAULT_DATA_DIR,
                        output_root=output_root,
                        reuse_summary=False,
                    )
                    run_records.append(compact_run(report, round_index))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[Warning] {dataset} round {round_index} attempt {attempt} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                    if attempt < max_attempts:
                        time.sleep(min(60, 5 * attempt))
            if last_exc is not None:
                failures.append({
                    "round": round_index,
                    "dataset": dataset,
                    "error": f"{type(last_exc).__name__}: {last_exc}",
                })
            write_summary(output_root, run_records, failures)

    summary = write_summary(output_root, run_records, failures)
    print("\n===== DeepSeek-V4 three-run summary =====", flush=True)
    for dataset, payload in summary["datasets"].items():
        avg = payload["average_metrics"]
        print(
            f"{dataset}: precision={avg['precision']} recall={avg['recall']} f1={avg['f1_score']} "
            f"runs={payload['run_count']}",
            flush=True,
        )
    print(json.dumps(summary.get("summary_files", {}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
