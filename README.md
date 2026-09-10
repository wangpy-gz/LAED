# An LLM Agent Framework for Token-Efficient Tabular Data Error Detection

This repository is the reproducibility package for the manuscript *An LLM
Agent Framework for Token-Efficient Tabular Data Error Detection*
submitted to *PeerJ*.  It contains the LAED implementation, five paired
dirty/clean tabular datasets, reproducible experiment entry points, and saved
run artifacts.

LAED detects erroneous **cells** in a dirty table.  It is an error-detection
framework; it does not automatically repair values.

## Contents

- [Workflow](#workflow)
- [Repository and code structure](#repository-and-code-structure)
- [Datasets](#datasets)
- [Requirements and environment](#requirements-and-environment)
- [API configuration](#api-configuration)
- [Run the experiments](#run-the-experiments)
- [Outputs and verification](#outputs-and-verification)
- [Archived experiment results and logs](#archived-experiment-results-and-logs)
- [Reproducibility notes](#reproducibility-notes)
## Workflow

For one dirty table, `LAED_Demo.py` executes the following three-stage
workflow and then evaluates the detection result.

```text
dirty CSV
  |
  +--> 1. Data summary: Summarizer.summarize()
  |       - local column profiles and observed-value evidence
  |       - LLM field semantics and dataset description
  |       - field-relationship 
  |
  +--> 2. Initial screening: initial_screening()
  |       - format-rule checks
  |       - relationship-validator candidates
  |       - output: correct-cell mask and suspicious-cell scope
  |
  +--> 3. LLM error detection: DetectionExplorer.generate()
  |       - organize suspicious values by column
  |       - batch LLM diagnosis of candidates
  |       - output: detailed errors and final cell coordinates
  |
  +--> Evaluation: evaluate_detection()
          - compare aligned dirty and clean cells to derive ground truth
          - calculate TP, FP, FN, precision, recall, and F1
```

The clean CSV is deliberately **not** passed to the summary, initial-screening,
or detection modules.  It is read only after prediction, inside
`evaluate_detection()`, to calculate ground-truth metrics.

## Repository and code structure

The following is the core execution path for reproduction.  The source file
`evaluation_metrics.py` is not part of this path; the active evaluation
implementation is in `LAED_Demo.py`.

```text
pythonProject1/
├─ Data/
│  ├─ beers_{error,clean}.csv
│  ├─ flights_{error,clean}.csv
│  ├─ hospital_{error,clean}.csv
│  ├─ movies_{error,clean}.csv
│  └─ rayyan_{error,clean}.csv
├─ API_invocation/
│  └─ qwen_gen.py                         # shared API client and token accounting
├─ main_code/
│  ├─ LAED_Demo.py                        # one complete LAED run
│  ├─ run_laed_datasets.py                # generic multi-dataset/multi-round launcher
│  ├─ run_laed_deepseek_v4.py             # DeepSeek full-workflow launcher
│  ├─ run_laed_strict_ablations.py        # strict-ablation launcher
│  ├─ ablation_strict_no_summary.py       # wrapper for no_summary
│  ├─ ablation_strict_no_screening.py     # wrapper for no_screening
│  ├─ ablation_strict_no_summary_screening.py
│  ├─ ablation_core.py                    # strict-ablation implementations
│  ├─ summarizer.py                       # Summarizer
│  ├─ Initial_Screening_all.py            # initial_screening
│  ├─ Error_Detection_update.py           # DetectionExplorer
│  └─ utils.py                            # shared utility functions
├─ Run_Results/                           # default generic-main-experiment output root
├─ README.md
└─ requirements.txt
```

### Main experiment entry points

| File | Purpose | Model configuration behaviour |
|---|---|---|
| `main_code/LAED_Demo.py` | Runs the complete workflow on one dataset. | Calls `configure_qwen()` and uses the source constants: `qwen2.5-72b-instruct`, temperature `0.0`, maximum generated length `8192`. |
| `main_code/run_laed_datasets.py` | Starts one independent `LAED_Demo.py` subprocess per dataset and round, then aggregates actual run reports. | Inherits the generic Qwen configuration through each `LAED_Demo.py` subprocess. |
| `main_code/run_laed_deepseek_v4.py` | Performs a DeepSeek preflight and runs the complete LAED workflow for all five datasets and three rounds. | Reconfigures the shared client and `LAED_Demo` constants so that DeepSeek is used in every LLM-backed stage of that runner. |
| `main_code/run_laed_strict_ablations.py` | Runs the three strict ablations, validates their artifacts, and writes formal tables. | Uses the generic Qwen configuration through `ablation_core.py`; it is not a command-line DeepSeek launcher. |

Although the shared client source is named `qwen_gen.py`,
`run_laed_deepseek_v4.py` overrides that shared client for the entire DeepSeek
workflow before each run.

## Datasets

`Data/` contains five aligned dirty/clean CSV pairs.  The dirty file is the
only input to LAED.  The corresponding clean file is used only to calculate
evaluation metrics after detection.

| Dataset | Rows | Columns | Dirty input | Clean evaluation reference |
|---|---:|---:|---|---|
| Beers | 2,410 | 9 | `beers_error.csv` | `beers_clean.csv` |
| Flights | 2,376 | 6 | `flights_error.csv` | `flights_clean.csv` |
| Hospital | 1,000 | 17 | `hospital_error.csv` | `hospital_clean.csv` |
| Movies | 7,390 | 17 | `movies_error.csv` | `movies_clean.csv` |
| Rayyan | 1,000 | 11 | `rayyan_error.csv` | `rayyan_clean.csv` |

The archive already contains the paired CSV files; no data download or manual
preprocessing is required for the supplied experiments.  The loader reads the
CSV files as strings using UTF-8-with-BOM handling, so that literal missing
markers, leading zeros, identifiers, and value formatting are preserved.

### Dataset provenance and references

The five public benchmark pairs are distributed with the [Raha benchmark
repository](https://github.com/BigDaMa/raha/tree/7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c/datasets)
at the fixed commit `7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c`. Hospital and
Flights are established error-detection benchmarks. Beers is derived from the
[Craft Beers dataset](https://www.kaggle.com/datasets/nickhould/craft-cans),
and Movies is derived from the [Magellan Data
Repository](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository).
Rayyan contains bibliographic records paired with a reference cleaned by the
source-data maintainers.

The supplied CSV files are schema-normalized copies of those benchmark pairs.
Non-semantic record-index columns were omitted while preserving row order and
dirty/clean alignment. The Hospital copy retains one address field and omits
empty or redundant secondary address fields; Flights omits `tuple_id`; Beers
omits `index` and `ibu`; Rayyan and Movies retain the benchmark attributes
used in the experiment. Every method receives the same transformed dirty
table. No clean-reference value is used to construct a summary, rule,
candidate set, or prediction.

## Requirements and environment

The recorded software environment uses **Python 3.11** on Windows.

### Computing infrastructure

LAED calls hosted LLM APIs. A local GPU is therefore not a functional
requirement, and the code does not impose a fixed CPU, memory, storage, or GPU
model. A computer capable of running Python 3.11, an Internet connection to
the selected API provider, and sufficient local storage for CSV inputs and
timestamped result/log directories are required. CPU, memory, and storage
primarily affect runtime and the amount of retained output.

### Data preparation and evaluation

The five aligned dirty/clean CSV pairs are already included in `Data/`; no
additional download, manual cleaning, or preprocessing command is required.
The loader reads the supplied CSV files as strings with UTF-8-with-BOM
handling to preserve literal missing markers, leading zeros, identifiers, and
value formatting. During a run, LAED reads only the dirty CSV for summary
generation, screening, and detection. It reads the paired clean CSV only after
prediction in `evaluate_detection()` to derive ground truth and report
precision, recall, and F1.

Run the following commands in PowerShell from the `pythonProject1` directory.
The first block follows the Conda workflow used in the accompanying setup
guide.

```powershell
Set-Location "D:\path\to\Quality_ab\pythonProject1"

# Create and activate a Python 3.11 environment.
conda create -n laed python=3.11 -y
conda activate laed

# Install the pinned dependencies supplied with this archive.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the core dependencies used by the supplied
experiment entry points (`dashscope`, `numpy`, `openai`, `pandas`, and
`requests`) and additional SDKs used by provider-wrapper modules under
`API_invocation/`.  Install the complete file for the packaged environment.

If Conda is unavailable, create a Python 3.11 virtual environment instead:

```powershell
Set-Location "D:\path\to\Quality_ab\pythonProject1"
python -m venv .venv-laed
.\.venv-laed\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## API configuration

Never place an API key in source code, README files, saved command logs, or a
public archive.  Set it only in the PowerShell session used to launch a run.

### Generic Qwen main experiment and strict ablations

`LAED_Demo.py`, `run_laed_datasets.py`, and
`run_laed_strict_ablations.py` use the generic Qwen configuration fixed in the
source code.  Before using these entry points, set a valid DashScope key:

```powershell
$env:DASHSCOPE_API_KEY = "<your DashScope API key>"
```

The active source constants are:

```text
model:       qwen2.5-72b-instruct
temperature: 0.0
max tokens:  8192
```

### GPT-4o configuration and archived results

The shared OpenAI-compatible client recognises the following PowerShell
credentials:

```powershell
$env:OPENAI_API_KEY = "<your API key>"
$env:OPENAI_API_BASE = "<OpenAI-compatible API base URL>"
```

The active source constants are:

```text
model:       gpt-4o
temperature: 0.0
max tokens:  8192
```


### DeepSeek full-workflow experiment

Use `run_laed_deepseek_v4.py` when DeepSeek is intended to replace Qwen for
the complete workflow.  This runner does not perform a detection-stage-only
substitution: it configures the shared client before `LAED_Demo.demo_LAED()`
runs, thereby covering summary generation, rule generation, and LLM error
detection.

```powershell
$env:DEEPSEEK_API_KEY = "<your DeepSeek API key>"
$env:DEEPSEEK_MODEL = "deepseek-v4-flash
$env:DEEPSEEK_API_BASE = "https://api.deepseek.com"
$env:DEEPSEEK_MAX_TOKENS = "8192"
$env:DEEPSEEK_TEMPERATURE = "0.0"
$env:DEEPSEEK_OUTPUT_ROOT = (Join-Path $PWD "Run_Results_DeepSeekV4_Reproduction")
```

Use the exact model identifier accepted by the provider for the actual run;
model availability is controlled by the provider, not by this repository.

> `DEEPSEEK_*` variables alone do not make `LAED_Demo.py` or
> `run_laed_datasets.py` use DeepSeek.  Use the dedicated
> `run_laed_deepseek_v4.py` entry point for the supported DeepSeek full run.

## Run the experiments

All commands below assume that the environment is active and PowerShell is in
`Quality_ab\pythonProject1`.

### 1. Run one generic LAED dataset

After configuring `DASHSCOPE_API_KEY`, run one dataset as follows:

```powershell
python main_code\LAED_Demo.py --dataset hospital
```

The supported dataset names are `beers`, `flights`, `hospital`, `rayyan`, and
`movies`.  

### 2. Run the generic five-dataset main experiment

`run_laed_datasets.py` starts a new `LAED_Demo.py` subprocess for every
dataset/round combination.  It accepts dataset names as positional arguments,
not through a `--datasets` option.

```powershell
# One round for a quick execution check.
python main_code\run_laed_datasets.py hospital --rounds 1

# Five datasets, three independent rounds each (15 complete LAED runs).
python main_code\run_laed_datasets.py beers flights hospital rayyan movies --rounds 3
```

With no positional datasets, it defaults to all five datasets and three
rounds.  Generic results are written beneath `Run_Results/`.

### 3. Run the DeepSeek five-dataset, three-round experiment

After setting the `DEEPSEEK_*` variables above, run:

```powershell
python main_code\run_laed_deepseek_v4.py
```

This entry point has no dataset or round command-line parameters.  It performs
one API preflight and then runs all five datasets for exactly three rounds,
with a fresh summary (`reuse_summary=False`) for each run.  Set
`DEEPSEEK_OUTPUT_ROOT` before execution to keep a new reproduction separate
from archived results.

### Strict ablation experiments

After configuring `DASHSCOPE_API_KEY`, the default formal command runs three
strict configurations over five datasets for three rounds:

```powershell
python main_code\run_laed_strict_ablations.py `
  --rounds 3 `
  --output-root .\Run_Results_Ablation_Reproduction
```

For a full pre-formal smoke gate followed by the formal run, use all default
datasets and configurations:

```powershell
python main_code\run_laed_strict_ablations.py `
  --run-smoke-gate `
  --rounds 3 `
  --smoke-output-root .\Run_Results_Ablation_Smoke `
  --output-root .\Run_Results_Ablation_Reproduction
```

The smoke-gate implementation expects all 3 configurations × all 5 datasets.
Do not combine `--run-smoke-gate` or `--smoke-only` with a subset specified by
`--configs` or `--datasets`.

For a limited diagnostic run without a smoke gate, subsets are supported:

```powershell
python main_code\run_laed_strict_ablations.py `
  --configs no_summary `
  --datasets hospital `
  --rounds 1 `
  --output-root .\Run_Results_Ablation_Diagnostic
```

## Outputs and verification

### One complete LAED run

For a run with identifier `<run_id>`, the output directory is normally:

```text
Run_Results/<dataset>/<run_id>/
```

Important files are:

| File | Meaning |
|---|---|
| `data_summary.json` | Compact readable dataset summary: field semantics, format rules, relationships, and validator scope. |
| `correct_cells_mask.csv` | Boolean mask of cells screened as correct. |
| `suspicious_cells_for_error_detection.csv` | Table retaining cells that remain in the detection scope. |
| `detailed_errors.json` | Detailed detected-error records. |
| `detected_errors.json` | Final detected cell coordinates. |
| `errors_with_context.json` | Detection result, context, rules, and final coordinates. |
| `ground_truth_errors.json` | Cell coordinates derived from aligned dirty/clean differences. |
| `evaluation_metrics.json` | TP, FP, FN, precision, recall, and F1. |
| `metrics_tokens_<run_id>.json` | Model configuration, token usage, runtime, metrics, and output paths. |
| `latest_metrics_tokens.json` | Copy of the latest run report for the dataset. |

`evaluation_metrics.json` is a **result artifact** written by the active
`LAED_Demo.py` evaluation code.  It is not dependent on a source module named
`evaluation_metrics.py`.

### Generic batch output

`run_laed_datasets.py` additionally writes under `Run_Results/`:

```text
command_logs/<dataset>_<timestamp>.stdout.log
command_logs/<dataset>_<timestamp>.stderr.log
command_logs/<dataset>_<timestamp>.log
command_logs/run_laed_datasets_<timestamp>.log
batch_run_summary_<timestamp>.json
latest_batch_run_summary.json
three_run_average_metrics_<timestamp>.json
latest_three_run_average_metrics.json
latest_three_run_average_metrics.csv
latest_three_run_average_metrics.md
```

The batch script records each child process return code and continues with the
remaining tasks if an individual child fails.  Before reporting a three-round
result, verify that each dataset has `successful_run_count: 3`, that expected
metrics are present, and that the per-run logs show `returncode=0`.

### DeepSeek batch output

The DeepSeek runner writes an API preflight record in
`<DEEPSEEK_OUTPUT_ROOT>/command_logs/`, one complete LAED output directory per
dataset/round, and these aggregate files at the chosen output root:

```text
deepseek_v4_three_run_metrics_<timestamp>.json
latest_deepseek_v4_three_run_metrics.json
deepseek_v4_three_run_metrics_<timestamp>.csv
latest_deepseek_v4_three_run_metrics.csv
deepseek_v4_three_run_metrics_<timestamp>.md
latest_deepseek_v4_three_run_metrics.md
```

Inspect the `failures` list and confirm `run_count: 3` for every dataset before
using an aggregate result.

## Archived experiment results and logs

The following directories are preserved execution evidence supplied with this
archive. Their timestamped files should not be overwritten, deleted, or
combined with newly generated runs.

| Directory | Contents and interpretation | Key files to inspect |
|---|---|---|
| `Run_Results/` | Default output root for the generic main launcher. It contains per-dataset, per-round LAED outputs; child-process stdout, stderr, and merged logs; batch summaries; and three-round aggregate metrics. The existing `latest_batch_run_summary.json` records GPT-4o for its archived runs. It is historical evidence, because the current generic source is fixed to Qwen. |`latest_batch_run_summary.json`, `latest_three_run_average_metrics.{json,csv,md}`, and `command_logs/` |
| `Run_Results_Ablation_Strict_GPT4o_20260901_1824/` | Historical GPT-4o strict-ablation archive. It contains one formal round for the three ablation configurations (`no_screening`, `no_summary`, and `no_summary_screening`) across the five datasets, together with task logs and formal summary tables. | `latest_strict_ablation_formal_tables.{json,md}`, `latest_strict_ablation_long_metrics.csv`, and `command_logs/` |
| `Run_Results_Ablation_Strict_SmokeFull/` | Historical complete strict-ablation smoke-validation archive. It contains the 15 task outputs and logs for three configurations multiplied by five datasets. It is a smoke-validation record, not a formal multi-round statistical result. | `no_screening/`, `no_summary/`, `no_summary_screening/`, and `command_logs/` |
| `Run_Results_DeepSeekV4_Final/` | DeepSeek full-workflow archive: all five datasets, three rounds per dataset, per-run outputs, a preflight/command log, and JSON/CSV/Markdown aggregates. Its aggregate report records `deepseek-v4-flash`. | `deepseek_v4_three_run_metrics_20260528_220346.{json,csv,md}`, `all_metrics_tokens.json`, and `command_logs/` |


## Reproducibility notes

- Preserve the archived result directories above. Use a fresh output directory
  for a new reproduction whenever the selected launcher supports one;
  `run_laed_datasets.py` is the exception and must be run from a separate copy
  of the archive if its fixed `Run_Results/` root must remain unchanged.
- Run all five datasets for all three required main-experiment rounds after
  freezing the code.  If relevant code changes, repeat the complete protocol.
- Do not pool results obtained with different source revisions, models,
  provider versions, API endpoints, dates, temperatures, or token limits.
- Preserve the command logs, run reports, preflight files (for DeepSeek), and
  aggregate summaries.  They provide the execution evidence for a reported
  result.
- Keep API credentials outside the repository.  Rotate any credential that
  has ever been exposed.
