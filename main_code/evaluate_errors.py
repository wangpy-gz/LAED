import json
import pandas as pd
import os
from typing import Set, Tuple

def load_ground_truth_errors(clean_path: str, dirty_path: str) -> Set[Tuple[int, str]]:
    """
    Read “clean” and “dirty” CSVs with keep_default_na=False, so that strings like "N/A" or "nan"
    remain literal strings. Wherever the clean and dirty cells differ (by string comparison),
    record (row_index, column_name) as a true error.
    """
    df_clean = pd.read_csv(clean_path, keep_default_na=False)
    df_dirty = pd.read_csv(dirty_path, keep_default_na=False)

    if df_clean.shape != df_dirty.shape:
        raise ValueError("Clean and Dirty files have different shapes. Please ensure rows/columns align.")

    ground_truth: Set[Tuple[int, str]] = set()
    for idx in df_clean.index:
        for col in df_clean.columns:
            val_clean = df_clean.at[idx, col]
            val_dirty = df_dirty.at[idx, col]
            if str(val_clean) != str(val_dirty):
                ground_truth.add((int(idx), col))

    return ground_truth

def save_ground_truth_errors(ground_truth: Set[Tuple[int, str]], output_path: str):
    """
    Save the ground‐truth error set as JSON: [[row_index, column_name], ...]
    """
    gt_list = [[row, col] for (row, col) in sorted(ground_truth)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gt_list, f, ensure_ascii=False, indent=2)

def load_detected_errors_from_context(detected_context_path: str) -> Set[Tuple[int, str]]:
    """
    Read errors_with_context.json, extract the "final_errors" list
    (each entry is [row_index, column_name]), and return as a set of tuples.
    """
    with open(detected_context_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_list = data.get("final_errors", [])
    detected: Set[Tuple[int, str]] = set()
    for entry in final_list:
        row_idx = entry[0]
        cols = entry[1]
        if isinstance(cols, list):
            for c in cols:
                detected.add((int(row_idx), c))
        else:
            detected.add((int(row_idx), cols))
    return detected

def compute_metrics(ground_truth: Set[Tuple[int, str]], detected: Set[Tuple[int, str]]):
    """
    Compute TP, FP, FN, precision, recall, and F1 between ground_truth and detected sets.
    Also returns a dict of all relevant counts and scores.
    """
    tp = len(ground_truth & detected)
    fp = len(detected - ground_truth)
    fn = len(ground_truth - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "ground_truth_count": len(ground_truth),
        "detected_count": len(detected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    print("===== Evaluation Results =====")
    print(f"Ground-Truth Error Cells: {metrics['ground_truth_count']}")
    print(f"Detected Error Cells:       {metrics['detected_count']}")
    print(f"True Positives (TP):        {metrics['true_positives']}")
    print(f"False Positives (FP):       {metrics['false_positives']}")
    print(f"False Negatives (FN):       {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")

    return metrics

if __name__ == "__main__":
    # Adjust these paths as needed
    clean_csv = "F:/Quality/pythonProject1/Data/beers_clean-01.csv"
    dirty_csv = "F:/Quality/pythonProject1/Data/beers_error-01.csv"
    detected_context_json = "errors_with_context.json"
    gt_output_json = "ground_truth_errors.json"
    metrics_output_json = "evaluation_metrics.json"

    # 1. Check that all required files exist
    for path in (clean_csv, dirty_csv, detected_context_json):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # 2. Build and save ground-truth errors
    gt_errors = load_ground_truth_errors(clean_csv, dirty_csv)
    save_ground_truth_errors(gt_errors, gt_output_json)
    print(f"Saved ground-truth error cells to: {gt_output_json}")

    # 3. Load detected errors from errors_with_context.json → the "final_errors" field
    detected_errors = load_detected_errors_from_context(detected_context_json)

    # 4. Compute metrics
    metrics = compute_metrics(gt_errors, detected_errors)

    # 5. Save metrics to JSON
    with open(metrics_output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation metrics to: {metrics_output_json}")