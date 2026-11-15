"""Benchmark full-history versus sliding-window retraining on a data stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.data_loader import load_dataset
from harness.metrics import compute_f1, compute_pr_auc, compute_recall
from harness.splitter import split_dataset, window_stream
from full_retrain import FullHistoryRetrainer
from sliding_window import SlidingWindowRetrainer


def _json_metric(value):
    return float(value) if np.isfinite(value) else None


def run_benchmark(dataset_name="fraud", model_key="lr", seed=42, max_windows=12, window_size=500, train_window_size=5000):
    dataset = load_dataset(dataset_name)
    splits = split_dataset(dataset, seed=seed)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    stream = window_stream(splits["test"]["X"], splits["test"]["y"], window_size)
    stream = stream[:max_windows]
    if not stream:
        raise ValueError("The selected test split produced no stream windows")

    strategies = {
        "full_retrain": FullHistoryRetrainer(model_key, seed),
        "sliding_window": SlidingWindowRetrainer(model_key, train_window_size, seed),
    }
    rows = []
    drift_window = max(1, len(stream) // 3)
    baseline_values = {name: [] for name in strategies}
    X_seen = X_train.copy()
    y_seen = y_train.copy()
    for window in stream:
        for name, strategy in strategies.items():
            try:
                strategy.retrain(X_seen, y_seen)
            except ValueError as error:
                rows.append({
                    "strategy": name,
                    "window_id": window["window_id"],
                    "status": "NOT_RUN",
                    "reason": str(error),
                    "n_training_samples": strategy.n_samples,
                    "fit_time_s": 0.0,
                    "f1": None,
                    "pr_auc": None,
                    "minority_recall": None,
                })
                continue
            proba = strategy.predict_proba(window["X"])
            pred = strategy.predict(window["X"])
            primary_value = (
                compute_pr_auc(window["y"], proba)
                if dataset["meta"]["primary_metric"] == "pr_auc"
                else compute_f1(window["y"], pred)
            )
            if window["window_id"] < drift_window and np.isfinite(primary_value):
                baseline_values[name].append(primary_value)
            rows.append({
                "strategy": name,
                "window_id": window["window_id"],
                "status": "OK",
                "n_training_samples": strategy.n_samples,
                "fit_time_s": round(strategy.fit_time_s, 6),
                "f1": _json_metric(compute_f1(window["y"], pred)),
                "pr_auc": _json_metric(compute_pr_auc(window["y"], proba)),
                "minority_recall": _json_metric(compute_recall(window["y"], pred)),
            })
        X_seen = pd.concat([X_seen, window["X"]], ignore_index=True)
        y_seen = pd.concat([y_seen, window["y"]], ignore_index=True)
    for row in rows:
        values = baseline_values[row["strategy"]]
        baseline = float(np.mean(values)) if values else float("nan")
        row["baseline_primary"] = _json_metric(baseline)
        row["primary_metric"] = dataset["meta"]["primary_metric"]
        row["primary_value"] = (
            row["pr_auc"] if row["primary_metric"] == "pr_auc" else row["f1"]
        )
    recovery = {}
    for name in strategies:
        values = baseline_values[name]
        baseline = float(np.mean(values)) if values else float("nan")
        threshold = 0.95 * baseline
        recovered = [
            row["window_id"] - drift_window
            for row in rows
            if row["strategy"] == name and row["status"] == "OK" and row["window_id"] >= drift_window and row["primary_value"] is not None and row["primary_value"] >= threshold
        ]
        recovery[name] = min(recovered) if recovered else None
    return {"dataset": dataset_name, "model_key": model_key, "drift_window": drift_window, "recovery_speed_windows": recovery, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fraud")
    parser.add_argument("--model", default="lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--train-window-size", type=int, default=5000)
    args = parser.parse_args()
    result = run_benchmark(args.dataset, args.model, args.seed, args.max_windows, args.window_size, args.train_window_size)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
