"""Orchestrate the four comparison-study components into one JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for path in (STUDY, STUDY / "part1_detect", STUDY / "part2_degradation", STUDY / "part3_heal", STUDY / "part4_retrain"):
    sys.path.insert(0, str(path))

from harness.data_loader import load_dataset
from harness.metrics import compute_f1, compute_pr_auc, compute_recall
from harness.model_wrapper import build_model
from harness.splitter import split_dataset, window_stream
from harness import config
from evidently_style_detector import EvidentlyStyleDriftDetector
from alibi_style_detector import AlibiStyleDriftDetector
from nannyml_cbpe import CBPEDegradationDetector
from ddm_detector import DDMDegradationDetector, EDDMDegradationDetector
from full_retrain import FullHistoryRetrainer
from sliding_window import SlidingWindowRetrainer
from run_part3 import run_benchmark as run_healing_benchmark


def _metric(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _run_part1(dataset, splits, windows, max_mmd_rows=100):
    reference = splits["train"]["X"].iloc[-min(1000, len(splits["train"]["X"])):]
    numerical = list(reference.columns)
    ref_records = reference.to_dict(orient="records")
    evidently = EvidentlyStyleDriftDetector()
    bh = AlibiStyleDriftDetector(method="columnwise_bh", alpha=0.05)
    mmd = AlibiStyleDriftDetector(method="multivariate_mmd", alpha=0.05, n_permutations=100, seed=42)
    rows = []
    for window in windows:
        current = window["X"]
        current_records = current.to_dict(orient="records")
        ev = evidently.detect(ref_records, current_records, numerical, [])
        bh_result = bh.detect(ref_records, current_records, numerical, [])
        ref_mmd = reference.iloc[:max_mmd_rows].to_dict(orient="records")
        cur_mmd = current.iloc[:max_mmd_rows].to_dict(orient="records")
        mmd_result = mmd.detect(ref_mmd, cur_mmd, numerical, [])
        rows.append({
            "window_id": window["window_id"],
            "evidently_alarm": bool(ev.get("dataset_drift_detected", False)),
            "alibi_bh_alarm": bool(bh_result.get("dataset_drift_detected", False)),
            "alibi_mmd_alarm": bool(mmd_result.get("dataset_drift_detected", False)),
            "alibi_mmd_p_value": _metric(mmd_result.get("p_value")),
            "feature_attribution_count": sum(
                bool(item.get("drift_detected", False))
                for item in bh_result.get("feature_results", {}).values()
            ),
        })
    return {
        "status": "OK",
        "reference_rows": len(reference),
        "rows": rows,
        "false_positive_rate": None,
        "detection_latency_windows": None,
    }


def _run_part2(model, splits, windows, window_size):
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    limit = min(5000, len(y_val))
    cbpe = CBPEDegradationDetector(chunk_size=window_size, alert_threshold_f1=0.08, alert_threshold_auc=0.08)
    cbpe.fit(y_val.iloc[:limit].tolist(), val_proba[:limit].tolist(), val_pred[:limit].tolist())
    y_stream = pd.concat([window["y"] for window in windows], ignore_index=True)
    pred_stream = np.concatenate([model.predict(window["X"]) for window in windows])
    proba_stream = np.concatenate([model.predict_proba(window["X"]) for window in windows])
    cbpe_result = cbpe.detect(proba_stream.tolist(), pred_stream.tolist())
    ddm = DDMDegradationDetector()
    ddm.fit(y_val.iloc[:limit].tolist(), val_pred[:limit].tolist())
    eddm = EDDMDegradationDetector()
    eddm.fit(y_val.iloc[:limit].tolist(), val_pred[:limit].tolist())
    return {
        "status": "OK",
        "cbpe": cbpe_result,
        "ddm": ddm.detect_chunks(y_stream.tolist(), pred_stream.tolist(), window_size),
        "eddm": eddm.detect_chunks(y_stream.tolist(), pred_stream.tolist(), window_size),
        "mean_absolute_estimation_error": None,
        "label_delay_latency": None,
    }


def _run_part4(model_key, seed, splits, windows, train_window_size):
    strategies = {
        "full_history": FullHistoryRetrainer(model_key, seed),
        "sliding_window": SlidingWindowRetrainer(model_key, train_window_size, seed),
    }
    models = {name: build_model(model_key, seed=seed) for name in strategies}
    for model in models.values():
        model.fit(splits["train"]["X"], splits["train"]["y"])
    history_X = splits["train"]["X"].copy()
    history_y = splits["train"]["y"].copy()
    rows = []
    for window in windows:
        for name, strategy in strategies.items():
            current = models[name]
            pred = current.predict(window["X"])
            proba = current.predict_proba(window["X"])
            f1 = compute_f1(window["y"].values, pred)
            pr_auc = compute_pr_auc(window["y"].values, proba)
            recall = compute_recall(window["y"].values, pred)
            status = "OK"
            reason = None
            try:
                if name == "full_history":
                    updated = strategy.retrain(history_X, history_y)
                else:
                    updated = strategy.retrain(window["X"], window["y"])
                models[name] = updated
                fit_time = strategy.fit_time_s
            except ValueError as error:
                fit_time = 0.0
                status = "NOT_RUN"
                reason = str(error)
            row = {
                "strategy": name,
                "window_id": window["window_id"],
                "f1": _metric(f1),
                "pr_auc": _metric(pr_auc),
                "minority_recall": _metric(recall),
                "fit_time_s": fit_time,
                "n_training_samples": strategy.n_samples,
                "status": status,
            }
            if reason:
                row["reason"] = reason
            rows.append(row)
        history_X = pd.concat([history_X, window["X"]], ignore_index=True)
        history_y = pd.concat([history_y, window["y"]], ignore_index=True)
    return {"status": "OK", "rows": rows, "recovery_speed": None}


def run_full_study(dataset_name="fraud", model_key="lr", max_windows=15,
                   window_size=500, train_window_size=5000, seed=42, out_dir=None,
                   pretrain_episodes=500):
    if dataset_name == "elec2":
        return {"dataset": dataset_name, "status": "NOT_RUN", "reason": "elec2_clean.csv is unavailable locally"}
    dataset = load_dataset(dataset_name)
    splits = split_dataset(dataset, seed=seed)
    windows = window_stream(splits["test"]["X"], splits["test"]["y"], window_size)[:max_windows]
    model = build_model(model_key, seed=seed)
    model.fit(splits["train"]["X"], splits["train"]["y"])
    result = {
        "dataset": dataset_name,
        "model_key": model_key,
        "seed": seed,
        "window_size": window_size,
        "n_windows": len(windows),
        "status": "OK",
        "part1": _run_part1(dataset, splits, windows),
        "part2": _run_part2(model, splits, windows, window_size),
        "part4": _run_part4(model_key, seed, splits, windows, train_window_size),
    }
    healing = run_healing_benchmark(
        dataset_name, model_key, seed, max_windows, window_size,
        train_window_size, pretrain_episodes,
    )
    result["part3"] = healing
    output_dir = Path(out_dir) if out_dir else STUDY / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"unified_{dataset_name}_{model_key}.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    result["output_path"] = str(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fraud", choices=["fraud", "lendingclub"])
    parser.add_argument("--model", default="lr")
    parser.add_argument("--max-windows", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--train-window-size", type=int, default=5000)
    parser.add_argument("--out-dir", default=str(STUDY / "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-episodes", type=int, default=500)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_full_study(
        args.dataset, args.model, args.max_windows, args.window_size,
        args.train_window_size, args.seed, args.out_dir, args.pretrain_episodes,
    )
    print(json.dumps({"status": result.get("status"), "output_path": result.get("output_path"), "elapsed_s": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
