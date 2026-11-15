"""Closed-loop Part 3 healing benchmark runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_STUDY = _HERE.parent
sys.path.insert(0, str(_STUDY))
sys.path.insert(0, str(_STUDY / "part1_detect"))
sys.path.insert(0, str(_STUDY / "part2_degradation"))
sys.path.insert(0, str(_STUDY / "part4_retrain"))

from harness.data_loader import load_dataset
from harness.metrics import compute_f1, compute_pr_auc, compute_recall
from harness.model_wrapper import build_model
from harness.splitter import split_dataset, window_stream
from harness import config
from alibi_style_detector import AlibiStyleDriftDetector
from nannyml_cbpe import CBPEDegradationDetector
from full_retrain import FullHistoryRetrainer
from sliding_window import SlidingWindowRetrainer
from baselines import AlwaysRetrain, FixedSchedule, NeverRetrain, RetrainOnAlarm
from qtable_engine import QTableEngine
from regol_engine import RegolCostAwareEngine


def _metric_or_none(value):
    return float(value) if np.isfinite(value) else None


def _action_for_engine(name, engine, window_id, drift_alarm, degradation_alarm, state, current_metric):
    if name == "qtable":
        return engine.ACTION_NAMES[engine.select_action(state)]
    if name == "regol":
        return "sliding_window" if engine.should_retrain(float(drift_alarm), current_metric, state[2]) else "defer"
    if name == "never_retrain":
        return engine.decide(drift_alarm, degradation_alarm)
    if name == "always_retrain":
        return engine.decide()
    if name == "fixed_schedule":
        return engine.decide(window_id)
    return engine.decide(drift_alarm, degradation_alarm)


def _primary_value(dataset, f1, pr_auc):
    return pr_auc if dataset["meta"]["primary_metric"] == "pr_auc" else f1


def _execute_action(action, full_retrainer, sliding_retrainer, X_history, y_history, window):
    if action in ("full_retrain", "adversarial_hardening"):
        return full_retrainer.retrain(X_history, y_history)
    if action in ("sliding_window", "weighted_retrain", "model_switch"):
        return sliding_retrainer.retrain(window["X"], window["y"])
    return None


def _make_engines(model_key, seed, train_window_size, pretrain_episodes):
    qtable = QTableEngine()
    qtable.pretrain(pretrain_episodes, seed)
    return {
        "qtable": qtable,
        "regol": RegolCostAwareEngine(cost_budget="medium"),
        "never_retrain": NeverRetrain(),
        "always_retrain": AlwaysRetrain(),
        "fixed_schedule": FixedSchedule(k=5),
        "retrain_on_alarm": RetrainOnAlarm(),
    }


def run_benchmark(dataset_name="fraud", model_key="lr", seed=42, max_windows=15,
                  window_size=500, train_window_size=5000, pretrain_episodes=500):
    dataset = load_dataset(dataset_name)
    splits = split_dataset(dataset, seed=seed)
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    windows = window_stream(splits["test"]["X"], splits["test"]["y"], window_size)[:max_windows]
    if not windows:
        raise ValueError("No stream windows were produced")

    feature_names = list(X_train.columns)
    reference = X_train.iloc[-min(1000, len(X_train)):]
    reference_records = reference.to_dict(orient="records")
    drift_detector = AlibiStyleDriftDetector(method="columnwise_bh", alpha=0.05)

    reference_model = build_model(model_key, seed=seed)
    reference_model.fit(X_train, y_train)
    val_pred = reference_model.predict(X_val)
    val_proba = reference_model.predict_proba(X_val)
    # CBPE is intentionally pure Python; cap calibration data deterministically
    # so the closed-loop runner remains practical on the large datasets.
    cbpe_limit = min(5000, len(y_val))
    val_pred = val_pred[:cbpe_limit]
    val_proba = val_proba[:cbpe_limit]
    y_val_cbpe = y_val.iloc[:cbpe_limit]
    cbpe = CBPEDegradationDetector(
        chunk_size=window_size,
        alert_threshold_f1=0.08,
        alert_threshold_auc=0.08,
    )
    cbpe.fit(y_val_cbpe.tolist(), val_proba.tolist(), val_pred.tolist())

    engines = _make_engines(model_key, seed, train_window_size, pretrain_episodes)
    names = list(engines.keys())
    full_retrainers = {name: FullHistoryRetrainer(model_key, seed) for name in names}
    sliding_retrainers = {
        name: SlidingWindowRetrainer(model_key, train_window_size, seed)
        for name in names
    }
    # All policies begin from the same fitted baseline. Once a policy retrains,
    # its model is replaced independently and no longer shares this object.
    current_models = {name: reference_model for name in names}

    X_history = X_train.copy()
    y_history = y_train.copy()
    previous_metrics = {name: None for name in names}
    last_retrain = {name: -3 for name in names}
    drift_history = []
    rows = []

    for window in windows:
        window_id = window["window_id"]
        current_records = window["X"].to_dict(orient="records")
        drift_result = drift_detector.detect(reference_records, current_records, feature_names, [])
        drift_alarm = bool(drift_result.get("dataset_drift_detected", False))
        drift_history.append(drift_alarm)
        recent_drift = any(drift_history[-3:])
        drift_level = 2 if drift_alarm else 1 if recent_drift else 0

        for name in names:
            model = current_models[name]
            y_pred = model.predict(window["X"])
            y_proba = model.predict_proba(window["X"])
            f1 = compute_f1(window["y"].values, y_pred)
            pr_auc = compute_pr_auc(window["y"].values, y_proba)
            recall = compute_recall(window["y"].values, y_pred)
            primary = _primary_value(dataset, f1, pr_auc)
            current_signal = primary if np.isfinite(primary) else 0.0

            cbpe_result = cbpe.detect(y_proba.tolist(), y_pred.tolist())
            degradation_alarm = bool(cbpe_result.get("overall_alert", False))
            time_since_retrain = min(max(window_id - last_retrain[name], 0), 10)
            time_state = 0 if time_since_retrain <= 2 else 1 if time_since_retrain < 10 else 2
            state = (drift_level, 0, time_state)
            action = _action_for_engine(
                name, engines[name], window_id, drift_alarm,
                degradation_alarm, state, current_signal,
            )

            retrain_triggered = action != "defer"
            fit_time = 0.0
            status = "OK" if np.isfinite(primary) else "NOT_RUN"
            reason = None if status == "OK" else "Primary metric undefined: window contains one class"
            try:
                retrained_model = _execute_action(
                    action, full_retrainers[name], sliding_retrainers[name],
                    X_history, y_history, window,
                )
                if retrained_model is not None:
                    current_models[name] = retrained_model
                    fit_time = full_retrainers[name].fit_time_s if action in ("full_retrain", "adversarial_hardening") else sliding_retrainers[name].fit_time_s
                    last_retrain[name] = window_id
            except ValueError as error:
                status = "NOT_RUN"
                reason = str(error)
                retrain_triggered = False

            if previous_metrics[name] is not None and primary is not None and np.isfinite(primary):
                observed_delta = primary - previous_metrics[name]
                if name == "regol":
                    engines[name].observe(float(drift_alarm), current_signal, time_since_retrain, observed_delta)
            if primary is not None and np.isfinite(primary):
                previous_metrics[name] = primary

            row = {
                "engine": name,
                "window_id": window_id,
                "action_taken": action,
                "drift_alarm": drift_alarm,
                "degradation_alarm": degradation_alarm,
                "f1": _metric_or_none(f1),
                "pr_auc": _metric_or_none(pr_auc),
                "minority_recall": _metric_or_none(recall),
                "retrain_triggered": retrain_triggered,
                "fit_time_s": round(float(fit_time), 6),
                "cumulative_retrains": sum(1 for old in rows if old["engine"] == name and old["retrain_triggered"]) + int(retrain_triggered),
                "time_since_retrain": time_since_retrain,
                "primary_metric": dataset["meta"]["primary_metric"],
                "primary_value": _metric_or_none(primary),
                "status": status,
            }
            if reason:
                row["reason"] = reason
            rows.append(row)

        X_history = pd.concat([X_history, window["X"]], ignore_index=True)
        y_history = pd.concat([y_history, window["y"]], ignore_index=True)

    result = {
        "dataset": dataset_name,
        "model_key": model_key,
        "seed": seed,
        "window_size": window_size,
        "n_windows": len(windows),
        "rows": rows,
        "summary": _summarize(rows),
    }
    output = config.RESULTS_DIR / f"part3_results_{dataset_name}.json"
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def _summarize(rows):
    summary = []
    for engine in sorted(set(row["engine"] for row in rows)):
        group = [row for row in rows if row["engine"] == engine]
        valid_f1 = [row["f1"] for row in group if row["f1"] is not None]
        valid_pr = [row["pr_auc"] for row in group if row["pr_auc"] is not None]
        recalls = [row["minority_recall"] for row in group if row["minority_recall"] is not None]
        retrains = [row for row in group if row["retrain_triggered"]]
        unnecessary = [row for row in retrains if not row["drift_alarm"] and not row["degradation_alarm"]]
        drift_positions = [idx for idx, row in enumerate(group) if row["drift_alarm"]]
        recovery = None
        if drift_positions:
            first_drift = drift_positions[0]
            pre_drift = [
                row["primary_value"] for row in group[:first_drift]
                if row["primary_value"] is not None
            ]
            if pre_drift:
                threshold = 0.95 * float(np.mean(pre_drift))
                for later_idx in range(first_drift + 1, len(group)):
                    value = group[later_idx]["primary_value"]
                    if value is not None and value >= threshold:
                        recovery = later_idx - first_drift
                        break
        summary.append({
            "engine": engine,
            "retrains": len(retrains),
            "avg_f1": _metric_or_none(np.mean(valid_f1) if valid_f1 else float("nan")),
            "avg_pr_auc": _metric_or_none(np.mean(valid_pr) if valid_pr else float("nan")),
            "min_recall": _metric_or_none(min(recalls) if recalls else float("nan")),
            "unnecessary_rate": len(unnecessary) / len(retrains) if retrains else 0.0,
            "recovery": recovery,
        })
    return summary


def _print_summary(result):
    print("Engine             Retrains  Avg F1  Avg PR-AUC  Min Recall  Unnecessary%  Recovery")
    for row in result["summary"]:
        print(
            f"{row['engine']:<18} {row['retrains']:>8}  "
            f"{str(row['avg_f1']):>7}  {str(row['avg_pr_auc']):>10}  "
            f"{str(row['min_recall']):>10}  {row['unnecessary_rate'] * 100:>12.1f}  "
            f"{str(row['recovery']):>8}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fraud", choices=["fraud", "lendingclub"])
    parser.add_argument("--model", default="lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--train-window-size", type=int, default=5000)
    parser.add_argument("--pretrain-episodes", type=int, default=500)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_benchmark(
        args.dataset, args.model, args.seed, args.max_windows,
        args.window_size, args.train_window_size, args.pretrain_episodes,
    )
    _print_summary(result)
    print(f"Results saved to {config.RESULTS_DIR / f'part3_results_{args.dataset}.json'}")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
