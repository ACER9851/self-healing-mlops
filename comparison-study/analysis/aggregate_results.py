"""Ingest benchmark JSON files and synthesize cross-part metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from statistical_tests import compute_bootstrap_ci


def _finite(values):
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def load_result(path):
    """Load one JSON result file without changing its raw structure."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_result_schema(result):
    """Return a small validation report for a result object."""
    if not isinstance(result, dict):
        return {"valid": False, "reason": "root must be an object"}
    if "rows" in result and not isinstance(result["rows"], list):
        return {"valid": False, "reason": "rows must be a list"}
    if "summary" in result and not isinstance(result["summary"], list):
        return {"valid": False, "reason": "summary must be a list"}
    if "parts" in result and not isinstance(result["parts"], dict):
        return {"valid": False, "reason": "parts must be an object"}
    return {"valid": True, "reason": None}


def compute_unnecessary_retrain_rate(rows):
    retrains = [row for row in rows if row.get("retrain_triggered")]
    if not retrains:
        return 0.0
    unnecessary = [
        row for row in retrains
        if not row.get("drift_alarm", False) and not row.get("degradation_alarm", False)
    ]
    return len(unnecessary) / len(retrains)


def compute_recovery_speed(rows, primary_key="primary_value"):
    """Return windows from first alarm to 95% of the pre-alarm baseline."""
    ordered = sorted(rows, key=lambda row: row.get("window_id", 0))
    alarm_positions = [idx for idx, row in enumerate(ordered) if row.get("drift_alarm")]
    if not alarm_positions:
        return None
    first_alarm = alarm_positions[0]
    baseline = _finite([
        row.get(primary_key) for row in ordered[:first_alarm]
    ])
    if not baseline:
        return None
    threshold = 0.95 * float(np.mean(baseline))
    for idx in range(first_alarm + 1, len(ordered)):
        value = ordered[idx].get(primary_key)
        if value is not None and np.isfinite(value) and value >= threshold:
            return idx - first_alarm
    return None


def _aggregate_rows(rows):
    by_engine = {}
    for row in rows:
        by_engine.setdefault(row.get("engine", "unknown"), []).append(row)
    output = []
    for engine, group in sorted(by_engine.items()):
        f1 = _finite([row.get("f1") for row in group])
        pr_auc = _finite([row.get("pr_auc") for row in group])
        recall = _finite([row.get("minority_recall") for row in group])
        fit_times = _finite([row.get("fit_time_s") for row in group if row.get("fit_time_s", 0) > 0])
        retrains = sum(bool(row.get("retrain_triggered")) for row in group)
        output.append({
            "engine": engine,
            "n_windows": len(group),
            "retrains": retrains,
            "avg_f1": float(np.mean(f1)) if f1 else None,
            "avg_pr_auc": float(np.mean(pr_auc)) if pr_auc else None,
            "avg_minority_recall": float(np.mean(recall)) if recall else None,
            "avg_f1_ci": compute_bootstrap_ci(f1),
            "avg_pr_auc_ci": compute_bootstrap_ci(pr_auc),
            "avg_minority_recall_ci": compute_bootstrap_ci(recall),
            "geometric_mean_fit_time_s": float(np.exp(np.mean(np.log(fit_times)))) if fit_times else None,
            "total_compute_cost_s": float(np.sum(fit_times)) if fit_times else 0.0,
            "unnecessary_retrain_rate": compute_unnecessary_retrain_rate(group),
            "recovery_speed_windows": compute_recovery_speed(group),
            "cumulative_reward": None,
            "missed_heal_rate": None,
        })
    frontier = pareto_frontier(
        output,
        maximize=("avg_pr_auc", "avg_minority_recall"),
        minimize=("total_compute_cost_s", "unnecessary_retrain_rate"),
    )
    return {"engines": output, "pareto_frontier": frontier}


def pareto_frontier(records, maximize=(), minimize=()):
    """Return non-dominated records for the requested objective directions."""
    usable = [
        record for record in records
        if all(record.get(key) is not None for key in tuple(maximize) + tuple(minimize))
    ]
    frontier = []
    for candidate in usable:
        dominated = False
        for other in usable:
            if other is candidate:
                continue
            at_least_as_good = True
            strictly_better = False
            for key in maximize:
                if other[key] < candidate[key]:
                    at_least_as_good = False
                if other[key] > candidate[key]:
                    strictly_better = True
            for key in minimize:
                if other[key] > candidate[key]:
                    at_least_as_good = False
                if other[key] < candidate[key]:
                    strictly_better = True
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate.get("engine", "unknown"))
    return frontier


def aggregate_result(result, source="memory"):
    """Normalize one result object into analysis sections."""
    report = validate_result_schema(result)
    if not report["valid"]:
        return {"source": source, "status": "INVALID", "reason": report["reason"]}
    if "parts" in result:
        output = {"source": source, "status": "OK"}
        output.update(result["parts"])
        return output
    part_keys = ("part1", "part2", "part3", "part4")
    if any(key in result for key in part_keys):
        output = {
            "source": source,
            "status": result.get("status", "OK"),
            "dataset": result.get("dataset"),
            "model_key": result.get("model_key"),
        }
        for key in part_keys:
            if key not in result:
                continue
            section = result[key]
            if key == "part3" and isinstance(section, dict) and isinstance(section.get("rows"), list):
                section = {"status": "OK", **_aggregate_rows(section["rows"])}
            output[key] = section
        return output
    rows = result.get("rows", [])
    if rows:
        return {
            "source": source,
            "status": "OK",
            "dataset": result.get("dataset"),
            "model_key": result.get("model_key"),
            "part3": {"status": "OK", **_aggregate_rows(rows)},
        }
    return {"source": source, "status": "OK", "metadata": result}


def aggregate_directory(results_dir):
    """Aggregate all JSON files currently available in a results directory."""
    directory = Path(results_dir)
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    reports = []
    for path in files:
        try:
            reports.append(aggregate_result(load_result(path), path.name))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reports.append({"source": path.name, "status": "INVALID", "reason": str(error)})
    return {
        "results_dir": str(directory),
        "source_files": [path.name for path in files],
        "reports": reports,
        "status": "OK" if reports else "NO_RESULTS",
    }


def save_aggregate(report, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = aggregate_directory(args.results_dir)
    output = args.output or str(Path(args.results_dir) / "aggregate_results.json")
    save_aggregate(report, output)
    print(json.dumps(report, indent=2))
