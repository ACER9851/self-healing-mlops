"""Generate publication-ready tables and the Semester 2 decision log."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from aggregate_results import aggregate_directory


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "NOT_RUN"}]
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value):
    text = "NA" if value is None else str(value)
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _format_metric(value, interval=None):
    if value is None:
        return "NA"
    formatted = f"{value:.4f}"
    if interval and interval.get("lower") is not None:
        formatted += f" ({interval['lower']:.4f}, {interval['upper']:.4f})"
    return formatted


def _latex_table(title, columns, rows):
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + _latex_escape(title) + r"}",
        r"\begin{tabular}{" + alignment + r"}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row.get(column)) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _part3_rows(report):
    rows = []
    for item in report.get("reports", []):
        part3 = item.get("part3", {})
        for engine in part3.get("engines", []):
            rows.append({
                "Dataset": item.get("dataset", "unknown"),
                "Engine": engine.get("engine"),
                "Retrains": engine.get("retrains"),
                "Avg PR-AUC": _format_metric(engine.get("avg_pr_auc"), engine.get("avg_pr_auc_ci")),
                "Recall": _format_metric(engine.get("avg_minority_recall"), engine.get("avg_minority_recall_ci")),
                "Fit time (s)": _format_metric(engine.get("geometric_mean_fit_time_s")),
                "Unnecessary rate": _format_metric(engine.get("unnecessary_retrain_rate")),
                "Recovery (windows)": engine.get("recovery_speed_windows"),
            })
    return rows


def _unified_reports(report):
    return [item for item in report.get("reports", []) if "part1" in item]


def _part1_rows(report):
    rows = []
    for item in _unified_reports(report):
        section = item.get("part1", {})
        detector_rows = section.get("rows", [])
        rows.append({
            "Dataset": item.get("dataset", "unknown"),
            "Status": "SMOKE",
            "Windows": len(detector_rows),
            "Evidently alarms": sum(row.get("evidently_alarm", False) for row in detector_rows),
            "Alibi BH alarms": sum(row.get("alibi_bh_alarm", False) for row in detector_rows),
            "Alibi MMD alarms": sum(row.get("alibi_mmd_alarm", False) for row in detector_rows),
        })
    return rows


def _part2_rows(report):
    rows = []
    for item in _unified_reports(report):
        section = item.get("part2", {})
        rows.append({
            "Dataset": item.get("dataset", "unknown"),
            "Status": "SMOKE",
            "CBPE alert": section.get("cbpe", {}).get("overall_alert"),
            "DDM alarm": section.get("ddm", {}).get("overall_degradation_detected"),
            "EDDM alarm": section.get("eddm", {}).get("overall_degradation_detected"),
            "Estimation error": section.get("mean_absolute_estimation_error"),
        })
    return rows


def _part4_rows(report):
    rows = []
    for item in _unified_reports(report):
        for row in item.get("part4", {}).get("rows", []):
            rows.append({
                "Dataset": item.get("dataset", "unknown"),
                "Status": "SMOKE",
                "Strategy": row.get("strategy"),
                "Window": row.get("window_id"),
                "PR-AUC": row.get("pr_auc"),
                "Recall": row.get("minority_recall"),
                "Fit time (s)": row.get("fit_time_s"),
            })
    return rows


def _status_row(part, reason):
    return [{"Part": part, "Status": reason}]


def generate_tables(report=None, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    report = report or aggregate_directory(results_dir)
    table_dir = results_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    healing_rows = _part3_rows(report)
    drift_rows = _part1_rows(report) or _status_row("Part 1: Drift detection", "NOT_RUN: no Part 1 JSON artifact present")
    degradation_rows = _part2_rows(report) or _status_row("Part 2: Degradation monitoring", "NOT_RUN: no Part 2 JSON artifact present")
    retraining_rows = _part4_rows(report) or _status_row("Part 4: Retraining", "NOT_RUN: no Part 4 JSON artifact present")
    if not healing_rows:
        healing_rows = _status_row("Part 3: Healing", "NOT_RUN: no Part 3 JSON artifact present")
    decision_rows = [
        {"Decision area": "Drift detector", "Evidence status": "UNVERIFIED", "Semester 2 decision": "Collect full Part 1 benchmark"},
        {"Decision area": "Degradation estimator", "Evidence status": "UNVERIFIED", "Semester 2 decision": "Collect full Part 2 benchmark"},
        {"Decision area": "Retraining strategy", "Evidence status": "UNVERIFIED", "Semester 2 decision": "Collect multi-window Part 4 benchmark"},
        {"Decision area": "Healing engine", "Evidence status": "SMOKE ONLY", "Semester 2 decision": "Do not select from smoke results"},
    ]

    tables = {
        "table1_drift_detection.csv": drift_rows,
        "table2_degradation_monitoring.csv": degradation_rows,
        "table3_retraining_strategies.csv": retraining_rows,
        "table4_healing_policies.csv": healing_rows,
        "table5_decision_matrix.csv": decision_rows,
    }
    for filename, rows in tables.items():
        _write_csv(table_dir / filename, rows)

    latex_sections = [r"\usepackage{booktabs}", ""]
    latex_sections.append(_latex_table("Drift detection", ["Part", "Status"], drift_rows))
    latex_sections.append(_latex_table("Degradation monitoring", ["Part", "Status"], degradation_rows))
    latex_sections.append(_latex_table("Retraining strategies", ["Part", "Status"], retraining_rows))
    latex_sections.append(_latex_table(
        "Healing policies",
        ["Dataset", "Engine", "Retrains", "Avg PR-AUC", "Recall", "Fit time (s)", "Unnecessary rate", "Recovery (windows)"],
        healing_rows,
    ))
    latex_sections.append(_latex_table(
        "Decision matrix",
        ["Decision area", "Evidence status", "Semester 2 decision"],
        decision_rows,
    ))
    with open(table_dir / "tables_all.tex", "w", encoding="utf-8") as handle:
        handle.write("\n".join(latex_sections))

    decision_log = """# Semester 2 Architectural Decision Log

## Evidence Policy

This log records only evidence available in the results directory. Missing or smoke-test-only results are explicitly marked and are not treated as final scientific findings.

## Decisions

| Area | Evidence | Decision |
|---|---|---|
| Drift detector | UNVERIFIED: no full Part 1 aggregate artifact | Do not select a final detector yet. Compare false-positive control and multivariate sensitivity after the full benchmark. |
| Degradation estimator | UNVERIFIED: no full Part 2 aggregate artifact | CBPE remains the candidate for label-free monitoring, but the final choice requires measured estimation error and delayed-label latency. |
| Retraining strategy | UNVERIFIED: no full Part 4 aggregate artifact | Do not select Full History or Sliding Window from the current smoke run. |
| Healing engine | SMOKE ONLY: Part 3 Fraud run completed | Do not claim Q-learning, Regol, or a baseline is superior until multi-window, multi-seed comparisons are available. |

## Next Required Evidence

1. Run unified benchmarks on Fraud and LendingClub.
2. Mark Elec2 as NOT_RUN until its dataset is available.
3. Run paired bootstrap comparisons on identical windows and seeds.
4. Update this log only when confidence intervals and missing-data status support a defensible decision.
"""
    with open(table_dir / "DECISION_LOG.md", "w", encoding="utf-8") as handle:
        handle.write(decision_log)

    return {"table_dir": str(table_dir), "files": sorted(path.name for path in table_dir.iterdir())}


if __name__ == "__main__":
    result = generate_tables()
    print(json.dumps(result, indent=2))
