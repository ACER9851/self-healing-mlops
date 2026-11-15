"""Direct-execution tests for Task 6 analysis utilities."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_results import (
    aggregate_result,
    compute_recovery_speed,
    compute_unnecessary_retrain_rate,
    pareto_frontier,
    validate_result_schema,
)
from generate_tables import generate_tables, _latex_table
from statistical_tests import compute_bootstrap_ci, paired_bootstrap_difference


def test_bootstrap_ci_bounds():
    result = compute_bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_boot=200)
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert result["n"] == 4


def test_paired_identical_series():
    result = paired_bootstrap_difference([1, 2, 3], [1, 2, 3], n_boot=200)
    assert result["mean_diff"] == 0.0
    assert not result["significant"]


def test_paired_divergent_series():
    result = paired_bootstrap_difference([1, 2, 3, 4], [0, 0, 0, 0], n_boot=200)
    assert result["mean_diff"] > 0.0
    assert result["ci_lower"] > 0.0
    assert result["significant"]


def test_json_schema_validation():
    assert validate_result_schema({"rows": []})["valid"]
    assert not validate_result_schema({"rows": {}})["valid"]
    report = aggregate_result({"dataset": "fraud", "rows": []})
    assert report["status"] == "OK"


def test_aggregator_rates_and_recovery():
    rows = [
        {"window_id": 0, "drift_alarm": False, "degradation_alarm": False, "primary_value": 1.0, "retrain_triggered": False},
        {"window_id": 1, "drift_alarm": True, "degradation_alarm": False, "primary_value": 0.5, "retrain_triggered": True},
        {"window_id": 2, "drift_alarm": False, "degradation_alarm": False, "primary_value": 0.96, "retrain_triggered": True},
    ]
    assert compute_unnecessary_retrain_rate(rows) == 0.5
    assert compute_recovery_speed(rows) == 1


def test_pareto_frontier():
    records = [
        {"engine": "balanced", "score": 2.0, "cost": 2.0},
        {"engine": "cheap", "score": 1.0, "cost": 1.0},
        {"engine": "dominated", "score": 0.5, "cost": 2.0},
    ]
    frontier = pareto_frontier(records, maximize=("score",), minimize=("cost",))
    assert frontier == ["balanced", "cheap"]


def test_latex_format_and_output_files():
    latex = _latex_table("Test", ["A", "B"], [{"A": "x", "B": "y"}])
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    with tempfile.TemporaryDirectory() as directory:
        result = generate_tables({"reports": []}, directory)
        assert len(result["files"]) == 7
        assert (Path(directory) / "tables" / "tables_all.tex").exists()
        assert (Path(directory) / "tables" / "DECISION_LOG.md").exists()


if __name__ == "__main__":
    tests = [
        test_bootstrap_ci_bounds,
        test_paired_identical_series,
        test_paired_divergent_series,
        test_json_schema_validation,
        test_aggregator_rates_and_recovery,
        test_pareto_frontier,
        test_latex_format_and_output_files,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as error:
            print(f"  FAIL  {test.__name__}: {error}")
            failed += 1
    print(f"\n{passed}/{passed + failed} Analysis tests passed")
    if failed:
        raise SystemExit(1)
