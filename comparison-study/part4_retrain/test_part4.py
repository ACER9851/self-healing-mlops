"""Focused tests for Part 4 retraining strategies."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.metrics import compute_f1, compute_pr_auc
from full_retrain import FullHistoryRetrainer
from sliding_window import SlidingWindowRetrainer


def _data(seed=7):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.normal(size=(240, 2)), columns=["x1", "x2"])
    y = (X["x1"] + 0.5 * X["x2"] > 0).astype(int)
    return X, y


def test_full_history_grows_and_retrains():
    X, y = _data()
    retrainer = FullHistoryRetrainer()
    retrainer.retrain(X.iloc[:100], y.iloc[:100])
    assert retrainer.n_samples == 100
    retrainer.retrain(X.iloc[:180], y.iloc[:180])
    assert retrainer.n_samples == 180
    assert retrainer.retrain_count == 2


def test_sliding_buffer_is_bounded_and_keeps_latest_rows():
    X, y = _data()
    retrainer = SlidingWindowRetrainer(window_size=50)
    retrainer.append(X.iloc[:40], y.iloc[:40])
    retrainer.append(X.iloc[40:120], y.iloc[40:120])
    assert retrainer.n_samples == 50
    pd.testing.assert_frame_equal(retrainer.X_buffer, X.iloc[70:120].reset_index(drop=True))
    pd.testing.assert_series_equal(retrainer.y_buffer, y.iloc[70:120].reset_index(drop=True), check_names=False)


def test_retraining_restores_shifted_stream_performance():
    X, y = _data()
    initial = FullHistoryRetrainer()
    initial.retrain(X.iloc[:100], y.iloc[:100])

    shifted_X = X.iloc[100:].copy()
    shifted_y = (shifted_X["x1"] + 0.5 * shifted_X["x2"] < 0).astype(int)
    before = compute_f1(shifted_y, initial.predict(shifted_X))

    retrainer = SlidingWindowRetrainer(window_size=100)
    retrainer.retrain(pd.concat([X.iloc[:20], shifted_X.iloc[:80]]), pd.concat([y.iloc[:20], shifted_y.iloc[:80]]))
    after_pred = retrainer.predict(shifted_X.iloc[80:])
    after_proba = retrainer.predict_proba(shifted_X.iloc[80:])
    after = compute_f1(shifted_y.iloc[80:], after_pred)
    pr_auc = compute_pr_auc(shifted_y.iloc[80:], after_proba)

    assert before < 0.95
    assert after > before
    assert pr_auc > 0.7


def test_invalid_inputs_are_rejected():
    X, y = _data()
    try:
        SlidingWindowRetrainer(window_size=0)
        raise AssertionError("zero window size should fail")
    except ValueError:
        pass
    try:
        FullHistoryRetrainer().retrain(X.iloc[:10], np.zeros(10))
        raise AssertionError("single-class history should fail")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_full_history_grows_and_retrains,
        test_sliding_buffer_is_bounded_and_keeps_latest_rows,
        test_retraining_restores_shifted_stream_performance,
        test_invalid_inputs_are_rejected,
    ]
    for test in tests:
        test()
    print("4/4 Part 4 tests passed")
