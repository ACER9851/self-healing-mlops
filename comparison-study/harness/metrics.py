"""
metrics.py — Evaluation metrics with bootstrap confidence intervals.

Primary metric: PR-AUC (for imbalanced datasets) or F1 (for balanced).
Always reports both.  Never accuracy alone.

Usage:
    from harness.metrics import evaluate, paired_bootstrap_test

    result = evaluate(y_true, y_pred, y_proba, primary="pr_auc")
    # result = {"pr_auc": 0.82, "pr_auc_ci": (0.79, 0.85), "f1": 0.71, ...}

    delta = paired_bootstrap_test(y_true, proba_A, proba_B, metric="pr_auc")
    # delta = {"mean_diff": 0.03, "ci_lo": 0.01, "ci_hi": 0.05, "significant": True}
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)

from . import config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Core metric computation
# ────────────────────────────────────────────────────────────────────

def compute_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Area under the Precision-Recall curve.

    Uses ``average_precision_score`` which is equivalent to PR-AUC
    computed with the trapezoidal rule over the step function.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(average_precision_score(y_true, y_proba))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score (binary, positive = 1)."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC-AUC (for reference, not primary)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


# ────────────────────────────────────────────────────────────────────
#  Bootstrap confidence intervals
# ────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: callable,
    n_iterations: int = None,
    ci_level: float = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric bootstrap 95% CI for a metric.

    Parameters
    ----------
    y_true : array
    y_score : array
        Probabilities (for PR-AUC, ROC-AUC) or predictions (for F1).
    metric_fn : callable
        Function(y_true, y_score) → float.
    n_iterations : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (0.95 for 95% CI).
    seed : int

    Returns
    -------
    (ci_low, ci_high) : tuple of float
    """
    n_iterations = n_iterations or config.BOOTSTRAP_ITERATIONS
    ci_level = ci_level or config.BOOTSTRAP_CI_LEVEL
    rng = np.random.RandomState(seed)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        # Skip degenerate samples
        if len(np.unique(y_t)) < 2:
            continue
        scores.append(metric_fn(y_t, y_s))

    if len(scores) < 10:
        logger.warning("Bootstrap: too few valid resamples for reliable CI")
        return (float("nan"), float("nan"))

    scores = np.array(scores)
    alpha = 1 - ci_level
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


# ────────────────────────────────────────────────────────────────────
#  Full evaluation
# ────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    primary: str = "pr_auc",
    bootstrap: bool = True,
    seed: int = 42,
) -> dict:
    """Compute all metrics + optional bootstrap CI.

    Parameters
    ----------
    y_true : array of 0/1
    y_pred : array of 0/1 (hard predictions)
    y_proba : array of float (P(y=1))
    primary : str
        ``"pr_auc"`` or ``"f1"`` — highlighted in the result dict.
    bootstrap : bool
        Whether to compute bootstrap CIs (slower).
    seed : int

    Returns
    -------
    dict with keys:
        pr_auc, pr_auc_ci, f1, f1_ci, roc_auc, roc_auc_ci,
        precision, recall, primary_metric, primary_value, primary_ci
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    result = {
        "pr_auc":    compute_pr_auc(y_true, y_proba),
        "f1":        compute_f1(y_true, y_pred),
        "roc_auc":   compute_roc_auc(y_true, y_proba),
        "precision": compute_precision(y_true, y_pred),
        "recall":    compute_recall(y_true, y_pred),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "pos_rate":  float(y_true.mean()),
    }

    if bootstrap:
        result["pr_auc_ci"] = bootstrap_ci(
            y_true, y_proba, compute_pr_auc, seed=seed
        )
        result["f1_ci"] = bootstrap_ci(
            y_true, y_pred, compute_f1, seed=seed
        )
        result["roc_auc_ci"] = bootstrap_ci(
            y_true, y_proba, compute_roc_auc, seed=seed
        )
    else:
        result["pr_auc_ci"] = (float("nan"), float("nan"))
        result["f1_ci"] = (float("nan"), float("nan"))
        result["roc_auc_ci"] = (float("nan"), float("nan"))

    # Highlight primary
    result["primary_metric"] = primary
    result["primary_value"] = result[primary]
    result["primary_ci"] = result[f"{primary}_ci"]

    return result


# ────────────────────────────────────────────────────────────────────
#  Paired bootstrap test (comparing two methods)
# ────────────────────────────────────────────────────────────────────

def paired_bootstrap_test(
    y_true: np.ndarray,
    score_A: np.ndarray,
    score_B: np.ndarray,
    metric_fn: callable = None,
    metric_name: str = "pr_auc",
    n_iterations: int = None,
    ci_level: float = None,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is method A significantly better than B?

    Computes metric(A) − metric(B) on the same bootstrap resample,
    then reports the CI of the difference.

    Parameters
    ----------
    y_true : array
    score_A, score_B : array
        Probabilities from method A and B.
    metric_fn : callable, optional
        Defaults to the function matching ``metric_name``.
    metric_name : str
        ``"pr_auc"`` or ``"f1"``.
    n_iterations, ci_level, seed : as in ``bootstrap_ci``.

    Returns
    -------
    dict with keys:
        mean_diff, ci_lo, ci_hi, significant (True if CI excludes 0),
        winner ("A", "B", or "tie")
    """
    n_iterations = n_iterations or config.BOOTSTRAP_ITERATIONS
    ci_level = ci_level or config.BOOTSTRAP_CI_LEVEL

    if metric_fn is None:
        metric_fn = {"pr_auc": compute_pr_auc, "f1": compute_f1, "roc_auc": compute_roc_auc}[metric_name]

    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    score_A = np.asarray(score_A)
    score_B = np.asarray(score_B)
    n = len(y_true)

    diffs = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        if len(np.unique(y_t)) < 2:
            continue
        m_a = metric_fn(y_t, score_A[idx])
        m_b = metric_fn(y_t, score_B[idx])
        diffs.append(m_a - m_b)

    diffs = np.array(diffs)
    alpha = 1 - ci_level
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))

    # Significant if CI excludes zero
    significant = (ci_lo > 0) or (ci_hi < 0)
    if significant:
        winner = "A" if mean_diff > 0 else "B"
    else:
        winner = "tie"

    return {
        "metric": metric_name,
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": significant,
        "winner": winner,
        "n_resamples": len(diffs),
    }

