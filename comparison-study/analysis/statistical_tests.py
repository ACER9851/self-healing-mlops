"""Statistical utilities for cross-part comparison results."""

from __future__ import annotations

import numpy as np


def _finite(values):
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def compute_bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Return a percentile bootstrap CI for the sample mean."""
    clean = _finite(values)
    if len(clean) == 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0}
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.RandomState(seed)
    samples = rng.choice(clean, size=(n_boot, len(clean)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "estimate": float(clean.mean()),
        "lower": float(np.percentile(means, 100.0 * alpha)),
        "upper": float(np.percentile(means, 100.0 * (1.0 - alpha))),
        "n": int(len(clean)),
    }


def paired_bootstrap_difference(series_A, series_B, n_boot=1000, ci=0.95, seed=42):
    """Estimate paired mean difference ``A - B`` with a bootstrap CI."""
    left = np.asarray(series_A, dtype=float).ravel()
    right = np.asarray(series_B, dtype=float).ravel()
    if len(left) != len(right):
        raise ValueError("paired series must have equal length")
    mask = np.isfinite(left) & np.isfinite(right)
    differences = left[mask] - right[mask]
    if len(differences) == 0:
        return {
            "mean_diff": None, "ci_lower": None, "ci_upper": None,
            "p_value": None, "significant": False, "n": 0,
        }
    rng = np.random.RandomState(seed)
    samples = rng.choice(differences, size=(n_boot, len(differences)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(means, 100.0 * alpha))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    p_left = float(np.mean(means <= 0.0))
    p_right = float(np.mean(means >= 0.0))
    p_value = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "mean_diff": float(differences.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "significant": bool(lower > 0.0 or upper < 0.0),
        "n": int(len(differences)),
    }
