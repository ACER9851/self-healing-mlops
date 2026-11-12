"""
runner.py — High-level experiment runner.

Orchestrates: load data → split → train model → evaluate → save results.

This is the single entry point the Part 1/2/3/4 scripts call to get a
trained model + metrics on any dataset.

Usage:
    from harness.runner import run_baseline_experiment, run_all_baselines

    result = run_baseline_experiment("fraud", "lr", seed=42)
    all_results = run_all_baselines(seed=42)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_dataset
from .splitter import split_dataset
from .model_wrapper import build_model, list_available_models, ModelWrapper
from .metrics import evaluate

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Single experiment
# ────────────────────────────────────────────────────────────────────

def run_baseline_experiment(
    dataset_name: str,
    model_key: str,
    seed: int = 42,
    bootstrap: bool = True,
    return_model: bool = False,
) -> dict:
    """Train one model on one dataset and evaluate on test split.

    Parameters
    ----------
    dataset_name : str
        ``"fraud"``, ``"elec2"``, or ``"lendingclub"``.
    model_key : str
        Key from ``config.MODEL_REGISTRY`` (e.g. ``"lr"``, ``"rf"``).
    seed : int
        Controls both the split randomisation and the model random_state.
    bootstrap : bool
        Whether to compute bootstrap CIs (adds ~10 s per run).
    return_model : bool
        If True, include the fitted ModelWrapper in the result dict.

    Returns
    -------
    dict with keys:
        dataset, model_key, model_name, seed, metrics (from evaluate()),
        fit_time_s, predict_time_s, n_train, n_test
    """
    logger.info(f"=== Experiment: {dataset_name} × {model_key} (seed={seed}) ===")

    # Load & split
    ds = load_dataset(dataset_name)
    splits = split_dataset(ds, seed=seed)

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_test,  y_test  = splits["test"]["X"],  splits["test"]["y"]

    # Build & train
    model = build_model(model_key, seed=seed)
    model.fit(X_train, y_train)

    # Predict
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Evaluate
    primary_metric = ds["meta"]["primary_metric"]
    metrics = evaluate(
        y_test.values, y_pred, y_proba,
        primary=primary_metric,
        bootstrap=bootstrap,
        seed=seed,
    )

    result = {
        "dataset": dataset_name,
        "model_key": model_key,
        "model_name": model.name,
        "tier": model.tier,
        "seed": seed,
        "metrics": metrics,
        "fit_time_s": round(model.fit_time_s, 4),
        "predict_time_s": round(model.predict_time_s, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    if return_model:
        result["model"] = model
        result["splits"] = splits

    logger.info(
        f"  {model.name}: {primary_metric}={metrics['primary_value']:.4f} "
        f"CI={metrics['primary_ci']}  fit={model.fit_time_s:.2f}s"
    )

    return result


# ────────────────────────────────────────────────────────────────────
#  Batch: all models × one dataset
# ────────────────────────────────────────────────────────────────────

def run_all_models_on_dataset(
    dataset_name: str,
    seed: int = 42,
    tier: str = None,
    bootstrap: bool = True,
) -> list[dict]:
    """Run all available models on a single dataset.

    Parameters
    ----------
    dataset_name : str
    seed : int
    tier : str, optional
        ``"core"``, ``"diversity"``, ``"lightweight"``, or None for all.
    bootstrap : bool

    Returns
    -------
    list of result dicts (one per model).
    """
    model_keys = list_available_models(tier=tier)
    results = []
    for key in model_keys:
        try:
            r = run_baseline_experiment(
                dataset_name, key, seed=seed, bootstrap=bootstrap
            )
            results.append(r)
        except Exception as e:
            logger.error(f"  FAILED: {key} on {dataset_name}: {e}")
            results.append({
                "dataset": dataset_name,
                "model_key": key,
                "model_name": config.MODEL_REGISTRY[key]["name"],
                "tier": config.MODEL_REGISTRY[key]["tier"],
                "seed": seed,
                "metrics": None,
                "error": str(e),
            })
    return results


# ────────────────────────────────────────────────────────────────────
#  Batch: all models × all datasets
# ────────────────────────────────────────────────────────────────────

def run_all_baselines(
    seed: int = 42,
    tier: str = None,
    bootstrap: bool = True,
    datasets: list[str] = None,
) -> dict[str, list[dict]]:
    """Run all available models on all (or selected) datasets.

    Returns
    -------
    dict mapping dataset_name → list of result dicts.
    """
    datasets = datasets or config.DATASET_NAMES
    all_results = {}
    for ds_name in datasets:
        try:
            all_results[ds_name] = run_all_models_on_dataset(
                ds_name, seed=seed, tier=tier, bootstrap=bootstrap
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            all_results[ds_name] = []
    return all_results


# ────────────────────────────────────────────────────────────────────
#  Multi-seed aggregate
# ────────────────────────────────────────────────────────────────────

def run_multi_seed(
    dataset_name: str,
    model_key: str,
    seeds: list[int] = None,
    bootstrap: bool = False,
) -> dict:
    """Run one model on one dataset over multiple seeds and aggregate.

    Returns
    -------
    dict with per-seed results and aggregated mean ± std.
    """
    seeds = seeds or config.FINAL_SEEDS
    per_seed = []
    for s in seeds:
        r = run_baseline_experiment(
            dataset_name, model_key, seed=s, bootstrap=bootstrap
        )
        per_seed.append(r)

    # Aggregate primary metric
    vals = [r["metrics"]["primary_value"] for r in per_seed if r["metrics"] is not None]
    agg = {
        "dataset": dataset_name,
        "model_key": model_key,
        "model_name": per_seed[0]["model_name"],
        "seeds": seeds,
        "n_runs": len(vals),
        "primary_metric": per_seed[0]["metrics"]["primary_metric"],
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "std":  float(np.std(vals))  if vals else float("nan"),
        "min":  float(np.min(vals))  if vals else float("nan"),
        "max":  float(np.max(vals))  if vals else float("nan"),
        "per_seed": per_seed,
    }
    return agg


# ────────────────────────────────────────────────────────────────────
#  Save / load results
# ────────────────────────────────────────────────────────────────────

def save_results(results: dict | list, filename: str):
    """Save results to JSON in the results directory."""
    path = config.RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serialisable
    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, ModelWrapper):
            return f"<ModelWrapper:{obj.key}>"
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_default)

    logger.info(f"Results saved to {path}")


def load_results(filename: str) -> dict | list:
    """Load results from JSON."""
    path = config.RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────
#  Pretty-print table
# ────────────────────────────────────────────────────────────────────

def results_to_table(results: list[dict]) -> str:
    """Format a list of result dicts as a readable table.

    Parameters
    ----------
    results : list of dict
        Each from ``run_baseline_experiment()``.

    Returns
    -------
    str  (printable table)
    """
    if not results:
        return "(no results)"

    lines = []
    header = (
        f"{'Model':<22s} {'Tier':<12s} "
        f"{'PR-AUC':>8s} {'F1':>8s} {'ROC-AUC':>8s} "
        f"{'Fit(s)':>8s} {'n_train':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        m = r.get("metrics")
        if m is None:
            lines.append(f"{r['model_name']:<22s} {r.get('tier','?'):<12s}  ERROR: {r.get('error','?')}")
            continue

        ci = m.get("pr_auc_ci", (None, None))
        ci_str = ""
        if ci[0] is not None and not np.isnan(ci[0]):
            ci_str = f" [{ci[0]:.3f},{ci[1]:.3f}]"

        lines.append(
            f"{r['model_name']:<22s} {r['tier']:<12s} "
            f"{m['pr_auc']:>8.4f}{ci_str} "
            f"{m['f1']:>8.4f} {m['roc_auc']:>8.4f} "
            f"{r['fit_time_s']:>8.2f} {r['n_train']:>8,d}"
        )

    return "\n".join(lines)

