"""
model_wrapper.py — Unified interface for all 11 classifiers.

Handles:
  - Dynamic import (graceful skip if xgboost/lightgbm not installed).
  - StandardScaler preprocessing (KNN, SVM need it; others don't hurt).
  - Unified train / predict / predict_proba API.
  - Clone for fresh training (retraining experiments).

Usage:
    from harness.model_wrapper import ModelWrapper, build_model, list_available_models

    model = build_model("lr", seed=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
"""

from __future__ import annotations

import importlib
import logging
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Availability check
# ────────────────────────────────────────────────────────────────────

_UNAVAILABLE: dict[str, str] = {}   # model_key → reason


def _check_available(model_key: str) -> bool:
    """Check if the model's dependencies are importable."""
    entry = config.MODEL_REGISTRY[model_key]
    module_path = entry["class"].rsplit(".", 1)[0]
    try:
        importlib.import_module(module_path)
        return True
    except ImportError as e:
        _UNAVAILABLE[model_key] = str(e)
        return False


def list_available_models(tier: str = None) -> list[str]:
    """Return model keys whose dependencies are installed.

    Parameters
    ----------
    tier : str, optional
        Filter to ``"core"``, ``"diversity"``, or ``"lightweight"``.
    """
    keys = config.ALL_MODELS if tier is None else [
        k for k, v in config.MODEL_REGISTRY.items() if v["tier"] == tier
    ]
    available = [k for k in keys if _check_available(k)]
    skipped = [k for k in keys if k not in available]
    if skipped:
        logger.warning(
            f"Models unavailable (missing library): {skipped}. "
            f"Install xgboost / lightgbm to enable them."
        )
    return available


# ────────────────────────────────────────────────────────────────────
#  ModelWrapper
# ────────────────────────────────────────────────────────────────────

class ModelWrapper:
    """Unified wrapper around any sklearn-API classifier.

    Adds:
      - StandardScaler pre-processing.
      - Consistent predict_proba (returns P(y=1) as 1-D array).
      - Timing for fit / predict.
      - Clone method for retraining experiments.
    """

    def __init__(self, key: str, clf, name: str, tier: str, seed: int = 42):
        self.key = key
        self.name = name
        self.tier = tier
        self.seed = seed
        self._clf = clf
        self._scaler = StandardScaler()
        self._is_fitted = False
        self.fit_time_s: float = 0.0
        self.predict_time_s: float = 0.0

    def __repr__(self):
        fitted = "fitted" if self._is_fitted else "not fitted"
        return f"ModelWrapper({self.key!r}, {self.name!r}, {fitted})"

    # ── Fit ─────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        """Fit the scaler + classifier."""
        X_arr = self._to_array(X)
        y_arr = np.asarray(y).ravel()

        # Scale
        X_scaled = self._scaler.fit_transform(X_arr)

        # Fit classifier
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(X_scaled, y_arr)
        self.fit_time_s = time.perf_counter() - t0
        self._is_fitted = True

        logger.debug(
            f"[{self.key}] fit done in {self.fit_time_s:.2f}s on {len(y_arr):,} samples"
        )
        return self

    # ── Predict ─────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return binary predictions (0/1)."""
        self._check_fitted()
        X_scaled = self._scaler.transform(self._to_array(X))

        t0 = time.perf_counter()
        preds = self._clf.predict(X_scaled)
        self.predict_time_s = time.perf_counter() - t0
        return preds.astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return P(y=1) as a 1-D array.

        Falls back to decision_function → sigmoid for models without
        predict_proba (e.g., Perceptron, base SGD without loss='log_loss').
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(self._to_array(X))

        t0 = time.perf_counter()
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X_scaled)
            # Some classifiers return (n, 2), we want column 1
            if proba.ndim == 2:
                proba = proba[:, 1]
        elif hasattr(self._clf, "decision_function"):
            # Sigmoid fallback
            scores = self._clf.decision_function(X_scaled)
            proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            # Last resort: hard predictions as probabilities
            logger.warning(
                f"[{self.key}] No predict_proba or decision_function. "
                f"Using hard predictions as probabilities."
            )
            proba = self._clf.predict(X_scaled).astype(float)

        self.predict_time_s = time.perf_counter() - t0
        return np.asarray(proba).ravel()

    # ── Clone ───────────────────────────────────────────────────────

    def clone(self, new_seed: int = None) -> "ModelWrapper":
        """Create a fresh (unfitted) copy with optionally a new seed."""
        seed = new_seed if new_seed is not None else self.seed
        return build_model(self.key, seed=seed)

    # ── Helpers ─────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.key!r} not fitted. Call .fit() first.")

    @staticmethod
    def _to_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)


# ────────────────────────────────────────────────────────────────────
#  Factory
# ────────────────────────────────────────────────────────────────────

def build_model(key: str, seed: int = 42) -> ModelWrapper:
    """Build a fresh (unfitted) ModelWrapper by registry key.

    Parameters
    ----------
    key : str
        One of the keys in ``config.MODEL_REGISTRY`` (e.g. ``"lr"``, ``"rf"``).
    seed : int
        Random seed injected into the classifier if it accepts ``random_state``.

    Returns
    -------
    ModelWrapper
    """
    if key not in config.MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{key}'. Choose from {config.ALL_MODELS}."
        )

    entry = config.MODEL_REGISTRY[key]
    class_path = entry["class"]
    kwargs = dict(entry["kwargs"])  # copy so we don't mutate config

    # Resolve class
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import {module_path} for model '{key}': {e}\n"
            f"Install the missing package to use this model."
        ) from e

    cls = getattr(module, class_name)

    # Inject random_state if the classifier supports it
    import inspect
    sig = inspect.signature(cls.__init__)
    if "random_state" in sig.parameters:
        kwargs.setdefault("random_state", seed)

    # Instantiate
    clf = cls(**kwargs)

    return ModelWrapper(
        key=key,
        clf=clf,
        name=entry["name"],
        tier=entry["tier"],
        seed=seed,
    )


def build_all_available(
    seed: int = 42,
    tier: str = None,
) -> dict[str, ModelWrapper]:
    """Build all available models, skipping those with missing dependencies.

    Parameters
    ----------
    seed : int
        Shared seed for reproducibility.
    tier : str, optional
        ``"core"``, ``"diversity"``, ``"lightweight"``, or ``None`` for all.

    Returns
    -------
    dict mapping model key → ModelWrapper.
    """
    keys = list_available_models(tier=tier)
    models = {}
    for key in keys:
        try:
            models[key] = build_model(key, seed=seed)
            logger.info(f"  Built {key}: {models[key].name}")
        except Exception as e:
            logger.warning(f"  Failed to build {key}: {e}")
    return models

