"""Full-history retraining strategy for the comparison study."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from harness.model_wrapper import ModelWrapper, build_model


class FullHistoryRetrainer:
    """Keep every labeled observation and fit a fresh model on each retrain."""

    def __init__(self, model_key: str = "lr", seed: int = 42):
        self.model_key = model_key
        self.seed = seed
        self.model: ModelWrapper | None = None
        self.X_history: pd.DataFrame | None = None
        self.y_history: pd.Series | None = None
        self.fit_time_s = 0.0
        self.retrain_count = 0

    @property
    def n_samples(self) -> int:
        return 0 if self.y_history is None else len(self.y_history)

    def retrain(self, X_labeled, y_labeled) -> ModelWrapper:
        """Replace history with the supplied complete labeled history and fit."""
        X = _as_frame(X_labeled)
        y = _as_series(y_labeled, len(X))
        if len(X) == 0:
            raise ValueError("Cannot retrain on an empty labeled history")
        if len(X) != len(y):
            raise ValueError("X_labeled and y_labeled must have equal length")
        if y.nunique() < 2:
            raise ValueError("Retraining requires both target classes")

        model = build_model(self.model_key, seed=self.seed + self.retrain_count)
        started = time.perf_counter()
        model.fit(X, y)
        self.fit_time_s = time.perf_counter() - started
        self.X_history = X.reset_index(drop=True)
        self.y_history = y.reset_index(drop=True)
        self.model = model
        self.retrain_count += 1
        return model

    fit = retrain

    def predict(self, X):
        return self._require_model().predict(X)

    def predict_proba(self, X):
        return self._require_model().predict_proba(X)

    def _require_model(self) -> ModelWrapper:
        if self.model is None:
            raise RuntimeError("No fitted model. Call retrain() first.")
        return self.model


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X_labeled must be a two-dimensional array or DataFrame")
    return pd.DataFrame(array)


def _as_series(y, expected_length: int) -> pd.Series:
    values = np.asarray(y).ravel()
    if len(values) != expected_length:
        raise ValueError("X_labeled and y_labeled must have equal length")
    return pd.Series(values)
