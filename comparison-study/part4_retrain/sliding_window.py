"""Sliding-window retraining strategy for the comparison study."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from harness.model_wrapper import ModelWrapper, build_model


class SlidingWindowRetrainer:
    """Retain only the newest ``window_size`` labeled observations."""

    def __init__(self, model_key: str = "lr", window_size: int = 5000, seed: int = 42):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.model_key = model_key
        self.window_size = window_size
        self.seed = seed
        self.model: ModelWrapper | None = None
        self.X_buffer = pd.DataFrame()
        self.y_buffer = pd.Series(dtype=float)
        self.fit_time_s = 0.0
        self.retrain_count = 0

    @property
    def n_samples(self) -> int:
        return len(self.y_buffer)

    def append(self, X_labeled, y_labeled) -> None:
        """Append labeled observations and discard the oldest overflow."""
        X = _as_frame(X_labeled)
        y = _as_series(y_labeled, len(X))
        if len(X) == 0:
            return
        if self.n_samples == 0:
            self.X_buffer = X.iloc[-self.window_size :].reset_index(drop=True)
            self.y_buffer = y.iloc[-self.window_size :].reset_index(drop=True)
            return
        if self.n_samples and list(X.columns) != list(self.X_buffer.columns):
            raise ValueError("Feature columns must stay consistent across appends")
        combined_X = pd.concat([self.X_buffer, X], ignore_index=True)
        combined_y = pd.concat([self.y_buffer, y], ignore_index=True)
        self.X_buffer = combined_X.iloc[-self.window_size :].reset_index(drop=True)
        self.y_buffer = combined_y.iloc[-self.window_size :].reset_index(drop=True)

    def retrain(self, X_labeled=None, y_labeled=None) -> ModelWrapper:
        """Append an optional batch, then fit on the current bounded buffer."""
        if X_labeled is not None or y_labeled is not None:
            if X_labeled is None or y_labeled is None:
                raise ValueError("X_labeled and y_labeled must be supplied together")
            self.append(X_labeled, y_labeled)
        if self.n_samples == 0:
            raise ValueError("Cannot retrain with an empty buffer")
        if self.y_buffer.nunique() < 2:
            raise ValueError("Retraining requires both target classes")

        model = build_model(self.model_key, seed=self.seed + self.retrain_count)
        started = time.perf_counter()
        model.fit(self.X_buffer, self.y_buffer)
        self.fit_time_s = time.perf_counter() - started
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
