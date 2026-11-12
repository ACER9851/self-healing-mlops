"""
splitter.py — Create train / val / test / future_pool splits.

Splitting strategy:
  - Fraud:       Stratified random split (preserves 0.17 % positive rate).
  - Elec2:       Chronological split (NEVER shuffle — row order IS the drift).
  - LendingClub: Chronological split (already sorted by issue_d in data_loader).

Usage:
    from harness.data_loader import load_dataset
    from harness.splitter import split_dataset

    ds = load_dataset("fraud")
    splits = split_dataset(ds, seed=42)
    # splits["train"]["X"], splits["train"]["y"], etc.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: dict,
    seed: int = 42,
    ratios: Optional[dict] = None,
) -> dict[str, dict]:
    """Split a dataset dict (from data_loader) into train/val/test/future_pool.

    Parameters
    ----------
    dataset : dict
        Output of ``load_dataset()`` with keys ``X``, ``y``, ``meta``, ``name``.
    seed : int
        Random seed for stratified splits. Ignored for chronological splits.
    ratios : dict, optional
        Override split ratios. Defaults to ``config.SPLIT_RATIOS``.

    Returns
    -------
    dict[str, dict]
        Keys: ``"train"``, ``"val"``, ``"test"``, ``"future_pool"``.
        Each value is a dict with ``"X"`` (DataFrame), ``"y"`` (Series),
        ``"indices"`` (ndarray of original row positions).
    """
    ratios = ratios or config.SPLIT_RATIOS
    name = dataset["name"]
    X = dataset["X"]
    y = dataset["y"]
    n = len(X)

    # Choose strategy
    drift_type = dataset["meta"]["drift_type"]
    if drift_type in ("real_temporal", "real_gradual"):
        # Chronological — keep row order
        logger.info(f"[{name}] Chronological split (n={n:,})")
        splits = _chronological_split(X, y, ratios)
    else:
        # Stratified random
        logger.info(f"[{name}] Stratified random split (n={n:,}, seed={seed})")
        splits = _stratified_split(X, y, ratios, seed)

    # Log split stats
    for split_name, split_data in splits.items():
        n_split = len(split_data["y"])
        pos_rate = split_data["y"].mean()
        logger.info(
            f"  {split_name:12s}: {n_split:>8,} rows  "
            f"pos_rate={pos_rate:.4f}  ({split_data['y'].sum():,} positive)"
        )

    return splits


def _chronological_split(
    X: pd.DataFrame, y: pd.Series, ratios: dict
) -> dict[str, dict]:
    """Split by row position (chronological order preserved)."""
    n = len(X)

    # Compute cumulative cut points
    cuts = {}
    cum = 0
    for split_name in ("train", "val", "test", "future_pool"):
        start = cum
        end = start + int(n * ratios[split_name])
        if split_name == "future_pool":
            end = n  # take remainder
        cuts[split_name] = (start, end)
        cum = end

    splits = {}
    for split_name, (start, end) in cuts.items():
        idx = np.arange(start, end)
        splits[split_name] = {
            "X": X.iloc[start:end].reset_index(drop=True),
            "y": y.iloc[start:end].reset_index(drop=True),
            "indices": idx,
        }

    return splits


def _stratified_split(
    X: pd.DataFrame, y: pd.Series, ratios: dict, seed: int
) -> dict[str, dict]:
    """Stratified random split that preserves class balance in each split."""
    rng = np.random.RandomState(seed)
    n = len(X)

    # Separate positive and negative indices
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]

    # Shuffle each class independently
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    splits = {}
    pos_offset = 0
    neg_offset = 0

    split_names = list(ratios.keys())
    for i, split_name in enumerate(split_names):
        ratio = ratios[split_name]

        if i == len(split_names) - 1:
            # Last split gets the remainder
            p_idx = pos_idx[pos_offset:]
            n_idx = neg_idx[neg_offset:]
        else:
            p_count = int(len(pos_idx) * ratio)
            n_count = int(len(neg_idx) * ratio)
            p_idx = pos_idx[pos_offset : pos_offset + p_count]
            n_idx = neg_idx[neg_offset : neg_offset + n_count]
            pos_offset += p_count
            neg_offset += n_count

        combined_idx = np.concatenate([p_idx, n_idx])
        # Sort to keep original ordering within each split (useful for streaming)
        combined_idx.sort()

        splits[split_name] = {
            "X": X.iloc[combined_idx].reset_index(drop=True),
            "y": y.iloc[combined_idx].reset_index(drop=True),
            "indices": combined_idx,
        }

    return splits


# ────────────────────────────────────────────────────────────────────
#  Windowed streaming (for drift detection / monitoring)
# ────────────────────────────────────────────────────────────────────

def window_stream(
    X: pd.DataFrame,
    y: pd.Series,
    window_size: int = None,
) -> list[dict]:
    """Yield non-overlapping windows from X, y.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    window_size : int
        Defaults to ``config.WINDOW_SIZE``.

    Returns
    -------
    list of dict, each with ``"X"``, ``"y"``, ``"window_id"``,
    ``"start_idx"``, ``"end_idx"``.
    """
    window_size = window_size or config.WINDOW_SIZE
    n = len(X)
    windows = []

    for i, start in enumerate(range(0, n, window_size)):
        end = min(start + window_size, n)
        if end - start < window_size // 2:
            # Skip tiny trailing window
            break
        windows.append({
            "X": X.iloc[start:end].reset_index(drop=True),
            "y": y.iloc[start:end].reset_index(drop=True),
            "window_id": i,
            "start_idx": start,
            "end_idx": end,
        })

    return windows

