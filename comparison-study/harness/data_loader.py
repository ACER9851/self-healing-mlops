"""
data_loader.py — Load and clean the three datasets.

Returns a standardised dict:
    {
        "X":       pd.DataFrame   (features only, numeric, no NaN),
        "y":       pd.Series      (binary 0/1 target),
        "meta":    dict            (from config.DATASET_META),
        "name":    str,
    }

Usage:
    from harness.data_loader import load_dataset
    ds = load_dataset("fraud")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────────────

def load_dataset(name: str) -> dict:
    """Load one of the three datasets by name.

    Parameters
    ----------
    name : str
        One of ``"fraud"``, ``"elec2"``, ``"lendingclub"``.

    Returns
    -------
    dict with keys ``X``, ``y``, ``meta``, ``name``.
    """
    if name not in config.DATASET_META:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from {config.DATASET_NAMES}."
        )

    loader = _LOADERS[name]
    X, y = loader()

    # Sanity checks
    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, pd.Series), "y must be a Series"
    assert len(X) == len(y), "X and y length mismatch"
    assert set(y.unique()).issubset({0, 1}), f"y has non-binary values: {y.unique()}"
    assert X.isna().sum().sum() == 0, f"X still has NaN:\n{X.isna().sum()[X.isna().sum()>0]}"

    pos_rate = y.mean()
    logger.info(
        f"[{name}] Loaded {len(X):,} rows, {X.shape[1]} features, "
        f"positive rate={pos_rate:.4f} ({y.sum():,}/{len(y):,})"
    )

    return {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True).astype(int),
        "meta": config.DATASET_META[name],
        "name": name,
    }


def load_all_datasets() -> dict[str, dict]:
    """Load all three datasets. Returns dict keyed by name."""
    results = {}
    for name in config.DATASET_NAMES:
        try:
            results[name] = load_dataset(name)
        except FileNotFoundError as e:
            logger.warning(f"[{name}] Skipped — file not found: {e}")
        except Exception as e:
            logger.error(f"[{name}] Failed to load: {e}")
            raise
    return results


# ────────────────────────────────────────────────────────────────────
#  Private loaders
# ────────────────────────────────────────────────────────────────────

def _load_fraud() -> tuple[pd.DataFrame, pd.Series]:
    """Load Credit-Card Fraud dataset from CSV.

    Raw columns: Time, V1..V28, Amount, Class
    - Keep V1..V28 and Amount as features.
    - Target: Class (already 0/1).
    - Drop Time (seconds from first transaction — not a real feature).
    """
    csv_path = config.DATASET_META["fraud"]["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Fraud CSV not found at {csv_path}")

    logger.info(f"Reading fraud CSV from {csv_path} ...")
    df = pd.read_csv(csv_path)

    target_col = config.DATASET_META["fraud"]["target_col"]  # "Class"
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Amount can have outliers; keep raw (no scaling here — model_wrapper handles that)
    # No NaN expected in this dataset
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)

    y = y.astype(int)
    return X, y


def _load_elec2() -> tuple[pd.DataFrame, pd.Series]:
    """Load Elec2 dataset.

    Priority:
    1. CSV at config.ELEC2_CSV (if user downloaded from Colab)
    2. Generate from ``river`` library

    Features (7 numeric): date, day, period, nswprice, nswdemand, vicprice, vicdemand
    Target: UP/DOWN → 1/0
    IMPORTANT: row order IS the temporal signal — NEVER shuffle.
    """
    csv_path = config.DATASET_META["elec2"]["csv"]

    if csv_path.exists():
        logger.info(f"Reading elec2 CSV from {csv_path} ...")
        df = pd.read_csv(csv_path)
        return _parse_elec2_df(df)

    # Fallback: generate from river
    logger.info("Elec2 CSV not found, generating from river library ...")
    return _generate_elec2_from_river()


def _parse_elec2_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Parse an elec2 DataFrame (from CSV or river)."""

    # Detect target column name
    if "target" in df.columns:
        target_col = "target"
    elif "class" in df.columns:
        target_col = "class"
    else:
        raise KeyError(f"No target column in elec2. Columns: {list(df.columns)}")

    # Feature columns: everything numeric except target and split
    drop_cols = {target_col, "split"}
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and df[c].dtype in ("float64", "int64", "float32", "int32")
    ]

    if not feature_cols:
        # Try converting
        for c in df.columns:
            if c not in drop_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and df[c].dtype in ("float64", "int64", "float32", "int32")
        ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode target if string
    if y.dtype == object:
        mapping = {"UP": 1, "DOWN": 0, "1": 1, "0": 0}
        y = y.map(mapping)
        if y.isna().any():
            raise ValueError(f"Unmapped target values in elec2: {df[target_col].unique()}")

    y = y.astype(int)
    X = X.fillna(X.median())
    return X, y


def _generate_elec2_from_river() -> tuple[pd.DataFrame, pd.Series]:
    """Generate Elec2 from river library and save to CSV."""
    try:
        from river.datasets import Elec2
    except ImportError:
        raise ImportError(
            "river library not installed. Install it with: pip install river\n"
            "Or provide elec2_clean.csv at " + str(config.ELEC2_CSV)
        )

    dataset = Elec2()
    rows = []
    targets = []
    for x, y_val in dataset:
        rows.append(x)
        targets.append(int(y_val))

    df = pd.DataFrame(rows)
    df["target"] = targets

    # Save for next time
    csv_path = config.DATASET_META["elec2"]["csv"]
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved elec2 CSV to {csv_path}")

    return _parse_elec2_df(df)


def _load_lendingclub() -> tuple[pd.DataFrame, pd.Series]:
    """Load LendingClub dataset from the raw 151-column CSV.

    Steps:
    1. Filter to Fully Paid / Charged Off loans only.
    2. Create binary target: Charged Off = 1, Fully Paid = 0.
    3. Parse issue_d for chronological ordering.
    4. Select curated numeric features (LENDINGCLUB_FEATURES from config).
    5. Parse string-encoded numerics (term, int_rate, revol_util).
    6. Drop rows with NaN, or impute where safe.
    7. Sort by issue_d (temporal order).
    """
    csv_path = config.DATASET_META["lendingclub"]["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"LendingClub CSV not found at {csv_path}")

    logger.info(f"Reading LendingClub CSV from {csv_path} ...")
    # Read only the columns we need + loan_status + issue_d to save memory
    usecols = config.LENDINGCLUB_FEATURES + ["loan_status", "issue_d"]
    # Some columns might not exist; read all and filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(csv_path, low_memory=False)

    logger.info(f"  Raw shape: {df.shape}")

    # Step 1: filter to terminal loan statuses
    valid_statuses = ["Fully Paid", "Charged Off"]
    mask = df["loan_status"].isin(valid_statuses)
    df = df[mask].copy()
    logger.info(f"  After status filter: {len(df):,} rows")

    # Step 2: create target
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)

    # Step 3: parse issue_d for ordering
    # Format: "Dec-2011" or "2011-12-01"
    df["issue_date_parsed"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
    df = df.sort_values("issue_date_parsed").reset_index(drop=True)

    # Step 4: select features
    feature_cols = []
    for col in config.LENDINGCLUB_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
        else:
            logger.warning(f"  Feature '{col}' not found in CSV, skipping.")

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Step 5: parse string-encoded numerics
    X = _parse_lendingclub_strings(X)

    # Step 6: convert all to numeric, impute
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Impute with median (safer than dropping for 1.6 GB dataset)
    na_counts = X.isna().sum()
    na_cols = na_counts[na_counts > 0]
    if len(na_cols) > 0:
        logger.info(f"  Imputing NaN in {len(na_cols)} columns: {list(na_cols.index)}")
        X = X.fillna(X.median())

    # Final check: drop any remaining NaN rows
    valid_mask = ~X.isna().any(axis=1)
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        logger.warning(f"  Dropping {n_drop} rows with remaining NaN")
        X = X[valid_mask]
        y = y[valid_mask]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)

    return X, y


def _parse_lendingclub_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Parse string-encoded numeric columns in LendingClub data.

    - term: " 36 months" → 36
    - int_rate: "10.65%" → 10.65
    - revol_util: "50.5%" → 50.5
    """
    if "term" in X.columns:
        X["term"] = (
            X["term"]
            .astype(str)
            .str.replace("months", "", regex=False)
            .str.strip()
        )
        X["term"] = pd.to_numeric(X["term"], errors="coerce")

    for col in ("int_rate", "revol_util"):
        if col in X.columns:
            X[col] = (
                X[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


# ────────────────────────────────────────────────────────────────────
#  Loader dispatch
# ────────────────────────────────────────────────────────────────────

_LOADERS = {
    "fraud":       _load_fraud,
    "elec2":       _load_elec2,
    "lendingclub": _load_lendingclub,
}

