"""
dataset_wrapper.py — Dataset abstraction mirroring ModelWrapper (Part 4B).

Provides uniform data loading, splitting, feature typing, and preprocessing
across different dataset profiles. Config selects the active profile.

Profile 1 (tabular_fraud): Credit Card Fraud Detection — 30 features, binary label
Profile 2 (image_fashion_mnist): Fashion-MNIST binarized to 2 classes — stub for Step 9
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import DATASET_PROFILE, DATA_DIR


class DatasetWrapper:
    """
    Uniform dataset interface across profiles.

    Usage:
        ds = DatasetWrapper.create()
        train, val, test, future = ds.load()
        feature_types = ds.get_feature_types()
    """

    def __init__(self, profile: str):
        self._profile = profile
        self._feature_names = None
        self._feature_types = None

    @staticmethod
    def create(profile: str = None) -> "DatasetWrapper":
        """Factory: return the right wrapper for the active profile."""
        profile = profile or DATASET_PROFILE
        if profile == "tabular_fraud":
            return TabularFraudDataset(profile)
        elif profile == "image_fashion_mnist":
            return ImageFashionMNISTDataset(profile)
        else:
            raise ValueError(f"Unknown DATASET_PROFILE: '{profile}'")

    def load(self) -> Tuple[dict, dict, dict, dict]:
        """
        Return 4 splits, each as a dict with keys 'X' and 'y':
          train_initial (40%), validation (10%), test_clean (10%), future_pool (40%)
        """
        raise NotImplementedError

    def get_feature_types(self) -> Dict[str, str]:
        """
        Return {feature_name: 'continuous' | 'categorical'} map.
        Used by drift detection (tabular profile) to pick the right stat test.
        """
        raise NotImplementedError

    def get_feature_names(self) -> list:
        """Return ordered list of feature names."""
        raise NotImplementedError

    def get_input_shape(self) -> tuple:
        """Return input tensor shape (used by DL models in image profile)."""
        raise NotImplementedError

    def preprocess(self, batch: np.ndarray) -> np.ndarray:
        """Apply profile-specific normalization/reshaping to a batch."""
        raise NotImplementedError


class TabularFraudDataset(DatasetWrapper):
    """
    Profile 1 — Credit Card Fraud Detection (Kaggle).
    284,807 rows, 30 features (V1–V28 + Amount + Time), binary label (Class).
    """

    CSV_PATH = DATA_DIR / "tabular_fraud" / "creditcard.csv"

    def load(self) -> Tuple[dict, dict, dict, dict]:
        if not self.CSV_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.CSV_PATH}. "
                f"Download from Kaggle and place creditcard.csv in {self.CSV_PATH.parent}/"
            )

        df = pd.read_csv(str(self.CSV_PATH))

        # Separate features and label
        X = df.drop(columns=["Class"]).values.astype(np.float32)
        y = df["Class"].values.astype(np.int32)
        self._feature_names = [c for c in df.columns if c != "Class"]

        # Build feature type map — all continuous for this dataset
        self._feature_types = {name: "continuous" for name in self._feature_names}

        # Shuffle deterministically
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        # 4-way split: 40% train, 10% val, 10% test, 40% future
        n = len(X)
        cuts = [int(0.4 * n), int(0.5 * n), int(0.6 * n)]

        train_initial = {"X": X[:cuts[0]], "y": y[:cuts[0]]}
        validation    = {"X": X[cuts[0]:cuts[1]], "y": y[cuts[0]:cuts[1]]}
        test_clean    = {"X": X[cuts[1]:cuts[2]], "y": y[cuts[1]:cuts[2]]}
        future_pool   = {"X": X[cuts[2]:], "y": y[cuts[2]:]}

        return train_initial, validation, test_clean, future_pool

    def get_feature_types(self) -> Dict[str, str]:
        if self._feature_types is None:
            # Load just columns to build map without full data load
            df_cols = pd.read_csv(str(self.CSV_PATH), nrows=0)
            self._feature_names = [c for c in df_cols.columns if c != "Class"]
            self._feature_types = {name: "continuous" for name in self._feature_names}
        return self._feature_types

    def get_feature_names(self) -> list:
        if self._feature_names is None:
            self.get_feature_types()
        return self._feature_names

    def get_input_shape(self) -> tuple:
        return (30,)  # 30 features

    def preprocess(self, batch: np.ndarray) -> np.ndarray:
        """Tabular data: ensure float32, no reshape needed."""
        return np.asarray(batch, dtype=np.float32)


class ImageFashionMNISTDataset(DatasetWrapper):
    """
    Profile 2 — Fashion-MNIST binarized to 2 classes.
    Stub for Step 9 implementation.
    """

    def load(self) -> Tuple[dict, dict, dict, dict]:
        raise NotImplementedError(
            "Fashion-MNIST dataset loading will be implemented in Step 9 (Profile 2). "
            "Use DATASET_PROFILE='tabular_fraud' for now."
        )

    def get_feature_types(self) -> Dict[str, str]:
        # Image data — per-pixel types not meaningful (Part 4B)
        return {"pixels": "image_embedding"}

    def get_feature_names(self) -> list:
        return [f"pixel_{i}" for i in range(784)]

    def get_input_shape(self) -> tuple:
        return (1, 28, 28)  # (channels, height, width)

    def preprocess(self, batch: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] and reshape for CNN."""
        batch = np.asarray(batch, dtype=np.float32) / 255.0
        if batch.ndim == 2:
            # Flatten → (N, 1, 28, 28)
            batch = batch.reshape(-1, 1, 28, 28)
        return batch


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    ds = DatasetWrapper.create()
    print(f"Profile: {ds._profile}")
    print(f"Input shape: {ds.get_input_shape()}")
    try:
        train, val, test, future = ds.load()
        print(f"✓ Loaded: train={train['X'].shape}, val={val['X'].shape}, "
              f"test={test['X'].shape}, future={future['X'].shape}")
        print(f"  Feature types: {len(ds.get_feature_types())} features, "
              f"all {list(set(ds.get_feature_types().values()))}")
    except FileNotFoundError as e:
        print(f"⚠ {e}")
