"""
model_wrapper.py — Framework-agnostic model abstraction (Part 4).

Pipeline code NEVER touches the raw model directly — always through this wrapper.
Switching frameworks = changing only config.py (MODEL_TYPE), never pipeline code.

Supports: scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (via skorch), TF/Keras (via scikeras).
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import MODEL_TYPE, MODEL_PARAMS, MODELS_DIR, DATASET_PROFILE


class ModelWrapper:
    """
    Uniform interface over any supported ML framework.

    Usage:
        wrapper = ModelWrapper.create(model_type="xgboost", params={...})
        wrapper.train(X, y)
        preds = wrapper.predict(X)
        wrapper.save("models/v1.pkl")
        wrapper = ModelWrapper.load("models/v1.pkl")
    """

    def __init__(self, model, model_type: str):
        self._model = model
        self._model_type = model_type

    # ──────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────
    @staticmethod
    def create(model_type: str = None, params: dict = None) -> "ModelWrapper":
        """Create a new, untrained model wrapper from config."""
        model_type = model_type or MODEL_TYPE
        params = params or MODEL_PARAMS.copy()

        if model_type == "sklearn":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif model_type == "sklearn_lr":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**params, random_state=42, max_iter=1000)

        elif model_type == "xgboost":
            from xgboost import XGBClassifier
            xgb_params = params.copy()
            xgb_params.setdefault("use_label_encoder", False)
            xgb_params.setdefault("eval_metric", "logloss")
            xgb_params.setdefault("random_state", 42)
            model = XGBClassifier(**xgb_params)

        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            lgb_params = params.copy()
            lgb_params.setdefault("random_state", 42)
            lgb_params.setdefault("verbose", -1)
            model = LGBMClassifier(**lgb_params)

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            cb_params = params.copy()
            cb_params.setdefault("random_seed", 42)
            cb_params.setdefault("verbose", 0)
            model = CatBoostClassifier(**cb_params)

        elif model_type == "pytorch_cnn":
            # Profile 2 — CNN via skorch (Step 9 full implementation)
            raise NotImplementedError(
                "PyTorch CNN wrapper will be implemented in Step 9 (Profile 2). "
                "Use a tabular model type for now."
            )

        elif model_type == "tf_cnn":
            # Profile 2 — CNN via scikeras (Step 9 full implementation)
            raise NotImplementedError(
                "TensorFlow CNN wrapper will be implemented in Step 9 (Profile 2). "
                "Use a tabular model type for now."
            )

        else:
            raise ValueError(
                f"Unknown MODEL_TYPE: '{model_type}'. "
                f"Supported: sklearn, sklearn_lr, xgboost, lightgbm, catboost, "
                f"pytorch_cnn, tf_cnn"
            )

        return ModelWrapper(model, model_type)

    # ──────────────────────────────────────────
    # Core interface
    # ──────────────────────────────────────────
    def train(self, X, y, sample_weight=None):
        """Train/retrain the model. sample_weight used by weighted_retrain strategy."""
        fit_params = {}
        if sample_weight is not None:
            # CatBoost uses a different parameter name in some versions
            if self._model_type == "catboost":
                # CatBoost's fit accepts sample_weight directly
                fit_params["sample_weight"] = sample_weight
            elif self._model_type == "xgboost":
                fit_params["sample_weight"] = sample_weight
            else:
                fit_params["sample_weight"] = sample_weight

        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X) -> np.ndarray:
        """Return class predictions."""
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return probability estimates. Shape: (n_samples, n_classes)."""
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        else:
            # Fallback: convert predictions to one-hot-ish probabilities
            preds = self.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds.astype(int)] = 1.0
            return proba

    def get_framework_name(self) -> str:
        """Return the framework identifier string."""
        return self._model_type

    def get_raw_model(self):
        """Access the underlying model — use only for adversarial testing (gradient access)."""
        return self._model

    # ──────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────
    def save(self, path: str):
        """Save the model + metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self._model,
            "model_type": self._model_type,
        }

        if self._model_type in ("pytorch_cnn", "tf_cnn"):
            # Framework-native save for DL models (Step 9)
            raise NotImplementedError("DL model serialization — Step 9")
        else:
            joblib.dump(save_data, str(path))

    @staticmethod
    def load(path: str) -> "ModelWrapper":
        """Load a previously saved model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data = joblib.load(str(path))
        return ModelWrapper(
            model=save_data["model"],
            model_type=save_data["model_type"],
        )


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ModelWrapper factory...")
    for mt in ["sklearn", "xgboost", "lightgbm", "catboost"]:
        try:
            w = ModelWrapper.create(model_type=mt, params={"n_estimators": 10, "max_depth": 3}
                                   if mt != "catboost" else {"iterations": 10, "depth": 3})
            print(f"  ✓ {mt}: {w.get_framework_name()} — {type(w.get_raw_model()).__name__}")
        except Exception as e:
            print(f"  ✗ {mt}: {e}")
    print("Done.")
