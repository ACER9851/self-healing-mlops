"""
Centralized configuration for the Self-Healing MLOps Architecture.
All tunable values live here (Fix #17) — no magic numbers scattered across pipeline code.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATA_DIR / "mlops.db"

# ──────────────────────────────────────────────
# Active profile selection
# ──────────────────────────────────────────────
DATASET_PROFILE = "tabular_fraud"          # or "image_fashion_mnist"
MODEL_TYPE = "xgboost"                     # sklearn | xgboost | lightgbm | catboost | pytorch_cnn | tf_cnn
MODEL_PARAMS = {"n_estimators": 100, "max_depth": 6}

# ──────────────────────────────────────────────
# Buffer / drift detection (Fix #17)
# ──────────────────────────────────────────────
BUFFER_SIZE = 500
DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFT_MIN_FEATURES_FOR_FDR = 3             # Fix #1: require >=N features flagged, not just any 1
DRIFT_LEVEL_THRESHOLDS = {
    "low": 0.05,
    "medium": 0.01,
    "high": 0.001,
}

# Post-deployment settling window (Fix #13):
# skip drift detection for this many buffer cycles after a deployment
SETTLING_WINDOW_CYCLES = 1

# ──────────────────────────────────────────────
# Adversarial testing
# ──────────────────────────────────────────────
ADVERSARIAL_EPSILON = 0.1
ADVERSARIAL_VULNERABLE_RATIO = 0.7         # flag VULNERABLE if adv_acc < 0.7 * clean_acc

# ──────────────────────────────────────────────
# Q-learning (Fix #4)
# ──────────────────────────────────────────────
QTABLE_STATES = 18                         # 3 drift_level x 2 adversarial x 3 time_since_retrain
QTABLE_ACTIONS = 6                         # defer, full_retrain, sliding_window, weighted_retrain,
                                           # model_switch, adversarial_hardening (Fix #7)
ACTION_NAMES = [
    "defer",
    "full_retrain",
    "sliding_window",
    "weighted_retrain",
    "model_switch",
    "adversarial_hardening",
]

EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_FLOOR = 0.1
EPSILON_LIVE = 0.05                        # frozen low epsilon once in production (Fix #4)
LEARNING_RATE_ALPHA = 0.1
DISCOUNT_GAMMA = 0.95
CONFIDENCE_THRESHOLD_REAL_VISITS = 5       # Fix #4 sub-fix: real_visits only, never synthetic_visits
REAL_TRANSITION_REPLAY_EVERY_N = 50        # periodic_retrain_scheduler.py trigger interval

# ──────────────────────────────────────────────
# Validation gates (Fix #3: F1-based, not accuracy)
# ──────────────────────────────────────────────
GATE1_MAX_PVALUE = 0.01
GATE2_MAX_F1_DROP = 0.02
GATE3_MAX_ADV_DROP = 0.05

# ──────────────────────────────────────────────
# Deployment (Fix #9)
# ──────────────────────────────────────────────
CANARY_TRAFFIC_PERCENT = 10
CANARY_MIN_REQUESTS = 100                  # N requests to observe before promoting to 100%

# ──────────────────────────────────────────────
# Auth (Fix #16)
# ──────────────────────────────────────────────
API_KEY = os.environ.get("MLOPS_API_KEY", "dev-only-key-change-me")

# ──────────────────────────────────────────────
# Critical vs. auto-deployable models
# ──────────────────────────────────────────────
CRITICAL_MODELS = []                       # always ask human, ignore confidence threshold
AUTO_DEPLOY_MODELS = []                    # bypass human even sooner (use with caution)

# ──────────────────────────────────────────────
# App 1 Simulator settings
# ──────────────────────────────────────────────
SIMULATOR_SEND_INTERVAL = 1.5             # seconds between data sends
SIMULATOR_LABEL_DELAY = 1.0              # seconds delay before sending true label

# ──────────────────────────────────────────────
# Server ports
# ──────────────────────────────────────────────
APP1_BACKEND_PORT = 3002
APP1_FRONTEND_PORT = 3001
APP2_BACKEND_PORT = 5000
APP2_FRONTEND_PORT = 5173
