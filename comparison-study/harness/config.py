"""
config.py — All tunables in one place for the comparison study.

Every module imports from here.  Change a value once → it propagates everywhere.
"""

import os
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# 1.  PATHS
# ────────────────────────────────────────────────────────────────────

# Root of the comparison study
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # comparison-study/

# Where raw / cleaned CSVs live — override with env var if needed
DATA_DIR = Path(os.environ.get("DATA_DIR", r"C:\claude"))

FRAUD_CSV       = DATA_DIR / "fraud_clean.csv"
LENDINGCLUB_CSV = DATA_DIR / "lendingclub_clean.csv"
ELEC2_CSV       = DATA_DIR / "elec2_clean.csv"   # generated from river if missing

# Results go here
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 2.  WINDOW / STREAMING
# ────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 500          # rows per window for drift detection / monitoring
REFERENCE_WINDOW_MULT = 2  # reference window = this × WINDOW_SIZE  (1000 rows)

# ────────────────────────────────────────────────────────────────────
# 3.  STATISTICAL TESTING
# ────────────────────────────────────────────────────────────────────

SIGNIFICANCE_LEVEL     = 0.05   # default α for drift tests
BH_FDR_ALPHA           = 0.05   # Benjamini–Hochberg FDR level
MMD_PERMUTATIONS       = 200    # permutation test iterations for MMD
PSI_THRESHOLD          = 0.2    # Population Stability Index alarm threshold
DRIFT_VOTE_FRACTION    = 0.5    # Evidently-style: fraction of features needed to alarm

# ────────────────────────────────────────────────────────────────────
# 4.  BOOTSTRAP / EVALUATION
# ────────────────────────────────────────────────────────────────────

BOOTSTRAP_ITERATIONS = 1000     # for 95 % CI on metrics
BOOTSTRAP_CI_LEVEL   = 0.95     # confidence level
EVAL_SEEDS           = list(range(42, 52))   # 10 seeds: 42..51

# Separate seeds for tuning vs final results (never overlap)
TUNING_SEED   = 99
FINAL_SEEDS   = list(range(42, 52))

# ────────────────────────────────────────────────────────────────────
# 5.  DATASET SPLIT RATIOS
# ────────────────────────────────────────────────────────────────────

# For chronological / stratified splits:
#   train      – model fitting
#   val        – hyperparameter tuning & calibration
#   test       – held-out evaluation (main tables)
#   future_pool – simulates "new incoming data" for drift simulation

SPLIT_RATIOS = {
    "train":       0.50,
    "val":         0.20,
    "test":        0.25,
    "future_pool": 0.05,
}

# ────────────────────────────────────────────────────────────────────
# 6.  MODEL DEFINITIONS
# ────────────────────────────────────────────────────────────────────

# Each entry: (display_name, sklearn_class_path, default_kwargs, tier)
# sklearn_class_path uses dotted import so model_wrapper can resolve dynamically.

MODEL_REGISTRY = {
    # ── Core (always run) ──────────────────────────────────────────
    "lr": {
        "name": "Logistic Regression",
        "class": "sklearn.linear_model.LogisticRegression",
        "kwargs": {"max_iter": 1000, "solver": "lbfgs"},
        "tier": "core",
    },
    "rf": {
        "name": "Random Forest",
        "class": "sklearn.ensemble.RandomForestClassifier",
        "kwargs": {"n_estimators": 100, "n_jobs": -1},
        "tier": "core",
    },
    "xgb": {
        "name": "XGBoost",
        "class": "xgboost.XGBClassifier",
        "kwargs": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        },
        "tier": "core",
    },
    "lgbm": {
        "name": "LightGBM",
        "class": "lightgbm.LGBMClassifier",
        "kwargs": {"n_estimators": 100, "num_leaves": 31, "verbose": -1},
        "tier": "core",
    },

    # ── Diversity ──────────────────────────────────────────────────
    "svm": {
        "name": "SVM-RBF",
        "class": "sklearn.svm.SVC",
        "kwargs": {"kernel": "rbf", "probability": True, "max_iter": 5000},
        "tier": "diversity",
    },
    "knn": {
        "name": "KNN",
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "kwargs": {"n_neighbors": 5, "n_jobs": -1},
        "tier": "diversity",
    },
    "gnb": {
        "name": "Gaussian NB",
        "class": "sklearn.naive_bayes.GaussianNB",
        "kwargs": {},
        "tier": "diversity",
    },

    # ── Lightweight ────────────────────────────────────────────────
    "sgd": {
        "name": "SGDClassifier",
        "class": "sklearn.linear_model.SGDClassifier",
        "kwargs": {"loss": "log_loss", "max_iter": 1000, "random_state": 42},
        "tier": "lightweight",
    },
    "dt": {
        "name": "Decision Tree",
        "class": "sklearn.tree.DecisionTreeClassifier",
        "kwargs": {"max_depth": 10},
        "tier": "lightweight",
    },
    "perceptron": {
        "name": "Perceptron",
        "class": "sklearn.linear_model.Perceptron",
        "kwargs": {"max_iter": 1000},
        "tier": "lightweight",
    },
    "lda": {
        "name": "LDA",
        "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
        "kwargs": {},
        "tier": "lightweight",
    },
}

# Convenience lists
CORE_MODELS       = [k for k, v in MODEL_REGISTRY.items() if v["tier"] == "core"]
DIVERSITY_MODELS  = [k for k, v in MODEL_REGISTRY.items() if v["tier"] == "diversity"]
LIGHTWEIGHT_MODELS = [k for k, v in MODEL_REGISTRY.items() if v["tier"] == "lightweight"]
ALL_MODELS        = list(MODEL_REGISTRY.keys())
MAIN_MODELS       = CORE_MODELS + DIVERSITY_MODELS   # 7 for main tables

# ────────────────────────────────────────────────────────────────────
# 7.  DATASET METADATA
# ────────────────────────────────────────────────────────────────────

DATASET_META = {
    "fraud": {
        "csv": FRAUD_CSV,
        "target_col": "Class",          # original column name
        "target_rename": "target",       # we rename to this
        "time_col": None,                # no natural time column — use row order
        "drift_type": "synthetic",
        "primary_metric": "pr_auc",      # extreme imbalance → PR-AUC
        "positive_label": 1,
        "description": "Credit-card fraud (Kaggle), 0.17 % positive, synthetic drift.",
    },
    "elec2": {
        "csv": ELEC2_CSV,
        "target_col": "target",
        "target_rename": "target",
        "time_col": None,                # chronological by row order — NEVER shuffle
        "drift_type": "real_temporal",
        "primary_metric": "f1",          # balanced → F1
        "positive_label": 1,
        "description": "Elec2 (river library), 42.5 % positive, real temporal drift.",
    },
    "lendingclub": {
        "csv": LENDINGCLUB_CSV,
        "target_col": "loan_status",     # raw column
        "target_rename": "target",
        "time_col": "issue_d",           # used for chronological split
        "drift_type": "real_gradual",
        "primary_metric": "pr_auc",      # 20 % positive → PR-AUC
        "positive_label": 1,
        "description": "LendingClub loans, 20.2 % default, real year-based drift.",
    },
}

DATASET_NAMES = list(DATASET_META.keys())

# ────────────────────────────────────────────────────────────────────
# 8.  LENDING CLUB FEATURE SELECTION
# ────────────────────────────────────────────────────────────────────
# Keep only these numeric features (curated from the 151-column raw file).
# Excludes: IDs, text, dates, post-loan-outcome columns, sparse joint-app cols.

LENDINGCLUB_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "term",               # will be parsed: " 36 months" → 36
    "int_rate",           # will be parsed: "10.65%" → 10.65
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",         # may be string with % — parse
    "total_acc",
    "acc_now_delinq",
    "tot_coll_amt",
]

# ────────────────────────────────────────────────────────────────────
# 9.  DRIFT INJECTION (for fraud dataset — synthetic)
# ────────────────────────────────────────────────────────────────────

DRIFT_INJECTION = {
    "mean_shift":  2.0,      # shift top-5 features by +2σ
    "variance_mult": 1.5,    # inflate variance of 3 features
    "label_flip_rate": 0.10, # flip 10 % of labels
}

# ────────────────────────────────────────────────────────────────────
# 10. PART-SPECIFIC CONFIGS (stubs — each part fills these in)
# ────────────────────────────────────────────────────────────────────

PART1_DETECT = {
    "methods": ["evidently_style", "alibi_bh", "alibi_mmd"],
    "configs": ["defaults", "matched"],
}

PART2_DEGRADE = {
    "methods": ["cbpe", "ddm", "eddm"],
    "configs": ["defaults", "matched"],
}

PART3_HEAL = {
    "methods": [
        "adorn_qlearning", "regol_cost",
        "never_retrain", "always_retrain", "fixed_schedule", "retrain_on_alarm",
    ],
}

PART4_RETRAIN = {
    "methods": ["full_retrain", "sliding_window"],
    "sliding_window_size": 5000,   # rows to keep for sliding-window retrain
}

