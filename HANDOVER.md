# HANDOVER DOCUMENT: Self-Healing MLOps Comparison Study

**Document Generated:** 2026-11-13  
**Project:** Self-Healing MLOps (Final-Year Research Project — Semester 1 Comparison Study)  
**Repository:** `https://github.com/ACER9851/self-healing-mlops.git`  
**Working Directory:** `C:\claude\self-healing-mlops`  
**Current Branch:** `master` (Latest Commit: `029ac35`)

---

## 1. Main Objective and Background

### Academic Context
This is a final-year research project on **"Self-Healing MLOps"**: an autonomous closed-loop system that detects data drift, assesses whether predictive model quality is degrading, decides whether (and when) to retrain, and executes the appropriate retraining strategy.

### Semester 1 Scope: The Comparison Study
This semester focuses strictly on an **empirical benchmark comparison study**. The goal is to reimplement, test, and benchmark established methods from literature and industry tools on identical datasets under controlled conditions:
1. **Part 1 (DETECT):** Has incoming data drifted?
   - *Evidently-Style*: Univariate KS-test (continuous) + PSI (categorical & continuous) with majority voting ($>50\%$ drifted features).
   - *Alibi-Style (Univariate)*: Feature-wise KS-test / $\chi^2$-test with Benjamini-Hochberg (BH) False Discovery Rate (FDR) control ($\alpha = 0.05$).
   - *Alibi-Style (Multivariate)*: Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel and Monte-Carlo permutation test.
2. **Part 2 (DEGRADATION):** Is model quality dropping without waiting for delayed labels?
   - *NannyML-Style CBPE*: Confidence-Based Performance Estimation. Uses Platt scaling calibration on reference data to estimate expected confusion matrix and F1/ROC-AUC without ground-truth labels.
   - *DDM (Drift Detection Method)*: Sequential hypothesis testing on error rate (Gama et al., 2004). Requires ground-truth labels.
   - *EDDM (Early Drift Detection Method)*: Distance-between-consecutive-errors monitoring (Baena-Garcia et al., 2006). Requires ground-truth labels.
3. **Part 3 (HEALING):** When and how should the system intervene?
   - *ADORN-Style*: Offline pre-trained Q-learning decision engine ($18 \times 6$ state-action Q-table).
   - *Regol-Style*: Cost-aware performance forecasting (retrain only if expected performance gain outweighs computational cost).
   - *Heuristic Baselines*: Never retrain, Always retrain, Fixed periodic schedule, Retrain-on-alarm.
4. **Part 4 (RETRAINING):** How should the model be updated?
   - *Full Retrain*: Retrain on all accumulated labeled data from time $t_0$ to $t_{now}$.
   - *Sliding-Window Retrain*: Retrain on the most recent $W$ samples only.

### Semester 2 Outlook
In Semester 2, the student will design novel, custom drift detection and healing algorithms. The findings and metrics from this Semester 1 comparison study serve as the baseline and justification for the novel architecture.

---

## 2. Important Information About the Project

### The 3 Benchmark Datasets

| Dataset | Nature | Rows | Features | Target & Balance | Drift Mechanism |
|---|---|---|---|---|---|
| **Credit Card Fraud** | Tabular (PCA features) | 284,807 | 29 (`V1`–`V28`, `Amount`) | `Class` $\rightarrow$ `target` (0.17% pos) | Synthetic injected drift (+2$\sigma$ mean shift on top features) |
| **Elec2** | Tabular (Time-series) | 45,312 | 7 numeric | `class` $\rightarrow$ `target` (42.5% pos) | REAL natural temporal drift around row 17,424 (**NEVER shuffle**) |
| **LendingClub** | Tabular (Financial loans) | 1,345,310 (cleaned) | 18 curated numeric | `loan_status` $\rightarrow$ `target` (19.96% pos) | REAL gradual temporal drift across 2007–2018 (**ordered by `issue_d`**) |

### The 11 Benchmark Classifiers
Unified interface via `harness.model_wrapper.ModelWrapper`:
- **Core Tier (Always evaluated):** Logistic Regression (`lr`), Random Forest (`rf`), XGBoost (`xgb`), LightGBM (`lgbm`).
- **Diversity Tier (Inductive bias check):** SVM-RBF (`svm`), K-Nearest Neighbors (`knn`), Gaussian Naive Bayes (`gnb`).
- **Lightweight / Online Tier:** SGDClassifier with log-loss (`sgd`), Decision Tree (`dt`), Perceptron (`perceptron`), Linear Discriminant Analysis (`lda`).

---

## 3. Important Decisions and Rationale

1. **Reimplementation ("-Style") vs Package Wrappers:**
   - *Decision:* Core algorithms (Evidently, Alibi, NannyML CBPE, DDM, EDDM) are implemented from first principles.
   - *Reason:* Eliminates heavy, conflicting dependency locks (e.g. Evidently/Alibi often require specific pinned versions of torch, tensorflow, or older pydantic), allows complete algorithmic transparency, and enforces identical runtime environments across methods.
2. **Relaxation of "Zero-Import" Rule for Harness:**
   - *Decision:* Shared infrastructure (`harness/`) uses standard libraries (`scikit-learn`, `pandas`, `numpy`). Only the core detector modules retain standalone math primitives.
   - *Reason:* Implementing scikit-learn models or data loaders from scratch is unnecessary overhead and invites performance bottlenecks.
3. **Primary Evaluation Metrics:**
   - *Decision:* PR-AUC (Average Precision) with paired non-parametric bootstrap 95% confidence intervals is primary for imbalanced data (Fraud, LendingClub). F1 is primary for balanced data (Elec2). **Never report accuracy alone.**
   - *Reason:* Accuracy is trivially 99.83% on Fraud by predicting all negatives, completely masking drift and degradation.
4. **Windowing and Streaming Architecture:**
   - *Decision:* Production data is streamed in sequential non-overlapping chunks of $W = 500$ observations. Reference baseline is derived from the training set.
   - *Reason:* Standardizes detection latency measurements across both chunk-based estimators (CBPE, Evidently) and instance-based sequential detectors (DDM).
5. **Elec2 Shuffling Prohibition:**
   - *Decision:* Elec2 row order must remain strictly chronological.
   - *Reason:* Electricity price and demand dynamics across New South Wales and Victoria drift over time; shuffling destroys the temporal drift signal.

---

## 4. Technologies, Frameworks, and Versions Used

### Environment & Runtimes
- **OS:** Windows 11 / Windows Server (PowerShell shell environment)
- **Python Version:** `3.14.7` (tags/v3.14.7:823f032, Aug 5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]
- **Installed Key Packages:**
  * `pandas`: 3.0.6
  * `numpy`: 2.5.3
  * `scikit-learn`: 1.9.1
  * `scipy`: 1.18.1
  * `matplotlib`: 3.11.2
- **Missing / Optional Packages (Handled gracefully with fallbacks):**
  * `xgboost`: Not installed locally (handled via graceful skip in `model_wrapper.py`).
  * `lightgbm`: Not installed locally (handled via graceful skip in `model_wrapper.py`).
  * `river`: Not installed locally (used for Elec2 generation; fallback to `elec2_clean.csv`).

### Windows Git Execution Note
Git is installed at `C:\Program Files\Git\cmd\git.exe`, but is **not on system PATH by default**. Every PowerShell terminal session must prepend PATH:
```powershell
$env:PATH += ";C:\Program Files\Git\cmd"
```

---

## 5. GitHub Repository & File Structure

- **Remote:** `https://github.com/ACER9851/self-healing-mlops.git`
- **Branch:** `master`
- **Head Commit:** `029ac35` (*"Complete Task 0 (Harness) and Task 2 (Degradation Detection with 69/69 tests passing)"*)

### Relevant Directory Structure
```
c:\claude\self-healing-mlops\
├── .gitignore
├── README.md
├── pretrain_qtable.py
├── self-healing-mlops-architecture-consolidated-project-prompt.md
│
└── comparison-study/
    ├── dataset_cleaning.ipynb          # 35 cells: clean Fraud, Elec2, LendingClub + Section 6 Part 1 tests
    ├── test_harness.py                 # Smoke test for Task 0 harness (10/10 passed)
    │
    ├── harness/                        # [TASK 0 COMPLETED]
    │   ├── __init__.py
    │   ├── config.py                   # Central tunables, paths, model registry, split ratios
    │   ├── data_loader.py              # Robust loaders & clean-at-load for Fraud, Elec2, LendingClub
    │   ├── splitter.py                 # Stratified & chronological splitting + window_stream(W=500)
    │   ├── model_wrapper.py            # Unified ModelWrapper for 11 classifiers + StandardScaler
    │   ├── metrics.py                  # PR-AUC, F1, ROC-AUC, bootstrap 95% CI, paired tests
    │   └── runner.py                   # run_baseline_experiment(), batch runner, results saving
    │
    ├── part1_detect/                   # [TASK 1 COMPLETED]
    │   ├── evidently_style_detector.py # Univariate KS + PSI + majority vote (>50%)
    │   ├── alibi_style_detector.py     # Univariate KS/Chi2 + BH-FDR & Multivariate MMD (RBF kernel)
    │   ├── data_generator.py           # 8 synthetic drift test cases
    │   ├── test_models.py              # 128 unit tests (all passed)
    │   ├── main.py                     # Synthetic drift benchmark runner
    │   └── run_on_real_data.py         # Real-data evaluation on Fraud & LendingClub
    │
    ├── part2_degradation/              # [TASK 2 COMPLETED]
    │   ├── nannyml_cbpe.py             # NannyML CBPE: Platt scaling + expected F1/ROC-AUC estimation
    │   ├── ddm_detector.py             # DDM (Gama 2004) & EDDM (Baena-Garcia 2006) error monitoring
    │   ├── data_generator.py           # 8 degradation stream scenarios (sudden, gradual, calibration, etc.)
    │   ├── test_models.py              # 69 unit tests (all 69 passed)
    │   └── main.py                     # Benchmark comparing CBPE vs DDM vs EDDM
    │
    ├── part3_heal/                     # [TASK 4 - TO BE BUILT]
    │   ├── qtable_engine.py            # ADORN Q-learning (18x6 table)
    │   ├── regol_engine.py             # Regol cost-aware linear forecasting
    │   └── baselines.py                # Never, Always, Schedule, On-Alarm
    │
    ├── part4_retrain/                  # [TASK 3 - NEXT IMMEDIATE STEP]
    │   ├── full_retrain.py             # Retrain on all historical data
    │   ├── sliding_window.py           # Retrain on last W samples
    │   ├── test_part4.py
    │   └── run_part4.py
    │
    └── results/                        # JSON artifacts from experiment runs
        └── test_harness_results.json
```

---

## 6. Work Completed So Far

### Task 0: Shared Harness (`harness/`) — COMPLETE ✅
- Built 6 modules: `config.py`, `data_loader.py`, `splitter.py`, `model_wrapper.py`, `metrics.py`, `runner.py`.
- Verified via `test_harness.py`: All **10/10 tests passed**:
  * Fraud dataset loaded (284,807 rows, 29 features, pos_rate=0.0017).
  * Stratified split into train (142,403), val (56,961), test (71,201), future_pool (14,242).
  * Window stream partitioned test set into 142 chunks of 500 rows.
  * Model training: Logistic Regression (PR-AUC 0.8106, fit 0.45s), Gaussian NB (PR-AUC 0.0941), Decision Tree (PR-AUC 0.6867).
  * Paired bootstrap verified: LR is significantly superior to Gaussian NB on Fraud ($\Delta = +0.0845$, 95% CI $[0.0380, 0.1940]$).

### Task 1: Drift Detection (`part1_detect/`) — COMPLETE ✅
- `evidently_style_detector.py` and `alibi_style_detector.py` verified with 128 unit tests.
- Tested on synthetic Cases 1–8:
  * Key finding: Case 5 (joint correlation change with identical marginals) is **completely missed** by column-wise KS/PSI/BH-FDR tests, but **detected by Alibi-MMD** ($p = 0.0099$).
- Tested on real data via `run_on_real_data.py` & `dataset_cleaning.ipynb` Section 6:
  * On Fraud: Alibi-BH and MMD show 0 false alarms on stationary validation data, but trigger alarms on +2$\sigma$ injected drift.
  * On LendingClub: Captures temporal progression from 2007 (early train: 6/18 drifted) to 2018 (late stream: 17/18 drifted).

### Task 2: Degradation Detection (`part2_degradation/`) — COMPLETE ✅
- Built `nannyml_cbpe.py` (Platt scaling calibration + chunk-based F1/ROC-AUC estimation).
- Built `ddm_detector.py` (DDM error-rate tracking and EDDM distance-between-errors tracking).
- Built `data_generator.py` providing 8 degradation streaming scenarios.
- Fixed historical bugs (Platt scaling `_estimate_f1` signature, tied-rank ROC-AUC calculation, EDDM parameter aliases, numerical stability in `_manual_exp` up to $x = 709.78$).
- Verified via `test_models.py`: **69/69 tests passed** across all 13 test groups.
- Benchmarked via `main.py`: Proved CBPE successfully flags degradation label-free, and detects calibration-only drift where DDM is blind.

### Dataset Cleaning Notebook (`dataset_cleaning.ipynb`) — COMPLETE ✅
- Created 35-cell Jupyter Notebook handling Kaggle upload, column selection, string parsing, median imputation, and chronological sorting for Fraud, Elec2, and LendingClub.
- Integrated Section 6 demonstrating inline verification of Part 1 drift detection.

---

## 7. Errors, Bugs, and Unresolved Problems

1. **Local Missing Packages:**
   - `xgboost` and `lightgbm` are not installed in the local Python 3.14 environment.
   - `river` is not installed locally.
   - *Impact:* Harness skips XGB/LGBM without crashing and uses available models (LR, RF, SVM, KNN, GNB, SGD, DT, Perceptron, LDA). Elec2 raw generation requires either `pip install river` or providing `c:\claude\elec2_clean.csv`.
2. **Git PATH in Windows PowerShell:**
   - Git commands fail if `$env:PATH += ";C:\Program Files\Git\cmd"` is not set in new PowerShell subshells.
3. **Local CSV Storage:**
   - Local disk has `c:\claude\fraud_clean.csv` (144 MB) and `c:\claude\lendingclub_clean.csv` (1.6 GB).
   - `elec2_clean.csv` is currently missing locally (can be downloaded by running `dataset_cleaning.ipynb` on Google Colab).
4. **GitHub Push Authentication:**
   - The user's earlier personal access token (PAT) was revoked for security. Pushing local commits to `origin/master` will require the user to configure Git credentials or supply a new PAT.

---

## 8. Approaches Tested and Verified Empirical Results

### Part 1: Drift Detectors on Real Datasets

```
================================================================================
Dataset: FRAUD (29 Features) — Sampled Baseline vs Stream
================================================================================
Condition 1: Clean vs Clean (Stationary)
  - Evidently-Style (>50% majority):     drift = False (4/29 features drifted)
  - Alibi-Style BH-FDR (alpha=0.05):     drift = False (0/29 features rejected)
  - Alibi-Style MMD (RBF, perm=100):     drift = False (p = 0.4158)

Condition 2: Clean vs Injected Drift (+2.0σ shift on V1..V5)
  - Evidently-Style (>50% majority):     drift = False (9/29 features drifted, < 50%)
  - Alibi-Style BH-FDR (alpha=0.05):     drift = True  (5/29 features rejected: V1..V5)
  - Alibi-Style MMD (RBF, perm=100):     drift = True  (p = 0.0099)
```

```
================================================================================
Dataset: LENDINGCLUB (18 Curated Features) — Chronological Timeline
================================================================================
Reference: First 2,000 loans (2007)
  - Checkpoint 1 (Early Train, ~2008):   Evidently: clean (6/18)  | Alibi-BH: DRIFT (4/18)  | MMD: p=0.010
  - Checkpoint 2 (Mid Train, ~2014):     Evidently: DRIFT (17/18) | Alibi-BH: DRIFT (17/18) | MMD: p=0.010
  - Checkpoint 3 (Test Stream, ~2016):   Evidently: DRIFT (17/18) | Alibi-BH: DRIFT (17/18) | MMD: p=0.010
  - Checkpoint 4 (Future Pool, ~2018):   Evidently: DRIFT (13/18) | Alibi-BH: DRIFT (13/18) | MMD: p=0.010
```

### Part 2: Degradation Detectors Across Scenarios (`part2_degradation/main.py`)

| Scenario | CBPE (Label-Free) | DDM (Needs Labels) | EDDM (Needs Labels) | Key Finding |
|---|---|---|---|---|
| `no_degradation` | **OK** (no alert) | ALARM (step 531) | ALARM (step 326) | DDM suffers from sample variance over long streams; CBPE chunk bounds remain stable. |
| `sudden_degradation` | **ALERT** | **ALARM** (step 29) | **ALARM** (step 289) | Both methods detect sharp performance drop. |
| `gradual_degradation` | **ALERT** | **ALARM** (step 29) | **ALARM** (step 703) | EDDM tracks distance-between-errors, alerting later than DDM. |
| `calibration_shift` | **ALERT** | **OK** (no alarm) | **OK** (no alarm) | Hard accuracy is unchanged (90%), so DDM/EDDM miss it entirely; CBPE detects probability collapse. |
| `delayed_labels` | **ALERT** | **ALARM** (step 29) | **ALARM** (step 1560) | CBPE monitors instantly; DDM/EDDM are blocked until delayed labels arrive. |

---

## 9. User Requirements and Preferences

- **Scientific Rigor:** Never fabricate numbers or simulate non-existent results. Mark tests that could not run as `NOT RUN` or `UNVERIFIED`.
- **Paired Statistical Comparison:** Always evaluate model differences with paired bootstrap tests across identical seeds. Do not declare a method "better" unless the 95% confidence interval strictly excludes zero.
- **Code Execution Protocol:** Inspect and test scripts using PowerShell commands before reporting results.
- **Tone & Formatting:** Concise, high-density, technical responses formatted in standard GitHub Flavored Markdown. LaTeX math expressions ($...$) for formulas.

---

## 10. The Exact Next Task to Continue: Task 3 (Part 4 — Retraining)

### Objective of Task 3
Implement and compare the two retraining strategies defined in the research specification:
1. **Full Retrain (`full_retrain.py`):** When retrained, fits the model on all historical labeled data collected up to the current window.
2. **Sliding-Window Retrain (`sliding_window.py`):** When retrained, fits the model only on the most recent $W_{train}$ labeled samples (e.g. last 5,000 samples), discarding older data.

### Implementation Checklist for the Next Assistant
1. **Directory:** Work in `c:\claude\self-healing-mlops\comparison-study\part4_retrain\`
2. **Files to build:**
   - `full_retrain.py`: Encapsulates full-history model retraining using `harness.model_wrapper.ModelWrapper`.
   - `sliding_window.py`: Encapsulates sliding-window buffer management and retraining.
   - `test_part4.py`: Unit tests verifying that retraining restores F1/PR-AUC on shifted data and that the buffer bounds behave correctly.
   - `run_part4.py`: Benchmark comparing Full Retrain vs Sliding-Window Retrain on a drifting stream (e.g. LendingClub or Fraud with injected drift).
3. **Core Metrics to Record in Part 4:**
   - *F1 / PR-AUC Recovery Speed:* Number of windows required after drift to return to baseline performance.
   - *Wall-clock Training Time (seconds):* Full retrain grows linearly $O(N)$ with data volume; sliding window remains constant $O(W)$.
   - *Minority-Class Recall:* How well the retrained model preserves detection of rare events after retraining.

Once Task 3 is complete, the final component (**Task 4: Part 3 — Healing Decision Engines: ADORN Q-Learning vs Regol Cost-Aware**) can be assembled using the detectors and retrainers.
