"""
test_harness.py — Quick smoke test for the entire harness.
Run from comparison-study/:  python test_harness.py
"""
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── 1. Config ──────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Config import")
print("=" * 60)
from harness import config
print(f"  Models:   {config.ALL_MODELS}")
print(f"  Datasets: {config.DATASET_NAMES}")
print(f"  Window:   {config.WINDOW_SIZE}")
print(f"  Seeds:    {config.FINAL_SEEDS}")
print("  PASS\n")

# ── 2. Data Loader (fraud) ─────────────────────────────────────────
print("=" * 60)
print("TEST 2: Load fraud dataset")
print("=" * 60)
from harness.data_loader import load_dataset
t0 = time.time()
ds = load_dataset("fraud")
print(f"  Shape: X={ds['X'].shape}, y={ds['y'].shape}")
print(f"  Features: {list(ds['X'].columns[:5])}...")
print(f"  Positive rate: {ds['y'].mean():.4f}")
print(f"  NaN cells: {ds['X'].isna().sum().sum()}")
print(f"  Time: {time.time()-t0:.1f}s")
assert ds["X"].shape[1] == 29, f"Expected 29 features, got {ds['X'].shape[1]}"
assert ds["y"].mean() < 0.01, "Fraud should be extremely imbalanced"
print("  PASS\n")

# ── 3. Splitter ────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3: Split fraud dataset")
print("=" * 60)
from harness.splitter import split_dataset, window_stream
splits = split_dataset(ds, seed=42)
for name, sp in splits.items():
    print(f"  {name:12s}: {len(sp['y']):>8,} rows  pos_rate={sp['y'].mean():.4f}")
total = sum(len(sp['y']) for sp in splits.values())
assert total == len(ds['y']), f"Split total {total} != dataset {len(ds['y'])}"
print("  PASS\n")

# ── 4. Window Stream ──────────────────────────────────────────────
print("=" * 60)
print("TEST 4: Window stream on test split")
print("=" * 60)
windows = window_stream(splits["test"]["X"], splits["test"]["y"])
print(f"  {len(windows)} windows of size {config.WINDOW_SIZE}")
print(f"  First window: {windows[0]['X'].shape}")
print("  PASS\n")

# ── 5. Model Wrapper ──────────────────────────────────────────────
print("=" * 60)
print("TEST 5: Build & train models")
print("=" * 60)
from harness.model_wrapper import build_model, list_available_models
avail = list_available_models()
print(f"  Available: {avail}")

# Train LR on a small subset for speed
X_train = splits["train"]["X"].iloc[:2000]
y_train = splits["train"]["y"].iloc[:2000]
X_test  = splits["test"]["X"].iloc[:500]
y_test  = splits["test"]["y"].iloc[:500]

for key in ["lr", "gnb", "dt"]:
    model = build_model(key, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print(f"  {model.name}: preds={preds.shape}, proba={proba.shape}, "
          f"fit={model.fit_time_s:.3f}s, pred={model.predict_time_s:.4f}s")
    assert preds.shape == (500,), f"Wrong pred shape: {preds.shape}"
    assert proba.shape == (500,), f"Wrong proba shape: {proba.shape}"
    assert 0 <= proba.min() <= proba.max() <= 1, "Proba out of [0,1]"
print("  PASS\n")

# ── 6. Metrics ────────────────────────────────────────────────────
print("=" * 60)
print("TEST 6: Evaluate metrics with bootstrap CI")
print("=" * 60)
from harness.metrics import evaluate, paired_bootstrap_test
import numpy as np

model_lr = build_model("lr", seed=42)
model_lr.fit(X_train, y_train)
proba_lr = model_lr.predict_proba(X_test)
preds_lr = model_lr.predict(X_test)

result = evaluate(y_test.values, preds_lr, proba_lr, primary="pr_auc", seed=42)
print(f"  PR-AUC:  {result['pr_auc']:.4f}  CI={result['pr_auc_ci']}")
print(f"  F1:      {result['f1']:.4f}  CI={result['f1_ci']}")
print(f"  ROC-AUC: {result['roc_auc']:.4f}")
print(f"  Prec:    {result['precision']:.4f}  Recall: {result['recall']:.4f}")
assert not np.isnan(result['pr_auc']), "PR-AUC should not be NaN"
print("  PASS\n")

# ── 7. Paired Bootstrap ──────────────────────────────────────────
print("=" * 60)
print("TEST 7: Paired bootstrap test (LR vs GNB)")
print("=" * 60)
model_gnb = build_model("gnb", seed=42)
model_gnb.fit(X_train, y_train)
proba_gnb = model_gnb.predict_proba(X_test)

delta = paired_bootstrap_test(y_test.values, proba_lr, proba_gnb, metric_name="pr_auc", seed=42)
print(f"  LR − GNB: mean_diff={delta['mean_diff']:.4f}  CI=[{delta['ci_lo']:.4f}, {delta['ci_hi']:.4f}]")
print(f"  Significant: {delta['significant']}  Winner: {delta['winner']}")
print("  PASS\n")

# ── 8. Runner (full pipeline) ────────────────────────────────────
print("=" * 60)
print("TEST 8: Full pipeline via runner (fraud × lr)")
print("=" * 60)
from harness.runner import run_baseline_experiment, results_to_table, save_results
r = run_baseline_experiment("fraud", "lr", seed=42, bootstrap=True)
print(f"  Dataset: {r['dataset']}")
print(f"  Model:   {r['model_name']} ({r['tier']})")
print(f"  PR-AUC:  {r['metrics']['pr_auc']:.4f}")
print(f"  F1:      {r['metrics']['f1']:.4f}")
print(f"  Fit:     {r['fit_time_s']:.2f}s on {r['n_train']:,} samples")
print("  PASS\n")

# ── 9. Results table ─────────────────────────────────────────────
print("=" * 60)
print("TEST 9: Multi-model results table")
print("=" * 60)
results = []
for key in ["lr", "gnb", "dt"]:
    results.append(run_baseline_experiment("fraud", key, seed=42, bootstrap=False))
print(results_to_table(results))
print("\n  PASS\n")

# ── 10. Save results ─────────────────────────────────────────────
print("=" * 60)
print("TEST 10: Save results to JSON")
print("=" * 60)
save_results(results, "test_harness_results.json")
from harness.runner import load_results
loaded = load_results("test_harness_results.json")
assert len(loaded) == 3, f"Expected 3 results, got {len(loaded)}"
print("  PASS\n")

print("=" * 60)
print("ALL 10 TESTS PASSED ✅")
print("=" * 60)

