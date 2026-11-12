"""
run_on_real_data.py — Run Part 1 drift detectors on the real cleaned datasets.

Uses the harness to load + split data, then feeds windows into
the Evidently-style and Alibi-style detectors.

Run from comparison-study/:
    python part1_detect/run_on_real_data.py
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from harness.data_loader import load_dataset
from harness.splitter import split_dataset, window_stream
from harness import config

from evidently_style_detector import EvidentlyStyleDriftDetector
from alibi_style_detector import AlibiStyleDriftDetector


def df_to_dicts(X_df):
    """Convert a DataFrame to list of dicts (format the detectors expect)."""
    return X_df.to_dict(orient="records")


def run_drift_on_dataset(dataset_name, max_windows=20, mmd_sample=100):
    """Run all 3 drift detectors on a real dataset using windowed streaming.

    Strategy:
      - Reference window = train split's LAST 1000 rows (stable baseline)
      - Current windows  = sequential non-overlapping windows from test split
      - We expect drift to emerge in later windows (esp. for lendingclub/elec2)
    """
    print("=" * 78)
    print(f"  DRIFT DETECTION ON REAL DATA: {dataset_name.upper()}")
    print("=" * 78)

    # Load and split
    t0 = time.time()
    ds = load_dataset(dataset_name)
    splits = split_dataset(ds, seed=42)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Train: {len(splits['train']['y']):,} rows")
    print(f"  Test:  {len(splits['test']['y']):,} rows")
    print(f"  Future pool: {len(splits['future_pool']['y']):,} rows")

    feature_names = list(ds["X"].columns)
    print(f"  Features ({len(feature_names)}): {feature_names[:5]}...")

    # Reference = last 1000 rows of train (the "stable" baseline)
    ref_size = min(1000, len(splits["train"]["X"]))
    X_ref = splits["train"]["X"].iloc[-ref_size:]
    ref_dicts = df_to_dicts(X_ref)
    print(f"  Reference: last {ref_size} rows of train split")

    # Instantiate detectors
    ev_det = EvidentlyStyleDriftDetector(ks_alpha=0.05, psi_threshold=0.20)
    alibi_bh_det = AlibiStyleDriftDetector(method='columnwise_bh', alpha=0.05)
    alibi_mmd_det = AlibiStyleDriftDetector(
        method='multivariate_mmd', alpha=0.05,
        n_permutations=100, seed=42
    )

    # Create windows from test + future_pool combined
    import pandas as pd
    X_stream = pd.concat([splits["test"]["X"], splits["future_pool"]["X"]], ignore_index=True)
    y_stream = pd.concat([splits["test"]["y"], splits["future_pool"]["y"]], ignore_index=True)

    windows = window_stream(X_stream, y_stream, window_size=config.WINDOW_SIZE)
    n_windows = min(len(windows), max_windows)
    print(f"  Streaming: {len(windows)} windows total, running first {n_windows}")

    # Header
    print()
    print(f"  {'Win':>4s} {'Rows':>6s} {'PosRate':>8s} | "
          f"{'Evidently':>10s} {'Alibi-BH':>10s} {'Alibi-MMD':>10s} | "
          f"{'Ev Detail':>30s} {'Time':>6s}")
    print("  " + "-" * 100)

    results = []
    for i, w in enumerate(windows[:n_windows]):
        t1 = time.time()
        cur_dicts = df_to_dicts(w["X"])
        pos_rate = w["y"].mean()

        # For MMD, subsample if window is large (pure-Python MMD is O(n²))
        if len(cur_dicts) > mmd_sample:
            import random
            random.seed(42)
            mmd_cur_dicts = random.sample(cur_dicts, mmd_sample)
            mmd_ref_dicts = random.sample(ref_dicts, min(mmd_sample, len(ref_dicts)))
        else:
            mmd_cur_dicts = cur_dicts
            mmd_ref_dicts = ref_dicts

        # Run Evidently
        ev_result = ev_det.detect(ref_dicts, cur_dicts, feature_names, [])
        ev_drift = ev_result["dataset_drift_detected"]
        ev_frac = ev_result["drift_fraction"]
        ev_n_drift = ev_result["n_features_drifted"]
        ev_n_test = ev_result["n_features_tested"]

        # Run Alibi-BH
        bh_result = alibi_bh_det.detect(ref_dicts, cur_dicts, feature_names, [])
        bh_drift = bh_result["dataset_drift_detected"]
        bh_n_drift = bh_result["n_features_drifted"]

        # Run Alibi-MMD (subsampled)
        mmd_result = alibi_mmd_det.detect(mmd_ref_dicts, mmd_cur_dicts, feature_names, [])
        mmd_drift = mmd_result["dataset_drift_detected"]
        mmd_p = mmd_result.get("p_value", 1.0)

        elapsed = time.time() - t1

        ev_str = "DRIFT" if ev_drift else "clean"
        bh_str = "DRIFT" if bh_drift else "clean"
        mmd_str = f"DRIFT p={mmd_p:.3f}" if mmd_drift else f"clean p={mmd_p:.3f}"
        detail = f"{ev_n_drift}/{ev_n_test} feats ({ev_frac*100:.0f}%)"

        print(f"  {i:4d} {len(cur_dicts):6d} {pos_rate:8.4f} | "
              f"{ev_str:>10s} {bh_str:>10s} {mmd_str:>10s} | "
              f"{detail:>30s} {elapsed:6.1f}s")

        results.append({
            "window": i,
            "n_rows": len(cur_dicts),
            "pos_rate": pos_rate,
            "evidently_drift": ev_drift,
            "evidently_frac": ev_frac,
            "alibi_bh_drift": bh_drift,
            "alibi_bh_n_drift": bh_n_drift,
            "alibi_mmd_drift": mmd_drift,
            "alibi_mmd_p": mmd_p,
        })

    # Summary
    print()
    print("  " + "=" * 78)
    print(f"  SUMMARY: {dataset_name.upper()}")
    print("  " + "=" * 78)
    n_total = len(results)
    ev_alarms = sum(1 for r in results if r["evidently_drift"])
    bh_alarms = sum(1 for r in results if r["alibi_bh_drift"])
    mmd_alarms = sum(1 for r in results if r["alibi_mmd_drift"])
    print(f"  Windows tested: {n_total}")
    print(f"  Evidently alarms: {ev_alarms}/{n_total} ({ev_alarms/n_total*100:.0f}%)")
    print(f"  Alibi-BH alarms:  {bh_alarms}/{n_total} ({bh_alarms/n_total*100:.0f}%)")
    print(f"  Alibi-MMD alarms: {mmd_alarms}/{n_total} ({mmd_alarms/n_total*100:.0f}%)")

    # Show which features drift most (from last window's Evidently result)
    if ev_result.get("feature_results"):
        drifted_feats = [
            f for f, r in ev_result["feature_results"].items()
            if r.get("drift_detected", False)
        ]
        if drifted_feats:
            print(f"\n  Features drifted in last window (Evidently): {drifted_feats}")

    print()
    return results


def main():
    print()
    print("=" * 78)
    print("  PART 1: DRIFT DETECTION ON REAL DATASETS")
    print("  Self-Healing MLOps Comparison Study")
    print("=" * 78)
    print()

    all_results = {}

    # Run on fraud (available locally)
    if config.FRAUD_CSV.exists():
        all_results["fraud"] = run_drift_on_dataset(
            "fraud", max_windows=15, mmd_sample=80
        )
    else:
        print(f"  SKIP fraud: {config.FRAUD_CSV} not found\n")

    # Run on elec2 (from river or CSV)
    try:
        all_results["elec2"] = run_drift_on_dataset(
            "elec2", max_windows=15, mmd_sample=80
        )
    except Exception as e:
        print(f"  SKIP elec2: {e}\n")

    # Run on lendingclub (large — fewer windows, smaller MMD sample)
    if config.LENDINGCLUB_CSV.exists():
        all_results["lendingclub"] = run_drift_on_dataset(
            "lendingclub", max_windows=10, mmd_sample=60
        )
    else:
        print(f"  SKIP lendingclub: {config.LENDINGCLUB_CSV} not found\n")

    # Final cross-dataset comparison
    print("=" * 78)
    print("  CROSS-DATASET COMPARISON")
    print("=" * 78)
    print(f"\n  {'Dataset':<15s} {'Windows':>8s} {'Evidently':>12s} {'Alibi-BH':>12s} {'Alibi-MMD':>12s}")
    print("  " + "-" * 62)
    for name, results in all_results.items():
        n = len(results)
        ev = sum(1 for r in results if r["evidently_drift"])
        bh = sum(1 for r in results if r["alibi_bh_drift"])
        mmd = sum(1 for r in results if r["alibi_mmd_drift"])
        print(f"  {name:<15s} {n:>8d} {ev:>6d}/{n:<5d} {bh:>6d}/{n:<5d} {mmd:>6d}/{n:<5d}")

    print("\n  Key:")
    print("  - Fraud: synthetic drift (injected). Expect few alarms on clean test data.")
    print("  - Elec2: REAL temporal drift. Expect alarms to increase in later windows.")
    print("  - LendingClub: gradual real drift. Expect moderate alarms.")
    print()


if __name__ == "__main__":
    main()

