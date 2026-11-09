# =============================================================================
# main.py
# Runs both drift detectors on all 8 test cases and prints a comparison report.
# No external imports.
# =============================================================================

from evidently_style_detector import EvidentlyStyleDriftDetector
from alibi_style_detector      import AlibiStyleDriftDetector
from data_generator            import (
    generate_all_cases,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
)


# ------------------------------------------------------------------
# PRINT HELPERS
# ------------------------------------------------------------------

def _hline(char="─", width=78):
    print(char * width)


def _header(text, char="═", width=78):
    print(char * width)
    side = (width - len(text) - 2) // 2
    print(f"{char * side} {text} {char * (width - side - len(text) - 2)}")
    print(char * width)


def _bool_to_badge(val):
    return "DRIFT ✓" if val else "clean  "


def _short_result(result):
    """Extract the key numbers from a result dict for display."""
    drift = result.get("dataset_drift_detected", False)
    method = result.get("detection_method", result.get("detection_method", ""))

    if "multivariate_mmd" in result.get("detection_method", ""):
        p = result.get("p_value", None)
        mmd = result.get("mmd_statistic", None)
        return (drift, f"MMD²={round(mmd,5)}, p={round(p,4)}" if p is not None else "?")

    elif "columnwise_bh" in result.get("detection_method", ""):
        n_d = result.get("n_features_drifted", 0)
        n_t = result.get("n_features_tested",  0)
        return (drift, f"{n_d}/{n_t} features rejected (BH)")

    else:   # Evidently majority-vote
        n_d = result.get("n_features_drifted", 0)
        n_t = result.get("n_features_tested",  0)
        frac = result.get("drift_fraction", 0)
        return (drift, f"{n_d}/{n_t} features ({round(frac*100,0):.0f}%) drifted")


def _print_feature_table(result, label, num_feats, cat_feats):
    """Print per-feature breakdown."""
    fr = result.get("feature_results", {})
    if not fr:
        return
    print(f"\n  {label} — per-feature breakdown:")
    print(f"  {'Feature':<12} {'Type':<12} {'Drift?':<9} {'Key stat'}")
    print(f"  {'-'*12} {'-'*12} {'-'*9} {'-'*30}")
    for feat in num_feats + cat_feats:
        if feat not in fr:
            continue
        r = fr[feat]
        if "error" in r:
            print(f"  {feat:<12} {r.get('type','?'):<12} {'ERROR':<9} {r['error']}")
            continue
        flag = "DRIFT ✓" if r.get("drift_detected") else "clean  "
        typ  = r.get("type", "?")
        if typ == "numerical":
            ks_p = r.get("ks_pvalue", "?")
            psi  = r.get("psi", "?")
            ks_str  = str(ks_p) if not isinstance(ks_p, float) else str(round(ks_p, 4))
            psi_str = str(psi)  if not isinstance(psi,  float) else str(round(psi,  4))
            stat = f"KS p={ks_str}, PSI={psi_str}"
        else:
            psi = r.get("psi", "?")
            psi_str = str(psi) if not isinstance(psi, float) else str(round(psi, 4))
            stat = f"PSI={psi_str}"
            # For Alibi BH, also show chi2
            if "chi2_statistic" in r:
                chi2  = r["chi2_statistic"]
                chi2p = r.get("chi2_pvalue", "?")
                chi2p_str = str(chi2p) if not isinstance(chi2p, float) else str(round(chi2p, 4))
                stat = f"χ²={round(chi2,3)}, p={chi2p_str}"
        print(f"  {feat:<12} {typ:<12} {flag:<9} {stat}")


def _print_bh_detail(result):
    """Print BH FDR correction detail for Alibi columnwise mode."""
    detail = result.get("bh_fdr_detail", [])
    if not detail:
        return
    alpha = result.get("alpha", 0.05)
    m     = result.get("n_features_tested", len(detail))
    print(f"\n  BH-FDR correction (alpha={alpha}, m={m}):")
    print(f"  {'Rank':<6} {'Feature':<14} {'p-value':<12} {'BH threshold':<14} {'Rejected?'}")
    print(f"  {'-'*6} {'-'*14} {'-'*12} {'-'*14} {'-'*10}")
    for rank, item in enumerate(detail, 1):
        bh_thresh = round((rank / m) * alpha, 6)
        rej = "YES ✓" if item["rejected"] else "no"
        print(f"  {rank:<6} {item['feature']:<14} {item['p_value']:<12} {bh_thresh:<14} {rej}")


# ------------------------------------------------------------------
# PER-CASE RUNNER
# ------------------------------------------------------------------

def run_case(ref, cur, meta,
             ev_det, alibi_bh_det, alibi_mmd_det,
             num_feats, cat_feats,
             verbose=True):
    """
    Run all three detectors on one (reference, current) pair.
    Returns summary dict.
    """
    # Run detectors
    ev_result  = ev_det.detect(ref, cur, num_feats, cat_feats)
    bh_result  = alibi_bh_det.detect(ref, cur, num_feats, cat_feats)
    mmd_result = alibi_mmd_det.detect(ref, cur, num_feats, [])  # MMD: numerical only

    ev_drift,  ev_detail  = _short_result(ev_result)
    bh_drift,  bh_detail  = _short_result(bh_result)
    mmd_drift, mmd_detail = _short_result(mmd_result)

    if verbose:
        expected = meta.get("expected_drift", "?")
        exp_str  = (
            "YES (expected)" if expected is True  else
            "NO  (expected)" if expected is False else
            str(expected)
        )
        print(f"\n  Expected : {exp_str}")
        if meta.get("drifted_features"):
            print(f"  Ground truth drifted features : {meta['drifted_features']}")
        if meta.get("note"):
            print(f"  Note     : {meta['note']}")

        print()
        print(f"  Detector 1 ─ Evidently-style (KS + PSI + majority-vote)")
        print(f"    Result : {_bool_to_badge(ev_drift)}")
        print(f"    Detail : {ev_detail}")
        _print_feature_table(ev_result, "Evidently", num_feats, cat_feats)

        print()
        print(f"  Detector 2 ─ Alibi-style (KS + Chi² + BH-FDR)")
        print(f"    Result : {_bool_to_badge(bh_drift)}")
        print(f"    Detail : {bh_detail}")
        _print_bh_detail(bh_result)

        print()
        print(f"  Detector 3 ─ Alibi-style (Multivariate MMD)")
        print(f"    Result : {_bool_to_badge(mmd_drift)}")
        print(f"    Detail : {mmd_detail}")
        sigma = mmd_result.get("sigma_bandwidth", "?")
        n_v_r = mmd_result.get("n_valid_ref", "?")
        n_v_c = mmd_result.get("n_valid_cur", "?")
        print(f"    σ      : {sigma}   valid rows ref/cur: {n_v_r}/{n_v_c}")

        agree = (ev_drift == bh_drift)
        all_agree = (ev_drift == bh_drift == mmd_drift)
        print()
        print(f"  Agreement : Evidently vs Alibi-BH {'✓ agree' if agree else '✗ DIFFER'} | "
              f"All three {'✓ agree' if all_agree else '✗ DIFFER'}")

    return {
        "case":        meta["case"],
        "name":        meta["name"],
        "expected":    meta.get("expected_drift"),
        "evidently":   ev_drift,
        "alibi_bh":    bh_drift,
        "alibi_mmd":   mmd_drift,
    }


# ------------------------------------------------------------------
# COMPARISON ANALYSIS
# ------------------------------------------------------------------

def print_comparison_table(summaries):
    _header("SUMMARY COMPARISON TABLE")
    print(f"\n  {'Case':<4} {'Name':<32} {'Expected':<10} "
          f"{'Evidently':<12} {'Alibi-BH':<12} {'Alibi-MMD':<12} {'Agree?'}")
    print(f"  {'-'*4} {'-'*32} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*7}")

    for s in summaries:
        exp = s["expected"]
        ev  = s["evidently"]
        bh  = s["alibi_bh"]
        mmd = s["alibi_mmd"]

        exp_str = "YES" if exp is True else ("NO" if exp is False else str(exp))
        ev_str  = "DRIFT" if ev  else "clean"
        bh_str  = "DRIFT" if bh  else "clean"
        mmd_str = "DRIFT" if mmd else "clean"

        agree = (ev == bh == mmd)
        agree_str = "✓" if agree else "✗"

        print(f"  {s['case']:<4} {s['name']:<32} {exp_str:<10} "
              f"{ev_str:<12} {bh_str:<12} {mmd_str:<12} {agree_str}")


def print_method_analysis(summaries):
    _header("METHOD ANALYSIS")

    # Filter cases with known expected drift (bool)
    known = [s for s in summaries if isinstance(s["expected"], bool)]

    for detector, key in [
        ("Evidently  (KS+PSI+majority)", "evidently"),
        ("Alibi-BH   (KS+Chi²+BH-FDR)", "alibi_bh"),
        ("Alibi-MMD  (multivariate MMD)", "alibi_mmd"),
    ]:
        tp = sum(1 for s in known if s[key] and s["expected"])
        tn = sum(1 for s in known if not s[key] and not s["expected"])
        fp = sum(1 for s in known if s[key] and not s["expected"])
        fn = sum(1 for s in known if not s[key] and s["expected"])

        total_p = sum(1 for s in known if s["expected"])
        total_n = sum(1 for s in known if not s["expected"])

        tpr = tp / total_p if total_p > 0 else 0.0
        fpr = fp / total_n if total_n > 0 else 0.0

        print(f"\n  {detector}")
        print(f"    TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"    TPR (sensitivity) = {round(tpr,2)}   FPR = {round(fpr,2)}")


def print_why_different():
    _header("WHY THE MODELS MAY PRODUCE DIFFERENT RESULTS")

    reasons = [
        (
            "1. Multiple-testing correction",
            "Evidently uses majority voting (>50% features drift → alert).\n"
            "   Alibi-BH uses Benjamini-Hochberg FDR, which is more statistically rigorous.\n"
            "   BH rejects fewer features than raw α=0.05 when many features are tested,\n"
            "   making Alibi-BH LESS likely to false-alarm than Evidently on wide datasets."
        ),
        (
            "2. Categorical test method",
            "Evidently uses PSI (Population Stability Index) for categoricals.\n"
            "   Alibi-BH uses the chi-square goodness-of-fit test + Wilson-Hilferty p-value.\n"
            "   PSI is threshold-based (>0.20) and symmetric; chi-square is hypothesis-based.\n"
            "   PSI detects large shifts earlier; chi-square has calibrated Type I error."
        ),
        (
            "3. Joint vs. marginal testing",
            "Both column-wise methods (Evidently and Alibi-BH) test each feature's\n"
            "   MARGINAL distribution independently. They cannot detect drift that changes\n"
            "   only feature correlations while keeping marginals unchanged (Case 5).\n"
            "   Alibi-MMD tests the JOINT distribution via the kernel, so it catches\n"
            "   correlation drift that is completely invisible to column-wise methods."
        ),
        (
            "4. PSI bin construction",
            "Evidently PSI uses equal-frequency bins from the reference.\n"
            "   For a constant reference feature, all data collapses to one bin → PSI=0\n"
            "   even when the current has shifted. The KS test compensates for this case."
        ),
        (
            "5. Sample-size sensitivity",
            "KS power scales with sqrt(n). With small current samples (Case 8: n=50),\n"
            "   the KS test needs a larger D to reach p<0.05 than with n=200.\n"
            "   Alibi-BH's BH correction is tighter with more features (m larger → smaller\n"
            "   individual BH threshold), making it slightly more conservative."
        ),
        (
            "6. MMD bandwidth choice",
            "The MMD result depends strongly on σ (kernel bandwidth).\n"
            "   We use the median heuristic (σ = median pairwise distance / √2).\n"
            "   If the reference and current are far apart in feature space, σ is large and\n"
            "   the kernel 'blurs' — MMD may underestimate the distance. The permutation\n"
            "   test is distribution-free but has variance ~1/√n_permutations."
        ),
    ]

    for title, body in reasons:
        print(f"\n  {title}")
        for line in body.split("\n"):
            print(f"  {line}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    _header("DRIFT DETECTOR COMPARISON  —  Self-Healing MLOps Project")

    # Instantiate detectors
    ev_det       = EvidentlyStyleDriftDetector(ks_alpha=0.05, psi_threshold=0.20)
    alibi_bh_det = AlibiStyleDriftDetector(method='columnwise_bh', alpha=0.05)
    alibi_mmd_det = AlibiStyleDriftDetector(
        method='multivariate_mmd', alpha=0.05,
        n_permutations=200, seed=42   # reduce to 100 for faster runs
    )

    print("\nDetector configuration:")
    print(f"  Evidently  : KS α={ev_det.ks_alpha}, PSI>{ev_det.psi_threshold}, "
          f"majority>{ev_det.majority_threshold}")
    print(f"  Alibi-BH   : KS + chi² + BH-FDR α=0.05")
    print(f"  Alibi-MMD  : Gaussian RBF, median σ, {alibi_mmd_det.n_permutations} permutations")

    # Generate all 8 test cases
    all_cases = generate_all_cases()
    summaries = []

    for ref, cur, meta in all_cases:
        _header(
            f"CASE {meta['case']}:  {meta['name'].upper().replace('_',' ')}",
            char="─", width=78
        )
        print(f"  {meta['description']}")
        print(f"  Reference n={len(ref)}, Current n={len(cur)}")

        summary = run_case(
            ref, cur, meta,
            ev_det, alibi_bh_det, alibi_mmd_det,
            NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
            verbose=True,
        )
        summaries.append(summary)
        _hline()

    # Summary table
    print()
    print_comparison_table(summaries)

    print()
    print_method_analysis(summaries)

    print()
    print_why_different()

    _header("END OF REPORT")
    print()
    print("Next step: run  python test_models.py  to validate all")
    print("individual algorithm implementations before trusting these results.")
    print()


if __name__ == "__main__":
    main()
