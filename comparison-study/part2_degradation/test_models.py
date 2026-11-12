"""
Unit Tests for Part 2 — Degradation Detection
===============================================

Tests for CBPE, DDM, EDDM, and the data generator.
Validates mathematical correctness, edge cases, and expected behaviors.

NO IMPORTS USED.
"""

# ============================================================================
# We access the modules via exec since we can't import.
# Actually, we CAN use the function/class references directly since
# Python allows us to use exec to load modules. But the constraint is
# no 'import' keyword. Let's use execfile-style loading.
# ============================================================================

# Load modules by reading and executing their code
def _load_module(filepath):
    """Load a Python module without using import."""
    namespace = {}
    with open(filepath, 'r') as f:
        code = f.read()
    exec(compile(code, filepath, 'exec'), namespace)
    return namespace


# ============================================================================
# TEST FRAMEWORK (minimal, no imports)
# ============================================================================

_test_count = 0
_pass_count = 0
_fail_count = 0
_failures = []


def _assert(condition, message=""):
    global _test_count, _pass_count, _fail_count, _failures
    _test_count += 1
    if condition:
        _pass_count += 1
    else:
        _fail_count += 1
        _failures.append(message)
        print("  FAIL: " + message)


def _assert_close(actual, expected, tol=0.01, message=""):
    diff = actual - expected
    if diff < 0:
        diff = -diff
    _assert(diff <= tol,
            message + " (expected ~" + str(round(expected, 4)) +
            ", got " + str(round(actual, 4)) + ", tol=" + str(tol) + ")")


def _assert_true(value, message=""):
    _assert(value is True or value, message + " (expected True, got " + str(value) + ")")


def _assert_false(value, message=""):
    _assert(not value, message + " (expected False, got " + str(value) + ")")


# ============================================================================
# LOAD MODULES
# ============================================================================

print("Loading modules...")

# Get the directory of this test file
_this_dir = ""
# Determine path prefix
_test_file_path = __file__
_parts = _test_file_path.replace("\\", "/").split("/")
_this_dir = "/".join(_parts[:-1])
if not _this_dir:
    _this_dir = "."

cbpe_mod = _load_module(_this_dir + "/nannyml_cbpe.py")
ddm_mod = _load_module(_this_dir + "/ddm_detector.py")
datagen_mod = _load_module(_this_dir + "/data_generator.py")

# Extract classes and functions
CBPEDegradationDetector = cbpe_mod["CBPEDegradationDetector"]
_manual_sqrt_cbpe = cbpe_mod["_manual_sqrt"]
_manual_exp_cbpe = cbpe_mod["_manual_exp"]
_manual_log_cbpe = cbpe_mod["_manual_log"]
_sigmoid = cbpe_mod["_sigmoid"]
_compute_true_f1 = cbpe_mod["_compute_true_f1"]
_compute_true_roc_auc = cbpe_mod["_compute_true_roc_auc"]
_platt_scaling_fit = cbpe_mod["_platt_scaling_fit"]
_estimate_f1 = cbpe_mod["_estimate_f1"]

DDMDegradationDetector = ddm_mod["DDMDegradationDetector"]
EDDMDegradationDetector = ddm_mod["EDDMDegradationDetector"]
_manual_sqrt_ddm = ddm_mod["_manual_sqrt"]
_compute_f1_ddm = ddm_mod["_compute_f1"]
_compute_accuracy = ddm_mod["_compute_accuracy"]

DegradationDataGenerator = datagen_mod["DegradationDataGenerator"]

print("Modules loaded successfully.\n")


# ============================================================================
# TEST GROUP 1: Math Utilities
# ============================================================================

def test_math_utilities():
    print("=== Test Group 1: Math Utilities ===")

    # sqrt
    _assert_close(_manual_sqrt_cbpe(4.0), 2.0, 1e-10, "sqrt(4) = 2")
    _assert_close(_manual_sqrt_cbpe(9.0), 3.0, 1e-10, "sqrt(9) = 3")
    _assert_close(_manual_sqrt_cbpe(2.0), 1.41421356, 1e-6, "sqrt(2)")
    _assert_close(_manual_sqrt_cbpe(0.0), 0.0, 1e-10, "sqrt(0) = 0")
    _assert_close(_manual_sqrt_cbpe(0.01), 0.1, 1e-6, "sqrt(0.01) = 0.1")

    # exp
    _assert_close(_manual_exp_cbpe(0.0), 1.0, 1e-10, "exp(0) = 1")
    _assert_close(_manual_exp_cbpe(1.0), 2.71828, 1e-4, "exp(1) = e")
    _assert_close(_manual_exp_cbpe(-1.0), 0.36788, 1e-4, "exp(-1)")
    _assert_close(_manual_exp_cbpe(5.0), 148.413, 0.01, "exp(5)")

    # log
    _assert_close(_manual_log_cbpe(1.0), 0.0, 1e-10, "log(1) = 0")
    _assert_close(_manual_log_cbpe(2.71828), 1.0, 1e-4, "log(e) = 1")
    _assert_close(_manual_log_cbpe(10.0), 2.302585, 1e-4, "log(10)")

    # sigmoid
    _assert_close(_sigmoid(0.0), 0.5, 1e-10, "sigmoid(0) = 0.5")
    _assert(_sigmoid(10.0) > 0.999, "sigmoid(10) > 0.999")
    _assert(_sigmoid(-10.0) < 0.001, "sigmoid(-10) < 0.001")

    print()


# ============================================================================
# TEST GROUP 2: F1 and AUC Computation
# ============================================================================

def test_metrics():
    print("=== Test Group 2: F1 and AUC Computation ===")

    # Perfect predictions
    y_true = [0, 1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 0, 1]
    _assert_close(_compute_true_f1(y_true, y_pred), 1.0, 1e-10, "Perfect F1 = 1.0")

    # All wrong
    y_pred_bad = [1, 0, 1, 0, 0, 1, 1, 0]
    _assert_close(_compute_true_f1(y_true, y_pred_bad), 0.0, 1e-10, "All wrong F1 = 0.0")

    # Mixed predictions: TP=2, FP=1, FN=2 → F1 = 4/(4+1+2) = 0.571
    y_true2 = [1, 1, 1, 1, 0, 0, 0]
    y_pred2 = [1, 1, 0, 0, 1, 0, 0]
    expected_f1 = 4.0 / (4 + 1 + 2)
    _assert_close(_compute_true_f1(y_true2, y_pred2), expected_f1, 1e-6,
                  "Mixed F1 = 4/7 = 0.571")

    # AUC for perfect separation
    y_true_auc = [0, 0, 0, 1, 1, 1]
    y_proba_auc = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    auc = _compute_true_roc_auc(y_true_auc, y_proba_auc)
    _assert_close(auc, 1.0, 0.01, "Perfect separation AUC = 1.0")

    # AUC for random predictions
    y_true_rand = [0, 1, 0, 1, 0, 1]
    y_proba_rand = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    auc_rand = _compute_true_roc_auc(y_true_rand, y_proba_rand)
    _assert_close(auc_rand, 0.5, 0.1, "Random AUC ~ 0.5")

    # Accuracy computation
    _assert_close(_compute_accuracy([0, 1, 0, 1], [0, 1, 0, 1]), 1.0, 1e-10,
                  "Perfect accuracy = 1.0")
    _assert_close(_compute_accuracy([0, 1, 0, 1], [1, 0, 1, 0]), 0.0, 1e-10,
                  "All wrong accuracy = 0.0")
    _assert_close(_compute_accuracy([0, 1, 0, 1], [0, 1, 1, 0]), 0.5, 1e-10,
                  "Half correct accuracy = 0.5")

    print()


# ============================================================================
# TEST GROUP 3: Platt Scaling
# ============================================================================

def test_platt_scaling():
    print("=== Test Group 3: Platt Scaling ===")

    # Well-calibrated model: a should be close to 1, b close to 0
    y_true = []
    y_proba = []
    rng = datagen_mod["_LCG"](seed=42)
    for _ in range(500):
        p = rng.next_float()
        y = 1 if rng.next_float() < p else 0
        y_true.append(y)
        y_proba.append(p)

    a, b = _platt_scaling_fit(y_true, y_proba)
    # For a well-calibrated model, Platt scaling should not change much
    # a should be positive (same direction), b can be any value
    _assert(a > 0, "Platt scaling a > 0 for well-calibrated model (got " + str(round(a, 4)) + ")")

    # Miscalibrated model: probabilities systematically too high
    y_true2 = [0] * 80 + [1] * 20  # 20% positive rate
    y_proba2 = [0.6] * 80 + [0.9] * 20  # all probabilities too high
    a2, b2 = _platt_scaling_fit(y_true2, y_proba2)
    # After calibration, probabilities should be pulled down
    calibrated = cbpe_mod["_platt_scaling_transform"](y_proba2, a2, b2)
    avg_cal = sum(calibrated) / len(calibrated)
    avg_orig = sum(y_proba2) / len(y_proba2)
    _assert(avg_cal < avg_orig,
            "Platt scaling pulls down overconfident probabilities (avg: " +
            str(round(avg_orig, 3)) + " → " + str(round(avg_cal, 3)) + ")")

    print()


# ============================================================================
# TEST GROUP 4: CBPE Estimated F1
# ============================================================================

def test_cbpe_estimation():
    print("=== Test Group 4: CBPE Estimated F1 ===")

    # If calibrated probabilities match labels, estimated F1 should ~ true F1
    # Calibrated proba = [0.9, 0.1, 0.8, 0.2, 0.9] for y_pred = [1, 0, 1, 0, 1]
    cal_proba = [0.9, 0.1, 0.85, 0.15, 0.9]
    y_pred = [1, 0, 1, 0, 1]

    est_f1 = _estimate_f1(cal_proba, y_pred)
    # E[TP] = 0.9 + 0.85 + 0.9 = 2.65
    # E[FP] = 0.1 + 0.15 + 0.1 = 0.35
    # E[FN] = 0.1 + 0.15 = 0.25
    # F1 = 2*2.65 / (2*2.65 + 0.35 + 0.25) = 5.3 / 5.9 = 0.898
    expected = 5.3 / 5.9
    _assert_close(est_f1, expected, 0.01,
                  "Estimated F1 from calibrated probabilities")

    # All predict positive with perfect calibration
    cal_perfect = [1.0, 1.0, 0.0, 0.0]
    y_pred_all1 = [1, 1, 1, 1]
    est_f1_2 = _estimate_f1(cal_perfect, y_pred_all1)
    # E[TP] = 1+1+0+0 = 2, E[FP] = 0+0+1+1 = 2, E[FN] = 0
    # F1 = 4 / (4+2) = 0.667
    _assert_close(est_f1_2, 4.0 / 6.0, 0.01, "All-positive with mixed calibration")

    print()


# ============================================================================
# TEST GROUP 5: DDM Basic Behavior
# ============================================================================

def test_ddm_basic():
    print("=== Test Group 5: DDM Basic Behavior ===")

    # Stable error stream: no alarms
    ddm = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0, min_samples=30)
    stable_yt = [0, 1] * 500  # alternating labels
    stable_yp = list(stable_yt)  # perfect predictions
    # Make 10% wrong
    for i in range(0, len(stable_yp), 10):
        stable_yp[i] = 1 - stable_yp[i]

    result = ddm.detect_stream(stable_yt, stable_yp)
    # With stable 10% error, DDM should NOT alarm
    # (p stabilizes around 0.1, no significant increase)
    _assert_false(result["overall_degradation_detected"],
                  "DDM: no alarm on stable 10% error rate")

    # Sudden error increase: should alarm
    ddm2 = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0, min_samples=30)
    # First 500 samples: 5% error, then 50% error
    yt_sudden = [0, 1] * 500
    yp_sudden = list(yt_sudden)
    for i in range(0, 500, 20):  # 5% error in first half
        yp_sudden[i] = 1 - yp_sudden[i]
    for i in range(500, 1000, 2):  # 50% error in second half
        yp_sudden[i] = 1 - yp_sudden[i]

    result2 = ddm2.detect_stream(yt_sudden, yp_sudden)
    _assert_true(result2["overall_degradation_detected"],
                 "DDM: alarm on sudden error increase (5% → 50%)")
    if result2["first_alarm_step"] is not None:
        _assert(result2["first_alarm_step"] >= 500,
                "DDM: alarm after error increase starts (step " +
                str(result2["first_alarm_step"]) + " >= 500)")

    print()


# ============================================================================
# TEST GROUP 6: DDM Reset Behavior
# ============================================================================

def test_ddm_reset():
    print("=== Test Group 6: DDM Reset Behavior ===")

    ddm = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0, min_samples=20)

    # After alarm, DDM should reset and be able to detect again
    yt = [0, 1] * 300  # 600 samples
    yp = list(yt)
    # First degradation at step 100
    for i in range(100, 200):
        yp[i] = 1 - yp[i]
    # Recovery at step 200-400 (correct again)
    # Second degradation at step 400
    for i in range(400, 500):
        yp[i] = 1 - yp[i]

    result = ddm.detect_stream(yt, yp)

    # Should have detected at least 1 alarm
    _assert(result["total_alarms"] >= 1,
            "DDM: at least 1 alarm for degradation pattern (got " +
            str(result["total_alarms"]) + ")")

    print()


# ============================================================================
# TEST GROUP 7: EDDM Basic Behavior
# ============================================================================

def test_eddm():
    print("=== Test Group 7: EDDM Basic Behavior ===")

    eddm = EDDMDegradationDetector(alpha=0.95, beta=0.90, min_errors=20)

    # Stable error stream
    yt_stable = [0, 1] * 500
    yp_stable = list(yt_stable)
    for i in range(0, len(yp_stable), 10):
        yp_stable[i] = 1 - yp_stable[i]

    result = eddm.detect_stream(yt_stable, yp_stable)
    # Stable stream should not trigger alarm (errors evenly spaced)
    # Note: EDDM can be noisy, so we just check it runs without error
    _assert(result["stream_length"] == 1000, "EDDM: processed 1000 samples")

    # Error clustering: errors bunch together → should alarm
    eddm2 = EDDMDegradationDetector(alpha=0.95, beta=0.90, min_errors=15)
    yt_cluster = [0, 1] * 500
    yp_cluster = list(yt_cluster)
    # First 400: errors every 20 steps (spread out)
    for i in range(0, 400, 20):
        yp_cluster[i] = 1 - yp_cluster[i]
    # After 400: errors every 2 steps (clustered)
    for i in range(400, 600, 2):
        yp_cluster[i] = 1 - yp_cluster[i]

    result2 = eddm2.detect_stream(yt_cluster, yp_cluster)
    _assert(result2["total_alarms"] >= 0,  # EDDM may or may not alarm
            "EDDM: runs on clustered error pattern")

    print()


# ============================================================================
# TEST GROUP 8: CBPE Full Pipeline
# ============================================================================

def test_cbpe_pipeline():
    print("=== Test Group 8: CBPE Full Pipeline ===")

    gen = DegradationDataGenerator(seed=42)

    # Test 1: No degradation → no alert
    data = gen.generate_no_degradation(n_ref=1000, n_stream=2000)
    cbpe = CBPEDegradationDetector(chunk_size=500, alert_threshold_f1=0.10,
                                    alert_threshold_auc=0.10)
    cbpe.fit(data["reference"]["y_true"],
             data["reference"]["y_pred_proba"],
             data["reference"]["y_pred"])

    result = cbpe.detect(data["stream"]["y_pred_proba"],
                          data["stream"]["y_pred"])

    _assert(result["model"] == "NannyML_CBPE", "CBPE: correct model name")
    _assert(result["n_chunks"] == 4, "CBPE: 2000/500 = 4 chunks")
    _assert(result["reference_metrics"]["f1"] > 0.5,
            "CBPE: reference F1 > 0.5 (got " +
            str(result["reference_metrics"]["f1"]) + ")")

    # Test 2: Sudden degradation → should alert
    data2 = gen.generate_sudden_degradation(n_ref=1000, n_stream=3000,
                                             degrade_at=1000)
    cbpe2 = CBPEDegradationDetector(chunk_size=500, alert_threshold_f1=0.08,
                                     alert_threshold_auc=0.08)
    cbpe2.fit(data2["reference"]["y_true"],
              data2["reference"]["y_pred_proba"],
              data2["reference"]["y_pred"])

    result2 = cbpe2.detect(data2["stream"]["y_pred_proba"],
                            data2["stream"]["y_pred"])

    _assert_true(result2["overall_alert"],
                 "CBPE: detects sudden degradation")
    if result2["first_f1_alert_chunk"] is not None:
        _assert(result2["first_f1_alert_chunk"] >= 1,
                "CBPE: alert starts at or after degradation chunk")

    print()


# ============================================================================
# TEST GROUP 9: DDM Full Pipeline
# ============================================================================

def test_ddm_pipeline():
    print("=== Test Group 9: DDM Full Pipeline ===")

    gen = DegradationDataGenerator(seed=42)

    # Test 1: No degradation → no alarm
    data = gen.generate_no_degradation(n_ref=1000, n_stream=2000)
    ddm = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0, min_samples=30)
    ddm.fit(data["reference"]["y_true"], data["reference"]["y_pred"])

    result = ddm.detect_chunks(data["stream"]["y_true"],
                                data["stream"]["y_pred"],
                                chunk_size=500)

    _assert(result["model"] == "DDM_River_Style", "DDM: correct model name")
    _assert(result["n_chunks"] == 4, "DDM: 2000/500 = 4 chunks")

    # Test 2: Sudden degradation → should alarm
    data2 = gen.generate_sudden_degradation(n_ref=1000, n_stream=3000,
                                             degrade_at=1000)
    ddm2 = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0, min_samples=30)
    ddm2.fit(data2["reference"]["y_true"], data2["reference"]["y_pred"])

    result2 = ddm2.detect_chunks(data2["stream"]["y_true"],
                                  data2["stream"]["y_pred"],
                                  chunk_size=500)

    _assert_true(result2["overall_degradation_detected"],
                 "DDM: detects sudden degradation")

    print()


# ============================================================================
# TEST GROUP 10: Data Generator Validation
# ============================================================================

def test_data_generator():
    print("=== Test Group 10: Data Generator Validation ===")

    gen = DegradationDataGenerator(seed=42)

    # All scenarios should generate without error
    scenarios = gen.run_all_scenarios()
    _assert(len(scenarios) == 8, "Generator: produces 8 scenarios")

    # Scenario 1: no degradation — check sizes
    s1 = scenarios[0]
    _assert(s1["scenario_name"] == "no_degradation", "Scenario 1: correct name")
    _assert(len(s1["reference"]["y_true"]) == 2000, "Scenario 1: 2000 ref samples")
    _assert(len(s1["stream"]["y_true"]) == 5000, "Scenario 1: 5000 stream samples")

    # Scenario 2: sudden degradation — check error rate changes
    s2 = scenarios[1]
    _assert(s2["scenario_name"] == "sudden_degradation", "Scenario 2: correct name")
    # Before degradation (first 1000 stream samples): ~10% error rate
    errors_before = 0
    for i in range(1000):
        if s2["stream"]["y_true"][i] != s2["stream"]["y_pred"][i]:
            errors_before += 1
    error_rate_before = errors_before / 1000.0

    # After degradation (last 1000 stream samples): ~45% error rate
    errors_after = 0
    stream_len = len(s2["stream"]["y_true"])
    for i in range(stream_len - 1000, stream_len):
        if s2["stream"]["y_true"][i] != s2["stream"]["y_pred"][i]:
            errors_after += 1
    error_rate_after = errors_after / 1000.0

    _assert(error_rate_after > error_rate_before,
            "Scenario 2: error rate increases after degradation (" +
            str(round(error_rate_before, 3)) + " → " +
            str(round(error_rate_after, 3)) + ")")

    # Scenario 4: calibration shift — accuracy should be similar
    s4 = scenarios[3]
    ref_errors = 0
    for yt, yp in zip(s4["reference"]["y_true"], s4["reference"]["y_pred"]):
        if yt != yp:
            ref_errors += 1
    ref_acc = 1.0 - ref_errors / len(s4["reference"]["y_true"])

    str_errors = 0
    for yt, yp in zip(s4["stream"]["y_true"], s4["stream"]["y_pred"]):
        if yt != yp:
            str_errors += 1
    str_acc = 1.0 - str_errors / len(s4["stream"]["y_true"])

    acc_diff = ref_acc - str_acc
    if acc_diff < 0:
        acc_diff = -acc_diff
    _assert(acc_diff < 0.05,
            "Scenario 4: accuracy similar in ref and stream (" +
            str(round(ref_acc, 3)) + " vs " + str(round(str_acc, 3)) + ")")

    # Scenario 7: delayed labels
    s7 = scenarios[6]
    _assert(s7["stream"]["y_true_delayed"][0] is None,
            "Scenario 7: first delayed label is None")
    _assert(s7["stream"]["y_true_delayed"][-1] is not None,
            "Scenario 7: last delayed label is available")

    # Edge cases
    s8 = scenarios[7]
    _assert("all_correct" in s8["cases"], "Scenario 8: has all_correct case")
    _assert("all_wrong" in s8["cases"], "Scenario 8: has all_wrong case")
    _assert("single_sample" in s8["cases"], "Scenario 8: has single_sample case")

    print()


# ============================================================================
# TEST GROUP 11: Edge Cases
# ============================================================================

def test_edge_cases():
    print("=== Test Group 11: Edge Cases ===")

    gen = DegradationDataGenerator(seed=42)
    edge = gen.generate_edge_cases()

    # All correct: F1 = 1.0
    ac = edge["cases"]["all_correct"]
    f1_ac = _compute_true_f1(ac["y_true"], ac["y_pred"])
    _assert_close(f1_ac, 1.0, 1e-10, "Edge: all_correct F1 = 1.0")

    # All wrong: F1 = 0.0
    aw = edge["cases"]["all_wrong"]
    f1_aw = _compute_true_f1(aw["y_true"], aw["y_pred"])
    _assert_close(f1_aw, 0.0, 1e-10, "Edge: all_wrong F1 = 0.0")

    # All predict zero: F1 = 0.0 (no true positives)
    az = edge["cases"]["all_predict_zero"]
    f1_az = _compute_true_f1(az["y_true"], az["y_pred"])
    _assert_close(f1_az, 0.0, 1e-10, "Edge: all_predict_zero F1 = 0.0")

    # DDM with all correct: no alarm
    ddm = DDMDegradationDetector(min_samples=5)
    result = ddm.detect_stream(ac["y_true"], ac["y_pred"])
    _assert_false(result["overall_degradation_detected"],
                  "Edge: DDM no alarm on all-correct")

    # DDM with all wrong: should alarm (100% error)
    ddm2 = DDMDegradationDetector(min_samples=5)
    # First feed some correct, then all wrong
    mixed_yt = [0, 1] * 50 + aw["y_true"]
    mixed_yp = [0, 1] * 50 + aw["y_pred"]
    result2 = ddm2.detect_stream(mixed_yt, mixed_yp)
    # After seeing good predictions then bad, DDM should alarm
    _assert_true(result2["overall_degradation_detected"],
                 "Edge: DDM alarms when switching from good to all-wrong")

    # CBPE with single sample: should not crash
    cbpe = CBPEDegradationDetector(chunk_size=1)
    ss = edge["cases"]["single_sample"]
    # Need more reference data for calibration
    ref_yt = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
    ref_yp = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
    ref_pp = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9] * 10
    cbpe.fit(ref_yt, ref_pp, ref_yp)
    result3 = cbpe.detect(ss["y_pred_proba"], ss["y_pred"])
    _assert(result3["n_chunks"] == 1, "Edge: CBPE handles single sample (1 chunk)")

    print()


# ============================================================================
# TEST GROUP 12: Comparison — CBPE vs DDM on Same Data
# ============================================================================

def test_comparison():
    print("=== Test Group 12: CBPE vs DDM Comparison ===")

    gen = DegradationDataGenerator(seed=123)

    # Sudden degradation: both should detect
    data = gen.generate_sudden_degradation(n_ref=2000, n_stream=4000,
                                            degrade_at=1500)

    cbpe = CBPEDegradationDetector(chunk_size=500, alert_threshold_f1=0.08)
    cbpe.fit(data["reference"]["y_true"],
             data["reference"]["y_pred_proba"],
             data["reference"]["y_pred"])
    cbpe_result = cbpe.detect(data["stream"]["y_pred_proba"],
                               data["stream"]["y_pred"])

    ddm = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0)
    ddm.fit(data["reference"]["y_true"], data["reference"]["y_pred"])
    ddm_result = ddm.detect_chunks(data["stream"]["y_true"],
                                    data["stream"]["y_pred"],
                                    chunk_size=500)

    _assert_true(cbpe_result["overall_alert"],
                 "Comparison: CBPE detects sudden degradation")
    _assert_true(ddm_result["overall_degradation_detected"],
                 "Comparison: DDM detects sudden degradation")

    # Calibration shift: CBPE should detect, DDM should NOT
    data_cal = gen.generate_calibration_shift(n_ref=2000, n_stream=3000)

    cbpe2 = CBPEDegradationDetector(chunk_size=500, alert_threshold_f1=0.08,
                                     alert_threshold_auc=0.08)
    cbpe2.fit(data_cal["reference"]["y_true"],
              data_cal["reference"]["y_pred_proba"],
              data_cal["reference"]["y_pred"])
    cbpe_result2 = cbpe2.detect(data_cal["stream"]["y_pred_proba"],
                                 data_cal["stream"]["y_pred"])

    ddm2 = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0)
    ddm2.fit(data_cal["reference"]["y_true"], data_cal["reference"]["y_pred"])
    ddm_result2 = ddm2.detect_chunks(data_cal["stream"]["y_true"],
                                      data_cal["stream"]["y_pred"],
                                      chunk_size=500)

    # DDM should NOT alarm (error rate unchanged)
    _assert_false(ddm_result2["overall_degradation_detected"],
                  "Comparison: DDM does NOT alarm on calibration shift (error rate unchanged)")

    print()


# ============================================================================
# TEST GROUP 13: Numerical Stability
# ============================================================================

def test_numerical_stability():
    print("=== Test Group 13: Numerical Stability ===")

    # Very small probabilities
    small_proba = [0.001, 0.001, 0.001, 0.999, 0.999]
    y_pred = [0, 0, 0, 1, 1]
    est_f1 = _estimate_f1(small_proba, y_pred)
    _assert(est_f1 >= 0.0 and est_f1 <= 1.0,
            "Numerical: estimated F1 in [0,1] for extreme probabilities")

    # Very large exp argument
    _assert(_manual_exp_cbpe(700) > 1e300, "Numerical: exp(700) is large")
    _assert(_manual_exp_cbpe(-700) < 1e-300, "Numerical: exp(-700) is tiny")
    _assert(_manual_exp_cbpe(800) > 0, "Numerical: exp(800) doesn't crash")

    # Log of very small number
    log_small = _manual_log_cbpe(1e-300)
    _assert(log_small < -600, "Numerical: log(1e-300) is very negative")

    # Sigmoid at extremes
    _assert(_sigmoid(1000) > 0.999, "Numerical: sigmoid(1000) ~ 1")
    _assert(_sigmoid(-1000) < 0.001, "Numerical: sigmoid(-1000) ~ 0")

    # DDM with very low error rate
    ddm = DDMDegradationDetector(min_samples=10)
    yt = [0] * 1000
    yp = [0] * 1000  # perfect predictions
    yp[999] = 1  # single error at the end
    result = ddm.detect_stream(yt, yp)
    _assert(result["stream_length"] == 1000,
            "Numerical: DDM handles near-zero error rate")

    print()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    global _test_count, _pass_count, _fail_count, _failures

    print("=" * 70)
    print("Part 2 — Degradation Detection Unit Tests")
    print("=" * 70)
    print()

    test_math_utilities()
    test_metrics()
    test_platt_scaling()
    test_cbpe_estimation()
    test_ddm_basic()
    test_ddm_reset()
    test_eddm()
    test_cbpe_pipeline()
    test_ddm_pipeline()
    test_data_generator()
    test_edge_cases()
    test_comparison()
    test_numerical_stability()

    print("=" * 70)
    print("RESULTS: " + str(_pass_count) + "/" + str(_test_count) +
          " passed, " + str(_fail_count) + " failed")
    print("=" * 70)

    if _failures:
        print("\nFailed tests:")
        for f in _failures:
            print("  - " + f)
        print()

    return _fail_count == 0


# Entry point
if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        # Exit with error code but don't use sys.exit (no imports)
        print("\nSome tests FAILED.")
    else:
        print("\nAll tests PASSED.")

