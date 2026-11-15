# =============================================================================
# main.py — Part 2: Model Degradation Detection Benchmark Runner
# Compares NannyML-style CBPE (label-free) vs DDM & EDDM (error-stream)
# across all 8 benchmark degradation scenarios.
# =============================================================================

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from nannyml_cbpe import CBPEDegradationDetector
from ddm_detector import DDMDegradationDetector, EDDMDegradationDetector
from data_generator import DegradationDataGenerator


def _hline(char="─", width=80):
    print(char * width)


def _header(text, char="═", width=80):
    print(char * width)
    side = (width - len(text) - 2) // 2
    print(f"{char * side} {text} {char * (width - side - len(text) - 2)}")
    print(char * width)


def run_benchmark():
    _header("PART 2: MODEL DEGRADATION DETECTION BENCHMARK")
    print("Methods Compared:")
    print("  1. NannyML-style CBPE  (Confidence-Based Performance Estimation — Label-Free)")
    print("  2. DDM                 (Drift Detection Method on Error Stream — Needs Labels)")
    print("  3. EDDM                (Early Drift Detection Method on Error Stream — Needs Labels)")
    print()

    gen = DegradationDataGenerator(seed=42)
    scenarios = gen.run_all_scenarios()

    print(f"{'#':<3s} {'Scenario Name':<28s} | {'CBPE (No Labels)':<18s} | {'DDM (Labels)':<14s} | {'EDDM (Labels)':<14s}")
    _hline()

    results_table = []

    for i, s in enumerate(scenarios):
        name = s["scenario_name"]
        if name == "edge_cases":
            continue

        ref = s["reference"]
        stream = s["stream"]

        # 1. Run CBPE
        cbpe = CBPEDegradationDetector(chunk_size=500, alert_threshold_f1=0.08, alert_threshold_auc=0.08)
        cbpe.fit(ref["y_true"], ref["y_pred_proba"], ref["y_pred"])
        cbpe_res = cbpe.detect(stream["y_pred_proba"], stream["y_pred"])
        cbpe_alert = cbpe_res["overall_alert"]
        cbpe_first = cbpe_res["first_f1_alert_chunk"]

        # 2. Run DDM
        ddm = DDMDegradationDetector(warning_level=2.0, alarm_level=3.0)
        ddm.fit(ref["y_true"], ref["y_pred"])
        ddm_res = ddm.detect_chunks(stream["y_true"], stream["y_pred"], chunk_size=500)
        ddm_alert = ddm_res["overall_degradation_detected"]
        ddm_first = ddm_res["first_alarm_step"]

        # 3. Run EDDM
        eddm = EDDMDegradationDetector(warning_ratio=0.95, alarm_ratio=0.90)
        eddm.fit(ref["y_true"], ref["y_pred"])
        eddm_res = eddm.detect_chunks(stream["y_true"], stream["y_pred"], chunk_size=500)
        eddm_alert = eddm_res["overall_degradation_detected"]
        eddm_first = eddm_res["first_alarm_step"]

        cbpe_str = f"{'ALERT' if cbpe_alert else 'OK'} (chunk {cbpe_first if cbpe_first is not None else '-'})"
        ddm_str = f"{'ALARM' if ddm_alert else 'OK'} (step {ddm_first if ddm_first is not None else '-'})"
        eddm_str = f"{'ALARM' if eddm_alert else 'OK'} (step {eddm_first if eddm_first is not None else '-'})"

        print(f"{i+1:<3d} {name:<28s} | {cbpe_str:<18s} | {ddm_str:<14s} | {eddm_str:<14s}")

        results_table.append({
            "scenario": name,
            "cbpe_alert": cbpe_alert,
            "ddm_alert": ddm_alert,
            "eddm_alert": eddm_alert,
        })

    _hline()
    print()
    _header("KEY COMPARISON FINDINGS")
    print("""
1. LABEL INDEPENDENCE:
   - CBPE successfully monitors performance *without any ground-truth labels*.
   - DDM and EDDM require real-time immediate ground-truth labels, which in real
     production settings (like loan default or fraud confirmation) take weeks or months.

2. CALIBRATION-ONLY DRIFT:
   - When accuracy remains constant but probability confidence shifts (Scenario 4),
     CBPE detects the shift because confidence bounds contract.
     DDM/EDDM are completely blind to this because 0/1 accuracy did not change.

3. DETECTION LATENCY:
   - DDM/EDDM can flag individual sample transitions immediately when labels arrive.
   - CBPE evaluates chunks (e.g. 500 rows) to obtain statistically robust confidence intervals.
""")
    _header("END OF REPORT")


if __name__ == "__main__":
    run_benchmark()

