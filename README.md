# Self-Healing MLOps — Comparison Study

> **Semester 1 — Comparison Study**  
> Implements and benchmarks existing drift detection, degradation monitoring, healing policy, and retraining strategy methods from scratch before building the full self-healing pipeline in Semester 2.

---

## Repository Structure

```
comparison-study/
└── part1_detect/           ← Part 1: Drift Detection comparison
    ├── evidently_style_detector.py   # KS + PSI + majority-vote (Evidently-inspired)
    ├── alibi_style_detector.py       # KS+BH-FDR OR multivariate MMD (Alibi-inspired)
    ├── data_generator.py             # 8 synthetic test cases, zero imports
    ├── test_models.py                # 128 unit tests (all pass)
    └── main.py                       # Runs all 3 detectors, prints comparison report
```

---

## Part 1 — Drift Detection

### What is implemented
Two fully independent drift detectors built **from scratch** — no NumPy, no SciPy, no Evidently, no Alibi Detect. Every mathematical operation is hand-coded using Python built-ins only.

| Detector | Method | Tests |
|----------|--------|-------|
| `EvidentlyStyleDriftDetector` | KS test + PSI + majority vote | Per-feature + dataset level |
| `AlibiStyleDriftDetector` (`columnwise_bh`) | KS + Chi² + Benjamini-Hochberg FDR | Per-feature with FDR control |
| `AlibiStyleDriftDetector` (`multivariate_mmd`) | Gaussian RBF kernel + MMD² + permutation test | Joint distribution |

### Math implemented from scratch
- `exp(x)` — range reduction + Taylor series
- `ln(x)` — arctanh series with range reduction
- `sqrt(x)` — Newton's (Babylonian) method
- `cbrt(x)` — Newton's method
- `Φ(x)` — Abramowitz & Stegun rational approximation (error < 7.5×10⁻⁸)
- `χ²` p-value — Wilson-Hilferty normal approximation
- KS statistic — exact ECDF comparison
- KS p-value — Kolmogorov asymptotic series
- PSI — equal-frequency binning on reference
- BH-FDR — exact Benjamini-Hochberg procedure
- RBF kernel — Gaussian, with median bandwidth heuristic
- MMD² — unbiased two-sample estimator
- Permutation test — LCG PRNG + Fisher-Yates shuffle

### Key finding: Case 5 (Joint Drift)
When only the **correlation** between features changes (marginal distributions unchanged):

| Detector | Detects? |
|----------|---------|
| Evidently column-wise | ❌ Misses it |
| Alibi column-wise BH | ❌ Misses it |
| Alibi multivariate MMD | ✅ Detects it |

This is the core motivation for including MMD in the Self-Healing MLOps pipeline.

---

## How to Run

```bash
# No dependencies needed — zero external imports
cd comparison-study/part1_detect

# Run all 128 unit tests
python test_models.py

# Run full 8-case comparison report
python main.py
```

---

## Comparison Study Plan

| Part | Pick 1 | Pick 2 | Status |
|------|--------|--------|--------|
| 1. Detect | Evidently-style (KS+PSI) | Alibi-style (KS+BH / MMD) | ✅ Done |
| 2. Degradation | NannyML CBPE | DDM (River) | 🔜 Next |
| 3. Healing | ADORN Q-learning | Regol et al. simplified | 🔜 Next |
| 4. Retraining | Full retrain | Sliding-window retrain | 🔜 Next |

Datasets: Credit Card Fraud (synthetic drift) · Elec2 (real temporal drift) · LendingClub (hybrid)

---

## Project Context

This comparison study feeds into the **Self-Healing MLOps Architecture** (Semester 2):
a closed-loop system combining adaptive drift detection, adversarial testing, Q-learning decisions, multi-gate validation, and canary deployment — the first prototype to combine all five in one system.

