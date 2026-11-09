# =============================================================================
# data_generator.py
# Synthetic dataset generator for drift detection experiments.
# No external imports. All math and RNG implemented from scratch.
# =============================================================================
#
# PRNG  : Linear Congruential Generator (Numerical Recipes)
# Normal: Marsaglia polar method (no trig functions needed)
# Provides 8 canonical test cases covering the main failure modes.
# =============================================================================


# ------------------------------------------------------------------
# SECTION 1  MATH UTILITIES  (independent copy — no imports)
# ------------------------------------------------------------------

def _log(x):
    if x <= 0.0:
        return float('-inf')
    if x == 1.0:
        return 0.0
    LN2 = 0.6931471805599453
    e = 0
    while x >= 2.0:
        x /= 2.0; e += 1
    while x < 1.0:
        x *= 2.0; e -= 1
    u = (x - 1.0) / (x + 1.0)
    u2, s, term = u * u, u, u
    for k in range(1, 100):
        term *= u2
        contrib = term / (2 * k + 1)
        s += contrib
        if contrib < 1e-17:
            break
    return 2.0 * s + e * LN2


def _sqrt(x):
    if x <= 0.0:
        return 0.0
    g = x if x <= 1.0 else x / 2.0
    for _ in range(80):
        ng = (g + x / g) / 2.0
        if abs(ng - g) < 1e-13 * g:
            break
        g = ng
    return g


# ------------------------------------------------------------------
# SECTION 2  PSEUDO-RANDOM NUMBER GENERATOR
# ------------------------------------------------------------------

class _RNG:
    """
    Linear Congruential Generator + Marsaglia polar method for normals.

    LCG parameters: a=1664525, c=1013904223, m=2^32 (Numerical Recipes).
    Polar method: generate (u1, u2) in unit disk, then
        factor = sqrt(-2 ln(s) / s)   where s = u1^2 + u2^2
        z1, z2 = u1*factor, u2*factor  ~ N(0,1) independent pair.
    No trig functions required.
    """
    _A = 1664525
    _C = 1013904223
    _M = 4294967296   # 2^32

    def __init__(self, seed=42):
        self._state   = int(seed) % self._M
        self._spare   = None     # cached second normal from each polar pair
        self._has_spare = False

    def _raw(self):
        """Next float in [0, 1)."""
        self._state = (self._A * self._state + self._C) % self._M
        return self._state / self._M

    def uniform(self, lo=0.0, hi=1.0):
        return lo + (hi - lo) * self._raw()

    def normal(self, mean=0.0, std=1.0):
        """Standard normal via Marsaglia polar method."""
        if self._has_spare:
            self._has_spare = False
            return mean + std * self._spare

        # Rejection loop: draw (u1, u2) in unit disk
        while True:
            u1 = self._raw() * 2.0 - 1.0
            u2 = self._raw() * 2.0 - 1.0
            s  = u1 * u1 + u2 * u2
            if 0.0 < s < 1.0:
                break

        factor = _sqrt(-2.0 * _log(s) / s)
        z1, z2 = u1 * factor, u2 * factor
        self._spare     = z2
        self._has_spare = True
        return mean + std * z1

    def choice(self, options):
        """Pick one element uniformly from options."""
        idx = int(self._raw() * len(options)) % len(options)
        return options[idx]

    def randint(self, lo, hi):
        """Integer in [lo, hi)."""
        return lo + int(self._raw() * (hi - lo))

    def bernoulli(self, p):
        """Return True with probability p."""
        return self._raw() < p


# ------------------------------------------------------------------
# SECTION 3  ROW-BUILDER HELPERS
# ------------------------------------------------------------------

def _make_row(age, salary, score, city, risk):
    """Standard row schema used across all test cases."""
    return {
        "age":    age,
        "salary": salary,
        "score":  score,
        "city":   city,
        "risk":   risk,
    }


CITIES      = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"]
CITIES_SHIFT = ["Pune", "Hyderabad", "Ahmedabad", "Jaipur", "Lucknow"]
RISK_LABELS = ["low", "medium", "high"]


# ------------------------------------------------------------------
# SECTION 4  EIGHT CANONICAL TEST CASES
# ------------------------------------------------------------------

def generate_no_drift(n=200, seed=42):
    """
    Case 1 — No distributional change.
    Reference and current drawn from the same joint distribution.
    Expected: both detectors find NO drift.

    age    ~ N(30, 5)
    salary ~ N(50000, 10000)
    score  ~ N(0.6, 0.1)
    city   ~ Uniform over CITIES (each ~20%)
    risk   ~ {low:60%, medium:30%, high:10%}
    """
    rng = _RNG(seed)
    risk_weights = [0.6, 0.3, 0.1]

    def _risk(r):
        p = r.uniform()
        if p < 0.6:  return "low"
        if p < 0.9:  return "medium"
        return "high"

    def _one(r):
        return _make_row(
            r.normal(30, 5), r.normal(50000, 10000),
            r.normal(0.6, 0.1), r.choice(CITIES), _risk(r)
        )

    reference = [_one(rng) for _ in range(n)]
    current   = [_one(rng) for _ in range(n)]   # same distribution, new draws
    return reference, current, {
        "case": 1, "name": "no_drift",
        "description": "Both samples from identical distribution",
        "expected_drift": False,
    }


def generate_numerical_drift(n=200, seed=42):
    """
    Case 2 — Large shift in numerical features only.
    age:    30→50, salary: 50k→80k, score: 0.6→0.8
    city and risk are unchanged.
    Expected: both detectors find drift (numerical features flagged).
    PSI >> 0.20 for age and salary; KS D close to 1.
    """
    rng = _RNG(seed)

    def _risk(r):
        p = r.uniform()
        return "low" if p < 0.6 else ("medium" if p < 0.9 else "high")

    reference = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES), _risk(rng))
        for _ in range(n)
    ]
    current = [
        _make_row(rng.normal(50, 5), rng.normal(80000, 10000),
                  rng.normal(0.8, 0.1), rng.choice(CITIES), _risk(rng))
        for _ in range(n)
    ]
    return reference, current, {
        "case": 2, "name": "numerical_drift",
        "description": "age +20, salary +30k, score +0.2 — categoricals unchanged",
        "expected_drift": True,
        "drifted_features": ["age", "salary", "score"],
    }


def generate_categorical_drift(n=200, seed=42):
    """
    Case 3 — Shift in categorical features only.
    city:  CITIES → CITIES_SHIFT (completely new city set)
    risk:  60/30/10 → 10/30/60  (risk profile inverted)
    Numerical features are unchanged.
    Expected: both detectors find drift (categorical features flagged).
    PSI > 0.20 for city and risk.
    """
    rng = _RNG(seed)

    def _risk_ref(r):
        p = r.uniform()
        return "low" if p < 0.6 else ("medium" if p < 0.9 else "high")

    def _risk_cur(r):
        p = r.uniform()
        return "high" if p < 0.6 else ("medium" if p < 0.9 else "low")

    reference = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES), _risk_ref(rng))
        for _ in range(n)
    ]
    current = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES_SHIFT), _risk_cur(rng))
        for _ in range(n)
    ]
    return reference, current, {
        "case": 3, "name": "categorical_drift",
        "description": "city set fully replaced; risk profile inverted",
        "expected_drift": True,
        "drifted_features": ["city", "risk"],
    }


def generate_multiple_features_drift(n=200, seed=42):
    """
    Case 4 — All five features shift simultaneously.
    Largest realistic combined drift scenario.
    Expected: overwhelming evidence; every test flags drift.
    """
    rng = _RNG(seed)

    reference = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES),
                  "low" if rng.uniform() < 0.6 else "high")
        for _ in range(n)
    ]
    current = [
        _make_row(rng.normal(55, 8), rng.normal(90000, 15000),
                  rng.normal(0.3, 0.15), rng.choice(CITIES_SHIFT),
                  "high" if rng.uniform() < 0.7 else "low")
        for _ in range(n)
    ]
    return reference, current, {
        "case": 4, "name": "multiple_features_drift",
        "description": "All features shift simultaneously",
        "expected_drift": True,
        "drifted_features": ["age", "salary", "score", "city", "risk"],
    }


def generate_joint_multivariate_drift(n=200, seed=42):
    """
    Case 5 — Joint (correlation) drift only; marginals unchanged.

    Reference:
        age   ~ N(0, 1)   (standardised)
        salary ~ N(0, 1)  independently
        → correlation rho = 0

    Current:
        age   ~ N(0, 1)
        salary = 0.90 * age + 0.44 * N(0, 1)   → rho ≈ 0.90, marginal still N(0,1)

    score, city, risk unchanged.

    WHY THIS IS THE HARD CASE
    --------------------------
    Per-column KS on age and per-column KS on salary each see N(0,1)
    in both reference and current → p-value ≈ 1 → NO per-column drift.
    MMD sees the full joint (age, salary) vector, whose distribution
    has changed from (N(0,1) × N(0,1)) to a bivariate normal with
    rho=0.9 → MMD^2 is large → p-value small → drift detected.

    Expected:
        Evidently columnwise_bh : NO drift (misses it)
        Alibi columnwise_bh     : NO drift (misses it)
        Alibi multivariate_mmd  : YES drift (detects correlation change)
    """
    rng = _RNG(seed)

    # Reference: independent age and salary
    reference = []
    for _ in range(n):
        age    = rng.normal(0, 1)
        salary = rng.normal(0, 1)
        score  = rng.normal(0.6, 0.1)
        city   = rng.choice(CITIES)
        risk   = "low" if rng.uniform() < 0.6 else "medium"
        reference.append(_make_row(age, salary, score, city, risk))

    # Current: correlated age and salary (rho ≈ 0.90), same marginals
    current = []
    for _ in range(n):
        age    = rng.normal(0, 1)
        salary = 0.90 * age + 0.4359 * rng.normal(0, 1)   # 0.4359 ≈ sqrt(1-0.81)
        score  = rng.normal(0.6, 0.1)
        city   = rng.choice(CITIES)
        risk   = "low" if rng.uniform() < 0.6 else "medium"
        current.append(_make_row(age, salary, score, city, risk))

    return reference, current, {
        "case": 5,
        "name": "joint_multivariate_drift",
        "description": (
            "age and salary correlation changes from 0 to 0.90; "
            "marginals unchanged. Per-column tests miss this."
        ),
        "expected_drift": True,
        "expected_to_miss": ["Evidently columnwise", "Alibi columnwise_bh"],
        "expected_to_detect": ["Alibi multivariate_mmd"],
        "drifted_features": ["age-salary joint structure (not marginals)"],
    }


def generate_small_drift(n=200, seed=42):
    """
    Case 6 — Small, potentially sub-threshold drift.
    age: mean shifts by +2 (< 0.5 standard deviation).
    salary: mean shifts by +3000 (0.3 standard deviations).
    Score, city, risk unchanged.

    Expected behaviour (depends on n and thresholds):
        With n=200: borderline — KS may or may not reach p<0.05.
        PSI will be < 0.10 (noise range). Useful for sensitivity comparison.
    """
    rng = _RNG(seed)

    reference = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES),
                  "low" if rng.uniform() < 0.6 else "medium")
        for _ in range(n)
    ]
    current = [
        _make_row(rng.normal(32, 5), rng.normal(53000, 10000),   # +2, +3k
                  rng.normal(0.6, 0.1), rng.choice(CITIES),
                  "low" if rng.uniform() < 0.6 else "medium")
        for _ in range(n)
    ]
    return reference, current, {
        "case": 6, "name": "small_drift",
        "description": "age +2 (0.4σ), salary +3k (0.3σ) — subtle shift",
        "expected_drift": "borderline",
        "note": "Tests detector sensitivity; result depends on n and thresholds",
    }


def generate_missing_values(n=200, seed=42, missing_rate=0.20):
    """
    Case 7 — 20% of values replaced with None in the current dataset.
    Reference is clean.  Current has missing at random (MAR) across all features.

    Expected: detectors should handle missing values gracefully.
    Drift decisions depend on the remaining 80% of data.
    """
    rng = _RNG(seed)

    reference = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES),
                  "low" if rng.uniform() < 0.6 else "medium")
        for _ in range(n)
    ]

    current_raw = [
        _make_row(rng.normal(30, 5), rng.normal(50000, 10000),
                  rng.normal(0.6, 0.1), rng.choice(CITIES),
                  "low" if rng.uniform() < 0.6 else "medium")
        for _ in range(n)
    ]

    # Inject missing values at random
    keys = ["age", "salary", "score", "city", "risk"]
    current = []
    for row in current_raw:
        new_row = dict(row)
        for k in keys:
            if rng.bernoulli(missing_rate):
                new_row[k] = None
        current.append(new_row)

    total_missing = sum(
        1 for row in current for k in keys if row.get(k) is None
    )

    return reference, current, {
        "case": 7, "name": "missing_values",
        "description": f"{missing_rate*100:.0f}% of values replaced with None in current",
        "expected_drift": False,
        "note": f"~{total_missing} values missing in current; detectors must skip gracefully",
    }


def generate_unequal_sizes(n_ref=300, n_cur=50, seed=42):
    """
    Case 8 — Unequal reference (n=300) and current (n=50) sample sizes.
    Both drawn from same distribution. No drift.

    Tests that KS p-value formula correctly accounts for different n.
    With n_cur=50, the KS test has lower power; expect some false positives
    at the boundary. PSI may also be noisier.
    """
    rng = _RNG(seed)

    def _row(r):
        return _make_row(
            r.normal(30, 5), r.normal(50000, 10000),
            r.normal(0.6, 0.1), r.choice(CITIES),
            "low" if r.uniform() < 0.6 else "medium"
        )

    reference = [_row(rng) for _ in range(n_ref)]
    current   = [_row(rng) for _ in range(n_cur)]

    return reference, current, {
        "case": 8, "name": "unequal_sizes",
        "description": f"reference n={n_ref}, current n={n_cur}; same distribution",
        "expected_drift": False,
        "note": "Tests KS effective-n correction; smaller current → less power",
    }


# ------------------------------------------------------------------
# SECTION 5  CONVENIENCE: ALL CASES AT ONCE
# ------------------------------------------------------------------

def generate_all_cases():
    """
    Returns a list of (reference, current, metadata) tuples
    for all 8 test cases with default parameters.
    """
    return [
        generate_no_drift(),
        generate_numerical_drift(),
        generate_categorical_drift(),
        generate_multiple_features_drift(),
        generate_joint_multivariate_drift(),
        generate_small_drift(),
        generate_missing_values(),
        generate_unequal_sizes(),
    ]


# Feature schema (same for all cases)
NUMERICAL_FEATURES   = ["age", "salary", "score"]
CATEGORICAL_FEATURES = ["city", "risk"]

