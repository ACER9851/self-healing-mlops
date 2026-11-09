# =============================================================================
# evidently_style_detector.py
# Evidently-style Data Drift Detector — built entirely from scratch.
# No external imports. All math implemented via Python built-ins only.
# =============================================================================
#
# DETECTION METHODOLOGY
# ---------------------
# Numerical features : KS test  +  PSI  (flag if EITHER exceeds threshold)
# Categorical features: PSI only         (chi-square needs gamma CDF — PSI is
#                                         industry-standard for categoricals)
# Dataset decision    : majority vote    (drift if > majority_threshold % drift)
#
# STATISTICAL APPROXIMATION NOTES
# --------------------------------
# KS p-value  : Kolmogorov asymptotic formula. Reliable for n,m >= 5.
#               For tiny samples the KS stat is still correct; p-value is
#               approximate and labelled as such in results.
# PSI binning : equal-frequency (percentile) bins on the reference.
#               More robust than equal-width for skewed distributions.
#               Limitation: a constant reference feature collapses to one bin,
#               making PSI insensitive to shifts in that feature. KS catches it.
# =============================================================================


# ------------------------------------------------------------------
# SECTION 1  MATHEMATICAL UTILITIES
# All implemented from first principles; none imported.
# ------------------------------------------------------------------

def _exp(x):
    """
    Compute e^x.

    Algorithm
    ---------
    1. Clamp extreme values to avoid overflow / underflow.
    2. Reflect: exp(-x) = 1 / exp(x), so handle x < 0 recursively.
    3. Range-reduce: exp(x) = exp(x/2)^2 for x > 1 so the series only
       ever sees x in [0, 1] where it converges in < 30 terms.
    4. Taylor series: exp(u) = Σ_{k=0}^{inf} u^k / k!
    """
    if x > 709.78:
        return float('inf')
    if x < -709.78:
        return 0.0
    if x < 0.0:
        return 1.0 / _exp(-x)
    if x > 1.0:
        h = _exp(x / 2.0)
        return h * h
    # Taylor series for x in [0, 1]
    result = 1.0
    term = 1.0
    for k in range(1, 70):
        term *= x / k
        result += term
        if term < 1e-17:
            break
    return result


def _log(x):
    """
    Compute natural logarithm ln(x).

    Algorithm
    ---------
    1. Guard: ln(x <= 0) = -inf.
    2. Range-reduce to m in [1, 2) by repeated halving/doubling; track
       the exponent e so that original x = m * 2^e.
    3. Use the identity  ln(m) = 2 * arctanh((m-1)/(m+1))
       and the series    arctanh(u) = Σ_{k=0}^{inf} u^{2k+1} / (2k+1).
       For m in [1, 2), u = (m-1)/(m+1) in [0, 1/3), fast convergence.
    4. Recombine: ln(x) = 2*arctanh(u) + e * ln(2).

    LN2 = 0.6931471805599453 (pre-computed to full double precision).
    """
    if x <= 0.0:
        return float('-inf')
    if x == 1.0:
        return 0.0

    LN2 = 0.6931471805599453
    e = 0
    while x >= 2.0:
        x /= 2.0
        e += 1
    while x < 1.0:
        x *= 2.0
        e -= 1
    # x now in [1, 2)
    u = (x - 1.0) / (x + 1.0)   # u in [0, 1/3)
    u2 = u * u
    s = u
    term = u
    for k in range(1, 100):
        term *= u2
        contrib = term / (2 * k + 1)
        s += contrib
        if contrib < 1e-17:
            break
    return 2.0 * s + e * LN2


def _sqrt(x):
    """
    Compute sqrt(x) via Newton's (Babylonian) method.
    Converges quadratically; 50 iterations is far more than sufficient.
    """
    if x < 0.0:
        return float('nan')
    if x == 0.0:
        return 0.0
    guess = x if x <= 1.0 else x / 2.0
    for _ in range(80):
        new_g = (guess + x / guess) / 2.0
        if abs(new_g - guess) < 1e-13 * guess:
            break
        guess = new_g
    return guess


# ------------------------------------------------------------------
# SECTION 2  TWO-SAMPLE KOLMOGOROV-SMIRNOV TEST
# ------------------------------------------------------------------

def _build_ecdf(data):
    """
    Build the empirical CDF as a sorted list of (value, cumulative_prob).

    F_n(x) = (number of observations <= x) / n

    The ECDF is a step function; we store only the jump points.
    At each sorted observation x_i, the ECDF jumps to (i+1)/n.
    """
    n = len(data)
    if n == 0:
        return []
    return [(v, (i + 1) / n) for i, v in enumerate(sorted(data))]


def _ecdf_at(ecdf_pairs, x):
    """
    Evaluate the ECDF at point x.  O(n) linear scan (safe for n <= 10k).

    Returns F(x) = P(X <= x).
    ecdf_pairs must be sorted ascending by value (guaranteed by _build_ecdf).
    """
    prob = 0.0
    for val, cp in ecdf_pairs:
        if val <= x:
            prob = cp
        else:
            break
    return prob


def _ks_statistic(reference, current):
    """
    Compute the two-sample KS statistic.

    D = max_{x} |F_ref(x) - F_cur(x)|

    We only need to evaluate at the finite set of unique values from both
    samples, because both ECDFs are step functions that only jump there.
    """
    ecdf_ref = _build_ecdf(reference)
    ecdf_cur = _build_ecdf(current)
    all_values = sorted(set(reference + current))
    D = 0.0
    for x in all_values:
        diff = abs(_ecdf_at(ecdf_ref, x) - _ecdf_at(ecdf_cur, x))
        if diff > D:
            D = diff
    return D


def _ks_pvalue_approx(D, n, m):
    """
    Approximate the KS p-value using the Kolmogorov asymptotic distribution.

    Under H0, sqrt(n*m/(n+m)) * D converges in distribution to K.
    The survival function of K is:
        P(K > lambda) = 2 * Σ_{k=1}^{inf} (-1)^{k-1} * exp(-2 k^2 lambda^2)

    This series is alternating and converges very quickly (usually < 10 terms).

    Reliability: good for min(n, m) >= 5. For smaller samples the statistic is
    still correct; only the p-value approximation degrades.

    Returns p-value in [0, 1].
    """
    if D == 0.0:
        return 1.0
    effective_n = (n * m) / (n + m)
    lam = D * _sqrt(effective_n)
    if lam < 1e-9:
        return 1.0

    p = 0.0
    for k in range(1, 150):
        sign = 1.0 if k % 2 == 1 else -1.0
        term = sign * _exp(-2.0 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-13:
            break

    p = 2.0 * p
    # Clamp due to asymptotic approximation imprecision
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p


def _ks_test(reference, current):
    """
    Full two-sample KS test.

    Returns
    -------
    D       : KS statistic (exact for discrete/continuous data)
    p_value : approximate p-value (Kolmogorov asymptotic)
    """
    n, m = len(reference), len(current)
    if n == 0 or m == 0:
        return 0.0, 1.0
    D = _ks_statistic(reference, current)
    p = _ks_pvalue_approx(D, n, m)
    return D, p


# ------------------------------------------------------------------
# SECTION 3  POPULATION STABILITY INDEX (PSI)
# ------------------------------------------------------------------

def _percentile(sorted_data, pct):
    """
    Linear-interpolation percentile from pre-sorted data.
    pct is in [0, 100].
    """
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(sorted_data[-1])
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def _equal_freq_bin_edges(reference, n_bins):
    """
    Build bin edges using equal-frequency (quantile) binning on the reference.

    Returns a list of n_bins+1 edges where:
      - First edge  = -inf  (captures any current value below ref minimum)
      - Last  edge  = +inf  (captures any current value above ref maximum)
      - Internal edges = percentile boundaries of the reference

    Duplicate edges (from constant or highly discrete data) are removed;
    the number of effective bins may be < n_bins in those cases.
    """
    sref = sorted(reference)
    edges = [float('-inf')]
    for i in range(1, n_bins):
        edges.append(_percentile(sref, i * 100.0 / n_bins))
    edges.append(float('inf'))

    # Remove duplicates to avoid zero-width bins
    deduped = [edges[0]]
    for e in edges[1:]:
        if e > deduped[-1]:
            deduped.append(e)
    return deduped


def _bin_counts(values, edges):
    """
    Count how many values fall in each bin.
    Bin i covers (edges[i], edges[i+1]].
    Values equal to the lower -inf edge go into bin 0.
    """
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    for v in values:
        placed = False
        for i in range(n_bins):
            if edges[i] < v <= edges[i + 1]:
                counts[i] += 1
                placed = True
                break
        if not placed and n_bins > 0:
            # v == -inf boundary or below: assign to first bin
            counts[0] += 1
    return counts


def _psi_numerical(reference, current, n_bins=10, epsilon=1e-4):
    """
    Population Stability Index for a numerical feature.

    PSI = Σ_i (p_cur_i - p_ref_i) * ln(p_cur_i / p_ref_i)

    where p_ref_i, p_cur_i are the fraction of observations in bin i
    from the reference and current datasets respectively.

    Zero-frequency bins get epsilon substitution (industry standard).

    Interpretation
    --------------
    PSI < 0.10  : negligible shift
    PSI 0.10-0.20: moderate shift (investigate)
    PSI > 0.20  : significant shift (action needed)

    Limitation: if the reference feature is nearly constant, all values
    collapse into one bin and PSI will be near 0 even after a large shift.
    The KS test complements this case.
    """
    if not reference or not current:
        return 0.0
    edges = _equal_freq_bin_edges(reference, n_bins)
    if len(edges) < 2:
        return 0.0
    ref_counts = _bin_counts(reference, edges)
    cur_counts = _bin_counts(current, edges)
    n_ref = len(reference)
    n_cur = len(current)
    psi = 0.0
    for rc, cc in zip(ref_counts, cur_counts):
        rp = max(rc / n_ref, epsilon)
        cp = max(cc / n_cur, epsilon)
        psi += (cp - rp) * _log(cp / rp)
    return psi


def _psi_categorical(ref_counts, cur_counts, epsilon=1e-4):
    """
    Population Stability Index for a categorical feature.

    ref_counts, cur_counts : dict {category_label: count}

    Categories present in current but not reference get epsilon in reference
    (and vice versa), so PSI captures newly appearing categories.
    """
    all_cats = set(list(ref_counts.keys()) + list(cur_counts.keys()))
    n_ref = sum(ref_counts.values())
    n_cur = sum(cur_counts.values())
    if n_ref == 0 or n_cur == 0:
        return 0.0
    psi = 0.0
    for cat in all_cats:
        rp = max(ref_counts.get(cat, 0) / n_ref, epsilon)
        cp = max(cur_counts.get(cat, 0) / n_cur, epsilon)
        psi += (cp - rp) * _log(cp / rp)
    return psi


# ------------------------------------------------------------------
# SECTION 4  DATA HELPERS
# ------------------------------------------------------------------

def _extract_numerical(dataset, feature):
    """
    Pull non-None numerical values for a feature from a list of row dicts.
    Missing values are silently skipped (counted separately).
    """
    vals, n_missing = [], 0
    for row in dataset:
        v = row.get(feature)
        if v is None:
            n_missing += 1
        else:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                n_missing += 1
    return vals, n_missing


def _count_categorical(dataset, feature):
    """
    Count occurrences of each category value.
    Returns (counts_dict, n_missing).
    """
    counts, n_missing = {}, 0
    for row in dataset:
        v = row.get(feature)
        if v is None:
            n_missing += 1
        else:
            counts[v] = counts.get(v, 0) + 1
    return counts, n_missing


# ------------------------------------------------------------------
# SECTION 5  MAIN DETECTOR CLASS
# ------------------------------------------------------------------

class EvidentlyStyleDriftDetector:
    """
    Evidently-style data drift detector.

    For each numerical feature:
        - Run two-sample KS test  → p-value
        - Compute PSI             → index value
        - Flag feature as drifted if KS p-value < ks_alpha  OR  PSI > psi_threshold

    For each categorical feature:
        - Compute PSI over category distribution
        - Flag feature as drifted if PSI > psi_threshold

    Dataset-level decision:
        Drift declared when the fraction of drifted features > majority_threshold.

    Parameters
    ----------
    ks_alpha          : float  KS significance level (default 0.05)
    psi_threshold     : float  PSI threshold (default 0.20, industry standard)
    majority_threshold: float  fraction of drifted features for overall drift
    n_psi_bins        : int    equal-frequency bins for numerical PSI
    """

    def __init__(self, ks_alpha=0.05, psi_threshold=0.20,
                 majority_threshold=0.50, n_psi_bins=10):
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.majority_threshold = majority_threshold
        self.n_psi_bins = n_psi_bins

    # ----------------------------------------------------------------
    def detect(self, reference_data, current_data,
               numerical_features=None, categorical_features=None):
        """
        Run drift detection on a reference vs current dataset.

        Parameters
        ----------
        reference_data        : list of dicts  — baseline observations
        current_data          : list of dicts  — production observations
        numerical_features    : list of str    — feature names to test via KS+PSI
        categorical_features  : list of str    — feature names to test via PSI

        Returns
        -------
        dict  Full detection report (see structure below).
        """
        numerical_features   = numerical_features   or []
        categorical_features = categorical_features or []

        feature_results = {}
        n_drifted = 0
        n_tested  = 0

        # ---- Numerical features ----------------------------------------
        for feat in numerical_features:
            ref_vals, ref_miss = _extract_numerical(reference_data, feat)
            cur_vals, cur_miss = _extract_numerical(current_data,   feat)

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                feature_results[feat] = {
                    "type": "numerical",
                    "error": "too_few_values",
                    "n_reference": len(ref_vals),
                    "n_current":   len(cur_vals),
                    "drift_detected": False,
                }
                n_tested += 1
                continue

            ks_stat, ks_pval = _ks_test(ref_vals, cur_vals)
            psi_val          = _psi_numerical(ref_vals, cur_vals, self.n_psi_bins)

            ks_drifted  = ks_pval < self.ks_alpha
            psi_drifted = psi_val > self.psi_threshold
            drifted     = ks_drifted or psi_drifted   # union: either flags = drift

            feature_results[feat] = {
                "type":           "numerical",
                "n_reference":    len(ref_vals),
                "n_current":      len(cur_vals),
                "n_missing_ref":  ref_miss,
                "n_missing_cur":  cur_miss,
                "ks_statistic":   round(ks_stat, 6),
                "ks_pvalue":      round(ks_pval, 6),
                "ks_pvalue_note": "Kolmogorov asymptotic approx (reliable n>=5)",
                "ks_alpha":       self.ks_alpha,
                "ks_drift":       ks_drifted,
                "psi":            round(psi_val, 6),
                "psi_threshold":  self.psi_threshold,
                "psi_drift":      psi_drifted,
                "drift_detected": drifted,
            }
            n_tested  += 1
            if drifted:
                n_drifted += 1

        # ---- Categorical features ----------------------------------------
        for feat in categorical_features:
            ref_counts, ref_miss = _count_categorical(reference_data, feat)
            cur_counts, cur_miss = _count_categorical(current_data,   feat)

            if not ref_counts or not cur_counts:
                feature_results[feat] = {
                    "type": "categorical",
                    "error": "no_data",
                    "drift_detected": False,
                }
                n_tested += 1
                continue

            psi_val = _psi_categorical(ref_counts, cur_counts)
            drifted = psi_val > self.psi_threshold

            feature_results[feat] = {
                "type":                  "categorical",
                "n_reference":           sum(ref_counts.values()),
                "n_current":             sum(cur_counts.values()),
                "n_missing_ref":         ref_miss,
                "n_missing_cur":         cur_miss,
                "reference_distribution": ref_counts,
                "current_distribution":  cur_counts,
                "psi":                   round(psi_val, 6),
                "psi_threshold":         self.psi_threshold,
                "drift_detected":        drifted,
            }
            n_tested  += 1
            if drifted:
                n_drifted += 1

        # ---- Dataset-level majority vote ---------------------------------
        drift_fraction = (n_drifted / n_tested) if n_tested > 0 else 0.0
        dataset_drift  = drift_fraction > self.majority_threshold

        psi_note = (
            "0.00-0.10: negligible | 0.10-0.20: warning | >0.20: action needed"
        )

        return {
            "model":                "EvidentlyStyleDriftDetector",
            "detection_method":     "KS-test + PSI + majority-vote",
            "n_reference":          len(reference_data),
            "n_current":            len(current_data),
            "thresholds": {
                "ks_alpha":           self.ks_alpha,
                "psi_threshold":      self.psi_threshold,
                "majority_threshold": self.majority_threshold,
                "n_psi_bins":         self.n_psi_bins,
            },
            "n_features_tested":    n_tested,
            "n_features_drifted":   n_drifted,
            "drift_fraction":       round(drift_fraction, 4),
            "dataset_drift_detected": dataset_drift,
            "psi_interpretation":   psi_note,
            "feature_results":      feature_results,
            "explanation": (
                f"{n_drifted}/{n_tested} features drifted "
                f"({round(drift_fraction * 100, 1)}%); "
                f"majority threshold = {self.majority_threshold * 100:.0f}%. "
                f"Dataset drift: {'YES' if dataset_drift else 'NO'}."
            ),
        }

