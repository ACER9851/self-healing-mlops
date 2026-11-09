# =============================================================================
# alibi_style_detector.py
# Alibi-Detect-style Drift Detector — built entirely from scratch.
# No external imports. All math implemented via Python built-ins only.
# Fully independent of evidently_style_detector.py.
# =============================================================================
#
# TWO DETECTION MODES (user selects at construction time)
# -------------------------------------------------------
# 'columnwise_bh'   Per-column KS test (numerical) and Chi-square test
#                   (categorical), p-values corrected with Benjamini-Hochberg
#                   FDR procedure. Detects per-feature marginal drift.
#
# 'multivariate_mmd' Maximum Mean Discrepancy with a Gaussian RBF kernel,
#                    permutation-test p-value. Tests the JOINT distribution
#                    of all features simultaneously. Can detect drift that is
#                    invisible to any single-feature test (e.g. correlation
#                    changes with unchanged marginals).
#
# STATISTICAL APPROXIMATION NOTES
# --------------------------------
# KS p-value     : Kolmogorov asymptotic formula. Reliable for n,m >= 5.
# Chi-square p   : Wilson-Hilferty normal approximation of chi2 CDF.
#                  Reliable for df >= 1 and chi2_stat not extremely large.
# BH-FDR         : Exact procedure on the computed p-values.
# MMD permutation: Monte-Carlo p-value. Variance decreases as 1/sqrt(perms).
#                  Default 200 permutations gives ~95% CI half-width ~0.07.
# PRNG           : Linear Congruential Generator (Numerical Recipes params).
# =============================================================================


# ------------------------------------------------------------------
# SECTION 1  MATHEMATICAL UTILITIES  (independent copy)
# All implemented from first principles; none imported.
# ------------------------------------------------------------------

def _exp(x):
    """
    Compute e^x.
    Range reduction to [0,1] + Taylor series. See evidently_style_detector
    for detailed comments on the algorithm. (Independent implementation.)
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
    result, term = 1.0, 1.0
    for k in range(1, 70):
        term *= x / k
        result += term
        if term < 1e-17:
            break
    return result


def _log(x):
    """
    Compute ln(x) via arctanh series with range reduction to [1, 2).
    Independent implementation. See evidently_style_detector for details.
    """
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
    """Compute sqrt(x) via Newton's method."""
    if x < 0.0:
        return float('nan')
    if x == 0.0:
        return 0.0
    g = x if x <= 1.0 else x / 2.0
    for _ in range(80):
        ng = (g + x / g) / 2.0
        if abs(ng - g) < 1e-13 * g:
            break
        g = ng
    return g


def _cbrt(x):
    """
    Cube root of x (supports negative x).
    Newton's method: g_{n+1} = (2g_n + x/g_n^2) / 3
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return -_cbrt(-x)
    g = x if x <= 1.0 else x / 3.0
    for _ in range(80):
        g2 = g * g
        if g2 == 0.0:
            break
        ng = (2.0 * g + x / g2) / 3.0
        if abs(ng - g) < 1e-13 * abs(g):
            break
        g = ng
    return g


# ------------------------------------------------------------------
# SECTION 2  NORMAL DISTRIBUTION CDF
# Used to convert chi-square stats → p-values via Wilson-Hilferty.
# ------------------------------------------------------------------

# 1 / sqrt(2 * pi)  (pre-computed)
_INV_SQRT_2PI = 0.3989422804014327

def _normal_cdf(x):
    """
    Standard normal CDF  Φ(x) = P(Z ≤ x),  Z ~ N(0,1).

    Uses the rational polynomial approximation from Abramowitz & Stegun
    formula 26.2.17.  Maximum absolute error: 7.5 × 10^-8.

    Coefficients:
        p=0.2316419, b1..b5 from A&S Table 26.2 row 1 (order-5 Horner form).
    """
    b1, b2, b3, b4, b5 = (0.319381530, -0.356563782,
                           1.781477937, -1.821255978, 1.330274429)
    p_coeff = 0.2316419

    neg = (x < 0.0)
    if neg:
        x = -x

    t   = 1.0 / (1.0 + p_coeff * x)
    pdf = _INV_SQRT_2PI * _exp(-0.5 * x * x)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    cdf_pos = 1.0 - pdf * poly

    return (1.0 - cdf_pos) if neg else cdf_pos


def _chi2_pvalue(chi2_stat, df):
    """
    Chi-square p-value  P(χ²(df) > chi2_stat)  using the
    Wilson-Hilferty cube-root normal approximation.

    Approximation:
        χ²(df) ≈ df * (1 - 2/(9df) + Z * sqrt(2/(9df)))^3
    Solving for Z:
        Z = ((chi2_stat/df)^{1/3} - (1 - 2/(9df))) / sqrt(2/(9df))
    p-value = 1 - Φ(Z)

    Reliable for df ≥ 1.  For df = 1, prefer the exact formula where
    chi2_stat = z^2 and p = 2*(1-Φ(|z|)); both give similar results here.
    """
    if df <= 0 or chi2_stat <= 0.0:
        return 1.0
    h     = 1.0 - 2.0 / (9.0 * df)
    sigma = _sqrt(2.0 / (9.0 * df))
    z     = (_cbrt(chi2_stat / df) - h) / sigma
    p     = 1.0 - _normal_cdf(z)
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return p


# ------------------------------------------------------------------
# SECTION 3  TWO-SAMPLE KS TEST  (independent copy)
# ------------------------------------------------------------------

def _build_ecdf(data):
    n = len(data)
    if n == 0:
        return []
    return [(v, (i + 1) / n) for i, v in enumerate(sorted(data))]


def _ecdf_at(pairs, x):
    prob = 0.0
    for val, cp in pairs:
        if val <= x:
            prob = cp
        else:
            break
    return prob


def _ks_test(reference, current):
    """
    Two-sample KS test.  Returns (D_statistic, p_value_approx).
    D = max_x |F_ref(x) - F_cur(x)|.
    p-value: Kolmogorov asymptotic.
    """
    n, m = len(reference), len(current)
    if n < 1 or m < 1:
        return 0.0, 1.0
    ecdf_r = _build_ecdf(reference)
    ecdf_c = _build_ecdf(current)
    D = 0.0
    for x in sorted(set(reference + current)):
        diff = abs(_ecdf_at(ecdf_r, x) - _ecdf_at(ecdf_c, x))
        if diff > D:
            D = diff
    if D == 0.0:
        return 0.0, 1.0
    eff_n = (n * m) / (n + m)
    lam   = D * _sqrt(eff_n)
    p     = 0.0
    for k in range(1, 150):
        sign = 1.0 if k % 2 == 1 else -1.0
        t    = sign * _exp(-2.0 * k * k * lam * lam)
        p   += t
        if abs(t) < 1e-13:
            break
    p = max(0.0, min(1.0, 2.0 * p))
    return D, p


# ------------------------------------------------------------------
# SECTION 4  CHI-SQUARE TEST FOR CATEGORICAL FEATURES
# ------------------------------------------------------------------

def _chi2_test(ref_counts, cur_counts):
    """
    Pearson chi-square goodness-of-fit test.

    Tests whether the current categorical distribution matches the reference.
    Expected frequencies = reference proportions × n_current.

    χ² = Σ_k (O_k - E_k)^2 / E_k

    Returns (chi2_stat, p_value, degrees_of_freedom).
    p-value via Wilson-Hilferty normal approximation.

    Note: cells with E_k < 5 reduce reliability; this is documented in the
    result dict. Categories absent from reference contribute directly to χ².
    """
    all_cats = sorted(set(list(ref_counts.keys()) + list(cur_counts.keys())))
    n_ref = sum(ref_counts.values())
    n_cur = sum(cur_counts.values())

    if n_ref == 0 or n_cur == 0 or len(all_cats) < 2:
        return 0.0, 1.0, 0

    df = len(all_cats) - 1
    chi2 = 0.0
    low_expected_cells = 0

    for cat in all_cats:
        observed = cur_counts.get(cat, 0)
        ref_prop = ref_counts.get(cat, 0) / n_ref
        expected = ref_prop * n_cur
        if expected >= 1e-10:
            chi2 += (observed - expected) ** 2 / expected
            if expected < 5:
                low_expected_cells += 1
        elif observed > 0:
            # Category not in reference but appears in current → penalise
            chi2 += observed

    p_value = _chi2_pvalue(chi2, df)
    return chi2, p_value, df


# ------------------------------------------------------------------
# SECTION 5  BENJAMINI-HOCHBERG FDR CORRECTION
# ------------------------------------------------------------------

def _bh_fdr(p_values, alpha=0.05):
    """
    Benjamini-Hochberg (1995) procedure for controlling the False Discovery Rate.

    Motivation: testing m features independently at level alpha gives an
    expected number of false positives = m * alpha.  For 30 features at
    alpha=0.05, that is 1.5 false alarms per run — unacceptable.
    BH controls the expected fraction of false discoveries to <= alpha.

    Algorithm
    ---------
    1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
    2. BH critical value at rank k: c_k = (k / m) * alpha
    3. Find k* = max{ k : p_(k) ≤ c_k }
    4. Reject H_0(j) for all j with rank ≤ k*.
       If no k satisfies step 3, reject nothing.

    Parameters
    ----------
    p_values : list of float  — one per feature, any order
    alpha    : float          — FDR level (default 0.05)

    Returns
    -------
    list of bool  — True where the null hypothesis is rejected (drift detected)
                    in the same order as the input p_values list.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value, tracking original positions
    indexed_sorted = sorted(enumerate(p_values), key=lambda t: t[1])

    # Find k* (using 0-based index; rank = idx + 1)
    max_k = -1   # -1 means no rejection
    for rank_0, (orig_idx, pval) in enumerate(indexed_sorted):
        bh_threshold = ((rank_0 + 1) / m) * alpha
        if pval <= bh_threshold:
            max_k = rank_0

    # Mark all features with rank <= k* as rejected
    rejected_sorted = [False] * m
    for i in range(max_k + 1):
        rejected_sorted[i] = True

    # Map back to original order
    result = [False] * m
    for sorted_rank, (orig_idx, _) in enumerate(indexed_sorted):
        result[orig_idx] = rejected_sorted[sorted_rank]

    return result


# ------------------------------------------------------------------
# SECTION 6  LINEAR CONGRUENTIAL GENERATOR (PRNG)
# ------------------------------------------------------------------

class _LCG:
    """
    Linear Congruential Generator for deterministic pseudo-random numbers.

    Parameters (Numerical Recipes, Press et al.):
        a = 1 664 525
        c = 1 013 904 223
        m = 2^32 = 4 294 967 296

    Period = m = 2^32 ≈ 4.3 × 10^9. Adequate for permutation tests with
    n_permutations ≤ 10 000 and combined sample sizes ≤ 10 000.

    Not cryptographically secure.  Statistical quality: passes most standard
    tests; sufficient for permutation-test purposes.
    """
    _A = 1664525
    _C = 1013904223
    _M = 4294967296   # 2^32

    def __init__(self, seed=42):
        self._state = int(seed) % self._M

    def next_float(self):
        """Next float in [0, 1)."""
        self._state = (self._A * self._state + self._C) % self._M
        return self._state / self._M

    def next_int(self, n):
        """Next integer in [0, n)."""
        return int(self.next_float() * n) % n

    def shuffle(self, lst):
        """
        Fisher-Yates in-place shuffle.  Returns a new shuffled list.
        O(n) with uniform distribution (using LCG for index generation).
        """
        result = list(lst)
        for i in range(len(result) - 1, 0, -1):
            j = self.next_int(i + 1)
            result[i], result[j] = result[j], result[i]
        return result


# ------------------------------------------------------------------
# SECTION 7  DATA NORMALISATION AND HELPERS
# ------------------------------------------------------------------

def _extract_numerical(dataset, feature):
    vals, n_miss = [], 0
    for row in dataset:
        v = row.get(feature)
        if v is None:
            n_miss += 1
        else:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                n_miss += 1
    return vals, n_miss


def _count_categorical(dataset, feature):
    counts, n_miss = {}, 0
    for row in dataset:
        v = row.get(feature)
        if v is None:
            n_miss += 1
        else:
            counts[v] = counts.get(v, 0) + 1
    return counts, n_miss


def _zscore_normalize(reference_rows, current_rows, numerical_features):
    """
    Z-score normalise numerical features using reference statistics.

    Returns
    -------
    X_ref : list of vectors (one list per row)
    X_cur : list of vectors
    stats : dict {feature: (mean, std)}  — from reference
    skipped_rows_ref : int
    skipped_rows_cur : int
    """
    # Compute mean and std per feature from reference
    stats = {}
    for feat in numerical_features:
        vals = []
        for row in reference_rows:
            v = row.get(feat)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            mean = sum(vals) / len(vals)
            var  = sum((v - mean) ** 2 for v in vals) / len(vals)
            std  = _sqrt(var)
        else:
            mean, std = 0.0, 1.0
        stats[feat] = (mean, max(std, 1e-8))

    def vectorise(rows):
        vecs, skipped = [], 0
        for row in rows:
            vec, valid = [], True
            for feat in numerical_features:
                v = row.get(feat)
                if v is None:
                    valid = False; break
                try:
                    fv = float(v)
                    mean, std = stats[feat]
                    vec.append((fv - mean) / std)
                except (TypeError, ValueError):
                    valid = False; break
            if valid:
                vecs.append(vec)
            else:
                skipped += 1
        return vecs, skipped

    X_ref, skip_r = vectorise(reference_rows)
    X_cur, skip_c = vectorise(current_rows)
    return X_ref, X_cur, stats, skip_r, skip_c


def _ordinal_encode_categorical(ref_rows, cur_rows, cat_features):
    """
    Encode categorical features as integers for inclusion in MMD vectors.
    Encoding: each category gets a code based on descending reference frequency.
    Unseen-in-reference categories → code = number of reference categories.
    None / missing values → code = number_of_cats + 1  (treated as unknown).

    Returns (cat_ref_vecs, cat_cur_vecs, encoding_maps).
    """
    enc_maps = {}
    for feat in cat_features:
        counts = {}
        for row in ref_rows:
            v = row.get(feat)
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        sorted_cats = sorted(counts.keys(), key=lambda k: -counts[k])
        enc_maps[feat] = {cat: i for i, cat in enumerate(sorted_cats)}

    def encode(rows):
        vecs = []
        for row in rows:
            vec = []
            for feat in cat_features:
                v = row.get(feat)
                n_cats = len(enc_maps[feat])
                if v is None:
                    vec.append(n_cats + 1)   # missing sentinel
                else:
                    vec.append(enc_maps[feat].get(v, n_cats))
            vecs.append(vec)
        return vecs

    return encode(ref_rows), encode(cur_rows), enc_maps


# ------------------------------------------------------------------
# SECTION 8  RBF KERNEL AND MMD
# ------------------------------------------------------------------

def _squared_euclidean(x, y):
    """
    ||x - y||^2 for two equal-length lists.
    Pure Python; O(d) where d = number of features.
    """
    return sum((a - b) * (a - b) for a, b in zip(x, y))


def _rbf_kernel(x, y, sigma):
    """
    Radial Basis Function (Gaussian) kernel.

    K(x, y) = exp( -||x - y||^2 / (2 * sigma^2) )

    Properties:
      - K(x, x) = 1  for all x.
      - K(x, y) ∈ (0, 1] for all x, y.
      - Symmetric: K(x,y) = K(y,x).
      - Positive definite: induces an RKHS in which MMD is a valid metric.

    sigma (bandwidth) controls the length scale.  Large sigma → smooth kernel
    (less sensitive to small differences).  Median heuristic: sigma ≈
    median pairwise distance / sqrt(2) balances sensitivity.
    """
    sq_dist = _squared_euclidean(x, y)
    return _exp(-sq_dist / (2.0 * sigma * sigma))


def _median_bandwidth(X, Y, max_pairs=2000):
    """
    Median heuristic for bandwidth selection.

    sigma = median( ||xi - yj|| for xi in X, yj in Y ) / sqrt(2)

    We subsample up to max_pairs cross-sample distances to control runtime.
    If len(X)*len(Y) > max_pairs we stride through the product evenly.

    Why divide by sqrt(2)?  The heuristic aims for sigma where
    exp(-d^2/(2sigma^2)) ≈ 0.5 at the median distance, which gives
    sigma = median / sqrt(2*ln2) ≈ median / sqrt(2).
    """
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 1.0

    total = n * m
    step  = max(1, total // max_pairs)
    dists = []
    idx   = 0
    for i in range(n):
        for j in range(m):
            if idx % step == 0:
                dists.append(_sqrt(_squared_euclidean(X[i], Y[j])))
            idx += 1

    if not dists:
        return 1.0
    dists_sorted = sorted(dists)
    k = len(dists_sorted) // 2
    median = (dists_sorted[k - 1] + dists_sorted[k]) / 2.0 if len(dists_sorted) % 2 == 0 else dists_sorted[k]
    return max(median / _sqrt(2.0), 1e-6)


def _build_kernel_matrix(samples, sigma):
    """
    Compute the full (N × N) kernel matrix for a combined sample list.
    Uses symmetry: only compute upper triangle, O(N^2/2) kernel evaluations.

    For N=200, this is 19 900 evaluations.  At ~1 µs each in CPython, ~20 ms.
    """
    N = len(samples)
    K = [[0.0] * N for _ in range(N)]
    for i in range(N):
        K[i][i] = 1.0   # K(x, x) = 1 for RBF
        for j in range(i + 1, N):
            kval = _rbf_kernel(samples[i], samples[j], sigma)
            K[i][j] = kval
            K[j][i] = kval
    return K


def _mmd_unbiased_from_K(K, x_indices, y_indices):
    """
    Unbiased two-sample MMD^2 estimator from a precomputed kernel matrix.

    MMD^2_u(X, Y) =
        (1 / (n*(n-1))) * Σ_{i≠j in X} K[i][j]      (k_xx term)
      + (1 / (m*(m-1))) * Σ_{i≠j in Y} K[i][j]      (k_yy term)
      - (2 / (n*m))     * Σ_{i in X, j in Y} K[i][j] (k_xy term)

    The unbiased estimator has E[MMD^2_u] = MMD^2 (true squared MMD)
    and is preferred over the biased estimator for hypothesis testing.

    For permutation testing, we index into the pre-built K to avoid
    re-computing kernel values, reducing O(N^2) per permutation to O(1)
    lookups (but still O(n^2 + m^2 + nm) sum operations).
    """
    n, m = len(x_indices), len(y_indices)
    # k_xx
    kxx = 0.0
    for i in x_indices:
        for j in x_indices:
            if i != j:
                kxx += K[i][j]
    kxx /= (n * (n - 1)) if n > 1 else 1.0

    # k_yy
    kyy = 0.0
    for i in y_indices:
        for j in y_indices:
            if i != j:
                kyy += K[i][j]
    kyy /= (m * (m - 1)) if m > 1 else 1.0

    # k_xy
    kxy = 0.0
    for i in x_indices:
        for j in y_indices:
            kxy += K[i][j]
    kxy /= (n * m) if (n * m) > 0 else 1.0

    return kxx - 2.0 * kxy + kyy


def _mmd_permutation_test(X, Y, sigma, n_permutations=200, seed=42):
    """
    Two-sample test based on the unbiased MMD^2 statistic with a
    Monte-Carlo permutation test for the p-value.

    WHY MMD DETECTS WHAT COLUMN-WISE TESTS MISS
    --------------------------------------------
    Per-column KS tests examine each feature's marginal distribution
    independently.  A distribution shift that changes only the correlation
    structure between features (joint distribution P(X1, X2, ...) changes
    while each marginal P(Xi) stays the same) is completely invisible to
    any single-column test.  MMD uses a kernel that evaluates all features
    simultaneously, so such joint changes increase the test statistic.

    PERMUTATION TEST
    ----------------
    Under H0 (same distribution), all assignments of combined samples to
    X and Y groups are equally likely.  We generate random permutations
    with the LCG, compute MMD^2 for each, then estimate:
        p-value = (# permutations with MMD^2 ≥ observed + 1) / (n_perm + 1)
    The +1 continuity correction ensures the p-value is never 0 (the
    observed sample itself counts as one permutation).

    RUNTIME NOTE
    ------------
    For N = n+m samples and P permutations, complexity is:
        O(N^2) kernel-matrix construction + O(N^2 * P) index lookups.
    For N=100, P=200: ~10^6 operations (seconds in CPython).
    For N=400, P=200: ~16×10^6 (minutes). Recommend N <= 150 for demos.

    Returns
    -------
    mmd_stat  : float  observed MMD^2
    p_value   : float  Monte-Carlo p-value
    sigma_used: float  bandwidth actually used
    n_extreme : int    # permutations with MMD^2 ≥ observed
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0, 1.0, sigma, 0

    # Build combined sample and kernel matrix once
    combined = X + Y           # list of d-dimensional vectors
    N        = len(combined)
    K        = _build_kernel_matrix(combined, sigma)

    # Observed MMD^2 (X is first n, Y is last m)
    x_idx = list(range(n))
    y_idx = list(range(n, N))
    observed_mmd = _mmd_unbiased_from_K(K, x_idx, y_idx)

    # Permutation test using LCG PRNG
    rng     = _LCG(seed)
    all_idx = list(range(N))
    n_extreme = 0

    for _ in range(n_permutations):
        perm    = rng.shuffle(all_idx)
        px_idx  = perm[:n]
        py_idx  = perm[n:]
        perm_mmd = _mmd_unbiased_from_K(K, px_idx, py_idx)
        if perm_mmd >= observed_mmd:
            n_extreme += 1

    # +1 / (P+1) continuity correction
    p_value = (n_extreme + 1) / (n_permutations + 1)
    return observed_mmd, p_value, sigma, n_extreme


# ------------------------------------------------------------------
# SECTION 9  MAIN DETECTOR CLASS
# ------------------------------------------------------------------

class AlibiStyleDriftDetector:
    """
    Alibi-Detect-style drift detector.

    Two modes, selected at construction:

    'columnwise_bh'
        KS test per numerical feature, chi-square per categorical.
        All p-values corrected via Benjamini-Hochberg FDR (alpha).
        Feature is flagged when its BH-corrected test rejects H0.
        Dataset drift when any feature is flagged.

    'multivariate_mmd'
        All numerical features (and optionally encoded categoricals)
        are normalised and concatenated into one vector per row.
        A single MMD^2 test with permutation p-value is run on those
        joint vectors.  Dataset drift when p-value < alpha.

    Parameters
    ----------
    method          : 'columnwise_bh' or 'multivariate_mmd'
    alpha           : significance threshold (default 0.05)
    sigma           : RBF bandwidth; None → median heuristic (MMD mode only)
    n_permutations  : number of permutations (MMD mode; default 200)
    include_cat_mmd : encode categoricals and include in MMD vectors
    seed            : PRNG seed for reproducibility (MMD mode)
    """

    VALID_METHODS = ('columnwise_bh', 'multivariate_mmd')

    def __init__(self, method='columnwise_bh', alpha=0.05,
                 sigma=None, n_permutations=200,
                 include_cat_mmd=False, seed=42):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got '{method}'"
            )
        self.method          = method
        self.alpha           = alpha
        self.sigma           = sigma        # None → auto via median heuristic
        self.n_permutations  = n_permutations
        self.include_cat_mmd = include_cat_mmd
        self.seed            = seed

    # ----------------------------------------------------------------
    def detect(self, reference_data, current_data,
               numerical_features=None, categorical_features=None):
        """
        Run drift detection.

        Parameters
        ----------
        reference_data        : list of dicts  — baseline observations
        current_data          : list of dicts  — production observations
        numerical_features    : list of str    — numerical feature names
        categorical_features  : list of str    — categorical feature names

        Returns
        -------
        dict  Full detection report.
        """
        numerical_features   = numerical_features   or []
        categorical_features = categorical_features or []

        if self.method == 'columnwise_bh':
            return self._detect_columnwise_bh(
                reference_data, current_data,
                numerical_features, categorical_features)
        else:
            return self._detect_mmd(
                reference_data, current_data,
                numerical_features, categorical_features)

    # ---- Mode A: column-wise KS + BH-FDR ----------------------------
    def _detect_columnwise_bh(self, ref, cur, num_feats, cat_feats):
        all_features = []
        p_values     = []
        raw_results  = {}

        # Numerical: KS test
        for feat in num_feats:
            rv, rm = _extract_numerical(ref, feat)
            cv, cm = _extract_numerical(cur, feat)
            if len(rv) < 2 or len(cv) < 2:
                raw_results[feat] = {
                    "type": "numerical", "error": "too_few_values",
                    "ks_stat": None, "ks_pvalue": None,
                }
                continue
            ks_stat, ks_pval = _ks_test(rv, cv)
            all_features.append(feat)
            p_values.append(ks_pval)
            raw_results[feat] = {
                "type": "numerical",
                "n_ref": len(rv), "n_cur": len(cv),
                "n_missing_ref": rm, "n_missing_cur": cm,
                "ks_statistic": round(ks_stat, 6),
                "ks_pvalue":    round(ks_pval, 6),
                "ks_pvalue_note": "Kolmogorov asymptotic approx",
            }

        # Categorical: chi-square
        for feat in cat_feats:
            rc, rm = _count_categorical(ref, feat)
            cc, cm = _count_categorical(cur, feat)
            if not rc or not cc:
                raw_results[feat] = {
                    "type": "categorical", "error": "no_data",
                    "chi2_stat": None, "chi2_pvalue": None,
                }
                continue
            chi2_stat, chi2_p, df = _chi2_test(rc, cc)
            all_features.append(feat)
            p_values.append(chi2_p)
            raw_results[feat] = {
                "type": "categorical",
                "n_ref": sum(rc.values()), "n_cur": sum(cc.values()),
                "n_missing_ref": rm, "n_missing_cur": cm,
                "chi2_statistic": round(chi2_stat, 6),
                "chi2_pvalue":    round(chi2_p, 6),
                "degrees_of_freedom": df,
                "chi2_pvalue_note": "Wilson-Hilferty normal approximation",
            }

        # BH-FDR correction
        rejections = _bh_fdr(p_values, alpha=self.alpha)

        # Assemble feature-level results
        feature_results = {}
        bh_detail = []
        for feat, pval, rej in zip(all_features, p_values, rejections):
            raw_results[feat]["bh_rejected"]    = rej
            raw_results[feat]["drift_detected"] = rej
            feature_results[feat] = raw_results[feat]
            bh_detail.append({
                "feature": feat, "p_value": round(pval, 6), "rejected": rej
            })

        # Features that errored are not in feature_results yet; add them
        for feat in list(raw_results.keys()):
            if feat not in feature_results:
                raw_results[feat]["drift_detected"] = False
                feature_results[feat] = raw_results[feat]

        n_drifted  = sum(1 for r in feature_results.values()
                         if r.get("drift_detected", False))
        n_tested   = len(all_features)
        # Dataset drift: ANY feature flagged after BH
        dataset_drift = any(rejections)

        return {
            "model":              "AlibiStyleDriftDetector",
            "detection_method":   "columnwise_bh",
            "n_reference":        len(ref),
            "n_current":          len(cur),
            "alpha":              self.alpha,
            "n_features_tested":  n_tested,
            "n_features_drifted": n_drifted,
            "bh_fdr_detail":      sorted(bh_detail, key=lambda d: d["p_value"]),
            "dataset_drift_detected": dataset_drift,
            "feature_results":    feature_results,
            "explanation": (
                f"BH-FDR (alpha={self.alpha}) applied to {n_tested} features. "
                f"{n_drifted} feature(s) rejected after correction. "
                f"Dataset drift: {'YES' if dataset_drift else 'NO'}."
            ),
        }

    # ---- Mode B: Multivariate MMD -----------------------------------
    def _detect_mmd(self, ref, cur, num_feats, cat_feats):
        # Build normalised numerical vectors
        X_ref, X_cur, stats, skip_r, skip_c = _zscore_normalize(
            ref, cur, num_feats)

        # Optionally append ordinal-encoded categoricals
        if self.include_cat_mmd and cat_feats:
            cat_ref, cat_cur, enc_maps = _ordinal_encode_categorical(
                ref, cur, cat_feats)
            # Zip with normalised numerical vectors
            # (rows already aligned since we iterate original data order)
            X_ref = [nr + cr for nr, cr in zip(X_ref, cat_ref[:len(X_ref)])]
            X_cur = [nc + cc for nc, cc in zip(X_cur, cat_cur[:len(X_cur)])]
            enc_detail = {f: list(enc_maps[f].keys()) for f in enc_maps}
        else:
            enc_detail = {}

        if len(X_ref) < 2 or len(X_cur) < 2:
            return {
                "model":  "AlibiStyleDriftDetector",
                "detection_method": "multivariate_mmd",
                "error":  "insufficient_valid_rows",
                "n_valid_ref": len(X_ref),
                "n_valid_cur": len(X_cur),
                "dataset_drift_detected": False,
            }

        # Bandwidth
        if self.sigma is None:
            sigma_used = _median_bandwidth(X_ref, X_cur)
        else:
            sigma_used = float(self.sigma)

        mmd_stat, p_value, sigma_ret, n_extreme = _mmd_permutation_test(
            X_ref, X_cur, sigma_used,
            n_permutations=self.n_permutations,
            seed=self.seed
        )

        dataset_drift = p_value < self.alpha

        return {
            "model":                  "AlibiStyleDriftDetector",
            "detection_method":       "multivariate_mmd",
            "n_reference":            len(ref),
            "n_current":              len(cur),
            "n_valid_ref":            len(X_ref),
            "n_valid_cur":            len(X_cur),
            "n_skipped_ref":          skip_r,
            "n_skipped_cur":          skip_c,
            "numerical_features":     num_feats,
            "categorical_features_included": cat_feats if self.include_cat_mmd else [],
            "normalization_stats":    {f: {"mean": round(v[0], 4),
                                           "std":  round(v[1], 4)}
                                       for f, v in stats.items()},
            "sigma_bandwidth":        round(sigma_used, 6),
            "sigma_method":           "median heuristic" if self.sigma is None else "user-specified",
            "mmd_statistic":          round(mmd_stat, 8),
            "n_permutations":         self.n_permutations,
            "n_permutations_extreme": n_extreme,
            "p_value":                round(p_value, 6),
            "alpha":                  self.alpha,
            "dataset_drift_detected": dataset_drift,
            "mmd_note": (
                "Unbiased MMD^2 estimator. Tests joint distribution of all "
                "features simultaneously. Can detect correlation changes "
                "that are invisible to column-wise marginal tests."
            ),
            "explanation": (
                f"MMD^2 = {round(mmd_stat, 6)}  |  "
                f"permutation p-value = {round(p_value, 4)} "
                f"({n_extreme}/{self.n_permutations} permutations >= observed).  "
                f"Dataset drift: {'YES' if dataset_drift else 'NO'}."
            ),
        }

