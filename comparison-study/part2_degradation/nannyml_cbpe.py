"""
NannyML CBPE (Confidence-Based Performance Estimation) Degradation Monitor
==========================================================================

A zero-import implementation of NannyML's CBPE algorithm for estimating model
performance (F1, ROC-AUC) in production WITHOUT access to ground-truth labels.

CBPE Theory
-----------
In production ML systems, true labels often arrive with significant delay or
never at all. CBPE solves this by leveraging the model's own predicted
probabilities to *estimate* what the performance metrics would be if we had
true labels.

The key insight: if a model's predicted probability p for a sample is well-
calibrated (i.e., among all samples the model assigns probability p, roughly
p fraction are truly positive), then we can compute *expected* confusion matrix
entries:

    Expected TP for a positive prediction with calibrated prob p_cal:
        contribution = p_cal   (probability the sample is truly positive)

    Expected FP for a positive prediction with calibrated prob p_cal:
        contribution = 1 - p_cal   (probability it is truly negative)

    Expected FN for a negative prediction with calibrated prob p_cal:
        contribution = p_cal   (missed positive)

From these expected counts we derive estimated F1 and ROC-AUC.

Platt Scaling (Calibration)
---------------------------
Raw model probabilities are often miscalibrated. Platt Scaling fits a logistic
transform on a held-out reference set:

    p_calibrated = sigmoid(a * p_raw + b) = 1 / (1 + exp(-(a * p_raw + b)))

Parameters a and b are learned via gradient descent on the log-loss:

    L = -1/N * sum[ y*log(q) + (1-y)*log(1-q) ]

where q = sigmoid(a*p + b).

All mathematical functions (exp, log, sqrt) are implemented from scratch using
Taylor series and Newton's method approximations.

Author: CBPE Builder Agent
"""


# ---------------------------------------------------------------------------
# Mathematical primitives (zero-import: no math, numpy, scipy allowed)
# ---------------------------------------------------------------------------

def _manual_exp(x):
    """Compute e^x using a Taylor series expansion with range reduction.

    For large negative x (x < -709.78), returns 0.0 to avoid underflow.
    For large positive x (x > 709.78), returns float('inf') to avoid overflow.

    Taylor series: e^r = sum_{n=0}^{inf} r^n / n!
    """
    if x < -709.78:
        return 0.0
    if x > 709.78:
        return float('inf')

    # e constant approximated to high precision
    E = 2.718281828459045

    # Decompose x = k + r where k is integer part
    k = int(x)
    if x < 0 and x != k:
        k -= 1  # floor for negative numbers
    r = x - k

    # Taylor series for e^r (|r| < 1)
    result = 1.0
    term = 1.0
    for n in range(1, 35):
        term *= r / n
        result += term
        if abs(term) < 1e-16:
            break

    # Multiply by e^k via repeated squaring
    if k >= 0:
        e_power = 1.0
        base = E
        pk = k
        while pk > 0:
            if pk % 2 == 1:
                e_power *= base
            base *= base
            pk //= 2
        return result * e_power
    else:
        e_power = 1.0
        base = E
        pk = -k
        while pk > 0:
            if pk % 2 == 1:
                e_power *= base
            base *= base
            pk //= 2
        return result / e_power


def _manual_log(x):
    """Compute natural logarithm ln(x) using Newton's method.

    Uses the identity: if y = ln(x), then e^y = x.
    We solve f(y) = e^y - x = 0 via Newton iteration:
        y_{n+1} = y_n - (e^{y_n} - x) / e^{y_n}
                = y_n - 1 + x / e^{y_n}

    Initial guess uses the bit-length of x for rough log2 estimate.

    Args:
        x: positive float

    Returns:
        float: approximation of ln(x)

    Raises:
        ValueError: if x <= 0
    """
    if x <= 0:
        raise ValueError("log requires positive argument, got {}".format(x))
    if x == 1.0:
        return 0.0

    # Initial guess: rough log2 * ln(2)
    # For x >= 1: count integer part magnitude
    ln2 = 0.6931471805599453
    if x >= 1.0:
        mag = 0
        tmp = x
        while tmp >= 2.0:
            tmp /= 2.0
            mag += 1
        y = mag * ln2
    else:
        mag = 0
        tmp = x
        while tmp < 1.0:
            tmp *= 2.0
            mag += 1
        y = -mag * ln2

    # Newton iterations: y = y - 1 + x * exp(-y)
    for _ in range(60):
        ey = _manual_exp(y)
        if ey == 0.0:
            y += 1.0
            continue
        y_new = y - 1.0 + x / ey
        if abs(y_new - y) < 1e-15:
            break
        y = y_new

    return y


def _manual_sqrt(x):
    """Compute square root using Newton's method (Heron's method).

    Solves f(y) = y^2 - x = 0:
        y_{n+1} = (y_n + x / y_n) / 2

    Args:
        x: non-negative float

    Returns:
        float: approximation of sqrt(x)
    """
    if x < 0:
        raise ValueError("sqrt requires non-negative argument, got {}".format(x))
    if x == 0.0:
        return 0.0

    # Initial guess
    y = x if x >= 1.0 else 1.0
    for _ in range(100):
        y_new = 0.5 * (y + x / y)
        if abs(y_new - y) < 1e-15 * abs(y_new):
            break
        y = y_new
    return y


def _sigmoid(z):
    """Compute sigmoid(z) = 1 / (1 + exp(-z)) in a numerically stable way."""
    if z >= 0:
        ez = _manual_exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = _manual_exp(z)
        return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# Platt Scaling — calibration via logistic regression on predicted probs
# ---------------------------------------------------------------------------

def _platt_scaling_fit(y_true, y_pred_proba, lr=0.05, n_iter=2000):
    """Fit Platt Scaling parameters (a, b) via gradient descent.

    Platt Scaling models calibrated probability as:
        q_i = sigmoid(a * p_i + b)

    We minimize the mean log-loss:
        L = -1/N * sum_i [ y_i * log(q_i) + (1 - y_i) * log(1 - q_i) ]

    Gradients:
        dL/da = 1/N * sum_i [ (q_i - y_i) * p_i ]
        dL/db = 1/N * sum_i [ (q_i - y_i) ]

    Args:
        y_true: list of true binary labels (0 or 1)
        y_pred_proba: list of raw predicted probabilities
        lr: learning rate for gradient descent
        n_iter: number of gradient descent iterations

    Returns:
        tuple: (a, b) calibration parameters
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 0.0

    # Platt's recommended target encoding to avoid overfitting
    n_pos = sum(y_true)
    n_neg = n - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0) if n_pos > 0 else 0.5
    t_neg = 1.0 / (n_neg + 2.0) if n_neg > 0 else 0.5
    targets = [t_pos if yi == 1 else t_neg for yi in y_true]

    a = 1.0
    b = 0.0

    for iteration in range(n_iter):
        grad_a = 0.0
        grad_b = 0.0
        for i in range(n):
            z = a * y_pred_proba[i] + b
            q = _sigmoid(z)
            err = q - targets[i]
            grad_a += err * y_pred_proba[i]
            grad_b += err
        grad_a /= n
        grad_b /= n

        a -= lr * grad_a
        b -= lr * grad_b

    return a, b


def _platt_scaling_transform(y_pred_proba, a, b):
    """Apply Platt Scaling calibration to a list of predicted probabilities.

    q_i = sigmoid(a * p_i + b)

    Args:
        y_pred_proba: list of raw predicted probabilities
        a: calibration slope
        b: calibration intercept

    Returns:
        list: calibrated probabilities
    """
    return [_sigmoid(a * p + b) for p in y_pred_proba]


# ---------------------------------------------------------------------------
# Performance metric estimation from calibrated probabilities
# ---------------------------------------------------------------------------

def _compute_true_f1(y_true, y_pred):
    """Compute actual F1 score from true labels and predictions.

    F1 = 2 * precision * recall / (precision + recall)
       = 2*TP / (2*TP + FP + FN)

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels

    Returns:
        float: F1 score
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def _compute_true_roc_auc(y_true, y_pred_proba):
    """Compute actual ROC-AUC from true labels and predicted probabilities.

    Uses the trapezoidal rule on the ROC curve constructed by sweeping
    thresholds from high to low probability.

    ROC curve: plot TPR (sensitivity) vs FPR (1 - specificity) at each
    threshold, then integrate the area under the curve.

    Args:
        y_true: list of true binary labels
        y_pred_proba: list of predicted probabilities

    Returns:
        float: ROC-AUC score
    """
    n = len(y_true)
    if n == 0:
        return 0.5

    total_pos = sum(y_true)
    total_neg = n - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    # Group by distinct probability thresholds to properly handle ties
    thresholds = {}
    for p, y in zip(y_pred_proba, y_true):
        if p not in thresholds:
            thresholds[p] = [0, 0]  # [pos, neg]
        if y == 1:
            thresholds[p][0] += 1
        else:
            thresholds[p][1] += 1

    auc = 0.0
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for p in sorted(thresholds.keys(), reverse=True):
        pos_cnt, neg_cnt = thresholds[p]
        tp += pos_cnt
        fp += neg_cnt
        tpr = tp / total_pos
        fpr = fp / total_neg
        # Trapezoidal integration
        auc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr

    return auc


def _estimate_f1(calibrated_proba, y_pred):
    """Estimate F1 score using calibrated probabilities (no true labels).

    The key CBPE insight: calibrated probability p_cal represents the
    probability that the true label is 1. Therefore:

    For each sample predicted positive (y_pred=1):
        Expected TP contribution = p_cal     (prob it's truly positive)
        Expected FP contribution = 1 - p_cal (prob it's truly negative)

    For each sample predicted negative (y_pred=0):
        Expected FN contribution = p_cal     (prob it's truly positive, missed)

    Estimated F1 = 2*E[TP] / (2*E[TP] + E[FP] + E[FN])

    Args:
        calibrated_proba: list of calibrated probabilities
        y_pred: list of predicted class labels

    Returns:
        float: estimated F1 score
    """
    expected_tp = 0.0
    expected_fp = 0.0
    expected_fn = 0.0

    for p_cal, pred in zip(calibrated_proba, y_pred):
        if pred == 1:
            expected_tp += p_cal
            expected_fp += (1.0 - p_cal)
        else:
            expected_fn += p_cal

    denom = 2.0 * expected_tp + expected_fp + expected_fn
    if denom == 0.0:
        return 0.0
    return (2.0 * expected_tp) / denom


def _estimate_roc_auc(calibrated_proba):
    """Estimate ROC-AUC using the probability ranking method.

    For estimated AUC without true labels, we use the Wilcoxon-Mann-Whitney
    interpretation: AUC = P(score_pos > score_neg). With calibrated
    probabilities, each sample with calibrated prob p contributes:
        - p weight to the "positive" distribution
        - (1-p) weight to the "negative" distribution

    We approximate by constructing a soft ROC curve:
    1. Sort samples by descending calibrated probability
    2. Accumulate expected positives (p_cal) and expected negatives (1-p_cal)
    3. Integrate the resulting curve via the trapezoidal rule

    Args:
        calibrated_proba: list of calibrated probabilities

    Returns:
        float: estimated ROC-AUC
    """
    n = len(calibrated_proba)
    if n == 0:
        return 0.5

    total_pos = sum(calibrated_proba)
    total_neg = sum(1.0 - p for p in calibrated_proba)

    if total_pos < 1e-12 or total_neg < 1e-12:
        return 0.5

    # Sort by descending calibrated probability
    sorted_proba = sorted(calibrated_proba, reverse=True)

    auc = 0.0
    cum_tp = 0.0
    cum_fp = 0.0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for p_cal in sorted_proba:
        cum_tp += p_cal
        cum_fp += (1.0 - p_cal)
        tpr = cum_tp / total_pos
        fpr = cum_fp / total_neg
        auc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr)
        prev_tpr = tpr
        prev_fpr = fpr

    return auc


# ---------------------------------------------------------------------------
# Bootstrap reference statistics
# ---------------------------------------------------------------------------

def _bootstrap_reference_stats(y_true, y_pred_proba, y_pred, chunk_size,
                                n_bootstrap=100, seed=42):
    """Compute reference metric distributions via bootstrap resampling.

    Splits the reference data into chunks of `chunk_size`, computes true
    F1 and ROC-AUC for each chunk, then derives mean and standard deviation
    of the reference performance distribution.

    Additionally performs bootstrap resampling to get more robust statistics:
    for each bootstrap iteration, samples chunks with replacement and
    computes the metric.

    Args:
        y_true: list of true labels (reference set)
        y_pred_proba: list of predicted probabilities
        y_pred: list of predicted labels
        chunk_size: size of each evaluation chunk
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        dict: {"f1_mean", "f1_std", "auc_mean", "auc_std"}
    """
    n = len(y_true)
    # Compute per-chunk metrics
    chunk_f1s = []
    chunk_aucs = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if end - start < max(10, chunk_size // 4):
            continue  # skip very small tail chunks
        chunk_yt = y_true[start:end]
        chunk_yp = y_pred[start:end]
        chunk_pp = y_pred_proba[start:end]
        chunk_f1s.append(_compute_true_f1(chunk_yt, chunk_yp))
        chunk_aucs.append(_compute_true_roc_auc(chunk_yt, chunk_pp))

    if len(chunk_f1s) == 0:
        return {"f1_mean": 0.0, "f1_std": 0.0, "auc_mean": 0.5, "auc_std": 0.0}

    # Simple LCG pseudo-random number generator (no imports)
    rng_state = [seed]

    def _lcg_next():
        rng_state[0] = (1103515245 * rng_state[0] + 12345) & 0x7FFFFFFF
        return rng_state[0]

    # Bootstrap resampling of chunk-level metrics
    boot_f1_means = []
    boot_auc_means = []
    nc = len(chunk_f1s)
    for _ in range(n_bootstrap):
        sampled_f1 = 0.0
        sampled_auc = 0.0
        for __ in range(nc):
            idx = _lcg_next() % nc
            sampled_f1 += chunk_f1s[idx]
            sampled_auc += chunk_aucs[idx]
        boot_f1_means.append(sampled_f1 / nc)
        boot_auc_means.append(sampled_auc / nc)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _std(lst):
        if len(lst) < 2:
            return 0.0
        m = _mean(lst)
        variance = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
        return _manual_sqrt(variance)

    return {
        "f1_mean": _mean(boot_f1_means),
        "f1_std": _std(boot_f1_means),
        "auc_mean": _mean(boot_auc_means),
        "auc_std": _std(boot_auc_means),
    }


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class CBPEDegradationDetector:
    """NannyML-style CBPE Degradation Detector.

    Confidence-Based Performance Estimation (CBPE) monitors model
    performance in production without requiring ground-truth labels.
    It works by:

    1. **Calibration (fit)**: Learning a Platt Scaling transform on a
       reference dataset where true labels ARE available. This ensures
       predicted probabilities are well-calibrated.

    2. **Estimation (estimate/detect)**: Using calibrated probabilities
       to compute *expected* confusion matrix entries and derive
       estimated F1 and ROC-AUC for each chunk of production data.

    3. **Alerting**: Comparing estimated metrics against the reference
       baseline and flagging significant deviations.

    Usage:
        detector = CBPEDegradationDetector(chunk_size=500)
        detector.fit(y_true_ref, y_proba_ref, y_pred_ref)
        result = detector.detect(y_proba_prod, y_pred_prod)
        if result["overall_alert"]:
            print("Performance degradation detected!")
    """

    def __init__(self, chunk_size=500, metrics=None,
                 alert_threshold_f1=0.05, alert_threshold_auc=0.05):
        """Initialize the CBPE detector.

        Args:
            chunk_size: number of predictions per evaluation chunk
            metrics: list of metrics to estimate (default: ['f1', 'roc_auc'])
            alert_threshold_f1: absolute deviation threshold for F1 alerts
            alert_threshold_auc: absolute deviation threshold for AUC alerts
        """
        self.chunk_size = chunk_size
        self.metrics = metrics if metrics is not None else ["f1", "roc_auc"]
        self.alert_threshold_f1 = alert_threshold_f1
        self.alert_threshold_auc = alert_threshold_auc

        # Calibration parameters (set during fit)
        self._cal_a = None
        self._cal_b = None

        # Reference metrics (set during fit)
        self._ref_f1 = None
        self._ref_auc = None
        self._ref_stats = None
        self._ref_size = None
        self._fitted = False

    def fit(self, y_true, y_pred_proba, y_pred):
        """Calibration phase: fit Platt Scaling and compute reference baseline.

        This method must be called with a reference/validation dataset where
        true labels are available. It:

        1. Fits Platt Scaling parameters (a, b) to calibrate probabilities
        2. Computes the true F1 and ROC-AUC as baseline metrics
        3. Bootstraps reference chunk statistics (mean, std) for alerting

        Args:
            y_true: list of true binary labels (0 or 1), length N
            y_pred_proba: list of raw predicted probabilities, length N
            y_pred: list of predicted binary labels (0 or 1), length N

        Raises:
            ValueError: if input lists have mismatched lengths
        """
        if not (len(y_true) == len(y_pred_proba) == len(y_pred)):
            raise ValueError(
                "Input length mismatch: y_true={}, y_pred_proba={}, y_pred={}".format(
                    len(y_true), len(y_pred_proba), len(y_pred)
                )
            )

        self._ref_size = len(y_true)

        # Step 1: Fit Platt Scaling calibration
        self._cal_a, self._cal_b = _platt_scaling_fit(y_true, y_pred_proba)

        # Step 2: Compute true reference metrics
        self._ref_f1 = _compute_true_f1(y_true, y_pred)
        self._ref_auc = _compute_true_roc_auc(y_true, y_pred_proba)

        # Step 3: Bootstrap reference distribution statistics
        self._ref_stats = _bootstrap_reference_stats(
            y_true, y_pred_proba, y_pred, self.chunk_size
        )

        self._fitted = True

    def estimate(self, y_pred_proba, y_pred):
        """Estimate performance metrics for production data WITHOUT true labels.

        Applies Platt Scaling calibration, then computes expected F1 and
        ROC-AUC for each chunk using probability-weighted confusion matrix
        entries.

        Args:
            y_pred_proba: list of raw predicted probabilities (production)
            y_pred: list of predicted class labels (production)

        Returns:
            list of dicts, one per chunk, each containing:
                - chunk_idx: int
                - estimated_f1: float
                - estimated_roc_auc: float
                - f1_alert: bool
                - auc_alert: bool
                - f1_deviation: float (estimated - reference)
                - auc_deviation: float

        Raises:
            RuntimeError: if fit() has not been called
            ValueError: if input lists have mismatched lengths
        """
        if not self._fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        if len(y_pred_proba) != len(y_pred):
            raise ValueError(
                "Input length mismatch: y_pred_proba={}, y_pred={}".format(
                    len(y_pred_proba), len(y_pred)
                )
            )

        # Apply Platt Scaling calibration
        calibrated = _platt_scaling_transform(y_pred_proba, self._cal_a, self._cal_b)

        n = len(calibrated)
        chunks = []

        for chunk_idx, start in enumerate(range(0, n, self.chunk_size)):
            end = min(start + self.chunk_size, n)
            chunk_cal = calibrated[start:end]
            chunk_pred = y_pred[start:end]

            # Estimate metrics for this chunk
            est_f1 = _estimate_f1(chunk_cal, chunk_pred)
            est_auc = _estimate_roc_auc(chunk_cal)

            f1_dev = est_f1 - self._ref_f1
            auc_dev = est_auc - self._ref_auc

            f1_alert = abs(f1_dev) > self.alert_threshold_f1
            auc_alert = abs(auc_dev) > self.alert_threshold_auc

            chunks.append({
                "chunk_idx": chunk_idx,
                "estimated_f1": round(est_f1, 6),
                "estimated_roc_auc": round(est_auc, 6),
                "f1_alert": f1_alert,
                "auc_alert": auc_alert,
                "f1_deviation": round(f1_dev, 6),
                "auc_deviation": round(auc_dev, 6),
            })

        return chunks

    def detect(self, y_pred_proba, y_pred):
        """Run full detection pipeline and return structured result.

        This is the primary user-facing method. It estimates performance
        for each chunk and produces a comprehensive result dictionary with
        alerts and explanations.

        Args:
            y_pred_proba: list of raw predicted probabilities (production)
            y_pred: list of predicted class labels (production)

        Returns:
            dict: structured detection result (see module docstring for schema)
        """
        chunks = self.estimate(y_pred_proba, y_pred)

        # Determine overall alert status
        any_f1_alert = any(c["f1_alert"] for c in chunks)
        any_auc_alert = any(c["auc_alert"] for c in chunks)
        overall_alert = any_f1_alert or any_auc_alert

        # Build alert reason
        alert_reasons = []
        if any_f1_alert:
            flagged = [c["chunk_idx"] for c in chunks if c["f1_alert"]]
            alert_reasons.append(
                "F1 deviation exceeds threshold in chunk(s): {}".format(flagged)
            )
        if any_auc_alert:
            flagged = [c["chunk_idx"] for c in chunks if c["auc_alert"]]
            alert_reasons.append(
                "ROC-AUC deviation exceeds threshold in chunk(s): {}".format(flagged)
            )

        alert_reason = "; ".join(alert_reasons) if alert_reasons else "No alerts"

        # Build explanation
        if overall_alert:
            explanation = (
                "CBPE detected performance degradation. "
                "Estimated metrics deviate from the reference baseline "
                "beyond the configured thresholds (F1: {}, AUC: {}). "
                "Reference F1={:.4f}, Reference AUC={:.4f}. "
                "This suggests a distribution shift in the production data "
                "that is affecting model performance.".format(
                    self.alert_threshold_f1, self.alert_threshold_auc,
                    self._ref_f1, self._ref_auc
                )
            )
        else:
            explanation = (
                "CBPE estimates are within acceptable range of the reference "
                "baseline (F1={:.4f}, AUC={:.4f}). No significant performance "
                "degradation detected. Thresholds: F1={}, AUC={}.".format(
                    self._ref_f1, self._ref_auc,
                    self.alert_threshold_f1, self.alert_threshold_auc
                )
            )

        first_f1_alert_chunk = None
        first_auc_alert_chunk = None
        for c in chunks:
            if c.get("f1_alert") and first_f1_alert_chunk is None:
                first_f1_alert_chunk = c.get("chunk_index")
            if c.get("auc_alert") and first_auc_alert_chunk is None:
                first_auc_alert_chunk = c.get("chunk_index")

        return {
            "model": "NannyML_CBPE",
            "detection_method": "confidence_based_performance_estimation",
            "reference_sample_size": self._ref_size,
            "current_sample_size": len(y_pred_proba),
            "n_chunks": len(chunks),
            "first_f1_alert_chunk": first_f1_alert_chunk,
            "first_auc_alert_chunk": first_auc_alert_chunk,
            "calibration_params": {
                "a": round(self._cal_a, 6),
                "b": round(self._cal_b, 6),
            },
            "reference_metrics": {
                "f1": round(self._ref_f1, 6),
                "roc_auc": round(self._ref_auc, 6),
            },
            "reference_distribution": {
                "f1_mean": round(self._ref_stats["f1_mean"], 6),
                "f1_std": round(self._ref_stats["f1_std"], 6),
                "auc_mean": round(self._ref_stats["auc_mean"], 6),
                "auc_std": round(self._ref_stats["auc_std"], 6),
            },
            "chunks": chunks,
            "overall_alert": overall_alert,
            "alert_reason": alert_reason,
            "explanation": explanation,
        }


# ---------------------------------------------------------------------------
# Demo / self-test (runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Pseudo-random number generator (LCG, no imports) ---
    _rng = [12345]

    def _rand():
        """Return a pseudo-random float in [0, 1)."""
        _rng[0] = (1103515245 * _rng[0] + 12345) & 0x7FFFFFFF
        return _rng[0] / 2147483648.0

    def _randn():
        """Approximate standard normal via Box-Muller (manual log/sqrt)."""
        u1 = _rand() + 1e-12
        u2 = _rand()
        r = _manual_sqrt(-2.0 * _manual_log(u1))
        # cos(2*pi*u2) approximated via Taylor
        theta = 6.283185307179586 * u2
        # Manual cosine via Taylor series
        cos_val = 1.0
        term = 1.0
        for k in range(1, 15):
            term *= -theta * theta / ((2 * k - 1) * (2 * k))
            cos_val += term
        return r * cos_val

    # Generate synthetic reference data (well-behaved)
    n_ref = 2000
    y_true_ref = []
    y_proba_ref = []
    y_pred_ref = []

    for _ in range(n_ref):
        label = 1 if _rand() > 0.5 else 0
        y_true_ref.append(label)
        # Model is reasonably well-calibrated on reference
        noise = _randn() * 0.15
        p = 0.3 + 0.4 * label + noise
        p = max(0.01, min(0.99, p))
        y_proba_ref.append(p)
        y_pred_ref.append(1 if p >= 0.5 else 0)

    # Generate synthetic production data (degraded: shifted probabilities)
    n_prod = 1500
    y_proba_prod = []
    y_pred_prod = []

    for _ in range(n_prod):
        label = 1 if _rand() > 0.5 else 0
        noise = _randn() * 0.25
        # Shift probabilities to simulate degradation
        p = 0.4 + 0.2 * label + noise
        p = max(0.01, min(0.99, p))
        y_proba_prod.append(p)
        y_pred_prod.append(1 if p >= 0.5 else 0)

    # Run CBPE detector
    detector = CBPEDegradationDetector(chunk_size=500)
    detector.fit(y_true_ref, y_proba_ref, y_pred_ref)
    result = detector.detect(y_proba_prod, y_pred_prod)

    # Print results
    print("=" * 70)
    print("NannyML CBPE Degradation Detection Results")
    print("=" * 70)
    print("Model:                  {}".format(result["model"]))
    print("Method:                 {}".format(result["detection_method"]))
    print("Reference samples:      {}".format(result["reference_sample_size"]))
    print("Production samples:     {}".format(result["current_sample_size"]))
    print("Calibration params:     a={}, b={}".format(
        result["calibration_params"]["a"],
        result["calibration_params"]["b"]
    ))
    print("\nReference Metrics:")
    print("  F1:      {}".format(result["reference_metrics"]["f1"]))
    print("  ROC-AUC: {}".format(result["reference_metrics"]["roc_auc"]))
    print("\nChunk-level Estimates:")
    for chunk in result["chunks"]:
        alert_marker = " ** ALERT **" if chunk["f1_alert"] or chunk["auc_alert"] else ""
        print("  Chunk {}: F1={:.4f} (dev={:+.4f}{}) | AUC={:.4f} (dev={:+.4f}{})".format(
            chunk["chunk_idx"],
            chunk["estimated_f1"], chunk["f1_deviation"],
            " !" if chunk["f1_alert"] else "",
            chunk["estimated_roc_auc"], chunk["auc_deviation"],
            " !" if chunk["auc_alert"] else "",
        ))
    print("\nOverall Alert: {}".format(result["overall_alert"]))
    print("Reason: {}".format(result["alert_reason"]))
    print("\n{}".format(result["explanation"]))
