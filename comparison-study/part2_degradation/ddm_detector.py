"""
DDM & EDDM Degradation Detectors — Zero-Import Implementation
==============================================================

Drift Detection Method (DDM) and Early Drift Detection Method (EDDM),
implemented entirely from scratch with NO imports.

DDM Mathematics
---------------
DDM (Gama et al., 2004) is a sequential hypothesis test on a Bernoulli
error stream.  For each incoming prediction we observe an error e_i in {0,1}.

At step i the running error rate is:

    p_i = (number of errors so far) / i

The standard deviation of p_i (treating errors as i.i.d. Bernoulli) is:

    s_i = sqrt( p_i * (1 - p_i) / i )

DDM maintains the historical minimum of p + s:

    p_min, s_min  — values at the step where p_i + s_i was smallest

Decision rules (after min_samples observations):

    WARNING :  p_i + s_i  >  p_min + warning_level * s_min
    ALARM   :  p_i + s_i  >  p_min + alarm_level   * s_min

On ALARM the detector resets: a concept drift has occurred and we begin
tracking the new concept from scratch.

EDDM Mathematics
----------------
EDDM (Baena-Garcia et al., 2006) monitors the *distance between errors*
rather than the raw error rate.  Let d_i be the number of observations
between the (i-1)-th and i-th error.  EDDM tracks:

    d_mean = running mean of d_i
    d_std  = running standard deviation of d_i
    metric = d_mean + 2 * d_std          (the "EDDM metric")

The maximum of this metric is tracked as metric_max.  Ratios of
metric / metric_max are used for warnings (< 0.95) and alarms (< 0.90).

Zero-import note
----------------
ALL math helpers (sqrt, F1, accuracy) are hand-rolled using Newton's
method and simple loops.  No ``import`` or ``from … import`` appears
anywhere in this file.
"""


# ---------------------------------------------------------------------------
# Math helpers (no imports)
# ---------------------------------------------------------------------------

def _manual_sqrt(x):
    """Compute square root via Newton-Raphson iteration.

    Uses the recurrence  g_{n+1} = (g_n + x / g_n) / 2  starting from
    an initial guess of x/2 (or 1.0 for small x).  Converges quadratically.
    """
    if x < 0:
        return 0.0  # guard; conceptually undefined
    if x == 0:
        return 0.0
    guess = x / 2.0 if x >= 1.0 else 1.0
    for _ in range(100):
        next_guess = (guess + x / guess) / 2.0
        if abs(next_guess - guess) < 1e-15:
            break
        guess = next_guess
    return guess


def _compute_error(y_true, y_pred):
    """Return 1 if prediction is wrong, 0 if correct."""
    return 0 if y_true == y_pred else 1


def _compute_f1(y_true_list, y_pred_list):
    """Manually compute binary F1 score.

    F1 = 2 * TP / (2 * TP + FP + FN)
    Treats label 1 as the positive class.
    """
    tp = fp = fn = 0
    for yt, yp in zip(y_true_list, y_pred_list):
        if yp == 1 and yt == 1:
            tp += 1
        elif yp == 1 and yt == 0:
            fp += 1
        elif yp == 0 and yt == 1:
            fn += 1
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2.0 * tp) / denom


def _compute_accuracy(y_true_list, y_pred_list):
    """Manually compute accuracy = correct / total."""
    if not y_true_list:
        return 0.0
    correct = 0
    for yt, yp in zip(y_true_list, y_pred_list):
        if yt == yp:
            correct += 1
    return correct / len(y_true_list)


# ---------------------------------------------------------------------------
# DDM Degradation Detector
# ---------------------------------------------------------------------------

class DDMDegradationDetector:
    """Drift Detection Method (DDM) for model degradation monitoring.

    DDM monitors a sequential binary error stream and raises warnings /
    alarms when the running error rate diverges significantly from the
    historically observed minimum.

    Parameters
    ----------
    warning_level : float
        Number of standard deviations above p_min for a warning (default 2.0).
    alarm_level : float
        Number of standard deviations above p_min for an alarm (default 3.0).
    min_samples : int
        Minimum observations before monitoring begins (default 30).
    """

    def __init__(self, warning_level=2.0, alarm_level=3.0, min_samples=30):
        self.warning_level = warning_level
        self.alarm_level = alarm_level
        self.min_samples = min_samples

        # Running state
        self._n = 0           # total samples seen
        self._error_sum = 0   # total errors
        self._p_min = float('inf')
        self._s_min = float('inf')

        # Reference (fit) statistics
        self._ref_error_rate = None
        self._ref_f1 = None
        self._ref_accuracy = None
        self._ref_n = 0

    # -- internal helpers ---------------------------------------------------

    def _reset(self):
        """Reset DDM state after an alarm (concept change detected)."""
        self._n = 0
        self._error_sum = 0
        self._p_min = float('inf')
        self._s_min = float('inf')

    def _current_stats(self):
        """Return (p_i, s_i) for the current step."""
        if self._n == 0:
            return 0.0, 0.0
        p = self._error_sum / self._n
        s = _manual_sqrt(p * (1.0 - p) / self._n)
        return p, s

    # -- public API ---------------------------------------------------------

    def fit(self, y_true, y_pred):
        """Calibrate DDM on a reference (validation) set.

        Feeds the reference predictions through the DDM algorithm to
        establish a baseline error rate and stores reference metrics.

        Parameters
        ----------
        y_true : list[int]
            True binary labels of the reference set.
        y_pred : list[int]
            Predicted binary labels of the reference set.
        """
        self._reset()
        for yt, yp in zip(y_true, y_pred):
            self.update(yt, yp)

        p, s = self._current_stats()
        self._ref_error_rate = p
        self._ref_n = len(y_true)
        self._ref_f1 = _compute_f1(y_true, y_pred)
        self._ref_accuracy = _compute_accuracy(y_true, y_pred)

        # Reset for the monitoring phase
        self._reset()

    def update(self, y_true_single, y_pred_single):
        """Process a single prediction and return the detection status.

        Parameters
        ----------
        y_true_single : int
            True label (0 or 1).
        y_pred_single : int
            Predicted label (0 or 1).

        Returns
        -------
        str
            One of ``'normal'``, ``'warning'``, or ``'alarm'``.
        """
        error = _compute_error(y_true_single, y_pred_single)
        self._n += 1
        self._error_sum += error

        p_i, s_i = self._current_stats()

        # Not enough samples yet — just track minimums
        if self._n < self.min_samples:
            if p_i + s_i < self._p_min + self._s_min:
                self._p_min = p_i
                self._s_min = s_i
            return 'normal'

        # Update minimum tracker
        if p_i + s_i < self._p_min + self._s_min:
            self._p_min = p_i
            self._s_min = s_i

        # Check alarm first (stricter condition)
        if p_i + s_i > self._p_min + self.alarm_level * self._s_min:
            self._reset()
            return 'alarm'

        # Check warning
        if p_i + s_i > self._p_min + self.warning_level * self._s_min:
            return 'warning'

        return 'normal'

    def detect_stream(self, y_true_list, y_pred_list):
        """Process an entire prediction stream and return a structured result.

        Feeds every (y_true, y_pred) pair through ``update`` and records
        transition points (status changes) in a detection timeline.

        Returns
        -------
        dict
            Structured result following the project output schema.
        """
        timeline = []
        total_warnings = 0
        total_alarms = 0
        first_alarm_step = None
        prev_status = 'normal'

        for i, (yt, yp) in enumerate(zip(y_true_list, y_pred_list)):
            status = self.update(yt, yp)

            if status == 'warning':
                total_warnings += 1
            elif status == 'alarm':
                total_alarms += 1
                if first_alarm_step is None:
                    first_alarm_step = i

            # Record only transition points
            if status != prev_status:
                p_i, s_i = self._current_stats()
                timeline.append({
                    "step": i,
                    "error": _compute_error(yt, yp),
                    "running_error_rate": round(p_i, 6),
                    "running_std": round(s_i, 6),
                    "p_min": round(self._p_min if self._p_min != float('inf') else 0.0, 6),
                    "s_min": round(self._s_min if self._s_min != float('inf') else 0.0, 6),
                    "status": status,
                })
                prev_status = status

        stream_len = len(y_true_list)
        ref_err = self._ref_error_rate if self._ref_error_rate is not None else 0.0
        ref_f1 = self._ref_f1 if self._ref_f1 is not None else 0.0

        return {
            "model": "DDM_River_Style",
            "detection_method": "drift_detection_method",
            "reference_sample_size": self._ref_n,
            "stream_length": stream_len,
            "reference_error_rate": round(ref_err, 6),
            "reference_f1": round(ref_f1, 6),
            "detection_timeline": timeline,
            "chunks": [],
            "total_warnings": total_warnings,
            "total_alarms": total_alarms,
            "first_alarm_step": first_alarm_step,
            "overall_degradation_detected": total_alarms > 0,
            "explanation": (
                "DDM monitors the running error rate p_i and its standard "
                "deviation s_i = sqrt(p_i*(1-p_i)/i).  When p_i + s_i exceeds "
                "the historical minimum by {} std devs a warning is raised; "
                "by {} std devs an alarm is raised and the detector resets.  "
                "{} warnings and {} alarms were observed over {} samples."
            ).format(
                self.warning_level, self.alarm_level,
                total_warnings, total_alarms, stream_len,
            ),
        }

    def detect_chunks(self, y_true_list, y_pred_list, chunk_size=500):
        """Divide a prediction stream into chunks and run DDM on each.

        For each chunk the detector is run independently and per-chunk
        metrics (F1, accuracy, warning/alarm counts) are reported.

        Parameters
        ----------
        y_true_list : list[int]
        y_pred_list : list[int]
        chunk_size : int

        Returns
        -------
        dict
            Structured result with per-chunk breakdown.
        """
        n = len(y_true_list)
        chunks = []
        total_warnings = 0
        total_alarms = 0
        first_alarm_step = None
        global_step = 0

        num_chunks = (n + chunk_size - 1) // chunk_size  # ceiling division

        for c in range(num_chunks):
            start = c * chunk_size
            end = start + chunk_size
            if end > n:
                end = n
            yt_chunk = y_true_list[start:end]
            yp_chunk = y_pred_list[start:end]

            # Run a fresh DDM on this chunk
            chunk_detector = DDMDegradationDetector(
                warning_level=self.warning_level,
                alarm_level=self.alarm_level,
                min_samples=self.min_samples,
            )
            w_count = 0
            a_count = 0
            for yt, yp in zip(yt_chunk, yp_chunk):
                status = chunk_detector.update(yt, yp)
                if status == 'warning':
                    w_count += 1
                elif status == 'alarm':
                    a_count += 1
                    if first_alarm_step is None:
                        first_alarm_step = global_step
                global_step += 1

            total_warnings += w_count
            total_alarms += a_count

            chunk_f1 = _compute_f1(yt_chunk, yp_chunk)
            chunk_acc = _compute_accuracy(yt_chunk, yp_chunk)

            if a_count > 0:
                cstatus = "alarm"
            elif w_count > 0:
                cstatus = "warning"
            else:
                cstatus = "normal"

            chunks.append({
                "chunk_idx": c,
                "actual_f1": round(chunk_f1, 6),
                "actual_accuracy": round(chunk_acc, 6),
                "warnings_in_chunk": w_count,
                "alarms_in_chunk": a_count,
                "chunk_status": cstatus,
            })

        ref_err = self._ref_error_rate if self._ref_error_rate is not None else 0.0
        ref_f1 = self._ref_f1 if self._ref_f1 is not None else 0.0

        return {
            "model": "DDM_River_Style",
            "detection_method": "drift_detection_method",
            "reference_sample_size": self._ref_n,
            "stream_length": n,
            "n_chunks": len(chunks),
            "reference_error_rate": round(ref_err, 6),
            "reference_f1": round(ref_f1, 6),
            "detection_timeline": [],
            "chunks": chunks,
            "total_warnings": total_warnings,
            "total_alarms": total_alarms,
            "first_alarm_step": first_alarm_step,
            "overall_degradation_detected": total_alarms > 0,
            "explanation": (
                "Chunk-based DDM analysis: {} chunks of size {} evaluated.  "
                "{} total warnings, {} total alarms across all chunks."
            ).format(len(chunks), chunk_size, total_warnings, total_alarms),
        }


# ---------------------------------------------------------------------------
# EDDM Degradation Detector
# ---------------------------------------------------------------------------

class EDDMDegradationDetector:
    """Early Drift Detection Method (EDDM) for model degradation monitoring.

    EDDM (Baena-Garcia et al., 2006) is a variant of DDM that monitors
    the *distance between consecutive errors* rather than the error rate
    itself.  This makes it more sensitive to slow, gradual drift.

    Let d_i be the number of observations between the (i-1)-th and i-th
    error.  EDDM tracks:

        d_mean  = running mean of distances
        d_var   = running variance (Welford's online algorithm)
        d_std   = sqrt(d_var)
        metric  = d_mean + 2 * d_std

    The maximum of this metric is recorded as ``metric_max``.

    Decision rules (after ``min_samples`` errors):

        ALARM   :  metric / metric_max  <  alarm_ratio   (default 0.90)
        WARNING :  metric / metric_max  <  warning_ratio  (default 0.95)

    On ALARM the detector resets.

    Parameters
    ----------
    warning_ratio : float
        Ratio threshold for warning (default 0.95).
    alarm_ratio : float
        Ratio threshold for alarm (default 0.90).
    min_samples : int
        Minimum number of errors before monitoring begins (default 30).
    """

    def __init__(self, warning_ratio=0.95, alarm_ratio=0.90, min_samples=30,
                 alpha=None, beta=None, min_errors=None):
        if alpha is not None:
            warning_ratio = alpha
        if beta is not None:
            alarm_ratio = beta
        if min_errors is not None:
            min_samples = min_errors

        self.warning_ratio = warning_ratio
        self.alarm_ratio = alarm_ratio
        self.min_samples = min_samples
        self.alpha = warning_ratio
        self.beta = alarm_ratio
        self.min_errors = min_samples

        self._reset_state()

        # Reference statistics
        self._ref_error_rate = None
        self._ref_f1 = None
        self._ref_accuracy = None
        self._ref_n = 0

    def _reset_state(self):
        """Reset EDDM internal tracking state."""
        self._n = 0               # total observations
        self._error_count = 0     # total errors seen
        self._last_error_pos = 0  # position of last error
        self._d_mean = 0.0        # running mean of distances
        self._d_m2 = 0.0          # Welford M2 accumulator
        self._metric_max = 0.0    # max of (d_mean + 2*d_std)

    def _current_metric(self):
        """Return (d_mean, d_std, metric) for the current state."""
        if self._error_count < 2:
            return 0.0, 0.0, 0.0
        d_var = self._d_m2 / (self._error_count - 1)
        d_std = _manual_sqrt(d_var)
        metric = self._d_mean + 2.0 * d_std
        return self._d_mean, d_std, metric

    def fit(self, y_true, y_pred):
        """Calibrate EDDM on a reference set.

        Parameters
        ----------
        y_true : list[int]
        y_pred : list[int]
        """
        self._reset_state()
        for yt, yp in zip(y_true, y_pred):
            self.update(yt, yp)

        self._ref_n = len(y_true)
        if self._n > 0:
            self._ref_error_rate = self._error_count / self._n
        else:
            self._ref_error_rate = 0.0
        self._ref_f1 = _compute_f1(y_true, y_pred)
        self._ref_accuracy = _compute_accuracy(y_true, y_pred)

        self._reset_state()

    def update(self, y_true_single, y_pred_single):
        """Process a single prediction.

        Returns
        -------
        str
            ``'normal'``, ``'warning'``, or ``'alarm'``.
        """
        self._n += 1
        error = _compute_error(y_true_single, y_pred_single)

        if error == 0:
            return 'normal'

        # Error occurred — compute distance from last error
        self._error_count += 1
        distance = self._n - self._last_error_pos
        self._last_error_pos = self._n

        if self._error_count >= 2:
            # Welford online update for mean and variance of distances
            old_mean = self._d_mean
            self._d_mean += (distance - self._d_mean) / (self._error_count - 1)
            self._d_m2 += (distance - old_mean) * (distance - self._d_mean)

        # Not enough error samples yet
        if self._error_count < self.min_samples:
            _, _, metric = self._current_metric()
            if metric > self._metric_max:
                self._metric_max = metric
            return 'normal'

        _, _, metric = self._current_metric()
        if metric > self._metric_max:
            self._metric_max = metric

        if self._metric_max == 0:
            return 'normal'

        ratio = metric / self._metric_max

        if ratio < self.alarm_ratio:
            self._reset_state()
            return 'alarm'

        if ratio < self.warning_ratio:
            return 'warning'

        return 'normal'

    def detect_stream(self, y_true_list, y_pred_list):
        """Process an entire prediction stream with EDDM.

        Returns
        -------
        dict
            Structured result following the project output schema.
        """
        timeline = []
        total_warnings = 0
        total_alarms = 0
        first_alarm_step = None
        prev_status = 'normal'

        for i, (yt, yp) in enumerate(zip(y_true_list, y_pred_list)):
            status = self.update(yt, yp)

            if status == 'warning':
                total_warnings += 1
            elif status == 'alarm':
                total_alarms += 1
                if first_alarm_step is None:
                    first_alarm_step = i

            if status != prev_status:
                d_mean, d_std, metric = self._current_metric()
                p_i = self._error_count / self._n if self._n > 0 else 0.0
                s_i = _manual_sqrt(p_i * (1 - p_i) / self._n) if self._n > 0 else 0.0
                timeline.append({
                    "step": i,
                    "error": _compute_error(yt, yp),
                    "running_error_rate": round(p_i, 6),
                    "running_std": round(s_i, 6),
                    "p_min": round(d_mean, 6),
                    "s_min": round(d_std, 6),
                    "status": status,
                })
                prev_status = status

        stream_len = len(y_true_list)
        ref_err = self._ref_error_rate if self._ref_error_rate is not None else 0.0
        ref_f1 = self._ref_f1 if self._ref_f1 is not None else 0.0

        return {
            "model": "EDDM_Early_DDM",
            "detection_method": "early_drift_detection_method",
            "reference_sample_size": self._ref_n,
            "stream_length": stream_len,
            "reference_error_rate": round(ref_err, 6),
            "reference_f1": round(ref_f1, 6),
            "detection_timeline": timeline,
            "chunks": [],
            "total_warnings": total_warnings,
            "total_alarms": total_alarms,
            "first_alarm_step": first_alarm_step,
            "overall_degradation_detected": total_alarms > 0,
            "explanation": (
                "EDDM monitors the distance between consecutive errors.  "
                "When the metric (d_mean + 2*d_std) drops below {} of its "
                "historical maximum a warning is raised; below {} an alarm.  "
                "{} warnings and {} alarms over {} samples."
            ).format(
                self.warning_ratio, self.alarm_ratio,
                total_warnings, total_alarms, stream_len,
            ),
        }

    def detect_chunks(self, y_true_list, y_pred_list, chunk_size=500):
        """Chunk-based EDDM analysis.

        Returns
        -------
        dict
            Structured result with per-chunk breakdown.
        """
        n = len(y_true_list)
        chunks = []
        total_warnings = 0
        total_alarms = 0
        first_alarm_step = None
        global_step = 0

        num_chunks = (n + chunk_size - 1) // chunk_size

        for c in range(num_chunks):
            start = c * chunk_size
            end = start + chunk_size
            if end > n:
                end = n
            yt_chunk = y_true_list[start:end]
            yp_chunk = y_pred_list[start:end]

            chunk_det = EDDMDegradationDetector(
                warning_ratio=self.warning_ratio,
                alarm_ratio=self.alarm_ratio,
                min_samples=self.min_samples,
            )
            w_count = 0
            a_count = 0
            for yt, yp in zip(yt_chunk, yp_chunk):
                status = chunk_det.update(yt, yp)
                if status == 'warning':
                    w_count += 1
                elif status == 'alarm':
                    a_count += 1
                    if first_alarm_step is None:
                        first_alarm_step = global_step
                global_step += 1

            total_warnings += w_count
            total_alarms += a_count

            chunk_f1 = _compute_f1(yt_chunk, yp_chunk)
            chunk_acc = _compute_accuracy(yt_chunk, yp_chunk)

            if a_count > 0:
                cstatus = "alarm"
            elif w_count > 0:
                cstatus = "warning"
            else:
                cstatus = "normal"

            chunks.append({
                "chunk_idx": c,
                "actual_f1": round(chunk_f1, 6),
                "actual_accuracy": round(chunk_acc, 6),
                "warnings_in_chunk": w_count,
                "alarms_in_chunk": a_count,
                "chunk_status": cstatus,
            })

        ref_err = self._ref_error_rate if self._ref_error_rate is not None else 0.0
        ref_f1 = self._ref_f1 if self._ref_f1 is not None else 0.0

        return {
            "model": "EDDM_Early_DDM",
            "detection_method": "early_drift_detection_method",
            "reference_sample_size": self._ref_n,
            "stream_length": n,
            "n_chunks": len(chunks),
            "reference_error_rate": round(ref_err, 6),
            "reference_f1": round(ref_f1, 6),
            "detection_timeline": [],
            "chunks": chunks,
            "total_warnings": total_warnings,
            "total_alarms": total_alarms,
            "first_alarm_step": first_alarm_step,
            "overall_degradation_detected": total_alarms > 0,
            "explanation": (
                "Chunk-based EDDM analysis: {} chunks of size {} evaluated.  "
                "{} total warnings, {} total alarms across all chunks."
            ).format(len(chunks), chunk_size, total_warnings, total_alarms),
        }
