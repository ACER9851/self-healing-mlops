# =============================================================================
# data_generator.py
# Synthetic Data Generator for Model Degradation Detection (Part 2)
# Generates realistic model prediction streams, reference sets, and
# degradation scenarios (sudden, gradual, calibration shift, etc.).
# =============================================================================

class _LCG:
    """Deterministic Linear Congruential Generator for reproducible streams."""
    def __init__(self, seed=42):
        self._a = 1664525
        self._c = 1013904223
        self._m = 2**32
        self.state = int(seed) % self._m

    def next_float(self):
        self.state = (self._a * self.state + self._c) % self._m
        return self.state / self._m

    def random(self):
        return self.next_float()

    def randint(self, a, b):
        return a + int(self.next_float() * (b - a + 1))

    def uniform(self, a, b):
        return a + self.next_float() * (b - a)


_SimpleRNG = _LCG


class DegradationDataGenerator:
    """
    Generates reference and streaming prediction data for benchmark testing
    of degradation detectors (CBPE, DDM, EDDM).
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = _SimpleRNG(seed)

    def _sample_binary_pair(self, accuracy=0.90, pos_rate=0.50, miscalibrated=False):
        """
        Generate a single (y_true, y_pred, y_pred_proba) tuple.
        """
        y_true = 1 if self.rng.random() < pos_rate else 0
        is_correct = (self.rng.random() < accuracy)
        y_pred = y_true if is_correct else (1 - y_true)

        if not miscalibrated:
            # Well-calibrated probabilities
            if y_pred == 1:
                # confident high probability
                p = self.rng.uniform(0.70, 0.95)
            else:
                # confident low probability
                p = self.rng.uniform(0.05, 0.30)
        else:
            # Miscalibrated / confused probabilities (under-confident or shifted)
            # Hard prediction is correct at rate `accuracy`, but proba is near 0.50
            if y_pred == 1:
                p = self.rng.uniform(0.51, 0.55)
            else:
                p = self.rng.uniform(0.45, 0.49)

        return y_true, y_pred, p

    def generate_no_degradation(self, n_ref=1000, n_stream=2000, accuracy=0.90, pos_rate=0.50):
        """
        Scenario 1: Model performance is stable across both reference and stream.
        """
        ref_yt, ref_yp, ref_pp = [], [], []
        for _ in range(n_ref):
            yt, yp, pp = self._sample_binary_pair(accuracy=accuracy, pos_rate=pos_rate)
            ref_yt.append(yt)
            ref_yp.append(yp)
            ref_pp.append(pp)

        str_yt, str_yp, str_pp = [], [], []
        for _ in range(n_stream):
            yt, yp, pp = self._sample_binary_pair(accuracy=accuracy, pos_rate=pos_rate)
            str_yt.append(yt)
            str_yp.append(yp)
            str_pp.append(pp)

        return {
            "scenario_name": "no_degradation",
            "reference": {
                "y_true": ref_yt,
                "y_pred": ref_yp,
                "y_pred_proba": ref_pp,
            },
            "stream": {
                "y_true": str_yt,
                "y_pred": str_yp,
                "y_pred_proba": str_pp,
            }
        }

    def generate_sudden_degradation(self, n_ref=1000, n_stream=3000, degrade_at=1000,
                                     accuracy_before=0.90, accuracy_after=0.55, pos_rate=0.50):
        """
        Scenario 2: Model suffers abrupt degradation at index `degrade_at`.
        """
        ref_yt, ref_yp, ref_pp = [], [], []
        for _ in range(n_ref):
            yt, yp, pp = self._sample_binary_pair(accuracy=accuracy_before, pos_rate=pos_rate)
            ref_yt.append(yt)
            ref_yp.append(yp)
            ref_pp.append(pp)

        str_yt, str_yp, str_pp = [], [], []
        for i in range(n_stream):
            acc = accuracy_before if i < degrade_at else accuracy_after
            yt, yp, pp = self._sample_binary_pair(accuracy=acc, pos_rate=pos_rate)
            str_yt.append(yt)
            str_yp.append(yp)
            str_pp.append(pp)

        return {
            "scenario_name": "sudden_degradation",
            "degrade_at": degrade_at,
            "reference": {
                "y_true": ref_yt,
                "y_pred": ref_yp,
                "y_pred_proba": ref_pp,
            },
            "stream": {
                "y_true": str_yt,
                "y_pred": str_yp,
                "y_pred_proba": str_pp,
            }
        }

    def generate_gradual_degradation(self, n_ref=1000, n_stream=3000, start_degrade=1000,
                                      end_degrade=2500, acc_start=0.90, acc_end=0.55):
        """
        Scenario 3: Linear decline in accuracy over time.
        """
        ref_yt, ref_yp, ref_pp = [], [], []
        for _ in range(n_ref):
            yt, yp, pp = self._sample_binary_pair(accuracy=acc_start)
            ref_yt.append(yt)
            ref_yp.append(yp)
            ref_pp.append(pp)

        str_yt, str_yp, str_pp = [], [], []
        for i in range(n_stream):
            if i < start_degrade:
                acc = acc_start
            elif i >= end_degrade:
                acc = acc_end
            else:
                frac = (i - start_degrade) / (end_degrade - start_degrade)
                acc = acc_start - frac * (acc_start - acc_end)

            yt, yp, pp = self._sample_binary_pair(accuracy=acc)
            str_yt.append(yt)
            str_yp.append(yp)
            str_pp.append(pp)

        return {
            "scenario_name": "gradual_degradation",
            "reference": {"y_true": ref_yt, "y_pred": ref_yp, "y_pred_proba": ref_pp},
            "stream": {"y_true": str_yt, "y_pred": str_yp, "y_pred_proba": str_pp}
        }

    def generate_calibration_shift(self, n_ref=2000, n_stream=3000, accuracy=0.90):
        """
        Scenario 4: Hard predictions maintain high accuracy, but probability
        calibration collapses. (Tests CBPE sensitivity vs DDM insensitivity).
        """
        ref_yt, ref_yp, ref_pp = [], [], []
        for i in range(n_ref):
            yt = 1 if (i % 2 == 0) else 0
            # 90% accuracy: every 10th sample is an error
            yp = (1 - yt) if (i % 10 == 0) else yt
            pp = 0.85 if yp == 1 else 0.15
            ref_yt.append(yt)
            ref_yp.append(yp)
            ref_pp.append(pp)

        str_yt, str_yp, str_pp = [], [], []
        for i in range(n_stream):
            yt = 1 if (i % 2 == 0) else 0
            # Same 90% accuracy: every 10th sample is an error
            yp = (1 - yt) if (i % 10 == 0) else yt
            # Miscalibrated probabilities (compressed to 0.52 / 0.48)
            pp = 0.52 if yp == 1 else 0.48
            str_yt.append(yt)
            str_yp.append(yp)
            str_pp.append(pp)

        return {
            "scenario_name": "calibration_shift",
            "reference": {"y_true": ref_yt, "y_pred": ref_yp, "y_pred_proba": ref_pp},
            "stream": {"y_true": str_yt, "y_pred": str_yp, "y_pred_proba": str_pp}
        }

    def generate_recurring_degradation(self, n_ref=1000, n_stream=4000, cycle=1000):
        """Scenario 5: Periodic degradation and recovery."""
        ref_yt, ref_yp, ref_pp = [], [], []
        for _ in range(n_ref):
            yt, yp, pp = self._sample_binary_pair(accuracy=0.90)
            ref_yt.append(yt)
            ref_yp.append(yp)
            ref_pp.append(pp)

        str_yt, str_yp, str_pp = [], [], []
        for i in range(n_stream):
            phase = (i % cycle) / cycle
            acc = 0.90 if phase < 0.50 else 0.60
            yt, yp, pp = self._sample_binary_pair(accuracy=acc)
            str_yt.append(yt)
            str_yp.append(yp)
            str_pp.append(pp)

        return {
            "scenario_name": "recurring_degradation",
            "reference": {"y_true": ref_yt, "y_pred": ref_yp, "y_pred_proba": ref_pp},
            "stream": {"y_true": str_yt, "y_pred": str_yp, "y_pred_proba": str_pp}
        }

    def generate_extreme_imbalance(self, n_ref=1000, n_stream=3000, pos_rate=0.01):
        """Scenario 6: Rare event / severe class imbalance."""
        return self.generate_no_degradation(n_ref=n_ref, n_stream=n_stream, pos_rate=pos_rate)

    def generate_delayed_labels(self, n_ref=1000, n_stream=3000, delay=500):
        """
        Scenario 7: Labels arrive with a temporal delay of `delay` samples.
        """
        base = self.generate_sudden_degradation(n_ref=n_ref, n_stream=n_stream, degrade_at=1000)
        y_true_stream = base["stream"]["y_true"]
        delayed_labels = [None] * delay + y_true_stream[:-delay]

        base["scenario_name"] = "delayed_labels"
        base["delay"] = delay
        base["stream"]["y_true_delayed"] = delayed_labels
        return base

    def generate_edge_cases(self):
        """
        Scenario 8: Boundary edge cases (all correct, all wrong, all 0, single sample).
        """
        return {
            "scenario_name": "edge_cases",
            "cases": {
                "all_correct": {
                    "y_true": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    "y_pred": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    "y_pred_proba": [0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95]
                },
                "all_wrong": {
                    "y_true": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    "y_pred": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    "y_pred_proba": [0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05]
                },
                "all_predict_zero": {
                    "y_true": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    "y_pred": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "y_pred_proba": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                },
                "single_sample": {
                    "y_true": [1],
                    "y_pred": [1],
                    "y_pred_proba": [0.85]
                }
            }
        }

    def run_all_scenarios(self):
        """
        Generate and return all 8 benchmark scenarios in fixed order.
        """
        return [
            self.generate_no_degradation(n_ref=2000, n_stream=5000),
            self.generate_sudden_degradation(n_ref=2000, n_stream=5000, degrade_at=2000),
            self.generate_gradual_degradation(n_ref=2000, n_stream=5000),
            self.generate_calibration_shift(n_ref=2000, n_stream=5000),
            self.generate_recurring_degradation(n_ref=2000, n_stream=5000),
            self.generate_extreme_imbalance(n_ref=2000, n_stream=5000),
            self.generate_delayed_labels(n_ref=2000, n_stream=5000, delay=500),
            self.generate_edge_cases(),
        ]
