"""Simple heuristic healing policies with no external imports."""


class NeverRetrain:
    @property
    def name(self):
        return "never_retrain"

    def decide(self, *args, **kwargs):
        return "defer"


class AlwaysRetrain:
    @property
    def name(self):
        return "always_retrain"

    def decide(self, *args, **kwargs):
        return "sliding_window"


class FixedSchedule:
    def __init__(self, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    @property
    def name(self):
        return "fixed_schedule"

    def decide(self, window_id):
        return "sliding_window" if window_id > 0 and window_id % self.k == 0 else "defer"

    def reset(self):
        return None


class RetrainOnAlarm:
    def __init__(self, retrain_action="sliding_window"):
        valid = {"full_retrain", "sliding_window", "weighted_retrain", "model_switch", "adversarial_hardening"}
        if retrain_action not in valid:
            raise ValueError("invalid retrain action")
        self.retrain_action = retrain_action

    @property
    def name(self):
        return "retrain_on_alarm"

    def decide(self, drift_alarm, degradation_alarm):
        return self.retrain_action if drift_alarm or degradation_alarm else "defer"
