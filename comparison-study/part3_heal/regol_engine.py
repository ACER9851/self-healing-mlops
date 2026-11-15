"""Regol-style cost-aware healing decision engine, implemented without imports."""


class RegolCostAwareEngine:
    """Forecast expected metric improvement with a small online OLS model."""

    BUDGET_LEVELS = {"low": 0.01, "medium": 0.03, "high": 0.08}

    def __init__(self, cost_budget="medium", min_fit_samples=10):
        if cost_budget not in self.BUDGET_LEVELS:
            raise ValueError("cost_budget must be low, medium, or high")
        if min_fit_samples < 1:
            raise ValueError("min_fit_samples must be positive")
        self.cost_budget_name = cost_budget
        self.cost_budget = self.BUDGET_LEVELS[cost_budget]
        self.min_fit_samples = min_fit_samples
        self.default_delta = 0.03
        self._X_hist = []
        self._y_hist = []
        self._beta = None

    def observe(self, drift_score, current_metric, time_since_retrain, observed_delta_metric):
        self._X_hist.append(self._features(drift_score, current_metric, time_since_retrain))
        self._y_hist.append(float(observed_delta_metric))
        if len(self._y_hist) >= self.min_fit_samples:
            self._beta = self._fit_ols(self._X_hist, self._y_hist)

    def predict_delta(self, drift_score, current_metric, time_since_retrain):
        features = self._features(drift_score, current_metric, time_since_retrain)
        if self._beta is None:
            return self.default_delta
        return self._dot(self._beta, features)

    def should_retrain(self, drift_score, current_metric, time_since_retrain):
        return self.predict_delta(drift_score, current_metric, time_since_retrain) > self.cost_budget

    def decision_report(self, drift_score, current_metric, time_since_retrain):
        predicted = self.predict_delta(drift_score, current_metric, time_since_retrain)
        return {
            "predicted_delta": predicted,
            "cost_budget": self.cost_budget,
            "should_retrain": predicted > self.cost_budget,
            "n_observations": len(self._y_hist),
        }

    @staticmethod
    def _features(drift_score, current_metric, time_since_retrain):
        tsr_norm = min(max(float(time_since_retrain), 0.0), 10.0) / 10.0
        return [1.0, float(drift_score), float(current_metric), tsr_norm]

    def _fit_ols(self, rows, targets):
        size = 4
        matrix = [[0.0 for _ in range(size + 1)] for _ in range(size)]
        for row, target in zip(rows, targets):
            for i in range(size):
                for j in range(size):
                    matrix[i][j] += row[i] * row[j]
                matrix[i][size] += row[i] * target
        for pivot in range(size):
            best = pivot
            for row in range(pivot + 1, size):
                if abs(matrix[row][pivot]) > abs(matrix[best][pivot]):
                    best = row
            if abs(matrix[best][pivot]) < 1e-12:
                matrix[pivot][pivot] += 1e-6
            else:
                matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
            divisor = matrix[pivot][pivot]
            for col in range(pivot, size + 1):
                matrix[pivot][col] /= divisor
            for row in range(size):
                if row == pivot:
                    continue
                factor = matrix[row][pivot]
                for col in range(pivot, size + 1):
                    matrix[row][col] -= factor * matrix[pivot][col]
        return [matrix[i][size] for i in range(size)]

    @staticmethod
    def _dot(left, right):
        total = 0.0
        for a, b in zip(left, right):
            total += a * b
        return total
