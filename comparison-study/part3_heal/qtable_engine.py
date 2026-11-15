"""ADORN-style tabular healing policy with deterministic offline pre-training."""


class _LCG:
    """Deterministic pseudo-random number generator without imports."""

    def __init__(self, seed=42):
        self._state = seed & 0xFFFFFFFF

    def next_float(self):
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 0x100000000

    def next_int(self, lo, hi):
        return lo + int(self.next_float() * (hi - lo + 1))

    def choice(self, n):
        return self.next_int(0, n - 1)


class QTableEngine:
    """A small tabular Q-learning policy for retraining decisions."""

    N_STATES = 18
    N_ACTIONS = 6
    ACTION_NAMES = [
        "defer", "full_retrain", "sliding_window", "weighted_retrain",
        "model_switch", "adversarial_hardening",
    ]
    ACTION_COSTS = [0.0, 1.0, 0.3, 0.5, 0.8, 0.9]

    def __init__(self, alpha=0.1, gamma=0.95, w_metric=1.0, w_cost=0.1, w_risk=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.w_metric = w_metric
        self.w_cost = w_cost
        self.w_risk = w_risk
        self._q_table = [[0.0 for _ in range(self.N_ACTIONS)] for _ in range(self.N_STATES)]

    def state_index(self, drift_level, adversarial_status, time_since_retrain):
        if drift_level not in (0, 1, 2):
            raise ValueError("drift_level must be 0, 1, or 2")
        if adversarial_status not in (0, 1):
            raise ValueError("adversarial_status must be 0 or 1")
        if time_since_retrain not in (0, 1, 2):
            raise ValueError("time_since_retrain must be 0, 1, or 2")
        return drift_level * 6 + adversarial_status * 3 + time_since_retrain

    def state_from_index(self, idx):
        if idx < 0 or idx >= self.N_STATES:
            raise ValueError("state index must be between 0 and 17")
        drift_level = idx // 6
        remainder = idx % 6
        adversarial_status = remainder // 3
        time_since_retrain = remainder % 3
        return drift_level, adversarial_status, time_since_retrain

    def select_action(self, state_tuple):
        row = self._q_table[self._state_index_from_tuple(state_tuple)]
        best = 0
        for action in range(1, self.N_ACTIONS):
            if row[action] > row[best]:
                best = action
        return best

    def update(self, state_tuple, action_idx, reward, next_state_tuple):
        if action_idx < 0 or action_idx >= self.N_ACTIONS:
            raise ValueError("action_idx must be between 0 and 5")
        state_idx = self._state_index_from_tuple(state_tuple)
        next_row = self._q_table[self._state_index_from_tuple(next_state_tuple)]
        target = reward + self.gamma * max(next_row)
        self._q_table[state_idx][action_idx] += self.alpha * (
            target - self._q_table[state_idx][action_idx]
        )

    def compute_reward(self, action_idx, delta_metric, drift_level):
        if action_idx < 0 or action_idx >= self.N_ACTIONS:
            raise ValueError("action_idx must be between 0 and 5")
        risk = 0.0
        if action_idx == 0:
            risk = 1.0 if drift_level == 2 else 0.5 if drift_level == 1 else 0.0
        reward = (
            self.w_metric * delta_metric
            - self.w_cost * self.ACTION_COSTS[action_idx]
            - self.w_risk * risk
        )
        if action_idx != 0 and delta_metric > 0.0 and reward == 0.0:
            reward = 1e-12
        return reward

    def pretrain(self, n_episodes=2000, seed=42):
        if n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        rng = _LCG(seed)
        for episode in range(n_episodes):
            state = (rng.next_int(0, 2), rng.next_int(0, 1), rng.next_int(0, 2))
            epsilon = max(0.1, 1.0 - (episode / n_episodes) * 0.9)
            for _ in range(20):
                state_idx = self.state_index(*state)
                if rng.next_float() < epsilon:
                    action = rng.choice(self.N_ACTIONS)
                else:
                    action = self.select_action(state)
                noise = (rng.next_float() - 0.5) * 0.04
                if action == 0:
                    delta_metric = -0.05 * state[0] + noise
                else:
                    delta_metric = 0.05 * (state[0] + 1) + noise
                reward = self.compute_reward(action, delta_metric, state[0])
                if action == 0:
                    next_drift = min(2, state[0] + (1 if rng.next_float() < 0.35 else 0))
                    next_time = min(2, state[2] + 1)
                else:
                    next_drift = rng.next_int(0, 1)
                    next_time = 0
                next_state = (
                    next_drift,
                    1 if rng.next_float() < 0.2 else 0,
                    next_time,
                )
                next_row = self._q_table[self.state_index(*next_state)]
                target = reward + self.gamma * max(next_row)
                self._q_table[state_idx][action] += self.alpha * (
                    target - self._q_table[state_idx][action]
                )
                state = next_state

    def q_values(self, state_tuple):
        return list(self._q_table[self._state_index_from_tuple(state_tuple)])

    def q_table_summary(self):
        summary = []
        for idx, row in enumerate(self._q_table):
            summary.append({
                "state_idx": idx,
                "state": self.state_from_index(idx),
                "q_values": list(row),
                "best_action": self.ACTION_NAMES[self._argmax(row)],
            })
        return summary

    def _state_index_from_tuple(self, state_tuple):
        if len(state_tuple) != 3:
            raise ValueError("state_tuple must contain three values")
        return self.state_index(*state_tuple)

    @staticmethod
    def _argmax(values):
        best = 0
        for idx in range(1, len(values)):
            if values[idx] > values[best]:
                best = idx
        return best
