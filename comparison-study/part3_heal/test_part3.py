"""Direct-execution tests for Part 3 healing decision engines."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qtable_engine import QTableEngine
from regol_engine import RegolCostAwareEngine
from baselines import AlwaysRetrain, FixedSchedule, NeverRetrain, RetrainOnAlarm


def test_state_indexing_all_18_states():
    engine = QTableEngine()
    indices = [engine.state_index(d, a, t) for d in range(3) for a in range(2) for t in range(3)]
    assert sorted(indices) == list(range(18))


def test_state_roundtrip():
    engine = QTableEngine()
    for d in range(3):
        for a in range(2):
            for t in range(3):
                assert engine.state_from_index(engine.state_index(d, a, t)) == (d, a, t)


def test_q_table_shape():
    engine = QTableEngine()
    assert len(engine._q_table) == 18
    assert all(len(row) == 6 for row in engine._q_table)
    assert all(value == 0.0 for row in engine._q_table for value in row)


def test_q_update_increases_value():
    engine = QTableEngine()
    engine.update((1, 0, 0), 1, 1.0, (0, 0, 0))
    assert engine._q_table[6][1] > 0.0


def test_bellman_target():
    engine = QTableEngine(alpha=0.1, gamma=0.95)
    engine._q_table[0][2] = 0.4
    engine._q_table[1][0] = 0.8
    engine._q_table[1][1] = 0.3
    engine.update((0, 0, 0), 2, 0.5, (0, 0, 1))
    expected = 0.4 + 0.1 * (0.5 + 0.95 * 0.8 - 0.4)
    assert abs(engine._q_table[0][2] - expected) < 1e-12


def test_pretrain_changes_qtable():
    engine = QTableEngine()
    engine.pretrain(n_episodes=100)
    assert any(value != 0.0 for row in engine._q_table for value in row)


def test_pretrain_prefers_retrain_in_severe_drift():
    engine = QTableEngine()
    engine.pretrain(n_episodes=100)
    assert engine.select_action((2, 0, 2)) != 0


def test_select_action_returns_argmax():
    engine = QTableEngine()
    engine._q_table[0] = [0.1, 0.2, 0.9, 0.3, 0.4, 0.5]
    assert engine.select_action((0, 0, 0)) == 2


def test_reward_defer_severe_drift_is_negative():
    engine = QTableEngine()
    assert engine.compute_reward(0, 0.0, 2) < 0.0


def test_reward_retrain_gives_credit():
    engine = QTableEngine()
    assert engine.compute_reward(1, 0.1, 0) > 0.0


def test_regol_initial_decision_uses_default_delta():
    engine = RegolCostAwareEngine()
    assert engine.predict_delta(0.5, 0.8, 2) == 0.03


def test_regol_learns_positive_correlation():
    engine = RegolCostAwareEngine()
    for index in range(20):
        drift = index % 3
        engine.observe(drift, 0.8 - 0.02 * drift, index % 5, 0.01 + 0.05 * drift)
    assert engine.predict_delta(2.0, 0.7, 5) > engine.predict_delta(0.0, 0.8, 5)


def test_regol_budget_low_retrains_easily():
    low = RegolCostAwareEngine(cost_budget="low")
    high = RegolCostAwareEngine(cost_budget="high")
    assert low.should_retrain(0.2, 0.8, 0)
    assert not high.should_retrain(0.2, 0.8, 0)


def test_never_retrain():
    assert NeverRetrain().decide(True, True) == "defer"


def test_always_retrain():
    assert AlwaysRetrain().decide() == "sliding_window"


def test_fixed_schedule_fires_on_multiples():
    schedule = FixedSchedule(k=3)
    assert schedule.decide(1) == "defer"
    assert schedule.decide(2) == "defer"
    assert schedule.decide(3) == "sliding_window"
    assert schedule.decide(4) == "defer"
    assert schedule.decide(6) == "sliding_window"
    assert schedule.decide(9) == "sliding_window"


def test_retrain_on_alarm_fires_when_drift():
    assert RetrainOnAlarm().decide(True, False) == "sliding_window"


def test_retrain_on_alarm_does_not_fire_clean():
    assert RetrainOnAlarm().decide(False, False) == "defer"


if __name__ == "__main__":
    tests = [
        test_state_indexing_all_18_states,
        test_state_roundtrip,
        test_q_table_shape,
        test_q_update_increases_value,
        test_bellman_target,
        test_pretrain_changes_qtable,
        test_pretrain_prefers_retrain_in_severe_drift,
        test_select_action_returns_argmax,
        test_reward_defer_severe_drift_is_negative,
        test_reward_retrain_gives_credit,
        test_regol_initial_decision_uses_default_delta,
        test_regol_learns_positive_correlation,
        test_regol_budget_low_retrains_easily,
        test_never_retrain,
        test_always_retrain,
        test_fixed_schedule_fires_on_multiples,
        test_retrain_on_alarm_fires_when_drift,
        test_retrain_on_alarm_does_not_fire_clean,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as error:
            print(f"  FAIL  {test.__name__}: {error}")
            failed += 1
    print(f"\n{passed}/{passed + failed} Part 3 tests passed")
    if failed:
        raise SystemExit(1)
