"""Latest review: target precision, nullable cap, and request state identity."""

from copy import deepcopy
import numpy as np
import pytest
from backend.investment_ai import baseline_execution as b
from backend.investment_ai.infer import predict_weights
from backend.investment_ai.portfolio_env import PortfolioEnv
from tests.test_contract_v3 import data, config, service, state


def cfg(**kw):
    c = config()
    c.update(execution_policy=b.POLICY)
    c.update(kw)
    return c


@pytest.mark.parametrize("policy", [b.POLICY, "monthly_risk_distance_filtered_v2"])
def test_float32_target_copy_normalized(policy):
    raw = np.array([0.6, 0.4], dtype=np.float32)
    original = raw.copy()
    e = PortfolioEnv(data(), cfg(execution_policy=policy), start=1)
    e.reset()
    row = e.step_weights(raw, force=True)[-1]
    assert sum(row["raw_target_weights"].values()) == pytest.approx(1, abs=1e-14)
    np.testing.assert_array_equal(raw, original)
    assert sum(row["post_trade_weights"].values()) <= 1 + 1e-12


def test_direct_executable_float32_and_strict_actual_budget():
    raw = np.array([0.6, 0.4], dtype=np.float32)
    desired, _, _ = b.executable_target(raw, [0.5, 0.5], cfg())
    assert desired.sum() == pytest.approx(1, abs=1e-14)
    with pytest.raises(ValueError, match="actual"):
        b.executable_target([0.6, 0.4], [0.6, 0.4000001], cfg())


@pytest.mark.parametrize(
    "raw", [[0.7, 0.4], [-0.1, 1.1], [float("nan"), 1], [0, 0], [0.5, 0.5, 0.0]]
)
def test_invalid_raw_rejected_before_environment_changes(raw):
    e = PortfolioEnv(data(), cfg(), start=1)
    e.reset()
    before = deepcopy(
        (e.t, e.value, e.weights, e.last_executed_weights, e.history, e.mdd)
    )
    with pytest.raises(ValueError):
        e.step_weights(raw, force=True)
    after = (e.t, e.value, e.weights, e.last_executed_weights, e.history, e.mdd)
    for left, right in zip(before, after):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


@pytest.mark.parametrize("policy", [b.POLICY, "monthly_risk_distance_filtered_v2"])
def test_nullable_holdings_environment_and_inference(policy):
    c = cfg(execution_policy=policy, max_holdings=None)
    e = PortfolioEnv(data(), c, start=1)
    e.reset()
    e.step_weights([0.6, 0.4], force=True)
    d = data()
    d.risk[:] = 0.8

    class Predictor:
        def predict(self, *args):
            return np.array([0.6, 0.4], dtype=np.float32)

    result = predict_weights(d, c, [], None, None, [Predictor()])
    assert result["should_rebalance"]


def test_nullable_holdings_allows_more_than_seven():
    desired, target, _ = b.executable_target(
        np.ones(10) / 10, np.ones(10) / 10, cfg(max_holdings=None)
    )
    assert np.count_nonzero(target) == 10
    np.testing.assert_allclose(desired, target)


class Model:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, **kwargs):
        self.calls += 1
        return np.array([1.0, -1.0]), (np.zeros((1, 1, 2)), np.zeros((1, 1, 2)))


def make_service(tmp_path, recurrent=False):
    obj = service(tmp_path)
    obj.config = cfg(model_type="recurrent_ppo" if recurrent else "ppo")
    obj.models = [Model()]
    obj.recurrent_sessions = {}
    return obj


def test_first_recurrent_cash_request_retry_is_cached(tmp_path):
    obj = make_service(tmp_path, True)
    first = obj.optimize()
    second = obj.optimize()
    assert first == second
    assert obj.models[0].calls == 1
    assert first["last_executed_weights"] == {"SPY": 0.0, "QQQ": 0.0}


@pytest.mark.parametrize("recurrent", [False, True])
def test_new_cash_does_not_restore_prior_invested_reference(tmp_path, recurrent):
    obj = make_service(tmp_path, recurrent)
    obj.execution_references = {
        "default": {"SPY": 0.6, "QQQ": 0.4},
        "other": {"SPY": 1.0, "QQQ": 0.0},
    }
    result = obj.optimize()
    assert result["portfolio_weight_distance"] == 0
    assert not result["should_rebalance"]
    assert result["cash_weight"] == 1
    assert result["last_executed_weights"] == {"SPY": 0.0, "QQQ": 0.0}
    assert obj.execution_references["other"] == {"SPY": 1.0, "QQQ": 0.0}


def test_explicit_held_portfolio_restores_only_its_episode(tmp_path):
    obj = make_service(tmp_path)
    obj.execution_references = {
        "test": {"SPY": 0.5, "QQQ": 0.5},
        "other": {"SPY": 0.0, "QQQ": 1.0},
    }
    s = state(obj.data, mdd=0.05)
    s["weights"] = {"SPY": 0.5, "QQQ": 0.5}
    result = obj.optimize(state=s)
    assert result["last_executed_weights"] == obj.execution_references["test"]
    assert result["portfolio_weight_distance"] == 0


def test_recurrent_cash_retry_after_restart(tmp_path):
    import json

    obj = make_service(tmp_path, True)
    first = obj.optimize()
    saved = json.loads(obj.latch_path.read_text())
    restarted = make_service(tmp_path, True)
    restarted.execution_references = saved["execution_references"]
    restarted.episode_clocks = saved["clocks"]
    restarted.latches = saved["latches"]
    restarted._restore_recurrent(saved["recurrent"])
    assert restarted.optimize() == first
    assert restarted.models[0].calls == 0


def test_existing_actual_budget_rejected_without_state_change():
    e = PortfolioEnv(data(), cfg(), start=1)
    e.reset()
    e.weights = np.array([0.6, 0.4000001])
    before = e.weights.copy()
    time = e.t
    with pytest.raises(ValueError, match="actual"):
        e.step_weights([0.6, 0.4], force=True)
    np.testing.assert_array_equal(e.weights, before)
    assert e.t == time and not e.history


def test_excess_existing_holdings_rejected_before_time_advance():
    e = PortfolioEnv(data(), cfg(max_holdings=1), start=1)
    e.reset()
    e.weights = np.array([0.5, 0.5])
    e.last_executed_weights = e.weights.copy()
    with pytest.raises(ValueError, match="holdings"):
        e.step_weights([0.6, 0.4], force=True)
    assert e.t == 1 and not e.history and e.value == 1


def test_new_cash_does_not_overwrite_invested_reference(tmp_path):
    obj = make_service(tmp_path)
    obj.execution_references = {"default": {"SPY": 0.6, "QQQ": 0.4}}
    obj.optimize()
    assert obj.execution_references["default"] == {"SPY": 0.6, "QQQ": 0.4}
    s = state(obj.data, mdd=0.05)
    s["episode_id"] = "default"
    s["weights"] = {"SPY": 0.6, "QQQ": 0.4}
    assert obj.optimize(state=s)["portfolio_weight_distance"] == 0


def test_explicit_new_cash_recurrent_retry(tmp_path):
    obj = make_service(tmp_path, True)
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    assert obj.optimize(state=s) == obj.optimize(state=s)
    assert obj.models[0].calls == 1


def test_new_cash_request_does_not_reuse_invested_recurrent_cache(tmp_path):
    obj = make_service(tmp_path, True)
    obj.optimize()
    # Simulate a previously supplied invested default-episode request cache.
    record = obj.recurrent_sessions["default"]
    record["signature"] = "invested-request"
    record["prediction"]["weights"] = {"SPY": 1.0, "QQQ": 0.0}
    obj.execution_references["default"] = {"SPY": 1.0, "QQQ": 0.0}
    result = obj.optimize()
    assert not result["should_rebalance"] and result["cash_weight"] == 1
    assert obj.models[0].calls == 2
    assert obj.optimize() == result
    assert obj.models[0].calls == 2
