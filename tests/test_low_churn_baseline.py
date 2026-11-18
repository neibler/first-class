import numpy as np
import pytest
from backend.investment_ai import baseline_execution as b
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.backtest import performance
from backend.investment_ai.training import schema_for
from tests.test_contract_v3 import config, data


def cfg(**kw):
    result = dict(config(), execution_policy=b.POLICY, **kw)
    result["reward"] = "return_mdd"
    return result


def env(returns=None, **kw):
    e = PortfolioEnv(data(returns), cfg(**kw), start=1)
    e.reset()
    e.weights = np.array([0.5, 0.5])
    e.last_executed_weights = e.weights.copy()
    return e


def test_daily_target_is_not_trigger_and_hold_preserves_reference():
    e = env()
    r = e.step_weights([1.0, 0.0])[-1]
    assert r["rebalance_reason"] == "hold"
    assert r["executed_turnover"] == r["transaction_cost"] == 0
    np.testing.assert_array_equal(e.last_executed_weights, [0.5, 0.5])
    np.testing.assert_array_equal(e.weights, [0.5, 0.5])


@pytest.mark.parametrize("drift,trigger", [(0.099, False), (0.10, True), (0.101, True)])
def test_drift_boundary(drift, trigger):
    reasons = b.reasons(
        "2024-01-04",
        "2024-01-03",
        [0.5 + drift, 0.5 - drift],
        [0.5, 0.5],
        [0.1, 0.1],
        0,
        cfg(),
    )
    assert ("portfolio_weight_distance" in reasons) == trigger


def test_cap_and_reward_use_actual_and_update_reference():
    e = env()
    r = e.step_weights([0.8, 0.2], force=True)[-1]
    assert r["desired_turnover"] == pytest.approx(0.30)
    assert r["executed_turnover"] == pytest.approx(0.15)
    assert r["reward"] == pytest.approx(-0.003 * 0.15)
    np.testing.assert_allclose(e.last_executed_weights, [0.65, 0.35])
    r = e.step_weights([0.8, 0.2])[-1]
    assert r["executed_turnover"] == 0  # no carry-over


@pytest.mark.parametrize(
    "raw,held,expected",
    [
        (0.049, False, False),
        (0.05, False, True),
        (0.051, False, True),
        (0.04, True, True),
        (0.03, True, True),
        (0.029, True, False),
    ],
)
def test_hysteresis(raw, held, expected):
    desired, _, _ = b.executable_target(
        [raw, 1 - raw], [0.1, 0.9] if held else [0.0, 1.0], cfg()
    )
    assert bool(desired[0] > 0) == expected


def test_partial_exit_never_opens_eighth_position():
    current = np.r_[np.ones(7) / 7, 0.0]
    raw = np.array([0.01, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.27])
    for cap in [0.01, 0.05, 0.15]:
        desired, target, distance = b.executable_target(
            raw, current, cfg(max_rebalance_turnover=cap)
        )
        assert np.count_nonzero(target > b.EPS) <= 7
        assert 0.5 * np.abs(target - current).sum() <= cap + 1e-12
        assert distance > 0


def test_price_drift_without_hidden_trades():
    returns = np.zeros((5, 2))
    returns[2] = [0.1, 0]
    e = env(returns)
    r = e.step_weights([0.99, 0.01])[-1]
    np.testing.assert_allclose(e.weights, [0.55 / 1.05, 0.5 / 1.05])
    np.testing.assert_array_equal(e.last_executed_weights, [0.5, 0.5])
    assert r["transaction_cost"] == r["executed_turnover"] == 0


def test_emergency_ignores_cap_and_cash_padding_has_no_trades():
    returns = np.zeros((5, 2))
    returns[2] = [-0.2, -0.2]
    e = env(returns)
    r = e.step_weights([0.8, 0.2], force=True)[-1]
    assert r["rebalance_reason"] == "mdd_stop"
    assert r["cash_weight"] == 1
    assert r["executed_turnover"] == pytest.approx(0.5)
    assert r["normal_executed_turnover"] == 0
    np.testing.assert_array_equal(e.last_executed_weights, [0, 0])


def test_self_financing_fee_uses_actual_notional_and_cash():
    current = np.array([0.5, 0.5])
    target = np.array([0.65, 0.35])
    rate = 0.0007
    q, cost = b.execute(current, target, rate)
    trades = (1 - cost) * target - current
    assert q == pytest.approx(np.abs(trades).sum())
    assert trades.sum() + cost == pytest.approx(0)
    e = env(transaction_cost_bps=5.0, slippage_bps=2.0)
    r = e.step_weights([0.8, 0.2], force=True)[-1]
    assert r["executed_turnover"] == pytest.approx(0.15)
    assert r["transaction_cost"] == pytest.approx(cost)
    assert r["reward"] == pytest.approx(
        r["portfolio_return"] - 0.1 * r["max_drawdown"] - 0.003 * 0.15
    )


def test_initial_cash_capped_entry_and_no_eligible_assets():
    desired, target, _ = b.executable_target([0.5, 0.5], [0.0, 0.0], cfg())
    assert target.sum() == pytest.approx(0.30)
    desired, target, _ = b.executable_target(
        [0.5, 0.5], [0.0, 0.0], cfg(buy_threshold=0.6, sell_threshold=0.03)
    )
    assert desired.sum() == target.sum() == 0


def test_metrics_and_schema_policy():
    e = env()
    e.step_weights([0.8, 0.2], force=True)
    import pandas as pd

    m = performance(pd.DataFrame(e.history))
    assert m["rebalance_count"] == 1
    assert m["total_executed_turnover"] == pytest.approx(0.15)
    assert m["max_holdings_count"] == 2
    assert schema_for(data(), cfg())["execution_policy"] == b.POLICY
    assert schema_for(data(), config())["execution_policy"] != b.POLICY


@pytest.mark.parametrize(
    "kw",
    [
        {"buy_threshold": 0.02, "sell_threshold": 0.03},
        {"max_rebalance_turnover": 0},
        {"turnover_penalty_lambda": -1},
        {"execution_policy": "typo"},
    ],
)
def test_invalid_baseline_settings(kw):
    c = cfg()
    c.update(kw)
    with pytest.raises(ValueError):
        PortfolioEnv(data(), c)


def test_inference_reference_and_partial_cash_state():
    from backend.investment_ai.infer import predict_weights
    from tests.test_contract_v3 import state

    d = data()
    c = cfg()

    class Predictor:
        def predict(self, obs, environment):
            return np.array([0.1, 0.9])

    s = state(d, mdd=0.05)
    s["weights"] = {"SPY": 0.2, "QQQ": 0.1}
    s["last_executed_weights"] = s["weights"].copy()
    result = predict_weights(d, c, [], None, s, [Predictor()])
    assert result["status"] == "hold_until_rebalance"
    assert result["cash_weight"] == pytest.approx(0.7)
    assert result["weights"] == s["weights"]
    s.pop("last_executed_weights")
    with pytest.raises(ValueError, match="last_executed_weights"):
        predict_weights(d, c, [], None, s, [Predictor()])


def test_safeguard_inference_does_not_require_reference_or_model():
    from backend.investment_ai.infer import predict_weights
    from tests.test_contract_v3 import state

    d = data()
    result = predict_weights(d, cfg(), [], None, state(d), [])
    assert result["cash_weight"] == 1


def test_service_persists_acknowledged_reference_not_proposal(tmp_path):
    import json
    from tests.test_contract_v3 import service, state

    obj = service(tmp_path)
    obj.config = cfg()

    class Model:
        def predict(self, obs, deterministic=True):
            return np.array([1.0, -1.0]), None

    obj.models = [Model()]
    s = state(obj.data, mdd=0.05)
    s["weights"] = {"SPY": 0.5, "QQQ": 0.5}
    s["last_executed_weights"] = {"SPY": 0.6, "QQQ": 0.4}
    first = obj.optimize(state=s)
    assert first["should_rebalance"]
    saved = json.loads(obj.latch_path.read_text())
    assert saved["execution_references"]["test"] == s["last_executed_weights"]
    restored = service(tmp_path)
    restored.config = cfg()
    restored.models = [Model()]
    restored.execution_references = saved["execution_references"]
    s.pop("last_executed_weights")
    second = restored.optimize(state=s)
    assert first["target_weights"] == second["target_weights"]
    assert first["last_executed_weights"] == second["last_executed_weights"]


def test_cap_recomputed_after_overnight_drift():
    returns = np.zeros((5, 2))
    returns[2] = [0.1, 0]
    e = env(returns)
    row = e.step_weights([0.9, 0.1], force=True)[-1]
    pre = np.array(list(row["pre_trade_weights"].values()))
    post = np.array(list(row["post_trade_weights"].values()))
    assert 0.5 * np.abs(post - pre).sum() == pytest.approx(0.15)


def test_randomized_execution_constraints():
    rng = np.random.default_rng(22)
    for _ in range(100):
        current = np.zeros(15)
        chosen = rng.choice(15, size=int(rng.integers(0, 8)), replace=False)
        if len(chosen):
            current[chosen] = rng.dirichlet(np.ones(len(chosen))) * rng.uniform(0.1, 1)
        raw = rng.dirichlet(np.ones(15))
        _, target, _ = b.executable_target(raw, current, cfg())
        assert target.min() >= 0 and target.sum() <= 1 + 1e-12
        assert np.count_nonzero(target > b.EPS) <= 7
        assert 0.5 * np.abs(target - current).sum() <= 0.15 + 1e-12
        q, cost = b.execute(current, target, 0.0007)
        assert q == pytest.approx(np.abs((1 - cost) * target - current).sum())
        assert (1 - cost) * (1 - target.sum()) >= -1e-12


def test_emergency_cash_padding_and_metrics_are_not_repeated():
    from backend.investment_ai.backtest import run_backtest

    d = data([[0, 0], [0, 0], [-0.8, -0.8], [0, 0], [0, 0]])
    d.risk[:] = 0.8

    class Model:
        def predict(self, obs, deterministic=True):
            return np.zeros(2), None

    history = run_backtest(d, cfg(), 0, 4, models=[Model()])
    assert history.iloc[-1].executed_turnover == 0
    assert history.iloc[-1].transaction_cost == 0
    assert history.iloc[-1].entry_count == history.iloc[-1].exit_count == 0
    m = performance(history)
    assert m["rebalance_reason_counts"]["mdd_stop"] == 1
    assert history.iloc[1].cash_weight == 1
    assert m["max_turnover_per_rebalance"] == pytest.approx(0.15)


def test_monthly_and_risk_still_trigger_without_drift():
    c = cfg()
    assert b.reasons(
        "2024-06-03", "2024-05-31", [0.5, 0.5], [0.5, 0.5], [0.1, 0.7], 0, c
    ) == ["monthly", "risk_score"]
    assert b.reasons(
        "2024-06-04", "2024-06-03", [0.5, 0.5], [0.5, 0.5], [0.69, 0.1], 0, c
    ) == ["hold"]


def test_observation_unchanged_and_schema_rejects_relabelled_old_bundle(tmp_path):
    import json
    from backend.investment_ai.training import load_bundle

    c = cfg()
    old = schema_for(data(), config())
    new = schema_for(data(), c)
    assert old["feature_names"] == new["feature_names"]
    assert old["observation_size"] == new["observation_size"]
    (tmp_path / "config.json").write_text(json.dumps(c))
    (tmp_path / "schema.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="execution policy"):
        load_bundle(tmp_path)
