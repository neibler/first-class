import numpy as np
import pandas as pd
import pytest
from backend.investment_ai import baseline_execution as b
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.backtest import performance
from tests.test_low_churn_baseline import cfg
from tests.test_contract_v3 import data


def cash_config(**kw):
    c = cfg(**kw)
    c["execution_policy"] = "executed_drift_capped_hysteresis_cash_v2"
    return c


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ([0, 0], [0.3, 0], 0.3),
        ([0, 0], [0.15, 0], 0.15),
        ([0.2, 0.1], [0.1, 0.2], 0.1),
        ([0.2, 0.1], [0.2, 0.1], 0),
    ],
)
def test_cash_inclusive_turnover(before, after, expected):
    assert b.cash_turnover(before, after) == pytest.approx(expected)


def test_cash_entry_cap_and_fee_notional():
    c = cash_config()
    desired, target, wanted = b.executable_target([0.6, 0.4], [0, 0], c)
    assert wanted == pytest.approx(1)
    assert target.sum() == pytest.approx(0.15)
    assert b.cash_turnover([0, 0], target) <= 0.15 + b.EPS
    q, fee = b.execute([0, 0], target, 0.0007)
    assert q == pytest.approx(np.abs((1 - fee) * target).sum())
    assert fee == pytest.approx(q * 0.0007)


def test_desired_thirty_percent_capped_at_fifteen():
    _, target, wanted = b.executable_target([0.5, 0.5], [0.35, 0.35], cash_config())
    assert wanted == pytest.approx(0.3)
    assert b.cash_turnover([0.35, 0.35], target) == pytest.approx(0.15)


def test_actual_reward_and_evaluation():
    e = PortfolioEnv(data(), cash_config(), start=1)
    e.reset()
    result = e.step_weights([0.6, 0.4], force=True)[-1]
    assert result["desired_turnover"] == pytest.approx(1)
    assert result["executed_turnover"] == pytest.approx(0.15)
    assert result["reward"] == pytest.approx(-0.003 * 0.15)
    m = performance(pd.DataFrame(e.history))
    assert m["total_executed_turnover"] == pytest.approx(0.15)
    assert m["total_turnover"] == pytest.approx(0.15)


def test_emergency_sale_bypasses_cap():
    e = PortfolioEnv(data(), cash_config(), start=1)
    e.reset()
    e.weights = np.array([0.6, 0.4])
    e.last_executed_weights = e.weights.copy()
    e.mdd = 0.2
    e.safeguard_triggered = True
    r = e.step_weights([0.6, 0.4])[-1]
    assert r["executed_turnover"] == pytest.approx(1)
    assert r["cash_weight"] == 1
    assert "mdd_stop" in r["rebalance_reasons"]


def test_old_policy_and_distance_trigger_unchanged():
    from backend.investment_ai.rebalancing import portfolio_weight_distance

    _, target, _ = b.executable_target([0.6, 0.4], [0, 0], cfg())
    assert target.sum() == pytest.approx(0.3)
    assert portfolio_weight_distance([0, 0], [0.3, 0]) == pytest.approx(0.15)
    assert b.reasons(
        "2024-01-04", "2024-01-03", [0.15, 0], [0, 0], [0.1, 0.1], 0, cash_config()
    ) == ["hold"]


@pytest.mark.parametrize(
    "before,after", [([1.1, 0], [0, 0]), ([0, 0], [-0.1, 0.2]), ([0], [0, 0])]
)
def test_invalid_actual_weights_rejected(before, after):
    with pytest.raises(ValueError):
        b.cash_turnover(before, after)


def test_inference_and_schema_use_new_policy(tmp_path):
    import json
    from backend.investment_ai.infer import predict_weights
    from backend.investment_ai.training import schema_for, load_bundle
    from tests.test_contract_v3 import state

    d = data()
    d.risk[:] = 0.8
    c = cash_config()
    s = state(d, mdd=0)
    s["portfolio_value"] = 1.0
    s.update(
        weights={"SPY": 0.0, "QQQ": 0.0}, last_executed_weights={"SPY": 0.0, "QQQ": 0.0}
    )

    class Predictor:
        def predict(self, obs, environment):
            return np.array([0.6, 0.4])

    result = predict_weights(d, c, [], None, s, [Predictor()])
    assert sum(result["weights"].values()) == pytest.approx(0.15)
    assert schema_for(d, c)["execution_policy"] == b.CASH_POLICY
    (tmp_path / "config.json").write_text(json.dumps(c))
    (tmp_path / "schema.json").write_text(json.dumps(schema_for(d, cfg())))
    with pytest.raises(ValueError, match="execution policy"):
        load_bundle(tmp_path)


def test_fee_induced_liquidation_records_both_legs():
    c = cash_config(transaction_cost_bps=5.0, slippage_bps=2.0, max_drawdown=0.00001)
    e = PortfolioEnv(data(), c, start=1)
    e.reset()
    r = e.step_weights([0.6, 0.4], force=True)[-1]
    assert r["normal_executed_turnover"] == pytest.approx(0.15)
    assert r["executed_turnover"] == pytest.approx(0.30)
    assert r["cash_weight"] == 1
    assert "mdd_stop" in r["rebalance_reasons"]
    assert r["transaction_cost"] > 0
    assert r["reward"] == pytest.approx(
        r["portfolio_return"] - 0.1 * r["max_drawdown"] - 0.003 * 0.30
    )
