"""Decision boundaries, delayed execution, and daily policy/hold contracts."""

import numpy as np
import pandas as pd
import pytest

from backend.investment_ai.rebalancing import (
    get_rebalance_reasons,
    is_monthly_rebalance_day,
)
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.infer import predict_weights
from backend.investment_ai.api import OptimizeResponse
from backend.investment_ai.prediction import PredictionSession
from tests.test_contract_v3 import config, data, state, service


def reasons(
    current=(0.5, 0.5),
    target=(0.5, 0.5),
    risk=(0.1, 0.1),
    mdd=0.0,
    date="2024-02-06",
    previous="2024-02-05",
    stopped=False,
):
    return get_rebalance_reasons(
        date, previous, current, target, risk, mdd, config(), stopped
    )


@pytest.mark.parametrize("risk,expected", [(0.7, ["risk_score"]), (0.69, ["hold"])])
def test_risk_boundary(risk, expected):
    assert reasons(risk=(0.1, risk)) == expected


@pytest.mark.parametrize(
    "difference,expected", [(0.10, ["portfolio_weight_distance"]), (0.099, ["hold"])]
)
def test_portfolio_distance_boundary(difference, expected):
    assert reasons(target=(0.5 + difference, 0.5 - difference)) == expected


@pytest.mark.parametrize(
    "mdd,expected", [(0.151, ["mdd_stop"]), (0.15, ["hold"]), (1 - 0.85, ["hold"])]
)
def test_mdd_boundary(mdd, expected):
    assert reasons(mdd=mdd) == expected


def test_monthly_uses_first_trading_day_not_fixed_interval():
    assert reasons(date="2024-06-03", previous="2024-05-31") == ["monthly"]
    assert not is_monthly_rebalance_day("2024-06-04", "2024-06-03")
    assert not is_monthly_rebalance_day("2024-06-15")


def test_simultaneous_reasons_and_stop_priority():
    args = dict(
        date="2024-06-03", previous="2024-05-31", target=(0.60, 0.40), risk=(0.7, 0.1)
    )
    assert reasons(**args) == ["monthly", "risk_score", "portfolio_weight_distance"]
    assert reasons(**args, mdd=0.151) == ["mdd_stop"]
    assert reasons(stopped=True) == ["mdd_stop"]


def environment(returns=None, dates=None):
    d = data(returns)
    if dates is not None:
        d.dates = np.array(dates, dtype="datetime64[ns]")
    c = config()
    c.update(transaction_cost_bps=10, slippage_bps=10)
    env = PortfolioEnv(d, c, start=1)
    env.reset()
    env.weights = np.array([0.5, 0.5])
    return env


def test_hold_keeps_weights_and_cost_zero_despite_different_daily_target():
    env = environment()
    row = env.step_weights([0.549, 0.451])[-1]
    assert row["rebalance_reasons"] == ["hold"]
    np.testing.assert_allclose(env.weights, [0.5, 0.5])
    assert row["cost_amount"] == row["turnover"] == row["exit_cost"] == 0
    assert not row["rebalanced"]
    assert row["model_target_weight_spy"] == 0.549


def test_hold_allows_natural_price_drift_without_trading():
    env = environment([[0, 0], [0, 0], [0.1, 0], [0, 0], [0, 0]])
    row = env.step_weights([0.5, 0.5])[-1]
    np.testing.assert_allclose(env.weights, [0.55 / 1.05, 0.5 / 1.05])
    assert row["cost_amount"] == 0 and row["rebalance_reason"] == "hold"
    assert row["portfolio_value"] == pytest.approx(1.05)


@pytest.mark.parametrize(
    "trigger", ["monthly", "risk_score", "portfolio_weight_distance"]
)
def test_environment_triggers_fill_next_close_and_charge_existing_cost(trigger):
    env = environment(
        dates=(
            ["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06"]
            if trigger == "monthly"
            else None
        )
    )
    target = [0.52, 0.48]
    if trigger == "risk_score":
        env.data.risk[env.t, 0] = 0.7
    if trigger == "portfolio_weight_distance":
        target = [0.60, 0.40]
    row = env.step_weights(target)[-1]
    assert row["rebalance_reasons"] == [trigger]
    assert row["rebalanced"]
    np.testing.assert_allclose(env.weights, target)
    assert row["cost"] == pytest.approx(np.abs(np.array(target) - 0.5).sum() * 0.002)
    assert pd.Timestamp(row["execution_date"]) > pd.Timestamp(row["decision_date"])


def test_next_day_news_is_not_used_for_current_decision():
    env = environment()
    env.data.risk[env.t + 1, 0] = 0.99
    assert env.step_weights([0.52, 0.48])[-1]["rebalance_reasons"] == ["hold"]
    assert env.step_weights([0.52, 0.48])[-1]["rebalance_reasons"] == ["risk_score"]


@pytest.mark.parametrize("already_cash", [False, True])
def test_stop_liquidates_once_and_does_not_charge_cash_again(already_cash):
    env = environment()
    env.mdd = 0.151
    if already_cash:
        env.weights[:] = 0
        env.safeguard_triggered = True
    row = env.step_weights([0.5, 0.5])[-1]
    assert row["rebalance_reasons"] == ["mdd_stop"]
    assert row["safeguard_triggered"] and row["cash_weight"] == 1
    assert not env.weights.any()
    assert row["exit_turnover"] == (0 if already_cash else 1)
    assert row["cost_amount"] == pytest.approx(0 if already_cash else 0.002)
    assert row["rebalanced"] == (not already_cash)


def test_environment_exact_mdd_limit_does_not_stop():
    env = environment()
    env.mdd = 0.15
    row = env.step_weights([0.5, 0.5])[-1]
    assert not row["safeguard_triggered"] and row["rebalance_reasons"] == ["hold"]


class Policy:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, **kwargs):
        self.calls += 1
        return np.zeros(2), None


def test_daily_inference_computes_target_even_when_holding():
    d, c, model = data(), config(), Policy()
    s = state(d, mdd=0)
    s.update(
        weights={"SPY": 0.52, "QQQ": 0.48}, portfolio_value=1.0, days_since_rebalance=3
    )
    result = predict_weights(d, c, [model], None, s)
    assert model.calls == 1
    assert result["model_target_weights"] == {"SPY": 0.5, "QQQ": 0.5}
    assert result["weights"] == s["weights"]
    assert result["rebalance_reasons"] == ["hold"] and not result["should_rebalance"]


def test_api_exposes_proposal_and_execution_decision(tmp_path):
    obj = service(tmp_path)
    obj.models = [Policy()]
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.52, "QQQ": 0.48}, portfolio_value=1.0)
    result = OptimizeResponse.model_validate(obj.optimize(state=s))
    assert result.rebalance_reasons == ["hold"]
    assert result.target_weights == s["weights"]
    assert result.model_target_weights == {"SPY": 0.5, "QQQ": 0.5}
    assert not result.should_rebalance


def test_recurrent_memory_advances_on_hold_day():
    from tests.test_model_support import RecurrentSpy

    c, d = config(), data()
    c["model_type"] = "recurrent_ppo"
    model = RecurrentSpy()
    sessions = [PredictionSession(model, "recurrent_ppo")]
    s = state(d, mdd=0)
    s.update(weights={"SPY": 0.5, "QQQ": 0.5}, portfolio_value=1.0)
    for _ in range(2):
        assert predict_weights(d, c, [model], None, s, sessions)[
            "rebalance_reasons"
        ] == ["hold"]
    assert model.calls[0] == (None, True)
    assert model.calls[1][0] is not None and not model.calls[1][1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("risk_rebalance_threshold", float("nan")),
        ("portfolio_weight_distance_threshold", -1),
        ("max_drawdown", 1),
    ],
)
def test_invalid_thresholds_rejected(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError, match=key):
        PortfolioEnv(data(), c)


def test_normal_signal_queues_next_close_even_if_target_matches_today():
    d, c, model = data(), config(), Policy()
    d.risk[-1, 0] = 0.7
    s = state(d, mdd=0)
    s.update(weights={"SPY": 0.5, "QQQ": 0.5}, portfolio_value=1.0)
    result = predict_weights(d, c, [model], None, s)
    assert result["should_rebalance"] and result["rebalance_reasons"] == ["risk_score"]
    env = environment([[0, 0], [0, 0], [0.1, 0], [0, 0], [0, 0]])
    env.data.risk[env.t, 0] = 0.7
    row = env.step_weights([0.5, 0.5])[-1]
    assert row["should_rebalance"] and row["rebalanced"]
    assert row["cost_amount"] > 0


def test_api_cash_stop_does_not_propose_repeated_liquidation():
    d, c, model = data(), config(), Policy()
    s = state(d, mdd=0.2)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, stopped=True)
    result = predict_weights(d, c, [model], None, s)
    assert not result["should_rebalance"]
    assert result["cash_weight"] == 1 and model.calls == 0
    assert result["execution_delay_trading_bars"] == 0
