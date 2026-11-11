import numpy as np
import pytest

from backend.investment_ai.rebalancing import (
    filter_target_weights,
    portfolio_weight_distance,
)
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.backtest import run_backtest
from backend.investment_ai.data_loader import MarketData
from backend.investment_ai.infer import predict_weights
from backend.performance.model_shap import policy_function
from tests.test_contract_v3 import config, data, state
from tests.test_rebalancing import reasons, environment


def test_filter_strict_boundary_and_normalization_without_mutation():
    raw = np.array([0.029, 0.03, 0.941])
    result = filter_target_weights(raw, 0.03)
    assert result[0] == 0 and result[1] > 0
    assert result.sum() == pytest.approx(1, abs=1e-15)
    np.testing.assert_allclose(result[1:], raw[1:] / raw[1:].sum())
    np.testing.assert_array_equal(raw, [0.029, 0.03, 0.941])


def test_all_below_threshold_keeps_highest_and_ties_are_stable():
    np.testing.assert_array_equal(
        filter_target_weights([0.2, 0.3, 0.5], 0.6), [0, 0, 1]
    )
    result = filter_target_weights(np.full(40, 1 / 40), 0.03)
    assert result[0] == 1 and np.count_nonzero(result) == 1


@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.03, 0.05])
def test_configurable_filter_threshold(threshold):
    raw = np.array([0.009, 0.02, 0.04, 0.931])
    result = filter_target_weights(raw, threshold)
    assert np.all(result[raw < threshold] == 0)
    assert result.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "distance,triggered", [(0.099, False), (0.10, True), (0.101, True)]
)
def test_distance_boundaries(distance, triggered):
    target = [0.5 + distance, 0.5 - distance]
    assert portfolio_weight_distance([0.5, 0.5], target) == pytest.approx(distance)
    assert ("portfolio_weight_distance" in reasons(target=target)) == triggered


def test_individual_five_point_difference_does_not_trigger():
    assert reasons(target=[0.56, 0.44]) == ["hold"]


def panel(n=3):
    base = data()
    shape = (5, n)
    return MarketData(
        base.dates,
        ["SPY"] + [f"ETF{i}" for i in range(1, n)],
        [],
        np.zeros(shape),
        np.full(shape, 50),
        np.zeros(shape),
        np.empty((5, 0)),
        np.full(shape, 0.2),
    )


def test_filter_precedes_distance_and_logs_decision_weights():
    d = panel()
    c = {**config(), "assets": d.assets}
    env = PortfolioEnv(d, c, start=1)
    env.reset()
    env.weights = np.array([0.11, 0.39, 0.5])
    raw = [0.029, 0.471, 0.5]
    assert portfolio_weight_distance(env.weights, raw) < 0.1
    row = env.step_weights(raw)[-1]
    assert row["rebalance_reasons"] == ["portfolio_weight_distance"]
    assert row["portfolio_weight_distance"] == pytest.approx(0.11)
    assert row["raw_target_weights"]["SPY"] == 0.029
    assert row["filtered_target_weights"]["SPY"] == 0
    assert row["current_weights"]["SPY"] == 0.11
    assert row["current_weight_spy"] == 0
    assert env.weights.sum() == pytest.approx(1)


def test_hold_keeps_quantity_and_has_no_cost_after_prices_change():
    env = environment([[0, 0], [0, 0], [0.1, 0], [0, 0], [0, 0]])
    before = env.weights.copy() * env.value
    row = env.step_weights([0.56, 0.44])[-1]
    assert row["rebalance_reasons"] == ["hold"]
    assert row["cost_amount"] == 0
    np.testing.assert_allclose(env.weights * env.value / [1.1, 1], before)
    assert env.weights[0] != 0.5


def test_mdd_overrides_large_distance_and_high_risk():
    env = environment()
    env.mdd = 0.151
    env.data.risk[env.t] = 0.9
    row = env.step_weights([0.01, 0.99])[-1]
    assert row["rebalance_reasons"] == ["mdd_stop"]
    assert row["cash_weight"] == 1 and not env.weights.any()
    assert row["turnover"] == 0  # queued target cancelled
    assert row["exit_turnover"] == 1


def test_ensemble_is_averaged_before_filtering():
    class Session:
        def __init__(self, weights):
            self.weights = weights

        def predict(self, obs, env):
            return np.array(self.weights)

    d, c = data(), config()
    s = state(d, mdd=0)
    s.update(portfolio_value=1.0, weights={"SPY": 0.5, "QQQ": 0.5})
    result = predict_weights(
        d, c, [], None, s, [Session([0.02, 0.98]), Session([0.04, 0.96])]
    )
    assert result["raw_target_weights"]["SPY"] == pytest.approx(0.03)
    assert result["filtered_target_weights"]["SPY"] == pytest.approx(0.03)
    assert result["weights"] == result["filtered_target_weights"]


def test_shap_explains_filtered_targets():
    env = PortfolioEnv(data(), config())

    class Model:
        def predict(self, rows, **kwargs):
            return np.tile([-1, 1], (len(rows), 1)), None

    predict = policy_function(Model(), "ppo", env)
    np.testing.assert_array_equal(
        predict(np.zeros((2, env.observation_space.shape[0]))), [[0, 1], [0, 1]]
    )


def test_buy_hold_enters_even_with_high_distance_threshold_and_is_not_filtered():
    d = panel(15)
    d.returns[2:, 0] = np.log1p(0.01)
    c = {
        **config(),
        "assets": d.assets,
        "min_weight_threshold": 0.9,
        "portfolio_weight_distance_threshold": 0.9,
        "transaction_cost_bps": 5,
    }
    h = run_backtest(d, c, 0, 4, strategy="equal_buy_hold")
    assert h.iloc[0].cash_weight == pytest.approx(0)
    assert h.iloc[0].turnover == pytest.approx(1)
    assert sum(h.iloc[0][f"held_{a.lower()}"] > 0 for a in d.assets) == 15
    assert (h.iloc[1:].turnover == 0).all() and (h.iloc[1:].cost_amount == 0).all()
    assert h.iloc[-1].portfolio_value != h.iloc[0].portfolio_value


def test_distance_is_based_on_current_price_drifted_weights():
    env = environment([[0, 0], [0, 0], [0.3, 0], [0, 0], [0, 0]])
    env.step_weights([0.5, 0.5])
    previous_actual = env.weights.copy()
    row = env.step_weights([0.5, 0.5])[-1]
    assert row["portfolio_weight_distance"] == pytest.approx(
        portfolio_weight_distance(previous_actual, [0.5, 0.5])
    )
    assert row["portfolio_weight_distance"] > 0
