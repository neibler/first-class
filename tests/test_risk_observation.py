from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.settings import validate_config
from backend.investment_ai.training import feature_names, schema_for
from backend.investment_ai.infer import predict_weights
from tests.test_contract_v3 import config, data, state


@pytest.mark.parametrize("include", [True, False])
def test_observation_schema_and_news_rule(include):
    c, d = config(), data()
    c["include_risk_score"] = include
    original = d.risk.copy()
    env = PortfolioEnv(d, c)
    obs, _ = env.reset()
    names = feature_names(d, c)
    assert len(obs) == len(names) == env.observation_space.shape[0]
    assert schema_for(d, c)["observation_size"] == len(obs)
    assert any(n.startswith("risk_score_") for n in names) == include
    assert ("risk_score" in schema_for(d, c)["order"]) == include
    assert "risk_score" in schema_for(d, c)["news_columns"]
    np.testing.assert_array_equal(d.risk, original)
    env.weights[:] = 0.5
    d.risk[env.t, 0] = 0.7
    row = env.step_weights([0.5, 0.5])[-1]
    assert row["rebalance_reasons"] == ["risk_score"]


def test_exclusion_removes_only_risk_features_and_is_reversible():
    c, d = config(), data()
    c.pop("include_risk_score", None)
    default = PortfolioEnv(d, c).reset()[0]
    default_schema = schema_for(d, c)
    enabled = {**c, "include_risk_score": True}
    assert schema_for(d, enabled) == default_schema
    np.testing.assert_array_equal(PortfolioEnv(d, enabled).reset()[0], default)
    disabled = {**c, "include_risk_score": False}
    keep = [
        i for i, n in enumerate(feature_names(d, c)) if not n.startswith("risk_score_")
    ]
    np.testing.assert_array_equal(PortfolioEnv(d, disabled).reset()[0], default[keep])
    d2 = deepcopy(d)
    d2.risk[:] = 0.9
    np.testing.assert_array_equal(PortfolioEnv(d2, disabled).reset()[0], default[keep])
    assert not np.array_equal(PortfolioEnv(d2, enabled).reset()[0], default)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_flag_requires_boolean(value):
    c = {**config(), "include_risk_score": value}
    with pytest.raises(ValueError, match="include_risk_score"):
        validate_config(c)
    with pytest.raises(ValueError, match="include_risk_score"):
        PortfolioEnv(data(), c)


def test_config_pair_differs_only_in_risk_input_flag():
    root = Path(__file__).resolve().parents[1] / "config"
    enabled = json.loads((root / "multi_model.json").read_text())
    disabled = json.loads((root / "multi_model_no_risk.json").read_text())
    assert enabled.pop("include_risk_score") is True
    assert disabled.pop("include_risk_score") is False
    assert enabled == disabled


def test_inference_uses_excluded_observation_but_keeps_risk_trigger():
    c, d = {**config(), "include_risk_score": False}, data()
    d.risk[-1, 0] = 0.7
    seen = []

    class Model:
        def predict(self, obs, **kwargs):
            seen.append(obs.copy())
            return np.zeros(2), None

    s = state(d, mdd=0)
    s.update(portfolio_value=1.0, weights={"SPY": 0.5, "QQQ": 0.5})
    result = predict_weights(d, c, [Model()], None, s)
    assert len(seen[0]) == len(feature_names(d, c))
    assert result["rebalance_reasons"] == ["risk_score"]
