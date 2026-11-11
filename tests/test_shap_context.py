import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from backend.performance.shap_context import context_at, attach_market_context


@pytest.fixture
def market():
    return SimpleNamespace(
        assets=["GLD"],
        macros=["vix"],
        dates=pd.bdate_range("2025-01-01", periods=8),
        returns=np.arange(8).reshape(-1, 1) / 100,
        rsi=np.full((8, 1), 31.0),
        macd=np.full((8, 1), -3.0),
        risk=np.full((8, 1), 0.72),
        macro=np.arange(20, 28).reshape(-1, 1),
    )


def test_context_uses_raw_market_and_only_past_dates(market):
    features = [
        "log_return_lag_0_gld",
        "log_return_lag_4_gld",
        "rsi_div_100_gld",
        "macd_gld",
        "macro_vix",
        "risk_score_gld",
        "current_weight_gld",
    ]
    result = context_at(market, features, [9] * 6 + [0.2], 6)
    assert result["values"] == dict(
        zip(features, [0.06, 0.02, 0.31, -3, 26, 0.72, 0.2])
    )
    assert result["previous_values"] == {"macro_vix": 21}
    assert result["comparison_trading_days"] == 5
    assert result["comparison_date"] == str(market.dates[1])[:10]
    assert result["as_of"] == str(market.dates[6])[:10]
    initial = context_at(market, features, [0] * len(features), 0)
    assert not initial["previous_values"]
    assert "log_return_lag_4_gld" not in initial["values"]
    with pytest.raises(ValueError, match="length"):
        context_at(market, features, [], 0)


def test_attach_checks_hash_features_and_preserves_saved_payload(
    tmp_path, monkeypatch, market
):
    from backend.investment_ai import data_loader, model_runs, training

    (tmp_path / "config.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    (tmp_path / "run_spec.json").write_text(
        json.dumps({"start_date": "2025-01-01", "end_date": "2025-01-10"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_loader, "load_data", lambda *a: (market, None))
    monkeypatch.setattr(data_loader, "fingerprint", lambda *a: "same")
    monkeypatch.setattr(model_runs, "subset", lambda data, start, end: data)
    monkeypatch.setattr(training, "feature_names", lambda *a: ["macro_vix"])
    payload = {
        "features": ["macro_vix"],
        "data_hash": "same",
        "assets": {
            "GLD": {
                "decisions": [
                    {
                        "date": "2025-01-09",
                        "feature_values_raw": [-9],
                        "shap_values": [0.1],
                    }
                ]
            }
        },
    }
    original = deepcopy(payload)
    result = attach_market_context(payload, tmp_path)
    assert result["market_context_status"] == "verified"
    row = result["assets"]["GLD"]["decisions"][0]
    assert row["market_context"]["values"]["macro_vix"] == 26
    assert row["feature_values_raw"] == [-9] and row["shap_values"] == [0.1]
    assert payload == original
    for key, bad in [("data_hash", "different"), ("features", ["macro_wti"])]:
        broken = {**payload, key: bad}
        result = attach_market_context(broken, tmp_path)
        assert result["market_context_status"] == "unavailable"
        assert "market_context" not in result["assets"]["GLD"]["decisions"][0]
