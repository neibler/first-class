import numpy as np
import pandas as pd
import pytest

from backend.investment_ai.prepare_training_inputs import adapt_inputs


def inputs():
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    market = pd.DataFrame(
        dict(
            date=dates,
            ticker="SPY",
            adj_close=[100.0, 101.0, 99.0],
            log_return=[0.0, np.log(1.01), np.log(99 / 101)],
            rsi_14=50.0,
            macd=1.0,
            macd_signal=0.5,
            macd_hist=0.5,
            vix=20.0,
        )
    )
    news = pd.DataFrame(
        dict(
            date=dates,
            ticker="SPY",
            risk_score=0.4,
            risk_type="market_risk",
            has_news=0,
            days_since_news=1,
            evidence="retained",
        )
    )
    return market, news


def test_raw_returns_carried_risk_and_metadata_preserved():
    market, news = inputs()
    panel, metadata, data = adapt_inputs(
        market, news, ["SPY"], "2021-01-04", "2021-01-06"
    )
    np.testing.assert_array_equal(panel.log_return, market.log_return)
    assert data.macros == ["vix"]
    assert (data.risk == 0.4).all()
    assert metadata.evidence.tolist() == ["retained"] * 3
    assert panel.rsi.tolist() == [50.0] * 3


def test_missing_news_fails_instead_of_zero_fill():
    market, news = inputs()
    with pytest.raises(ValueError, match="missing or extra"):
        adapt_inputs(market, news.iloc[1:], ["SPY"], "2021-01-04", "2021-01-06")


def test_scaled_returns_rejected():
    market, news = inputs()
    market["log_return_z"] = 0.0
    with pytest.raises(ValueError, match="pre-normalized"):
        adapt_inputs(market, news, ["SPY"], "2021-01-04", "2021-01-06")
