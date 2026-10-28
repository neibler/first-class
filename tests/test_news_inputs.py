import numpy as np
import pytest
from backend.investment_ai.data_loader import load_data, read_table
from backend.investment_ai.settings import read_config
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_real_news_is_preserved_in_hybrid_input():
    config = read_config(ROOT / "config/hybrid_news.json")
    if not (ROOT / "runtime" / config["news_path"]).exists():
        pytest.skip("Local shared dataset not distributed with source")
    data, _ = load_data(config, ROOT / "runtime")
    news = read_table(ROOT / "runtime" / config["news_path"])
    expected = news.pivot(index="date", columns="ticker", values="risk_score")
    np.testing.assert_allclose(data.risk, expected[data.assets].to_numpy())
    assert len(data.dates) == news.date.nunique()


def test_csv_preserves_literal_risk_type(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text("date,ticker,risk_score,risk_type\n2023-01-03,SPY,0.2,none\n")
    assert read_table(path).risk_type.iloc[0] == "none"


def test_missing_real_market_does_not_fall_back_to_dummy(tmp_path):
    config = read_config(ROOT / "config/real_market.json")
    with pytest.raises(FileNotFoundError):
        load_data(config, tmp_path)


def test_wide_conversion_preserves_asset_order_and_rejects_bad_returns():
    import pandas as pd
    from backend.investment_ai.prepare_market import convert_features

    wide = pd.DataFrame({"Date": ["2023-01-03", "2023-01-04"]})
    for ticker in ["QQQ", "SPY"]:
        wide[ticker + "_CLOSE"] = [100.0, 110.0]
        wide[ticker + "_LOG_RETURN"] = [0.01, np.log(1.1)]
        wide[ticker + "_RSI"] = [50.0, 60.0]
        wide[ticker + "_MACD"] = [1.0, 2.0]
    panel = convert_features(wide, ["QQQ", "SPY"])
    assert panel.ticker.tolist() == ["QQQ", "SPY", "QQQ", "SPY"]
    wide.loc[1, "SPY_LOG_RETURN"] = 9.0
    with pytest.raises(AssertionError):
        convert_features(wide, ["QQQ", "SPY"])
