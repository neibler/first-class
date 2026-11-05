import json
from pathlib import Path

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


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    from backend.investment_ai import prepare_training_inputs as module

    project = tmp_path / "project"
    runtime = tmp_path / "runtime"
    (project / "config").mkdir(parents=True)
    config = json.loads(Path("config/multi_model.json").read_text(encoding="utf-8"))
    config["assets"] = ["SPY", "QQQ"]
    for name in ("investment_ai", "multi_model"):
        (project / "config" / f"{name}.json").write_text(json.dumps(config))
    market, news = inputs()
    market = pd.concat([market, market.assign(ticker="QQQ")], ignore_index=True)
    news = pd.concat([news, news.assign(ticker="QQQ")], ignore_index=True)
    market.to_parquet(tmp_path / "market.parquet", index=False)
    news.to_parquet(tmp_path / "news.parquet", index=False)
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", runtime)
    args = [
        "--market",
        str(tmp_path / "market.parquet"),
        "--news",
        str(tmp_path / "news.parquet"),
    ]
    return module, project, runtime, args


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "field,value",
    [
        ("assets", ["QQQ", "SPY"]),
        ("assets", ["SPY", "SPY"]),
        ("assets", []),
        ("assets", "SPY"),
        ("assets", ["SPY", None]),
        ("assets", ["SPY", " "]),
        ("lookback", 0),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_all_configs_validated_before_any_writes(preparation, field, value, existing):
    module, project, runtime, args = preparation
    if existing:
        module.main(args)
    path = project / "config/multi_model.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    before = snapshot(project.parent)
    with pytest.raises(ValueError):
        module.main(args)
    assert snapshot(project.parent) == before
    if not existing:
        assert not runtime.exists()


def test_preparation_updates_both_configs_and_validates_panel(preparation):
    from backend.investment_ai.data_loader import load_data

    module, project, runtime, args = preparation
    module.main(args)
    first = json.loads((project / "config/investment_ai.json").read_text())
    second = json.loads((project / "config/multi_model.json").read_text())
    assert first == second
    panel, _ = load_data(first, runtime)
    assert panel.assets == ["SPY", "QQQ"]
    assert panel.macros == ["vix"]
    assert (panel.risk == 0.4).all()
    assert (runtime / first["data_path"]).with_name("provenance.json").exists()
