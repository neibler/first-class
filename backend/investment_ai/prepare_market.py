"""Convert the team's wide daily ETF features to the training panel."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data_loader import prepare_data, read_table, parse_dates
from .settings import ROOT, read_config

PROJECT = Path(__file__).resolve().parents[2]


def convert_features(wide, assets):
    expected = {"Date"} | {
        f"{a}_{f}" for a in assets for f in ("CLOSE", "LOG_RETURN", "RSI", "MACD")
    }
    if set(wide.columns) != expected or wide.columns.duplicated().any():
        raise ValueError("Unexpected wide feature columns")
    dates = parse_dates(wide["Date"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Market dates must be unique and sorted")
    frames = []
    for asset in assets:
        frame = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "ticker": asset})
        for source, target in [
            ("CLOSE", "close"),
            ("LOG_RETURN", "log_return"),
            ("RSI", "rsi"),
            ("MACD", "macd"),
        ]:
            frame[target] = wide[f"{asset}_{source}"].to_numpy()
        if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
            raise ValueError("Invalid closing price")
        # The first return uses the upstream warmup price, which is outside this CSV.
        np.testing.assert_allclose(
            frame.log_return.to_numpy()[1:], np.diff(np.log(frame.close)), atol=1e-10
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["ticker"] = pd.Categorical(panel.ticker, categories=assets, ordered=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel["ticker"] = panel.ticker.astype(str)
    return panel


def main():
    config = read_config(PROJECT / "config/real_market.json")
    panel = convert_features(
        read_table(PROJECT / "data/processed/etf_features.csv"), config["assets"]
    )
    news = read_table(ROOT / config["news_path"])
    market_keys = set(map(tuple, panel[["date", "ticker"]].to_numpy()))
    news_keys = set(map(tuple, news[["date", "ticker"]].to_numpy()))
    if market_keys != news_keys:
        raise ValueError("Market and news date/ticker keys differ; align explicitly")
    data = prepare_data(panel, config["assets"], news, macro_columns=[])
    data.news_metadata.to_parquet(
        ROOT / "data/news/validated_news.parquet", index=False
    )
    target = ROOT / "data/market.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    report = {
        "market_source": "real_team_data",
        "rows": len(panel),
        "days": len(data.dates),
        "news_keys_exact_match": True,
        "macro_columns": [],
        "training_executed": False,
        "price_basis": "Close; no added dividend return",
        "output": str(target),
    }
    (ROOT / "market_import_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
