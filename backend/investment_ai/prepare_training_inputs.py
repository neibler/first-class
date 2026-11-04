"""Adapt unscaled team data and dated news scores; never starts training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import prepare_data, read_table, parse_dates, fingerprint
from .settings import ROOT

PROJECT = Path(__file__).resolve().parents[2]
ETF_COLUMNS = {
    "date",
    "ticker",
    "adj_close",
    "log_return",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
}


def adapt_inputs(market, news, assets, start, end):
    market, news = market.copy(), news.copy()
    if not ETF_COLUMNS.issubset(market.columns):
        raise ValueError("Expected unscaled team ETF columns")
    macros = [c for c in market.columns if c not in ETF_COLUMNS]
    if any(c.endswith("_z") for c in macros):
        raise ValueError("Use features_unscaled, not pre-normalized split files")
    for frame in (market, news):
        frame["date"] = parse_dates(frame["date"])
    market = market[market.date.between(start, end)].copy()
    news = news[news.date.between(start, end)].copy()
    market = market.rename(columns={"rsi_14": "rsi", "adj_close": "close"})
    market = market[["date", "ticker", "close", "log_return", "rsi", "macd", *macros]]
    if (
        not len(market)
        or not np.isfinite(market.close).all()
        or (market.close <= 0).any()
    ):
        raise ValueError("Missing or invalid adjusted prices")
    data = prepare_data(
        market, assets, news, macro_columns=macros, news_mode="required"
    )
    for asset in assets:
        rows = market[market.ticker == asset].sort_values("date")
        np.testing.assert_allclose(
            rows.log_return.to_numpy()[1:],
            np.diff(np.log(rows.close)),
            atol=1e-10,
            rtol=1e-7,
        )
    market["ticker"] = pd.Categorical(market.ticker, categories=assets, ordered=True)
    market = market.sort_values(["date", "ticker"]).reset_index(drop=True)
    market["ticker"] = market.ticker.astype(str)
    return market, data.news_metadata, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Default: first common market/news date")
    parser.add_argument("--end", help="Default: last common market/news date")
    parser.add_argument(
        "--market",
        type=Path,
        default=PROJECT / "data_process/processed/features_unscaled.parquet",
    )
    parser.add_argument(
        "--news", type=Path, default=ROOT / "data/news/raw/news_risk_nyt_2021_2025.csv"
    )
    args = parser.parse_args()
    configs = [
        PROJECT / "config/investment_ai.json",
        PROJECT / "config/multi_model.json",
    ]
    config = json.loads(configs[0].read_text(encoding="utf-8"))
    raw_market, raw_news = read_table(args.market), read_table(args.news)
    market_dates, news_dates = parse_dates(raw_market.date), parse_dates(raw_news.date)
    args.start = args.start or str(max(market_dates.min(), news_dates.min()).date())
    args.end = args.end or str(min(market_dates.max(), news_dates.max()).date())
    market, news, data = adapt_inputs(
        raw_market,
        raw_news,
        config["assets"],
        args.start,
        args.end,
    )
    source_hashes = {
        "market": hashlib.sha256(args.market.read_bytes()).hexdigest(),
        "news": hashlib.sha256(args.news.read_bytes()).hexdigest(),
    }
    identity = hashlib.sha256(
        json.dumps([source_hashes, args.start, args.end, data.assets]).encode()
    ).hexdigest()[:12]
    folder = ROOT / "data/prepared" / identity
    folder.mkdir(parents=True, exist_ok=True)
    market.to_parquet(folder / "market.parquet", index=False)
    news.to_parquet(folder / "news.parquet", index=False)
    report = {
        "training_executed": False,
        "market_source": str(args.market),
        "news_source": str(args.news),
        "source_sha256": source_hashes,
        "start": str(data.dates[0])[:10],
        "end": str(data.dates[-1])[:10],
        "days": len(data.dates),
        "rows": len(market),
        "macro_columns": data.macros,
        "news_columns": list(news.columns),
        "fingerprint": fingerprint(data),
        "price_basis": "upstream adj_close",
        "scaling": (
            "FeatureScaler fits training partition only; "
            "raw log_return retained for accounting"
        ),
        "limitations": (
            "Upstream VIX/WTI availability uses approximations; "
            "not certified strictly point-in-time."
        ),
    }
    (folder / "provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for path in configs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg["assets"] != data.assets:
            raise ValueError("Config asset order mismatch")
        cfg.update(
            data_path=(folder / "market.parquet").relative_to(ROOT).as_posix(),
            news_path=(folder / "news.parquet").relative_to(ROOT).as_posix(),
            macro_columns=data.macros,
            news_mode="required",
            synthetic_news=False,
        )
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
