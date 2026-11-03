"""Causal, correlated synthetic scenarios, not calibrated market forecasts."""

import numpy as np
import pandas as pd

SCENARIOS = ("mixed", "bull", "bear", "sideways", "crash", "volatile")


def generate(config, scenario=None):
    scenario = scenario or config.get("scenario", "mixed")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}")
    rng = np.random.default_rng(config["seed"])
    dates = pd.bdate_range(config["dummy_start"], config["dummy_end"])
    assets = config["assets"]
    n, a = len(dates), len(assets)
    if n < 30:
        raise ValueError("Synthetic data requires at least 30 weekdays")
    labels = (
        np.array(
            [("bull", "sideways", "bear", "volatile")[(i // 90) % 4] for i in range(n)]
        )
        if scenario == "mixed"
        else np.repeat(scenario, n)
    )
    drift = np.array(
        [
            {
                "bull": 0.0007,
                "bear": -0.0007,
                "sideways": 0.0,
                "volatile": 0.0001,
                "crash": 0.0002,
            }[x]
            for x in labels
        ]
    )
    vol = np.where(
        labels == "volatile", 0.022, np.where(labels == "bear", 0.014, 0.008)
    )
    market = drift + rng.normal(size=n) * vol
    if scenario in ("crash", "mixed"):
        k = n // 2
        market[k : k + 3] -= np.array([0.13, 0.10, 0.08])
    rates = rng.normal(0, 0.003, n)
    returns = np.empty((n, a))
    for j, ticker in enumerate(assets):
        if ticker in {"SHY", "IEF", "TLT", "TIP", "LQD"}:
            duration = {"SHY": 0.15, "IEF": 1.0, "TLT": 2.0, "TIP": 0.7, "LQD": 0.8}[
                ticker
            ]
            returns[:, j] = (
                0.00008 - 0.12 * market + duration * rates + rng.normal(0, 0.002, n)
            )
        elif ticker == "GLD":
            returns[:, j] = 0.0001 - 0.15 * market + rng.normal(0, 0.007, n)
        elif ticker in {"DBB", "DBA"}:
            returns[:, j] = 0.0001 + 0.3 * market + rng.normal(0, 0.008, n)
        else:
            beta = 1.3 if ticker in {"QQQ", "SMH"} else 1.0
            returns[:, j] = beta * market + rng.normal(0, 0.004, n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)))
    delta = px.diff().fillna(0)
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 * gain / (gain + loss)).fillna(50).clip(0, 100).to_numpy()
    macd = (
        (px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean())
        / px
    ).to_numpy()
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, a),
            "ticker": assets * n,
            "log_return": returns.ravel(),
            "rsi": rsi.ravel(),
            "macd": macd.ravel(),
        }
    )
    mean = pd.Series(market).rolling(20, min_periods=1).mean().to_numpy()
    vol = pd.Series(market).rolling(20, min_periods=2).std().fillna(0).to_numpy()
    known = {
        "pmi": 50 + mean * 1000,
        "core_pce": 2.5 + np.cumsum(rates),
        "treasury_10y": 4 + np.cumsum(rates),
        "vix": 15 + vol * 1000,
        "wti": 70 * np.exp(np.cumsum(market * 0.2)),
    }
    for name in config["dummy_macro_columns"]:
        frame[name] = np.repeat(known.get(name, np.cumsum(rng.normal(0, 0.01, n))), a)
    if config.get("synthetic_news", False):
        frame["risk_score"] = np.clip(np.maximum(-returns, 0) / 0.1, 0, 1).ravel()
        frame["risk_type"] = np.where(
            frame.risk_score > 0.5, "synthetic_stress", "synthetic_normal"
        )
    return frame
