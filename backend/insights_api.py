"""Read-only dashboard insights built from the active model's saved data.

    GET /risk                      1-day historical VaR/CVaR of the saved backtest
    GET /rebalancing               Daily weight changes the model made
    GET /assets/{ticker}/history   Cumulative return, RSI and MACD of one ETF

Everything is read from the loaded InvestmentService (saved evaluation history and
installed market inputs). Nothing is trained, predicted or generated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["insights"])

RISK_LEVEL = 0.95


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TailRisk(Contract):
    var: float | None = Field(description="1-day VaR, loss size as a positive number")
    cvar: float | None = Field(description="Mean loss beyond VaR, positive number")


class RiskResponse(Contract):
    model_version: str
    as_of: str
    start: str
    observations: int
    level: float
    horizon_days: int
    method: str
    benchmark: str
    portfolio: TailRisk
    benchmark_risk: TailRisk


class WeightChange(Contract):
    ticker: str
    before: float
    after: float
    change: float


class RebalanceEvent(Contract):
    date: str
    decision_date: str
    reasons: list[str]
    turnover: float
    cash_weight: float
    changes: list[WeightChange]


class RebalancingResponse(Contract):
    model_version: str
    as_of: str
    observations: int
    rebalanced_days: int
    average_turnover: float
    min_change: float
    events: list[RebalanceEvent]


class AssetRow(Contract):
    date: str
    close: float | None = Field(description="Close price in the input data's currency")
    cumulative_return: float
    rsi: float
    macd: float


class AssetHistoryResponse(Contract):
    model_version: str
    ticker: str
    as_of: str
    observations: int
    rows: list[AssetRow]


def active_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            503, "Investment model is not ready; check server configuration"
        )
    return service


def saved_history(service, strategy: str, observations: int | None):
    history = service.histories.get(strategy)
    if history is None:
        raise HTTPException(503, f"No saved evaluation history for {strategy}")
    return history if observations is None else history.tail(observations)


@router.get("/risk", response_model=RiskResponse)
def risk(
    request: Request,
    benchmark: str = "spy_buy_hold",
    observations: int | None = Query(default=None, ge=2),
):
    """Historical 1-day VaR/CVaR (95%) using the performance module definitions."""
    from backend.performance.metrics import conditional_value_at_risk, value_at_risk

    service = active_service(request)
    if benchmark == "ppo" or benchmark not in service.histories:
        raise HTTPException(422, "Unknown benchmark")
    portfolio = saved_history(service, "ppo", observations)
    compared = saved_history(service, benchmark, observations)

    def tail(history):
        returns = history.portfolio_return.to_numpy(dtype=float)
        return {
            "var": value_at_risk(returns, RISK_LEVEL),
            "cvar": conditional_value_at_risk(returns, RISK_LEVEL),
        }

    return {
        "model_version": service.run_id,
        "as_of": str(portfolio.date.iloc[-1]),
        "start": str(portfolio.date.iloc[0]),
        "observations": len(portfolio),
        "level": RISK_LEVEL,
        "horizon_days": 1,
        "method": "historical_simple_daily_returns",
        "benchmark": benchmark,
        "portfolio": tail(portfolio),
        "benchmark_risk": tail(compared),
    }


@router.get("/rebalancing", response_model=RebalancingResponse)
def rebalancing(
    request: Request,
    observations: int | None = Query(default=None, ge=1),
    limit: int = Query(default=30, ge=1, le=500),
    min_change: float = Query(default=0.01, ge=0, le=1),
):
    """Newest-first weight changes per trading day from the saved PPO history."""
    service = active_service(request)
    history = saved_history(service, "ppo", observations)
    assets = service.config["assets"]
    rebalanced = history[history.rebalanced.astype(bool)]
    events = []
    for _, row in rebalanced.iloc[::-1].head(limit).iterrows():
        changes = [
            {
                "ticker": asset,
                "before": float(row[f"previous_weight_{asset.lower()}"]),
                "after": float(row[f"current_weight_{asset.lower()}"]),
                "change": float(row[f"weight_change_{asset.lower()}"]),
            }
            for asset in assets
            if abs(float(row[f"weight_change_{asset.lower()}"])) >= min_change
        ]
        changes.sort(key=lambda item: -abs(item["change"]))
        reason = row.get("rebalance_reason")
        events.append(
            {
                "date": str(row.date),
                "decision_date": str(row.decision_date),
                "reasons": (
                    [] if pd.isna(reason) else [r for r in str(reason).split("|") if r]
                ),
                "turnover": float(row.turnover),
                "cash_weight": float(row.cash_weight),
                "changes": changes,
            }
        )
    return {
        "model_version": service.run_id,
        "as_of": str(history.date.iloc[-1]),
        "observations": len(history),
        "rebalanced_days": len(rebalanced),
        "average_turnover": (
            float(rebalanced.turnover.mean()) if len(rebalanced) else 0.0
        ),
        "min_change": min_change,
        "events": events,
    }


def close_prices(service, symbol, dates):
    """Close per date from the loaded input frame; None where it is unavailable."""
    frame = getattr(service, "market_frame", None)
    if frame is None or "close" not in frame:
        return [None] * len(dates)
    rows = frame[frame["ticker"] == symbol]
    closes = pd.Series(
        rows["close"].to_numpy(dtype=float),
        index=pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"),
    )
    closes = closes[~closes.index.duplicated()].reindex(dates)
    return [float(v) if np.isfinite(v) else None for v in closes.to_numpy()]


@router.get("/assets/{ticker}/history", response_model=AssetHistoryResponse)
def asset_history(
    request: Request,
    ticker: str,
    observations: int = Query(default=252, ge=2, le=5000),
):
    """Cumulative return (start = 0), RSI and MACD from the installed market input."""
    service = active_service(request)
    data = service.data
    symbol = ticker.upper()
    if symbol not in data.assets:
        raise HTTPException(404, "Not an asset of the active model")
    column = data.assets.index(symbol)
    count = min(observations, len(data.dates))
    window = slice(len(data.dates) - count, None)
    returns = data.returns[window, column]
    # The first row is the start point; later rows compound daily log returns.
    cumulative = np.expm1(np.concatenate([[0.0], np.cumsum(returns[1:])]))
    dates = [str(np.datetime_as_string(d, unit="D")) for d in data.dates[window]]
    closes = close_prices(service, symbol, dates)
    rows = [
        {
            "date": day,
            "close": close,
            "cumulative_return": float(total),
            "rsi": float(rsi),
            "macd": float(macd),
        }
        for day, close, total, rsi, macd in zip(
            dates,
            closes,
            cumulative,
            data.rsi[window, column],
            data.macd[window, column],
        )
    ]
    return {
        "model_version": service.run_id,
        "ticker": symbol,
        "as_of": dates[-1],
        "observations": count,
        "rows": rows,
    }
