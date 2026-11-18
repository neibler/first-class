"""UI-only fixtures: no model files, training or network are used by these tests."""

import pytest
from frontend import investment_client
from frontend.sample_data import load_portfolio, performance


@pytest.fixture(autouse=True)
def investment_ui_fixture(request, monkeypatch):
    if request.node.module.__name__.split(".")[-1] not in {
        "test_dashboard",
        "test_ui_pages",
        "test_research_frontend",
        "test_statistics_frontend",
    }:
        return

    def payload(observations=None):
        data = load_portfolio()
        if observations is not None:
            data["returns"] = data["returns"][-(observations + 1) :]
        data["display_metrics"] = performance(
            data["returns"], data["periods_per_year"], data["annual_risk_free_rate"]
        )
        data.update(data_source="synthetic", stopped=False, cash_weight=0.0)
        return data

    monkeypatch.setattr(investment_client, "load_portfolio", payload)


RISK = {
    "model_version": "fixture",
    "as_of": "2026-09-30",
    "start": "2026-01-02",
    "observations": 187,
    "level": 0.95,
    "horizon_days": 1,
    "method": "historical_simple_daily_returns",
    "benchmark": "spy_buy_hold",
    "portfolio": {"var": 0.0147, "cvar": 0.0187},
    "benchmark_risk": {"var": 0.0142, "cvar": 0.0172},
}
REBALANCING = {
    "model_version": "fixture",
    "as_of": "2026-09-30",
    "observations": 187,
    "rebalanced_days": 187,
    "average_turnover": 1.43,
    "min_change": 0.01,
    "events": [
        {
            "date": "2026-09-30",
            "decision_date": "2026-09-29",
            "reasons": ["portfolio_weight_distance"],
            "turnover": 1.31,
            "cash_weight": 0.0,
            "changes": [
                {"ticker": "SHY", "before": 0.712, "after": 0.344, "change": -0.368},
                {"ticker": "GLD", "before": 0.0, "after": 0.344, "change": 0.344},
            ],
        },
        {
            "date": "2026-09-29",
            "decision_date": "2026-09-28",
            "reasons": ["monthly", "portfolio_weight_distance"],
            "turnover": 0.8,
            "cash_weight": 0.0,
            "changes": [],
        },
    ],
}


def asset_history(ticker, observations=252):
    rows = [
        {
            "date": "2026-09-28",
            "close": 765.61,
            "cumulative_return": 0.0,
            "rsi": 51.5,
            "macd": 1.9,
        },
        {
            "date": "2026-09-29",
            "close": 764.2,
            "cumulative_return": -0.002,
            "rsi": 50.2,
            "macd": 1.6,
        },
        {
            "date": "2026-09-30",
            "close": 762.63,
            "cumulative_return": -0.004,
            "rsi": 48.8,
            "macd": 1.3,
        },
    ]
    return {
        "model_version": "fixture",
        "ticker": ticker,
        "as_of": rows[-1]["date"],
        "observations": len(rows),
        "rows": rows,
    }


@pytest.fixture(autouse=True)
def insights_ui_fixture(request, monkeypatch):
    """Pages call /risk, /rebalancing and asset history; never reach a live server."""
    if request.node.module.__name__.split(".")[-1] not in {
        "test_dashboard",
        "test_ui_pages",
        "test_research_frontend",
        "test_statistics_frontend",
        "test_performance_frontend",
        "test_correlation",
    }:
        return
    from frontend import insights_client

    monkeypatch.setattr(insights_client, "load_risk", lambda observations=None: RISK)
    monkeypatch.setattr(
        insights_client,
        "load_rebalancing",
        lambda observations=None, limit=30: REBALANCING,
    )
    monkeypatch.setattr(insights_client, "load_asset_history", asset_history)
