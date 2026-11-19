"""Risk, rebalancing and asset history APIs on a small in-memory service."""

import json
import math
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insights_api import router
from backend.performance.metrics import conditional_value_at_risk, value_at_risk
from frontend import insights_client
from frontend.api_client import APIError

ASSETS = ["SPY", "GLD"]


def history(returns, changes=None):
    rows = []
    for i, value in enumerate(returns):
        spy_change, gld_change = (changes or {}).get(i, (0.0, 0.0))
        rows.append(
            {
                "date": f"2026-01-{i + 2:02d}",
                "decision_date": f"2026-01-{i + 1:02d}",
                "portfolio_return": value,
                "rebalanced": True,
                "rebalance_reason": "monthly|portfolio_weight_distance",
                "turnover": abs(spy_change) + abs(gld_change),
                "cash_weight": 0.0,
                "previous_weight_spy": 0.5,
                "current_weight_spy": 0.5 + spy_change,
                "weight_change_spy": spy_change,
                "previous_weight_gld": 0.5,
                "current_weight_gld": 0.5 + gld_change,
                "weight_change_gld": gld_change,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def service():
    ppo = [0.01, -0.02, 0.005, -0.03, 0.012, 0.004, -0.001, 0.02, -0.015, 0.003]
    spy = [0.004, -0.01, 0.006, -0.012, 0.01, 0.002, 0.001, 0.008, -0.006, 0.002]
    dates = pd.date_range("2026-01-01", periods=6).to_numpy(dtype="datetime64[ns]")
    returns = np.array([[0.0, 0.0], [0.01, -0.01], [0.02, 0.0], [-0.01, 0.01]] * 2)[:6]
    return SimpleNamespace(
        run_id="fixture",
        config={"assets": ASSETS},
        histories={
            "ppo": history(ppo, {9: (0.2, -0.2), 8: (0.005, -0.005)}),
            "spy_buy_hold": history(spy),
        },
        data=SimpleNamespace(
            dates=dates,
            assets=ASSETS,
            returns=returns,
            rsi=np.linspace(40, 60, 12).reshape(6, 2),
            macd=np.linspace(-1, 1, 12).reshape(6, 2),
        ),
        # GLD has no price on the last day; SPY prices exist for every date.
        market_frame=pd.DataFrame(
            [
                {"date": day, "ticker": ticker, "close": 100.0 + i}
                for i, day in enumerate(pd.date_range("2026-01-01", periods=6))
                for ticker in ASSETS
                if not (ticker == "GLD" and i == 5)
            ]
        ),
    )


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(router)
    app.state.service = service
    return TestClient(app)


def test_risk_uses_the_performance_definitions(client, service):
    body = client.get("/risk").json()
    returns = service.histories["ppo"].portfolio_return.to_numpy()
    assert body["portfolio"]["var"] == pytest.approx(value_at_risk(returns, 0.95))
    assert body["portfolio"]["cvar"] == pytest.approx(
        conditional_value_at_risk(returns, 0.95)
    )
    assert body["portfolio"]["var"] > 0 and body["level"] == 0.95
    recent = client.get("/risk", params={"observations": 4}).json()
    assert recent["observations"] == 4 and recent["start"] == "2026-01-08"


def test_risk_rejects_unknown_benchmark_and_missing_model(client, service):
    assert client.get("/risk", params={"benchmark": "ppo"}).status_code == 422
    assert client.get("/risk", params={"benchmark": "nope"}).status_code == 422
    client.app.state.service = None
    assert client.get("/risk").status_code == 503
    assert client.get("/rebalancing").status_code == 503
    assert client.get("/assets/SPY/history").status_code == 503


def test_rebalancing_is_newest_first_and_hides_small_changes(client):
    body = client.get("/rebalancing", params={"limit": 2}).json()
    assert body["rebalanced_days"] == 10 and len(body["events"]) == 2
    latest, previous = body["events"]
    assert latest["date"] == "2026-01-11"
    assert latest["reasons"] == ["monthly", "portfolio_weight_distance"]
    assert [c["ticker"] for c in latest["changes"]] == ["SPY", "GLD"]
    assert latest["changes"][0] == {
        "ticker": "SPY",
        "before": 0.5,
        "after": pytest.approx(0.7),
        "change": 0.2,
    }
    # 0.5%p changes are below the default 1%p threshold.
    assert previous["changes"] == []
    shown = client.get("/rebalancing", params={"limit": 2, "min_change": 0.001})
    assert len(shown.json()["events"][1]["changes"]) == 2


def test_asset_history_compounds_saved_log_returns(client, service):
    body = client.get("/assets/spy/history", params={"observations": 4}).json()
    assert body["ticker"] == "SPY" and body["observations"] == 4
    rows = body["rows"]
    assert rows[0]["cumulative_return"] == 0.0
    expected = np.expm1(np.cumsum(service.data.returns[-3:, 0]))
    assert [r["cumulative_return"] for r in rows[1:]] == pytest.approx(expected)
    assert rows[-1]["rsi"] == pytest.approx(service.data.rsi[-1, 0])
    assert rows[-1]["date"] == "2026-01-06"
    assert [r["close"] for r in rows] == [102.0, 103.0, 104.0, 105.0]
    capped = client.get("/assets/GLD/history", params={"observations": 999})
    assert capped.json()["observations"] == 6
    # A missing price is reported as null, never filled in.
    assert capped.json()["rows"][-1]["close"] is None
    service.market_frame = None
    assert client.get("/assets/SPY/history").json()["rows"][0]["close"] is None
    assert client.get("/assets/XYZ/history").status_code == 404


def test_risk_without_saved_evaluation_is_not_ready(client, service):
    # A bundle without evaluation.json loads with no histories: that is "not
    # ready" (503), not a wrong benchmark name (422).
    service.histories = {}
    response = client.get("/risk")
    assert response.status_code == 503
    assert "No saved evaluation history" in response.json()["detail"]


def test_gain_only_window_gives_negative_var(client, service):
    gains = [0.01, 0.02, 0.015, 0.012, 0.011, 0.013, 0.014, 0.016, 0.017, 0.018]
    service.histories["ppo"] = history(gains)
    body = client.get("/risk").json()
    assert body["portfolio"]["var"] < 0 and body["portfolio"]["cvar"] < 0


def test_exact_one_point_change_is_kept_despite_float_noise(client, service):
    frame = service.histories["ppo"]
    # Both are exact 1%p moves; binary floats put one just below 0.01.
    frame.loc[9, "weight_change_spy"] = 0.29 - 0.28  # 0.009999999999999953
    frame.loc[9, "weight_change_gld"] = -(0.51 - 0.5)  # -0.010000000000000009
    assert frame.loc[9, "weight_change_spy"] < 0.01
    changes = client.get("/rebalancing", params={"limit": 1}).json()["events"][0]
    # Both stay in; the larger raw magnitude (GLD) sorts first.
    assert sorted(c["ticker"] for c in changes["changes"]) == ["GLD", "SPY"]


def test_asset_history_refuses_an_inference_data_override(client, monkeypatch):
    monkeypatch.setenv("INVESTMENT_AI_DATA", "other_market.parquet")
    response = client.get("/assets/SPY/history")
    assert response.status_code == 503
    assert "training input" in response.json()["detail"]


def fake_get(payload):
    def get(url, params=None, timeout=None):
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    return get


def raw_get(body):
    """Serve an already-encoded body; json.dumps writes Infinity as-is."""

    def get(url, params=None, timeout=None):
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
            request=httpx.Request("GET", url),
        )

    return get


def test_client_rejects_nonfinite_or_out_of_range_values(monkeypatch):
    from tests.investment_ui_fixtures import REBALANCING, RISK

    monkeypatch.setattr(insights_client.httpx, "get", fake_get(RISK))
    assert insights_client.load_risk()["portfolio"]["var"] == 0.0147
    bad = json.dumps({**RISK, "portfolio": {"var": float("inf"), "cvar": 0.01}})
    # Precondition: the body decodes, so only load_risk's validation can reject it.
    assert math.isinf(json.loads(bad)["portfolio"]["var"])
    monkeypatch.setattr(insights_client.httpx, "get", raw_get(bad))
    with pytest.raises(APIError):
        insights_client.load_risk()
    event = {**REBALANCING["events"][0]}
    event["changes"] = [{"ticker": "SPY", "before": 1.5, "after": 0.1, "change": 0.1}]
    broken = {**REBALANCING, "events": [event]}
    monkeypatch.setattr(insights_client.httpx, "get", fake_get(broken))
    with pytest.raises(APIError):
        insights_client.load_rebalancing()


def test_client_rejects_asset_rows_without_dates(monkeypatch):
    from tests.investment_ui_fixtures import asset_history

    good = asset_history("SPY")
    monkeypatch.setattr(insights_client.httpx, "get", fake_get(good))
    assert insights_client.load_asset_history("SPY")["rows"]
    for broken in (
        {**good, "rows": [{k: v for k, v in good["rows"][0].items() if k != "date"}]},
        {**good, "rows": [{**good["rows"][0], "date": "not-a-date"}]},
        {**good, "as_of": None},
    ):
        monkeypatch.setattr(insights_client.httpx, "get", fake_get(broken))
        with pytest.raises(APIError):
            insights_client.load_asset_history("SPY")
    # A missing close price stays allowed.
    nullable = {**good, "rows": [{**good["rows"][0], "close": None}]}
    monkeypatch.setattr(insights_client.httpx, "get", fake_get(nullable))
    assert insights_client.load_asset_history("SPY")["rows"][0]["close"] is None


def test_client_reports_connection_errors(monkeypatch):
    def down(url, params=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(insights_client.httpx, "get", down)
    for call in (
        insights_client.load_risk,
        insights_client.load_rebalancing,
        lambda: insights_client.load_asset_history("SPY"),
    ):
        with pytest.raises(APIError):
            call()
