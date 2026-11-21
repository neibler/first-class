"""Recover rejected run-scoped sessions without discarding transient-error tokens."""

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from frontend import investment_client as client, api_client

UNKNOWN = "Unknown inference session; create a session first"


def response(status, body):
    return httpx.Response(
        status, json=body, request=httpx.Request("POST", "http://api/optimize")
    )


def test_unknown_session_is_typed_client_error(monkeypatch):
    monkeypatch.setattr(
        httpx, "request", lambda *a, **kw: response(422, {"detail": UNKNOWN})
    )
    with pytest.raises(client.InvestmentAPIError) as caught:
        client.optimize(session_token="old")
    assert type(caught.value).__name__ == "UnknownInvestmentSessionError"


@pytest.mark.parametrize(
    "status,body",
    [
        (422, {"detail": "Invalid weights"}),
        (503, {"detail": UNKNOWN}),
        (422, {"detail": [{"msg": UNKNOWN}]}),
        (422, []),
        (422, {"detail": "Unknown inference session OTHER"}),
    ],
)
def test_other_responses_remain_generic(monkeypatch, status, body):
    monkeypatch.setattr(httpx, "request", lambda *a, **kw: response(status, body))
    with pytest.raises(client.InvestmentAPIError) as caught:
        client.optimize(session_token="old")
    assert type(caught.value) is client.InvestmentAPIError


def prepare_page(monkeypatch):
    from frontend.sample_data import load_portfolio, performance

    def portfolio(*args):
        d = load_portfolio()
        d["display_metrics"] = performance(
            d["returns"], d["periods_per_year"], d["annual_risk_free_rate"]
        )
        d.update(data_source="synthetic", stopped=False, cash_weight=0.0)
        return d

    monkeypatch.setattr(client, "load_portfolio", portfolio)
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=True),
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=20).run()
    app.session_state["investment_session_token"] = "old"
    return app


def click(app):
    next(b for b in app.button if b.label == "목표 비중 계산").click().run()
    assert not app.exception


def test_unknown_token_removed_and_next_click_creates_session(monkeypatch):
    app = prepare_page(monkeypatch)
    calls = []

    def http(method, url, **kw):
        calls.append((url, kw.get("json")))
        if url.endswith("/inference-sessions"):
            return response(200, {"schema_version": "3.0", "session_token": "new"})
        token = kw["json"]["session_token"]
        if token == "old":
            return response(422, {"detail": UNKNOWN})
        return response(
            200,
            {
                "schema_version": "3.0",
                "session_token": "new",
                "as_of": "2026-01-02",
                "execution_delay_trading_bars": 1,
                "target_weights": {"SPY": 1.0},
                "cash_weight": 0.0,
            },
        )

    monkeypatch.setattr(httpx, "request", http)
    click(app)
    assert "investment_session_token" not in app.session_state
    assert len(calls) == 1  # No automatic retry or silent new session.
    assert app.error
    click(app)
    assert app.session_state["investment_session_token"] == "new"
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "optimize",
        "inference-sessions",
        "optimize",
    ]
    assert not app.error
    click(app)
    assert len(calls) == 4 and calls[-1][1]["session_token"] == "new"


@pytest.mark.parametrize(
    "failure", ["timeout", "connection", "server", "validation", "invalid_json"]
)
def test_other_errors_preserve_token_and_do_not_create_session(monkeypatch, failure):
    app = prepare_page(monkeypatch)
    calls = []

    def http(method, url, **kw):
        calls.append((url, kw["json"]["session_token"]))
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        if failure == "connection":
            raise httpx.ConnectError("offline")
        if failure == "invalid_json":
            return httpx.Response(
                422, text="invalid json", request=httpx.Request("POST", url)
            )
        return response(503 if failure == "server" else 422, {"detail": "Other error"})

    monkeypatch.setattr(httpx, "request", http)
    click(app)
    click(app)
    assert app.error
    assert app.session_state["investment_session_token"] == "old"
    assert len(calls) == 2 and all(
        url.endswith("/optimize") and token == "old" for url, token in calls
    )
