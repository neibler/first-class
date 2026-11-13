"""Performance page over the investment /backtest API (mock HTTP responses)."""

from copy import deepcopy

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client, performance_client as client
from frontend.performance_view import chart_rows, convention_text, metric_rows

CONVENTION = "simple_daily_returns_annualized_sharpe_positive_mdd"


def metrics(total, sharpe=0.5, stop=False):
    return {
        "days": 3,
        "total_return": total,
        "annualized_return": total * 2,
        "annualized_volatility": 0.14,
        "sharpe": sharpe,
        "max_drawdown": 0.0889,
        "total_turnover": 2.5,
        "total_cost_initial_capital": 0.002,
        "cash_days": 0,
        "stop_triggered": stop,
    }


@pytest.fixture
def answers():
    """One /backtest body per benchmark, shaped like the investment API."""
    dates = ["2025-12-31", "2026-01-02", "2026-01-05"]
    ppo = [0.0, 0.01, 0.0466]
    bench = {
        "spy_buy_hold": [0.0, 0.02, 0.1242],
        "equal_rebalanced": [0.0, 0.015, 0.1056],
        "equal_buy_hold": [0.0, 0.016, 0.1088],
    }
    return {
        key: {
            "schema_version": "3.0",
            "model_version": "ppo_return_mdd_seed22",
            "data_source": "historical_parquet",
            "as_of": "2026-01-05",
            "benchmark": key,
            "periods_per_year": 252,
            "annual_risk_free_rate": 0.0,
            "metric_convention": CONVENTION,
            "assets": [],
            "cash_weight": 0.0,
            "stopped": False,
            "safeguard_triggered": False,
            "allocation_kind": "target",
            "returns": [],
            "path": [
                {"date": d, "portfolio": p, "benchmark": b, "drawdown": 0.0}
                for d, p, b in zip(dates, ppo, values)
            ],
            "metrics": metrics(0.0466),
            "benchmark_metrics": metrics(values[-1], sharpe=1.2),
            "safeguard_applied_to_benchmarks": False,
        }
        for key, values in bench.items()
    }


def respond(monkeypatch, answers, status=200, error=None, statuses=None):
    calls = []

    def get(url, params=None, **kwargs):
        calls.append(params["benchmark"])
        assert url == "http://backtest.test/backtest"
        assert kwargs["follow_redirects"] is False
        if error:
            raise error("PRIVATE SERVER DETAIL")
        code = (statuses or {}).get(params["benchmark"], status)
        body = answers.get(params["benchmark"]) if code == 200 else {"detail": "x"}
        return httpx.Response(code, json=body)

    monkeypatch.setenv("API_BASE_URL", "http://backtest.test/")
    monkeypatch.setattr(client.httpx, "get", get)
    return calls


def test_comparison_combines_benchmarks_in_order(monkeypatch, answers):
    calls = respond(monkeypatch, answers)
    result = client.get_backtest()
    assert calls == [key for key, _ in client.BENCHMARKS]
    assert [s.strategy for s in result.strategies] == [
        "ppo",
        "spy_buy_hold",
        "equal_rebalanced",
        "equal_buy_hold",
    ]
    assert result.strategies[0].label == "PPO (ppo_return_mdd_seed22)"
    assert str(result.base_date) == "2025-12-31"
    assert str(result.start_date) == "2026-01-02"
    assert str(result.end_date) == "2026-01-05"
    assert not result.missing


def test_view_keeps_server_values_and_units(monkeypatch, answers):
    respond(monkeypatch, answers)
    result = client.get_backtest()
    rows = metric_rows(result)
    assert [r["성과지표"] for r in rows] == [label for _, label, _ in client.METRICS]
    first = rows[0]
    assert first["PPO (ppo_return_mdd_seed22)"] == "4.66%"
    assert first["S&P500 (SPY 매수 후 보유)"] == "12.42%"
    by_name = {r["성과지표"]: r for r in rows}
    assert by_name["Sharpe Ratio"]["동일가중 (리밸런싱)"] == "1.200"
    assert by_name["총 회전율"]["PPO (ppo_return_mdd_seed22)"] == "2.50배"
    assert by_name["현금 보유일"]["PPO (ppo_return_mdd_seed22)"] == "0일"
    points = chart_rows(result)
    assert len(points) == 4 * 3
    assert points[2]["cumulative_return"] == 0.0466
    assert "MDD는 손실 크기" in convention_text(result)


def test_none_sharpe_is_shown_as_dash(monkeypatch, answers):
    for body in answers.values():
        body["metrics"]["sharpe"] = None
    respond(monkeypatch, answers)
    rows = {r["성과지표"]: r for r in metric_rows(client.get_backtest())}
    assert rows["Sharpe Ratio"]["PPO (ppo_return_mdd_seed22)"] == "—"


def test_missing_benchmark_is_listed_not_fatal(monkeypatch, answers):
    respond(monkeypatch, answers, statuses={"equal_buy_hold": 422})
    result = client.get_backtest()
    assert len(result.strategies) == 3
    assert result.missing == ["동일가중 (매수 후 보유)"]


@pytest.mark.parametrize(
    "case",
    [
        lambda a: a["spy_buy_hold"].update(schema_version="2.0"),
        lambda a: a["equal_rebalanced"].update(model_version="other_model"),
        lambda a: a["equal_rebalanced"].update(benchmark="spy_buy_hold"),
        lambda a: a["equal_buy_hold"]["path"].pop(),
        lambda a: a["equal_buy_hold"]["metrics"].update(total_return=0.5),
        lambda a: a["spy_buy_hold"]["path"].reverse(),
        lambda a: a["spy_buy_hold"]["benchmark_metrics"].pop("sharpe"),
        lambda a: a["spy_buy_hold"].update(path=[]),
    ],
)
def test_invalid_or_inconsistent_answers_are_rejected(monkeypatch, answers, case):
    broken = deepcopy(answers)
    case(broken)
    respond(monkeypatch, broken)
    with pytest.raises(client.PerformanceAPIError) as error:
        client.get_backtest()
    assert "형식이 올바르지" in str(error.value)


def test_no_benchmark_at_all_is_invalid(monkeypatch, answers):
    respond(monkeypatch, answers, status=422)
    with pytest.raises(client.PerformanceAPIError):
        client.get_backtest()


def open_page(monkeypatch):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=True),
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=30)
    return app.switch_page("app_pages/performance.py").run()


def test_page_renders_comparison(monkeypatch, answers):
    respond(monkeypatch, answers)
    app = open_page(monkeypatch)
    assert not app.exception and not app.warning
    assert len(app.get("page_link")) == 7
    assert len(app.get("vega_lite_chart")) == 1
    assert app.dataframe[0].value.shape == (len(client.METRICS), 5)
    captions = " ".join(c.value for c in app.caption)
    assert (
        "모델: ppo_return_mdd_seed22 · 평가 기간: 2026-01-02 ~ 2026-01-05 (시작 기준일 2025-12-31)"
        in captions
    )
    assert "활성 투자 모델의 백테스트 API" in captions


@pytest.mark.parametrize(
    "status, error, message",
    [
        (503, None, "아직 준비"),
        (500, None, "불러오지 못"),
        (200, httpx.ConnectError, "연결할 수"),
        (200, httpx.ReadTimeout, "시간이 초과"),
    ],
)
def test_http_failure_keeps_menu_and_recovers(
    monkeypatch, answers, status, error, message
):
    respond(monkeypatch, answers, status, error)
    app = open_page(monkeypatch)
    assert not app.exception
    assert len(app.get("page_link")) == 7
    assert message in app.warning[0].value
    assert "PRIVATE" not in app.warning[0].value
    assert not app.dataframe
    respond(monkeypatch, answers)
    app.button(key="refresh_performance").click().run()
    assert not app.exception and not app.warning
    assert app.dataframe


def test_safeguard_synthetic_and_missing_notices(monkeypatch, answers):
    for body in answers.values():
        body.update(data_source="synthetic", safeguard_triggered=True)
    respond(monkeypatch, answers, statuses={"equal_buy_hold": 422})
    app = open_page(monkeypatch)
    warnings = " ".join(w.value for w in app.warning)
    assert "합성 데이터" in warnings and "Safe-Guard" in warnings
    assert any("동일가중 (매수 후 보유)" in i.value for i in app.info)
