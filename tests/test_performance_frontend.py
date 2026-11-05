"""HTTP contract, server value formatting, and page recovery (mock responses)."""

from copy import deepcopy

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client, performance_client as client
from frontend.performance_view import chart_rows, evaluation_text, metric_rows


@pytest.fixture
def payload():
    names = ["ppo", "mvo", "equal_weight", "benchmark", "benchmark_raw", "market_index"]
    return {
        "schema_version": "performance-1",
        "start_date": "2025-05-28",
        "end_date": "2025-05-29",
        "evaluation": "single_holdout",
        "smoke": False,
        "strategies": [
            {
                "strategy": name,
                "label": f"서버 라벨 {name}",
                "metrics": {key: 0.08 for key, _, _ in client.METRICS},
            }
            for name in names
        ],
        "path": [{"date": "2025-05-28", **{name: {"value": 1.08} for name in names}}],
    }


def respond(monkeypatch, payload=None, status=200, error=None):
    def get(url, **kwargs):
        assert url == "http://backtest.test/performance/backtest"
        assert "params" not in kwargs  # all strategies and default daily path
        if error:
            raise error("PRIVATE SERVER DETAIL")
        return httpx.Response(status, json=payload)

    monkeypatch.setenv("API_BASE_URL", "http://backtest.test/")
    monkeypatch.setattr(client.httpx, "get", get)


def test_server_values_order_units_and_labels(monkeypatch, payload):
    payload["strategies"][0]["metrics"]["beta"] = None
    payload["metric_labels"] = {"cagr": "서버 CAGR"}
    respond(monkeypatch, payload)
    result = client.get_backtest()
    rows = chart_rows(result)
    assert len(rows) == 6
    assert all(row["cumulative_return"] == pytest.approx(0.08) for row in rows)
    assert [row["label"] for row in rows] == [s["label"] for s in payload["strategies"]]
    table = metric_rows(result)
    assert len(table) == 12 and len(table[0]) == 7
    assert table[0]["서버 라벨 ppo"] == "8.00%"
    assert table[1]["성과지표"] == "서버 CAGR (%)"
    assert table[6]["서버 라벨 ppo"] == "0.080"
    assert table[10]["서버 라벨 ppo"] == "—"


def test_metric_order_matches_backend():
    from backend.performance.metrics import METRIC_LABELS

    assert [key for key, _, _ in client.METRICS] == list(METRIC_LABELS)


def test_duplicate_labels_remain_separate(payload):
    for strategy in payload["strategies"]:
        strategy["label"] = "벤치마크"
    result = client.BacktestResult.model_validate(payload)
    assert len(metric_rows(result)[0]) == 7
    assert len({row["label"] for row in chart_rows(result)}) == 6


@pytest.mark.parametrize(
    "case", ["schema", "missing_metric", "nan", "path", "date", "duplicate"]
)
def test_invalid_contract_is_user_error(monkeypatch, payload, case):
    if case == "schema":
        payload["schema_version"] = "3.0"
    elif case == "missing_metric":
        del payload["strategies"][0]["metrics"]["cagr"]
    elif case == "nan":
        payload["strategies"][0]["metrics"]["cagr"] = "NaN"
    elif case == "path":
        del payload["path"][0]["ppo"]
    elif case == "date":
        payload["path"][0]["date"] = "2024-01-01"
    else:
        payload["strategies"].append(deepcopy(payload["strategies"][0]))
    respond(monkeypatch, payload)
    with pytest.raises(client.PerformanceAPIError, match="응답 형식"):
        client.get_backtest()


@pytest.mark.parametrize(
    "status,error,message",
    [
        (503, None, "아직 준비"),
        (500, None, "불러오지 못"),
        (200, httpx.ConnectError, "연결할 수"),
        (200, httpx.ReadTimeout, "시간이 초과"),
    ],
)
def test_http_failure_keeps_menu_and_recovers(
    monkeypatch, payload, status, error, message
):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    respond(monkeypatch, payload, status, error)
    app = AppTest.from_file("frontend/app.py", default_timeout=30)
    app.switch_page("app_pages/performance.py").run()
    assert not app.exception
    assert len(app.get("page_link")) == 6
    assert message in app.warning[0].value
    assert "PRIVATE" not in app.warning[0].value
    assert not app.dataframe
    respond(monkeypatch, payload)
    app.button(key="refresh_performance").click().run()
    assert not app.exception and not app.warning
    assert len(app.get("vega_lite_chart")) == 1
    assert app.dataframe[0].value.shape == (12, 7)
    assert any(
        "2025-05-28 ~ 2025-05-29 · 평가 방식: 단일 홀드아웃" in c.value
        for c in app.caption
    )
    respond(monkeypatch, payload, status=503)
    app.button(key="refresh_performance").click().run()
    assert not app.dataframe and not app.get("vega_lite_chart")


@pytest.mark.parametrize("empty_strategies", [True, False])
def test_empty_results_keep_menu(monkeypatch, payload, empty_strategies):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    payload["path"] = []
    if empty_strategies:
        payload["strategies"] = []
    respond(monkeypatch, payload)
    app = AppTest.from_file("frontend/app.py", default_timeout=30)
    app.switch_page("app_pages/performance.py").run()
    assert not app.exception and app.info
    assert len(app.get("page_link")) == 6
    assert not app.get("vega_lite_chart")
    assert bool(app.dataframe) is not empty_strategies


def test_walk_forward_and_smoke_metadata(monkeypatch, payload):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    payload.update(
        evaluation="walk_forward_stitched_out_of_sample",
        smoke=True,
        window_plan={
            "used": {"train_days": 1008, "test_days": 252, "folds": 2},
            "warnings": ["테스트 설정 안내"],
        },
    )
    respond(monkeypatch, payload)
    app = AppTest.from_file("frontend/app.py", default_timeout=30)
    app.switch_page("app_pages/performance.py").run()
    assert not app.exception
    assert any("검증용 smoke" in w.value for w in app.warning)
    assert any("Walk-Forward" in c.value for c in app.caption)
    assert any("1008거래일" in c.value and "2개 구간" in c.value for c in app.caption)
    assert any(w.value == "테스트 설정 안내" for w in app.warning)
    payload["window_plan"]["used"]["folds"] = 1
    assert "단일 홀드아웃" in evaluation_text(
        client.BacktestResult.model_validate(payload)
    )
