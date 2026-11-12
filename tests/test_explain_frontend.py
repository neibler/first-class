"""Saved SHAP HTTP contract and the real Streamlit SHAP page."""

import json
from copy import deepcopy

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from frontend import api_client
from frontend import explain_client as client
from frontend import explain_view as view
from frontend.charts import summary_plot

# Captured before the shared UI fixture swaps in a "not ready" stub.
REAL_GET_OVERVIEW = client.get_overview
NOTE = "SHAP 값은 해당 자산 목표 비중에 대한 기여도입니다. 인과관계가 아닙니다."
# Same labels as backend/performance/shap_analysis.FEATURE_LABELS.
FEATURES = [
    "최근 5일 로그 수익률",
    "RSI (14일)",
    "MACD",
    "20일 변동성",
    "현재 비중",
    "리스크 태그 점수",
]
FEATURES_I = range(len(FEATURES))


def decision(date, step, base=0.15):
    """A synthetic, additive decision: base + sum(SHAP) == final."""
    shap = [round(0.004 * ((step + j) % 5 - 2), 4) for j in range(len(FEATURES))]
    return {
        "date": date,
        "base_value": base,
        "final_value": base + sum(shap),
        "feature_values": [round(((step * 3 + j) % 7) / 6, 4) for j in FEATURES_I],
        "shap_values": shap,
        "feature_values_raw": [round(j * 0.5 - step, 4) for j in FEATURES_I],
    }


@pytest.fixture
def server():
    """API bodies in the server's shape; synthetic values only for tests."""
    features = FEATURES
    assets = {}
    for offset, (ticker, name) in enumerate(
        [("SPY", "미국 대형주"), ("QQQ", "미국 기술주")]
    ):
        assets[ticker] = {
            "ticker": ticker,
            "name": name,
            "features": features,
            "output_unit": "target_weight",
            "note": NOTE,
            "decisions": [
                decision(f"2025-12-{day:02d}", step + offset)
                for step, day in enumerate((26, 29, 30, 31)[offset:])
            ],
        }
    overview = {
        "features": features,
        "feature_importance": [
            {"feature": f, "mean_abs_shap": 0.01 * (i + 1)}
            for i, f in enumerate(features)
        ],
        "assets": list(assets),
        "note": NOTE,
    }
    return {"overview": overview, "assets": assets}


def response(status, body=None, text=None):
    request = httpx.Request("GET", "http://shap.test/performance/explain")
    if text is not None:
        return httpx.Response(status, text=text, request=request)
    return httpx.Response(status, json=body, request=request)


@pytest.fixture
def get(monkeypatch, server):
    calls = []

    def fake(url, params=None, **kwargs):
        calls.append((url, params, kwargs))
        if not params:
            return response(200, server["overview"])
        asset = server["assets"].get(params["ticker"])
        return response(200, asset) if asset else response(404, {"detail": "없음"})

    monkeypatch.setattr(client, "get_overview", REAL_GET_OVERVIEW)
    monkeypatch.setattr(client.httpx, "get", fake)
    monkeypatch.setenv("API_BASE_URL", "http://shap.test/")
    return calls


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
def test_overview_and_asset_requests(get, server):
    overview = client.get_overview()
    ticker = overview.assets[0]
    asset = client.get_asset(ticker, overview.features)
    assert overview.features == server["overview"]["features"]
    assert asset.ticker == ticker and asset.output_unit == "target_weight"
    assert [d.date for d in asset.decisions] == sorted(d.date for d in asset.decisions)
    url, params, kwargs = get[1]
    assert url == "http://shap.test/performance/explain"
    assert params == {"ticker": ticker}
    assert kwargs["timeout"] == client.TIMEOUT
    assert kwargs["follow_redirects"] is False
    assert get[0][1] is None


def test_decisions_are_sorted_by_date(server):
    body = deepcopy(server["assets"]["SPY"])
    body["decisions"].reverse()
    asset = client.parse_asset(body, "SPY", body["features"])
    assert [d.date for d in asset.decisions] == sorted(d.date for d in asset.decisions)


def test_optional_fields_may_be_missing(server):
    overview = deepcopy(server["overview"])
    del overview["feature_importance"]
    assert client.parse_overview(overview).feature_importance is None
    body = deepcopy(server["assets"]["SPY"])
    for decision in body["decisions"]:
        del decision["feature_values_raw"]
    asset = client.parse_asset(body, "SPY", body["features"])
    assert asset.decisions[0].feature_values_raw is None


@pytest.mark.parametrize(
    "result, message, missing",
    [
        (response(503, {"detail": "/secret/path"}), "아직 준비되지", False),
        (response(404, {"detail": "/secret/path"}), "선택한 자산", True),
        (response(500, {"detail": "/secret/path"}), "불러오지 못했습니다", False),
        (response(200, text="not json /secret/path"), "형식이 올바르지", False),
        (httpx.ConnectError("refused /secret/path"), "연결할 수 없습니다", False),
        (httpx.ReadTimeout("slow /secret/path"), "시간이 초과", False),
    ],
)
def test_http_failures_are_safe_messages(monkeypatch, server, result, message, missing):
    def fake(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client.httpx, "get", fake)
    with pytest.raises(client.ExplainAPIError) as error:
        client.get_asset("SPY", server["overview"]["features"])
    assert message in str(error.value)
    assert "/secret/path" not in str(error.value)
    assert error.value.missing is missing


def test_overview_404_is_not_an_asset_message(monkeypatch):
    monkeypatch.setattr(client.httpx, "get", lambda *a, **k: response(404, {}))
    with pytest.raises(client.ExplainAPIError) as error:
        REAL_GET_OVERVIEW()
    assert error.value.missing is False


def broken(server, change):
    body = deepcopy(server["assets"]["SPY"])
    change(body)
    return body


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b["decisions"][0]["shap_values"].pop(),
        lambda b: b["decisions"][0]["feature_values"].pop(),
        lambda b: b["decisions"][0]["feature_values_raw"].pop(),
        lambda b: b["decisions"][0].update(
            final_value=b["decisions"][0]["final_value"] + 0.01
        ),
        lambda b: b["decisions"][0]["feature_values"].__setitem__(0, 1.2),
        lambda b: b["decisions"][0].update(date="2026/01/01"),
        lambda b: b["decisions"].append(deepcopy(b["decisions"][0])),
        lambda b: b.update(output_unit="probability"),
        lambda b: b.update(decisions=[]),
        lambda b: b.update(ticker="QQQ"),
        lambda b: b.update(features=b["features"][::-1]),
        lambda b: b["decisions"][0]["shap_values"].__setitem__(0, "0.1"),
    ],
)
def test_contract_violations_are_rejected(monkeypatch, server, change):
    body = broken(server, change)
    monkeypatch.setattr(client.httpx, "get", lambda *a, **k: response(200, body))
    with pytest.raises(client.ExplainAPIError) as error:
        client.get_asset("SPY", server["overview"]["features"])
    assert "형식이 올바르지" in str(error.value)


def test_additivity_allows_float_rounding(server):
    body = deepcopy(server["assets"]["SPY"])
    body["decisions"][0]["final_value"] += 1e-9
    client.parse_asset(body, "SPY", body["features"])


@pytest.mark.parametrize(
    "change",
    [
        lambda o: o.update(features=[]),
        lambda o: o.update(assets=["SPY", "SPY"]),
        lambda o: o["feature_importance"][0].update(mean_abs_shap=-0.1),
        lambda o: o.pop("note"),
    ],
)
def test_overview_contract_violations(server, change):
    overview = deepcopy(server["overview"])
    change(overview)
    with pytest.raises(ValueError):
        client.parse_overview(overview)


# --------------------------------------------------------------------------- #
# View helpers
# --------------------------------------------------------------------------- #
def test_view_keeps_server_values(server):
    body = server["assets"]["SPY"]
    asset = client.parse_asset(body, "SPY", body["features"])
    records = view.summary_records(asset)
    assert len(records) == len(asset.decisions) * len(asset.features)
    decision = asset.decisions[-1]
    rows = view.detail_rows(asset.features, decision)
    assert [r["SHAP 값"] for r in rows] == decision.shap_values
    assert rows[0]["비중 기여 (%p)"] == pytest.approx(decision.shap_values[0] * 100)
    assert rows[0]["모델 입력 값"] == decision.feature_values_raw[0]
    assert view.chart_decision(decision)["final_value"] == decision.final_value
    assert f"{decision.final_value:.2%}" in view.force_summary(decision)
    assert view.asset_label(asset) == f"SPY · {body['name']}"
    same = asset.model_copy(update={"name": "SPY"})
    assert view.asset_label(same) == "SPY"


def test_summary_colors_encode_feature_values_not_shap_sign(server):
    body = server["assets"]["SPY"]
    asset = client.parse_asset(body, "SPY", body["features"])
    spec = summary_plot(view.summary_records(asset), asset.features).to_dict()
    points = spec["layer"][1]
    assert points["mark"]["type"] == "circle"
    assert points["encoding"]["color"]["field"] == "value"
    assert points["encoding"]["x"]["field"] == "shap"


def test_importance_rows_are_sorted_and_optional(server):
    overview = client.parse_overview(server["overview"])
    rows = view.importance_rows(overview)
    values = [r["평균 |SHAP| (%p)"] for r in rows]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(6.0)
    empty = overview.model_copy(update={"feature_importance": None})
    assert view.importance_rows(empty) == []


# --------------------------------------------------------------------------- #
# Streamlit page
# --------------------------------------------------------------------------- #
@pytest.fixture
def page(monkeypatch, get):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=15).run()
    return app.switch_page("app_pages/shap.py").run()


def test_page_shows_server_shap_without_sample(page, get, server):
    assert not page.exception
    assert len(page.get("page_link")) == 6
    assert page.selectbox(key="shap_asset").value == "SPY"
    spy = server["assets"]["SPY"]["decisions"]
    latest = max(d["date"] for d in spy)
    assert page.selectbox(key="shap_date").value == latest
    assert len(page.get("vega_lite_chart")) == 2
    captions = " ".join(c.value for c in page.caption)
    assert NOTE in captions
    assert "샘플" not in captions and "합성" not in captions
    assert "저장된 SHAP 결과를 조회합니다" in captions
    assert not page.get("badge")
    detail = page.dataframe[0].value
    final = next(d for d in spy if d["date"] == latest)
    assert list(detail["SHAP 값"]) == final["shap_values"]
    assert len(get) == 2
    assert not page.toast, "first visit has no previous choice to replace"


def test_asset_and_date_change_results(page, server):
    before = page.dataframe[0].value.copy()
    page.selectbox(key="shap_date").set_value(
        server["assets"]["SPY"]["decisions"][0]["date"]
    ).run()
    assert not page.exception
    assert not page.dataframe[0].value.equals(before)
    other = list(server["assets"])[1]
    page.selectbox(key="shap_asset").set_value(other).run()
    assert not page.exception
    dates = [d["date"] for d in server["assets"][other]["decisions"]]
    assert page.selectbox(key="shap_date").value in dates


def test_missing_date_falls_back_to_latest(page, server):
    other = list(server["assets"])[1]
    server["assets"][other]["decisions"][0]["date"] = "2099-01-02"
    page.selectbox(key="shap_date").set_value(
        server["assets"]["SPY"]["decisions"][0]["date"]
    ).run()
    page.selectbox(key="shap_asset").set_value(other).run()
    assert not page.exception
    assert page.selectbox(key="shap_date").value == "2099-01-02"
    old = server["assets"]["SPY"]["decisions"][0]["date"]
    assert page.toast[0].value == (
        f"기존 선택 항목(결정일 {old})이 없어 최신 데이터(2099-01-02)로 변경되었습니다."
    )


def test_refresh_switches_a_removed_asset_and_says_so(page, server):
    page.selectbox(key="shap_asset").set_value("QQQ").run()
    assert not page.toast
    del server["assets"]["QQQ"]
    server["overview"]["assets"] = ["SPY"]
    page.button(key="refresh_explain").click().run()
    assert not page.exception
    assert page.selectbox(key="shap_asset").value == "SPY"
    notices = [t.value for t in page.toast]
    assert (
        "기존 선택 항목(자산 QQQ)이 없어 최신 데이터(SPY)로 변경되었습니다." in notices
    )
    assert page.get("vega_lite_chart")


def test_refresh_reads_server_again(page, get, server):
    calls = len(get)
    server["overview"]["note"] = "새 결과"
    server["assets"]["SPY"]["note"] = "새 결과 안내"
    page.button(key="refresh_explain").click().run()
    assert not page.exception
    assert len(get) == calls + 2
    assert "새 결과 안내" in " ".join(c.value for c in page.caption)
    assert not page.toast, "selection still exists, so nothing changes silently"


def test_removed_asset_after_refresh_keeps_navigation(page, server):
    server["overview"]["assets"] = ["SPY", "GONE"]
    page.button(key="refresh_explain").click().run()
    page.selectbox(key="shap_asset").set_value("GONE").run()
    assert not page.exception
    assert "선택한 자산의 SHAP 결과가 없습니다" in page.warning[0].value
    assert len(page.get("page_link")) == 6
    assert page.button(key="refresh_explain")


def test_empty_and_not_ready_states(monkeypatch, page, server):
    server["overview"]["assets"] = []
    page.button(key="refresh_explain").click().run()
    assert page.info[0].value == "표시할 SHAP 해석 결과가 없습니다."
    assert not page.selectbox

    monkeypatch.setattr(client.httpx, "get", lambda *a, **k: response(503, {}))
    page.button(key="refresh_explain").click().run()
    assert not page.exception
    assert "아직 준비되지" in page.warning[0].value
    assert not page.get("vega_lite_chart")
    assert len(page.get("page_link")) == 6


def test_missing_importance_is_explained(page, server):
    del server["overview"]["feature_importance"]
    page.button(key="refresh_explain").click().run()
    assert not page.exception
    captions = " ".join(c.value for c in page.caption)
    assert "전체 피처 중요도는 이 결과에 제공되지 않았습니다" in captions


# --------------------------------------------------------------------------- #
# Real FastAPI router with the server's own payload builder
# --------------------------------------------------------------------------- #
def test_real_router_and_report_payload_match_the_client(tmp_path, monkeypatch, server):
    from backend.performance.api import router
    from backend.performance.report import explain_payload

    shap_payload = {
        "features": server["overview"]["features"],
        "assets": {
            ticker: {
                "name": body["name"],
                "observations": [],
                "decisions": body["decisions"],
            }
            for ticker, body in server["assets"].items()
        },
    }
    importance = pd.DataFrame(
        [
            {"ticker": "SPY", "feature": f, "mean_abs_shap": 0.01}
            for f in shap_payload["features"]
        ]
    )
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        assert api.get("/performance/explain").status_code == 503
        (tmp_path / "api").mkdir()
        (tmp_path / "api/explain.json").write_text(
            json.dumps(explain_payload(shap_payload, importance)), encoding="utf-8"
        )
        assert api.get("/performance/health").json()["available"]["explain"] is True
        overview = client.parse_overview(api.get("/performance/explain").json())
        assert overview.assets == list(server["assets"])
        assert overview.feature_importance
        for ticker in overview.assets:
            body = api.get("/performance/explain", params={"ticker": ticker}).json()
            asset = client.parse_asset(body, ticker, overview.features)
            assert len(asset.decisions) == len(server["assets"][ticker]["decisions"])
        assert (
            api.get("/performance/explain", params={"ticker": "NONE"}).status_code
            == 404
        )
