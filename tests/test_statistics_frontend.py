"""Saved statistics HTTP contract and real Streamlit page behavior."""

from copy import deepcopy
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from frontend import api_client, statistics_client as client, statistics_view as view


@pytest.fixture
def payload():
    def values(p, eta=0.2):
        return {
            "f_statistic": 4.25,
            "p_value": p,
            "eta_squared": eta,
            "effect_size_label": "큼",
            "significant": p < 0.05,
        }

    def one_way(group, groups, p):
        return {
            "test": "one_way_anova",
            "label": f"{group} 비교",
            "groups_label": group,
            "groups": groups,
            "n_per_group": {g: 8 for g in groups},
            "group_mean": {g: (i + 1) / 100 for i, g in enumerate(groups)},
            "group_std": {g: 0.02 for g in groups},
            "posthoc": None,
            "sample_source": "테스트 표본 출처",
            **values(p),
        }

    tukey = {
        "test": "tukey_hsd",
        "alpha": 0.05,
        "value": "period_return",
        "pairs": [
            {
                "group1": "bull",
                "group2": "sideways",
                "mean_difference": -0.0116,
                "p_adjusted": 0.0344,
                "lower": -0.0222,
                "upper": -0.0009,
                "significant": True,
            }
        ],
    }
    return {
        "results": [
            one_way("reward", ["return", "sharpe", "return_mdd"], 0.3289),
            one_way("strategy", ["ppo", "mvo", "equal_weight"], 0.7096),
            {
                "test": "two_way_anova",
                "label": "전략 × 시장 국면 비교",
                "value": "period_return",
                "factors": ["strategy", "regime"],
                "terms": {
                    "strategy": values(0.68),
                    "regime": {**values(0.049397), "partial_eta_squared": 0.1978},
                    "strategy:regime": values(0.80),
                },
                "interaction_included": True,
                "n": 24,
                "cell_counts": {"('ppo', 'bull')": 4},
                "posthoc": {"regime": tukey},
                "posthoc_note": "시계열 표본의 독립성에 주의하세요.",
            },
        ]
    }


def response(data, status=200):
    return httpx.Response(
        status,
        json=data,
        request=httpx.Request("GET", "http://stats.test/performance/statistics"),
    )


@pytest.fixture
def get(monkeypatch, payload):
    mock = Mock(return_value=response(payload))
    monkeypatch.setattr(client.httpx, "get", mock)
    monkeypatch.setenv("API_BASE_URL", "http://stats.test/")
    return mock


def test_http_contract_and_server_values(get):
    result = client.get_statistics()
    assert len(result.results) == 3
    assert result.results[0].groups_label == "reward"
    assert result.results[2].terms["regime"].p_value == 0.049397
    assert get.call_count == 1
    assert get.call_args.args == ("http://stats.test/performance/statistics",)
    assert get.call_args.kwargs["timeout"].connect == 3
    assert get.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    "status,message", [(503, "준비되지"), (500, "불러오지"), (404, "불러오지")]
)
def test_http_failure_is_safe_and_not_retried(get, status, message):
    get.return_value = response({"detail": "PRIVATE /server/path"}, status)
    with pytest.raises(client.StatisticsAPIError, match=message) as caught:
        client.get_statistics()
    assert "PRIVATE" not in str(caught.value)
    assert get.call_count == 1


@pytest.mark.parametrize(
    "error,message",
    [(httpx.ConnectError, "연결할 수"), (httpx.ReadTimeout, "시간이 초과")],
)
def test_network_failure(get, error, message):
    get.side_effect = error("PRIVATE")
    with pytest.raises(client.StatisticsAPIError, match=message):
        client.get_statistics()
    assert get.call_count == 1


@pytest.mark.parametrize(
    "bad",
    [None, {}, {"results": None}, {"results": [{}]}, {"results": [1]}],
)
def test_invalid_envelope(get, bad):
    get.return_value = response(bad)
    with pytest.raises(client.StatisticsAPIError, match="형식"):
        client.get_statistics()


def test_invalid_json(get):
    get.return_value = httpx.Response(200, text="<html>PRIVATE</html>")
    with pytest.raises(client.StatisticsAPIError, match="형식"):
        client.get_statistics()


@pytest.mark.parametrize(
    "field,value", [("p_value", 1.1), ("p_value", "0.03"), ("eta_squared", True)]
)
def test_invalid_metric_rejected(get, payload, field, value):
    payload["results"][0][field] = value
    get.return_value = response(payload)
    with pytest.raises(client.StatisticsAPIError, match="형식"):
        client.get_statistics()


def test_group_mapping_must_match(get, payload):
    payload["results"][0]["group_mean"].pop("sharpe")
    get.return_value = response(payload)
    with pytest.raises(client.StatisticsAPIError, match="형식"):
        client.get_statistics()


def test_empty_and_unknown_tests(get, payload):
    get.return_value = response({"results": []})
    assert client.get_statistics().results == []
    payload["results"].insert(0, {"test": "future_test"})
    get.return_value = response(payload)
    result = client.get_statistics()
    assert len(result.results) == 3
    assert result.unsupported_count == 1


def test_null_values_and_rounding_do_not_change_verdict(payload):
    item = payload["results"][0]
    item.update(f_statistic=None, p_value=None, eta_squared=None)
    result = client.parse_statistics(payload).results[0]
    row = view.test_row("검증", result)
    assert row["F값"] == row["p값"] == row["η²"] == "—"
    assert row["판정"] == "계산 불가"
    assert "판단할 수 없습니다" in view.posthoc_status(result, None)
    item.update(f_statistic=4.2, p_value=0.04999999, significant=True)
    result = client.parse_statistics(payload).results[0]
    assert view.test_row("검증", result)["p값"] == "0.0500"
    assert view.verdict(result) == "유의"
    assert "제공되지" in view.posthoc_status(result, None)
    assert view.number(1e-9) == "1.000e-09"


def test_both_tukey_shapes_and_direction(payload):
    one = payload["results"][0]
    one.update(p_value=0.001, significant=True)
    one["posthoc"] = deepcopy(payload["results"][2]["posthoc"]["regime"])
    result = client.parse_statistics(payload)
    for tukey in (result.results[0].posthoc, result.results[2].posthoc["regime"]):
        row = view.tukey_rows(tukey)[0]
        assert row["평균 차이 (%p)"] == "-1.16"
        assert row["보정 p값"] == "0.0344"
        assert row["신뢰구간 하한 (%p)"] == "-2.22"
        assert row["신뢰구간 상한 (%p)"] == "-0.09"
    result.results[0].posthoc.pairs[0].significant = False
    assert "유의한 비교 쌍이 없습니다" in view.posthoc_status(
        result.results[0], result.results[0].posthoc
    )


@pytest.fixture
def page(monkeypatch, get):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=15).run()
    return app.switch_page("app_pages/anova.py").run()


def table_text(page):
    return "\n".join(table.value.to_string(index=False) for table in page.dataframe)


def test_page_renders_results_and_refreshes_without_model(page, get, payload):
    assert not page.exception
    assert len(page.get("page_link")) == 7
    assert len(page.get("vega_lite_chart")) == 2
    text = table_text(page)
    for value in ("0.3289", "0.7096", "0.0494", "0.1978", "-1.16", "0.0344"):
        assert value in text
    assert "API와 데이터는 아직 준비되지" not in " ".join(c.value for c in page.caption)
    assert get.call_count == 1
    payload["results"][0]["p_value"] = 0.4567
    get.return_value = response(payload)
    page.button(key="refresh_statistics").click().run()
    assert not page.exception
    assert "0.4567" in table_text(page)
    assert get.call_count == 2
    page.switch_page("app_pages/risk.py").run()
    page.switch_page("app_pages/anova.py").run()
    assert not page.exception
    assert len(page.get("page_link")) == 7
    assert get.call_count == 3


def test_partial_reordered_results_and_unknown_type(page, get, payload):
    two = payload["results"][2]
    two["interaction_included"] = False
    two["terms"].pop("strategy:regime")
    get.return_value = response(
        {"results": [two, {"test": "future"}, payload["results"][1]]}
    )
    page.button(key="refresh_statistics").click().run()
    assert not page.exception
    assert any("지원하지 않는" in w.value for w in page.warning)
    info = " ".join(i.value for i in page.info)
    assert "보상 함수별 검증 결과가 제공되지" in info
    assert "상호작용 항이 포함되지" in info
    assert "0.7096" in table_text(page)
    assert "하락장" not in table_text(page)


@pytest.mark.parametrize(
    "data,status,message",
    [
        ({"results": []}, 200, "표시할 통계"),
        ({}, 503, "아직 준비"),
        ({}, 200, "응답 형식"),
    ],
)
def test_page_failures_keep_refresh_and_navigation(page, get, data, status, message):
    get.return_value = response(data, status)
    page.button(key="refresh_statistics").click().run()
    assert not page.exception
    assert not page.dataframe
    assert message in " ".join(i.value for i in [*page.info, *page.warning])
    assert len(page.get("page_link")) == 7
    assert page.button(key="refresh_statistics")


def test_real_router_reads_saved_statistics_without_other_results(
    tmp_path, monkeypatch, payload
):
    import json
    from backend.performance.api import router

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        assert api.get("/performance/statistics").status_code == 503
        (tmp_path / "statistics").mkdir()
        (tmp_path / "statistics/anova.json").write_text(
            json.dumps(payload["results"]), encoding="utf-8"
        )
        health = api.get("/performance/health").json()
        assert health["available"]["statistics"] is True
        assert health["ready"] is False
        data = api.get("/performance/statistics")
        assert data.status_code == 200
        assert data.json() == payload
        assert len(client.parse_statistics(data.json()).results) == 3
