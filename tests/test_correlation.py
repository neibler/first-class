from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from backend.performance import correlation as module
from backend.performance.correlation import (
    analyze_correlations,
    build_series,
    calculate_rolling_correlation,
    describe_correlation,
    describe_correlation_change,
    rank_correlations,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.01, 600)
    return SimpleNamespace(
        dates=pd.bdate_range("2023-01-02", periods=600),
        assets=["SPY", "TLT", "GLD", "SHY"],
        returns=np.column_stack([x, -x, rng.normal(0, 0.01, 600), np.zeros(600)]),
        macros=["ust_10y", "vix", "wti", "core_pce_yoy", "usd_ret_20d"],
        macro=np.column_stack(
            [
                4 - np.cumsum(x),
                20 + np.cumsum(x),
                70 * np.exp(np.cumsum(x)),
                np.ones(600) * 3,
                np.ones(600) * 0.1,
            ]
        ),
    )


def test_series_use_returns_and_changes_not_levels(market):
    series, metadata, excluded = build_series(market)
    np.testing.assert_equal(series["SPY"], market.returns[:, 0])
    np.testing.assert_allclose(
        series["ust_10y_change"].iloc[1:], np.diff(market.macro[:, 0])
    )
    np.testing.assert_allclose(
        series["vix_change"].iloc[1:], np.diff(market.macro[:, 1])
    )
    np.testing.assert_allclose(
        series["wti_return"].iloc[1:], np.expm1(market.returns[1:, 0])
    )
    assert {r["name"] for r in excluded} == {"core_pce_yoy", "usd_ret_20d"}
    assert metadata["SPY"]["transformation"] == "daily_log_return"


@pytest.mark.parametrize("window", [20, 60, 120, 252])
def test_windows_self_exclusion_rank_and_rolling(market, window):
    result = analyze_correlations(market, "SPY", window)
    rows = result["relationships"]
    assert len(rows) == 5
    assert not any(r["name"] in {"SPY", "SHY"} for r in rows)
    magnitude = [abs(r["correlation"]) for r in rows]
    assert magnitude == sorted(magnitude, reverse=True)
    assert next(r for r in rows if r["name"] == "TLT")["correlation"] == pytest.approx(
        -1
    )
    actual = calculate_rolling_correlation(
        pd.Series(market.returns[:, 0]), pd.Series(market.returns[:, 2]), window
    )
    expected = (
        pd.Series(market.returns[:, 0])
        .rolling(window)
        .corr(pd.Series(market.returns[:, 2]))
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert all(r["observations"] == window for r in rows)
    assert all("previous_correlation" in r for r in rows)
    assert all(
        not any(word in r["description"] for word in ["때문", "반드시", "항상"])
        for r in rows
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "value,strength",
    [
        (0.19, "매우 약한 관계"),
        (0.2, "약한 관계"),
        (0.4, "보통 관계"),
        (0.6, "강한 관계"),
        (0.8, "매우 강한 관계"),
        (-0.8, "매우 강한 관계"),
    ],
)
def test_labels(value, strength):
    direction, actual, label = describe_correlation(value)
    assert actual == strength
    assert direction == ("positive" if value > 0 else "negative")
    assert label == ("같이 움직이는 경향" if value > 0 else "반대로 움직이는 경향")


def test_insufficient_nan_and_constant_are_not_zero_correlations(market):
    short = analyze_correlations(market, "SPY", 60, as_of=str(market.dates[19].date()))
    assert short["status"] == "insufficient_data" and not short["relationships"]
    market.returns[-1, 1] = np.nan
    result = analyze_correlations(market, "SPY", 60)
    assert "TLT" not in [r["name"] for r in result["all_relationships"]]
    assert "SHY" not in [r["name"] for r in result["all_relationships"]]
    assert all(np.isfinite(r["correlation"]) for r in result["all_relationships"])


def test_as_of_never_uses_future(market):
    cutoff = str(market.dates[300].date())
    before = analyze_correlations(market, "SPY", 60, as_of=cutoff)
    market.returns[301:] = 99
    market.macro[301:] = 999
    assert analyze_correlations(market, "SPY", 60, as_of=cutoff) == before
    assert all(
        x["date"] <= cutoff for series in before["rolling"].values() for x in series
    )


def test_change_comparison_and_topk():
    assert "강해" in describe_correlation_change(-0.68, -0.35)
    assert "약해" in describe_correlation_change(-0.25, -0.7)
    assert "방향이 바뀌" in describe_correlation_change(0.2, -0.7)
    assert describe_correlation_change(0.2, float("nan")) is None
    assert [
        r["name"]
        for r in rank_correlations(
            [{"name": "a", "correlation": 0.1}, {"name": "b", "correlation": -0.9}], 1
        )
    ] == ["b"]


def test_api_parameters_and_errors(market, monkeypatch):
    from backend.performance.api import router, correlation

    monkeypatch.setattr(module, "load_correlation_data", lambda: market)
    app = FastAPI()
    app.include_router(router)
    assert "/performance/correlation/{ticker}" in app.openapi()["paths"]
    assert correlation("TLT", 60, 5)["status"] == "ok"
    for args, status in [
        (("TLT", 30, 5), 422),
        (("UNKNOWN", 60, 5), 404),
        (("TLT", 60, 5, "20250101"), 422),
    ]:
        with pytest.raises(HTTPException) as error:
            correlation(*args)
        assert error.value.status_code == status
    assert correlation("TLT", 60, 5, "2020-01-01")["status"] == "insufficient_data"

    def missing():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(module, "load_correlation_data", missing)
    with pytest.raises(HTTPException) as error:
        correlation("TLT", 60, 5)
    assert error.value.status_code == 503


def test_correlation_ui_has_real_payload_and_details(market, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import correlation_client

    monkeypatch.setattr(
        correlation_client,
        "load_correlations",
        lambda ticker, window: analyze_correlations(market, ticker, window),
    )
    page = AppTest.from_file(
        "frontend/app_pages/correlation.py", default_timeout=20
    ).run()
    assert not page.exception
    assert page.metric[0].label == "현재 구간 상관계수"
    assert len(page.metric) == 3
    page.selectbox(key="correlation_pair").set_value("TLT").run()
    assert not page.exception
    assert any("정규화 가격 변화 비교" in m.value for m in page.markdown)
    assert any("60거래일 Rolling Correlation 변화" in m.value for m in page.markdown)
    assert len(page.selectbox(key="correlation_pair").options) == 5
    page.selectbox(key="correlation_window").set_value(252).run()
    assert not page.exception
    page.selectbox(key="correlation_asset").set_value("TLT").run()
    assert not page.exception
    assert len(page.dataframe) == 1

    def unavailable(*args):
        from frontend.api_client import APIError

        raise APIError("실제 시장 데이터 없음")

    monkeypatch.setattr(correlation_client, "load_correlations", unavailable)
    page.run()
    assert not page.exception and page.error
    assert not page.metric
