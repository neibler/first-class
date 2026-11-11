import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client


@pytest.fixture
def page(monkeypatch):
    from frontend import shap_client
    from tests.test_model_shap import payload

    monkeypatch.setattr(shap_client, "load_model_shap", lambda model: payload(model))
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    return AppTest.from_file("frontend/app.py", default_timeout=10).run()


def test_navigation_all_six_pages_and_placeholder_scope(page):
    assert not page.exception
    assert page.title[0].value == "포트폴리오 현황"
    for slug, title in [
        ("performance", "강화학습 성과"),
        ("shap", "SHAP 해석"),
        ("research", "에이전트 리서치"),
        ("anova", "ANOVA 검증"),
        ("risk", "리스크 모니터링"),
    ]:
        page.switch_page(f"app_pages/{slug}.py").run()
        assert not page.exception
        assert page.title[0].value == title
        assert len(page.get("page_link")) == 7
        if slug == "research":
            assert page.text_area(key="research_question")
            assert page.date_input(key="research_date")
            assert not page.selectbox
        elif slug != "shap":
            assert page.info[0].value == "화면 준비 중"
            assert not page.selectbox
            captions = " ".join(item.value for item in page.caption)
            assert "분석 API와 데이터는 아직 준비되지 않았습니다" in captions
            assert "위 분석은 UI 개발용 샘플" not in captions
    page.switch_page("app_pages/portfolio.py").run()
    assert page.title[0].value == "포트폴리오 현황"


def test_period_and_shap_filters_change_results(page):
    before = page.metric[0].value
    page.selectbox(key="period").set_value("최근 63개 관측일").run()
    assert not page.exception
    assert page.metric[0].value != before
    page.switch_page("app_pages/shap.py").run()
    page.button(key="shap_details_button").click().run()
    before = str(page.dataframe[0].value)
    page.selectbox(key="shap_asset").set_value("QQQ").run()
    page.selectbox(key="shap_date").set_value("2025-01-02").run()
    assert not page.exception
    assert str(page.dataframe[0].value) != before


def test_real_shap_unavailable_has_no_sample_fallback(page, monkeypatch):
    from frontend import shap_client

    def unavailable(model):
        raise api_client.APIError("실제 SHAP 결과를 불러오지 못했습니다.")

    monkeypatch.setattr(shap_client, "load_model_shap", unavailable)
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    assert page.error
    assert not page.dataframe
    assert page.button(key="retry_shap")


def test_state_query_is_ignored_without_dev_flag(page, monkeypatch):
    monkeypatch.delenv("FIRST_CLASS_UI_DEV", raising=False)
    page.query_params["ui_state"] = "error"
    page.run()
    assert not page.exception
    assert not page.error
    assert page.metric


def test_portfolio_api_failure_is_visible(page, monkeypatch):
    from frontend import investment_client

    def unavailable(*args, **kwargs):
        raise investment_client.InvestmentAPIError(
            "투자 분석 결과를 불러오지 못했습니다."
        )

    monkeypatch.setattr(investment_client, "load_portfolio", unavailable)
    page.switch_page("app_pages/portfolio.py").run()
    assert not page.exception
    assert page.error
    assert not page.metric
    assert len(page.get("page_link")) == 7


def test_shap_raw_features_are_opt_in_and_correlation_separate(page):
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    assert not page.dataframe
    assert not page.get("progress")
    assert not any("이번 결정에서" in c.value for c in page.caption)
    page.button(key="shap_details_button").click().run()
    assert not page.exception
    assert "macro_vix" in str(page.dataframe[2].value)
    assert "absolute contribution (%p)" in page.dataframe[2].value.columns
    assert any("Rolling Correlation" in c.value for c in page.caption)
    page.button(key="shap_details_button").click().run()
    assert not page.dataframe


def test_shap_market_state_and_raw_detail_values(page, monkeypatch):
    import math
    from frontend import shap_client
    from tests.test_model_shap import payload

    data = payload()
    features = ["log_return_lag_0_gld", "rsi_div_100_gld", "macro_vix"]
    data["features"] = features
    row = data["assets"]["SPY"]["decisions"][-1]
    row.update(
        feature_values_raw=[9.0, 0.31, -8.0],
        shap_values=[0.03, 0.02, -0.01],
        final_value=0.14,
    )
    row["market_context"] = {
        "source": "verified_market_data",
        "as_of": row["date"],
        "values": dict(zip(features, [math.log(0.98), 0.31, 20.0])),
        "previous_values": {},
        "comparison_date": None,
    }
    monkeypatch.setattr(shap_client, "load_model_shap", lambda model: data)
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    captions = " ".join(c.value for c in page.caption)
    assert "수익률 -2.00%" in captions and "RSI 31" in captions
    assert not page.dataframe
    page.button(key="shap_details_button").click().run()
    assert not page.exception
    raw = page.dataframe[2].value
    assert raw["raw feature"].tolist() == features
    assert raw["model input value"].tolist() == [9.0, 0.31, -8.0]
    assert raw["shap value"].tolist() == [0.03, 0.02, -0.01]
    assert "낮추는 방향" in str(page.dataframe[0].value["설명"].tolist())


def test_shap_contributions_are_not_trade_directions(page):
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    assert not page.metric
    captions = " ".join(x.value for x in page.caption)
    assert "거래 기록을 확인하지 못했습니다" in captions
    page.button(key="shap_details_button").click().run()
    assert any(m.label == "SHAP 기준 비중" for m in page.metric)


@pytest.mark.parametrize(
    "before,target,direction",
    [(0.3, 0.14, "낮추는"), (0.1, 0.14, "높이는"), (0.14, 0.14, "현재 SPY")],
)
def test_decision_focused_shap_ui(page, monkeypatch, before, target, direction):
    from frontend import shap_client
    from tests.test_model_shap import payload

    data = payload()
    data["features"] = ["macd_gld", "macro_vix"]
    row = data["assets"]["SPY"]["decisions"][-1]
    row.update(shap_values=[0.06, -0.02], final_value=0.14)
    row["execution_context"] = {
        "previous_weight": before,
        "target_weight": target,
        "current_weight": target,
        "traded": before != target,
        "execution_date": "2025-01-06",
    }
    monkeypatch.setattr(shap_client, "load_model_shap", lambda model: data)
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    assert direction in " ".join(x.value for x in page.subheader)
    assert [m.value for m in page.metric] == [f"{before:.2%}", "14.00%", "14.00%"]
    assert not any(m.label == "SHAP 기준 비중" for m in page.metric)
    displayed = " ".join(x.value for x in page.markdown)
    if before > target:
        assert "시장 변동성" in displayed and "금(GLD) 관련 신호" not in displayed
    elif before < target:
        assert "금(GLD) 관련 신호" in displayed and "시장 변동성" not in displayed
    page.button(key="shap_details_button").click().run()
    assert not page.exception
    assert any(m.label == "SHAP 기준 비중" for m in page.metric)
    assert "macd_gld" in page.dataframe[2].value["raw feature"].tolist()
    assert page.dataframe[2].value["shap value"].tolist() == [0.06, -0.02]
    assert len(page.dataframe[0].value) == 2
    assert any(
        e.label == "反" or e.label == "반대 방향으로 작용한 요인 보기"
        for e in page.expander
    )
