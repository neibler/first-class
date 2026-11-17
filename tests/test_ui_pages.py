import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client, performance_client, statistics_client


@pytest.fixture
def page(monkeypatch):
    def no_backtest():
        raise performance_client.PerformanceAPIError(
            "백테스트 분석 결과가 아직 준비되지 않았습니다."
        )

    monkeypatch.setattr(performance_client, "get_backtest", no_backtest)

    from frontend import shap_client
    from tests.test_model_shap import payload

    monkeypatch.setattr(shap_client, "load_model_shap", lambda model: payload(model))
    monkeypatch.setattr(
        statistics_client,
        "get_statistics",
        lambda: statistics_client.StatisticsResult([]),
    )
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    return AppTest.from_file("frontend/app.py", default_timeout=10).run()


def test_menu_pages_and_detail_pages(page):
    assert not page.exception
    assert page.title[0].value == "포트폴리오"
    # (page, title, links): 4 menu links + detail links on menu pages,
    # or + 1 back link on detail pages.
    for slug, title, links in [
        ("risk", "위험도·설명", 6),
        ("correlation", "자산 상세", 4),
        ("research", "AI 도우미", 4),
        ("performance", "강화학습 성과", 5),
        ("shap", "SHAP 해석", 5),
        ("anova", "ANOVA 검증", 5),
    ]:
        page.switch_page(f"app_pages/{slug}.py").run()
        assert not page.exception
        assert page.title[0].value == title
        assert len(page.get("page_link")) == links
        assert any(e.label == "API 준비 상태" for e in page.expander)
        if slug == "research":
            assert page.text_area(key="research_question")
            assert page.date_input(key="research_date")
        elif slug == "performance":
            assert "백테스트 분석 결과" in page.warning[0].value
            assert page.button(key="refresh_performance")
        elif slug == "anova":
            assert page.info[0].value == "표시할 통계 검증 결과가 없습니다."
            assert page.button(key="refresh_statistics")
    page.switch_page("app_pages/portfolio.py").run()
    assert page.title[0].value == "포트폴리오"
    assert len(page.get("page_link")) == 5


def test_portfolio_cards_use_negative_drawdown(page):
    labels = [m.label for m in page.metric]
    assert labels[:3] == [
        "누적 수익률",
        "Sharpe Ratio · 위험 대비 수익",
        "최대 하락폭 · MDD",
    ]
    assert page.metric[2].value.startswith("-")
    assert any("왜 포트폴리오가 바뀌었나요?" in h.value for h in page.subheader)
    assert any("준비 중" in i.value for i in page.info)


def test_risk_page_shows_drawdown_safeguard_factors_and_guide(page):
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert [m.label for m in page.metric] == [
        "최대 하락폭 · MDD",
        "현재 낙폭",
        "변동성 · 연환산",
    ]
    assert page.metric[0].value.startswith("-")
    assert any("Safe-Guard 미작동" in s.value for s in page.success)
    assert len(page.get("vega_lite_chart")) == 1
    assert page.selectbox(key="risk_shap_asset")
    text = " ".join(m.value for m in page.markdown)
    assert "RSI" in text and "MACD" in text
    captions = " ".join(c.value for c in page.caption)
    assert "70 이상이면" in captions and "인과관계나 투자 권고가 아닙니다" in captions


def test_risk_page_survives_missing_backtest_and_shap(page, monkeypatch):
    from frontend import investment_client, shap_client

    def no_portfolio(*args, **kwargs):
        raise investment_client.InvestmentAPIError(
            "투자 분석 결과를 불러오지 못했습니다."
        )

    def no_shap(model):
        raise api_client.APIError("실제 SHAP 결과를 불러오지 못했습니다.")

    monkeypatch.setattr(investment_client, "load_portfolio", no_portfolio)
    monkeypatch.setattr(shap_client, "load_model_shap", no_shap)
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert len(page.warning) >= 2
    assert not page.metric and not page.get("vega_lite_chart")
    assert len(page.get("page_link")) == 6


def test_asset_detail_has_etf_intro_for_every_asset():
    from frontend.guide import ETF_GUIDE
    from frontend.shap_explanations import ASSET_NAMES

    assert set(ETF_GUIDE) == set(ASSET_NAMES)
    assert all(
        text.endswith("예요.") or text.endswith("요.") for text in ETF_GUIDE.values()
    )


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
    # 4 menu links + the comparison link, which stays reachable on failure.
    assert len(page.get("page_link")) == 5


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
    [(0.3, 0.14, "감소"), (0.1, 0.14, "증가"), (0.14, 0.14, "유지")],
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
    assert f"SPY 비중 {direction} 목표" in " ".join(x.value for x in page.markdown)
    assert [m.value for m in page.metric] == [f"{before:.2%}", "14.00%", "14.00%"]
    assert not any(m.label == "SHAP 기준 비중" for m in page.metric)
    displayed = " ".join(x.value for x in page.markdown)
    assert "시장 변동성" in displayed and "금(GLD) 관련 신호" in displayed
    assert "SHAP 기준 비중 대비" in " ".join(x.value for x in page.caption)
    page.button(key="shap_details_button").click().run()
    assert not page.exception
    assert any(m.label == "SHAP 기준 비중" for m in page.metric)
    assert "macd_gld" in page.dataframe[2].value["raw feature"].tolist()
    assert page.dataframe[2].value["shap value"].tolist() == [0.06, -0.02]
    assert len(page.dataframe[0].value) == 2
    assert any(e.label == "나머지 요인 보기" for e in page.expander)


def test_portfolio_composition_lists_only_held_assets(page, monkeypatch):
    from frontend import investment_client

    original = investment_client.load_portfolio

    def with_unheld(observations=None):
        data = original(observations)
        data["assets"] = data["assets"] + [
            {"ticker": "ZERO", "name": "미보유", "weight": 0.0}
        ]
        return data

    monkeypatch.setattr(investment_client, "load_portfolio", with_unheld)
    page.switch_page("app_pages/portfolio.py").run()
    assert not page.exception
    table = page.dataframe[0].value
    assert "ZERO" not in table["자산"].tolist()
    assert len(table) == len(original()["assets"])
    assert any("보유하지 않은 자산 1개는 생략" in c.value for c in page.caption)
