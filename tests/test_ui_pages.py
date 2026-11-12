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
        assert len(page.get("page_link")) == 6
        if slug == "research":
            assert page.text_area(key="research_question")
            assert page.date_input(key="research_date")
            assert not page.selectbox
        elif slug == "performance":
            assert "백테스트 분석 결과" in page.warning[0].value
            assert page.button(key="refresh_performance")
        elif slug == "anova":
            assert page.info[0].value == "표시할 통계 검증 결과가 없습니다."
            assert page.button(key="refresh_statistics")
        elif slug == "shap":
            assert "SHAP 해석 결과가 아직 준비되지" in page.warning[0].value
            assert page.button(key="refresh_explain")
            assert not page.selectbox
            captions = " ".join(item.value for item in page.caption)
            assert "저장된 SHAP 결과를 조회합니다" in captions
        else:
            assert page.info[0].value == "화면 준비 중"
            assert not page.selectbox
            captions = " ".join(item.value for item in page.caption)
            assert "분석 API와 데이터는 아직 준비되지 않았습니다" in captions
            assert "위 분석은 UI 개발용 샘플" not in captions
    page.switch_page("app_pages/portfolio.py").run()
    assert page.title[0].value == "포트폴리오 현황"


def test_period_filter_changes_results(page):
    before = page.metric[0].value
    page.selectbox(key="period").set_value("최근 63개 관측일").run()
    assert not page.exception
    assert page.metric[0].value != before


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
    assert len(page.get("page_link")) == 6
