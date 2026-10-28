
import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client


@pytest.fixture
def page(monkeypatch):
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
        if slug != "shap":
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
    before = str(page.dataframe[0].value)
    page.selectbox(key="shap_asset").set_value("QQQ").run()
    page.selectbox(key="shap_date").set_value("2026-04-30").run()
    assert not page.exception
    assert str(page.dataframe[0].value) != before


@pytest.mark.parametrize("slug", ["shap"])
@pytest.mark.parametrize("state", ["empty", "error", "loading"])
def test_sample_states_and_retry_keep_navigation(page, monkeypatch, slug, state):
    monkeypatch.setenv("FIRST_CLASS_UI_DEV", "1")
    page.switch_page(f"app_pages/{slug}.py")
    page.query_params["ui_state"] = state
    page.run()
    assert not page.exception
    assert len(page.get("page_link")) == 6
    if state == "error":
        assert "샘플 데이터 오류" in page.error[0].value
    else:
        assert "샘플" in page.info[0].value
    key = f"finish_{slug}" if state == "loading" else f"retry_{slug}"
    page.button(key=key).click().run()
    assert not page.exception
    assert not page.error
    assert page.metric if slug == "portfolio" else page.dataframe


def test_state_query_is_ignored_without_dev_flag(page, monkeypatch):
    monkeypatch.delenv("FIRST_CLASS_UI_DEV", raising=False)
    page.query_params["ui_state"] = "error"
    page.run()
    assert not page.exception
    assert not page.error
    assert page.metric


@pytest.mark.parametrize("slug", ["shap"])
@pytest.mark.parametrize("error_type", [OSError, ValueError, KeyError])
def test_sample_errors_do_not_expose_exception_text(
    page, monkeypatch, caplog, slug, error_type
):
    from frontend import sample_data

    private_detail = "PRIVATE_TEST_DETAIL_NOT_FOR_DISPLAY"

    def unavailable():
        raise error_type(private_detail)

    monkeypatch.setattr(sample_data, f"load_{slug}", unavailable)
    page.switch_page(f"app_pages/{slug}.py").run()
    assert not page.exception
    assert "샘플 데이터 오류" in page.error[0].value
    assert "다시 시도" in page.error[0].value
    assert private_detail not in page.error[0].value
    assert "SAMPLE_LOAD_FAILED" in caplog.text
    assert private_detail not in caplog.text
    assert len(page.get("page_link")) == 6
    assert page.button(key=f"retry_{slug}")


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
