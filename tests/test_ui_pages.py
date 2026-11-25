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
    assert [link.label for link in page.get("page_link")] == [
        "포트폴리오",
        "ETF 분석",
        "운용 리포트",
        "AI 도우미",
    ]
    # (page, title, links): 4 menu links + detail links on menu pages,
    # or + 1 back link on detail pages.
    for slug, title, links in [
        ("correlation", "ETF 분석", 7),
        ("report", "운용 리포트", 8),
        ("research", "AI 도우미", 4),
        ("performance", "강화학습 성과", 5),
        ("shap", "AI 판단 근거", 5),
        ("price", "가격·지표 상세", 5),
        ("relations", "자산 관계 상세", 5),
        ("trades", "사고판 내역", 5),
        ("risk", "위험 지표 상세", 5),
        ("anova", "ANOVA 검증", 5),
    ]:
        page.switch_page(f"app_pages/{slug}.py").run()
        assert not page.exception, slug
        assert page.title[0].value == title
        assert len(page.get("page_link")) == links, slug
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
    assert len(page.get("page_link")) == 4


def test_portfolio_shows_only_the_current_state(page):
    # Three result numbers; volatility and drawdown details live in the report.
    assert [m.label for m in page.metric] == [
        "누적 수익률",
        "Sharpe Ratio · 위험 대비 수익",
        "최대 하락폭 · MDD",
    ]
    assert page.metric[2].value.startswith("-")
    titles = [h.value for h in page.subheader]
    assert titles == ["누적 수익률", "포트폴리오 구성", "핵심 지표"]
    assert not page.expander or all(e.label == "API 준비 상태" for e in page.expander)


def test_trades_page_shows_rebalancing_changes_from_the_api(page):
    page.switch_page("app_pages/trades.py").run()
    assert not page.exception
    assert page.selectbox(key="rebalance_date").value == "2026-09-30"
    text = " ".join(m.value for m in page.markdown)
    assert "비중 차이가 조정 기준 이상 벌어져서" in text
    table = next(d.value for d in page.dataframe if "변화" in d.value.columns)
    assert list(table["자산"]) == ["SHY", "GLD", "EFA", "DBB", "TIP", "SMH"]
    assert list(table["변화"]) == ["-36%p", "+34%p", "+27%p", "-24%p", "-5%p", "+4%p"]
    # Shown before/after each add up to 100%, and the shown changes to 0%p.
    percent = [int(v[:-1]) for v in table["이전"]], [int(v[:-1]) for v in table["이후"]]
    assert sum(percent[0]) == 100 and sum(percent[1]) == 100
    assert sum(int(v[:-2]) for v in table["변화"]) == 0
    page.selectbox(key="rebalance_date").select("2026-09-29").run()
    assert any("비중이 바뀐 자산이 없습니다" in c.value for c in page.caption)
    assert "월간 정기 점검일이라서" in " ".join(m.value for m in page.markdown)
    assert any(e.label == "오늘 새로 시작한다면? (참고용)" for e in page.expander)


def test_report_answers_three_questions(page):
    page.switch_page("app_pages/report.py").run()
    assert not page.exception
    assert [h.value for h in page.subheader] == [
        "AI는 언제, 왜 사고팔았나요?",
        "얼마나 오르내렸나요?",
        "단순한 방법보다 나았나요?",
    ]
    text = " ".join(m.value for m in page.markdown)
    assert "**187일** 비중을 조정했어요" in text
    # 13% against the market's 12.22% reads as about the same.
    assert "S&P500과 **비슷한 정도로** 오르내렸어요" in text
    assert "**+4.7%**" in text and "**+12.4%**" in text
    assert [m.label for m in page.metric] == [
        "비중을 조정한 날",
        "하루 평균 거래 비중",
        "거래 비용 합계",
        "최대 하락폭",
        "변동성 · 연환산",
    ]
    assert page.metric[2].value == "원금의 19.7%"
    links = [link.label for link in page.get("page_link")][4:]
    assert links == [
        "날짜별 사고판 내역 보기",
        "위험 지표 자세히 보기",
        "다른 전략과 자세히 비교하기",
        "차이가 통계적으로 의미 있는지 보기 (ANOVA)",
    ]


def test_report_survives_missing_rebalancing(page, monkeypatch):
    from frontend import insights_client

    def unavailable(*args, **kwargs):
        raise api_client.APIError("비중 변화 기록을 불러오지 못했습니다.")

    monkeypatch.setattr(insights_client, "load_rebalancing", unavailable)
    page.switch_page("app_pages/report.py").run()
    assert not page.exception
    assert any("비중 변화 기록" in w.value for w in page.warning)
    assert page.metric, "The risk card must still render"
    assert len(page.get("page_link")) == 8, "Detail pages stay reachable"


def test_risk_page_shows_only_portfolio_wide_risk(page):
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert [m.label for m in page.metric] == [
        "최대 하락폭 · MDD",
        "현재 낙폭",
        "변동성 · 연환산",
        "하루 예상 손실 · VaR 95%",
        "나쁜 날 평균 손실 · CVaR 95%",
    ]
    assert page.metric[0].value.startswith("-")
    assert [m.value for m in page.metric[3:]] == ["-1.47%", "-1.87%"]
    # Risk detail page of the report: portfolio-wide only, plus the assumptions.
    assert [h.value for h in page.subheader] == ["시장과 비교한 위험도", "하락폭 추이"]
    assert not page.selectbox, "No ETF selection on the risk page"
    market = [c.value for c in page.caption if c.value.startswith("S&P500 ")]
    # MDD, volatility, VaR, CVaR in card order; market MDD is negative like ours.
    assert market == [
        "S&P500 -7.31%",
        "S&P500 12.22%",
        "S&P500 -1.42%",
        "S&P500 -1.72%",
    ]
    assert any("Safe-Guard 미작동" in s.value for s in page.success)
    assert len(page.get("vega_lite_chart")) == 1


def test_gain_only_var_is_not_shown_as_a_loss(page, monkeypatch):
    from frontend import insights_client
    from tests.investment_ui_fixtures import RISK

    gains = {**RISK, "portfolio": {"var": -0.004, "cvar": -0.002}}
    monkeypatch.setattr(insights_client, "load_risk", lambda observations=None: gains)
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert [m.value for m in page.metric[3:]] == [
        "손실 없음 · +0.40%",
        "손실 없음 · +0.20%",
    ]


def test_anova_waiting_notice_names_the_operator_step(page, monkeypatch):
    from frontend import statistics_client

    def pending():
        raise statistics_client.StatisticsPendingError(
            "통계 분석 결과가 아직 준비되지 않았습니다."
        )

    monkeypatch.setattr(statistics_client, "get_statistics", pending)
    page.switch_page("app_pages/anova.py").run()
    assert not page.exception
    assert any("아직 준비되지" in w.value for w in page.warning)
    notice = " ".join(c.value for c in page.caption)
    assert "운영 담당" in notice and "아직 지원하지 않아" in notice
    # No command that cannot analyze the shared serving bundle.
    assert "run_analysis" not in notice


def test_risk_page_survives_missing_backtest(page, monkeypatch):
    from frontend import investment_client

    def no_portfolio(*args, **kwargs):
        raise investment_client.InvestmentAPIError(
            "투자 분석 결과를 불러오지 못했습니다."
        )

    monkeypatch.setattr(investment_client, "load_portfolio", no_portfolio)
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert page.warning
    assert not page.metric and not page.get("vega_lite_chart")
    page.switch_page("app_pages/report.py").run()
    assert not page.exception
    assert len(page.get("page_link")) == 8, "Comparison and ANOVA stay reachable"


def test_etf_analysis_answers_three_questions(page, monkeypatch):
    from frontend import correlation_client

    def unavailable(*args, **kwargs):
        raise api_client.APIError("자산 관계를 불러오지 못했습니다.")

    monkeypatch.setattr(correlation_client, "load_correlations", unavailable)
    page.switch_page("app_pages/correlation.py").run()
    page.selectbox(key="correlation_asset").set_value("QQQ").run()
    assert not page.exception
    assert [h.value for h in page.subheader] == [
        "AI는 왜 이 비중을 정했나요?",
        "가격은 어떻게 움직였나요?",
        "다른 자산과 같이 움직이나요?",
    ]
    captions = " ".join(c.value for c in page.caption)
    # The AI card shows the latest monthly decision, not the period below.
    assert "2025-01-03 결정 기준 (가장 최근)" in captions
    assert "지난 2번의 월별 판단도 볼 수 있습니다" in captions
    assert page.selectbox(key="correlation_window").label == "가격·관계 분석 기간"
    links = [link.label for link in page.get("page_link")]
    assert links[4:] == [
        "AI 판단 근거 자세히 보기 · 지난 결정일도 보기",
        "가격·지표(RSI·MACD) 자세히 보기",
        "자산 관계 자세히 보기",
    ]
    # The detail page opens on the same ETF and returns to ETF analysis.
    page.switch_page("app_pages/shap.py").run()
    assert not page.exception
    assert page.selectbox(key="shap_asset").value == "QQQ"
    assert "ETF 분석 화면으로 돌아가기" in [
        link.label for link in page.get("page_link")
    ]
    # A date chosen there survives going back and forth.
    page.selectbox(key="shap_date").set_value("2025-01-02").run()
    page.switch_page("app_pages/correlation.py").run()
    assert page.selectbox(key="correlation_asset").value == "QQQ"
    page.switch_page("app_pages/shap.py").run()
    assert page.selectbox(key="shap_date").value == "2025-01-02"
    # An ETF without saved decisions says so instead of failing.
    page.switch_page("app_pages/correlation.py").run()
    page.selectbox(key="correlation_asset").set_value("SMH").run()
    assert not page.exception
    assert any("SMH의 AI 판단 기록이 없습니다" in i.value for i in page.info)


def test_shared_choice_wins_over_a_stale_widget_value(page):
    # What the portfolio composition click stores before opening ETF analysis.
    page.session_state["etf_choice"] = "QQQ"
    page.switch_page("app_pages/correlation.py").run()
    assert not page.exception
    assert page.selectbox(key="correlation_asset").value == "QQQ"


def test_etf_analysis_survives_missing_shap(page, monkeypatch):
    from frontend import shap_client

    def no_shap(model):
        raise api_client.APIError("실제 SHAP 결과를 불러오지 못했습니다.")

    monkeypatch.setattr(shap_client, "load_model_shap", no_shap)
    page.switch_page("app_pages/correlation.py").run()
    assert not page.exception
    assert any("SHAP 결과" in w.value for w in page.warning)
    assert any("가격은 어떻게 움직였나요?" in h.value for h in page.subheader)


def test_risk_page_keeps_other_cards_when_var_is_missing(page, monkeypatch):
    from frontend import insights_client

    def unavailable(*args, **kwargs):
        raise api_client.APIError("위험 지표(VaR·CVaR)를 불러오지 못했습니다.")

    monkeypatch.setattr(insights_client, "load_risk", unavailable)
    page.switch_page("app_pages/risk.py").run()
    assert not page.exception
    assert len(page.metric) == 3
    assert any("위험 지표(VaR·CVaR)" in c.value for c in page.caption)


def test_price_story_and_price_detail_page(page, monkeypatch):
    from frontend import correlation_client

    def unavailable(*args, **kwargs):
        raise api_client.APIError("자산 관계를 불러오지 못했습니다.")

    monkeypatch.setattr(correlation_client, "load_correlations", unavailable)
    page.switch_page("app_pages/correlation.py").run()
    assert not page.exception
    # One plain sentence and one chart on the main page.
    assert len(page.get("vega_lite_chart")) == 1
    assert "가격이 **0.4% 내렸습니다.**" in " ".join(m.value for m in page.markdown)
    page.switch_page("app_pages/price.py").run()
    assert not page.exception
    assert len(page.get("vega_lite_chart")) == 3
    titles = [h.value for h in page.subheader]
    assert "RSI · 현재 49" in titles and "MACD · 현재 +1.30" in titles
    captions = " ".join(c.value for c in page.caption)
    # Escaped so Markdown shows the dollar signs instead of a formula.
    assert "가격 \\$762.63 · 기간 첫날 \\$765.61" in captions
    assert "70 이상이면" in captions


def test_asset_detail_has_etf_intro_for_every_asset():
    from frontend.guide import ETF_GUIDE
    from frontend.shap_explanations import ASSET_NAMES

    assert set(ETF_GUIDE) == set(ASSET_NAMES)
    assert all(
        text.endswith("예요.") or text.endswith("요.") for text in ETF_GUIDE.values()
    )


def test_period_and_shap_filters_change_results(page):
    period = page.selectbox(key="period")
    # Months that fit inside the fixture's daily returns, then the whole period.
    assert period.options == ["1개월", "3개월", "6개월", "전체"]
    assert period.value == "전체"
    before = page.metric[0].value
    period.set_value("3개월").run()
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
    assert len(page.get("page_link")) == 4


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
    # Base -> effect of the signals -> AI target; no trade record, no trade line.
    assert [m.label for m in page.metric] == [
        "기준 비중",
        "신호들의 효과",
        "AI 목표 비중",
    ]
    assert not any("실제 운용" in c.value for c in page.caption)
    assert any(h.value == "이 화면 읽는 법" for h in page.subheader)
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
    assert page.metric[2].value == "14.0%"
    captions = " ".join(x.value for x in page.caption)
    assert f"실제 운용: 거래 전 {before:.1%} → 거래 후 14.0%" in captions
    displayed = " ".join(x.value for x in page.markdown)
    assert "시장 변동성" in displayed and "금(GLD) 관련 신호" in displayed
    assert "+6.0%p" in displayed and "-2.0%p" in displayed
    page.button(key="shap_details_button").click().run()
    assert not page.exception
    assert f"비중 {direction} 목표" in " ".join(x.value for x in page.caption)
    assert any(m.label == "SHAP 기준 비중" for m in page.metric)
    assert "macd_gld" in page.dataframe[2].value["raw feature"].tolist()
    assert page.dataframe[2].value["shap value"].tolist() == [0.06, -0.02]
    assert len(page.dataframe[0].value) == 2
    assert any(e.label == "나머지 요인 보기" for e in page.expander)


def test_portfolio_composition_lists_only_held_assets(page, monkeypatch):
    import json

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
    # Donut and its weight table are one chart sharing hover and click.
    spec = json.loads(page.get("vega_lite_chart")[0].proto.spec)
    assert "ZERO" not in json.dumps(spec)
    assert [p["name"] for p in spec["params"]] == ["hover", "pick"]
    assert any("보유하지 않은 자산 1개는 생략" in c.value for c in page.caption)
    assert any("누르면 어떤 ETF인지" in c.value for c in page.caption)
