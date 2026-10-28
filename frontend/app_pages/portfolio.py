"""Cockpit: one return series drives the route and the instruments."""

import streamlit as st

from frontend.charts import allocation
from frontend.investment_client import InvestmentAPIError, load_portfolio, optimize
from frontend.ui import (
    ASSETS,
    page_heading,
    portfolio_navigation,
)

# Register in the page's runtime, not a cached imported module. AppTest creates
# separate runtimes; each needs its own component registration.
flight_path = st.components.v2.component(
    "flight_path",
    html='<div class="flight-root"></div>',
    js=(ASSETS / "flight.js").read_text(encoding="utf-8"),
)

# Shared width and gap-free flow anchor sky, housing and console geometry.
with st.container(key="cockpit-shell", gap=None):
    # Anchor the decorative pillars to the window, not the whole scrolling page.
    with st.container(key="flight-window"):
        st.html(
            '<div class="window-scene" aria-hidden="true">'
            '<div class="window-sky"></div><div class="window-pillars"></div></div>'
        )
        page_heading(
            "포트폴리오 현황",
            "FLIGHT DECK / 01",
            "포트폴리오의 구성과 수익률 경로를 한눈에 확인하세요.",
        )
        period = st.selectbox(
            "분석 기간", ["전체 기간", "최근 63개 관측일"], key="period"
        )
        try:
            data = load_portfolio(None if period == "전체 기간" else 63)
        except InvestmentAPIError as exc:
            st.error(str(exc))
            data = None
            if st.button("다시 시도"):
                st.rerun()
        if data:
            st.caption(
                "합성 데이터로 학습·검증한 결과"
                if data["data_source"] == "synthetic"
                else "저장된 과거 데이터 검증 결과"
            )
            if data["stopped"]:
                st.warning(
                    "손실 제한 규칙이 발동하여 현금으로 유지 중인 백테스트 결과입니다."
                )
        if data:
            with st.container(key="windshield"):
                title = st.container()
                rows = data["returns"]
                result = data["display_metrics"]
                with title:
                    st.subheader("누적 수익률")
                    st.caption(
                        f"{rows[0]['date']} — {rows[-1]['date']} · "
                        f"{result['observations']}개 일간 수익률"
                    )
                flight_path(
                    data={"points": result["path"]}, key="flight_path", height="content"
                )

    # Native containers keep the six routes available even in sample error/empty states.
    with st.container(key="instrument-deck", gap=None):
        st.html('<div class="instrument-top-bar" aria-hidden="true"></div>')
        with st.container(key="instruments"):
            menu, weights, metrics, right = st.columns(
                [1.5, 4.1, 2.3, 0.7], gap="small"
            )
            with menu:
                portfolio_navigation()
            if data:
                with right:
                    st.html(
                        '<div class="aviation-side aviation-right" '
                        'aria-hidden="true"></div>'
                    )
                with weights:
                    with st.container(key="allocation"):
                        st.subheader("백테스트 종료 시점 보유 비중")
                        st.caption(f"{data['as_of']} 기준 · 합계 100%")
                        donut, table = st.columns([1, 1], vertical_alignment="center")
                        with donut:
                            st.altair_chart(allocation(data["assets"]), theme=None)
                        with table:
                            st.dataframe(
                                [
                                    dict(자산=a["ticker"], 비중=a["weight"])
                                    for a in data["assets"]
                                ],
                                hide_index=True,
                                column_config={
                                    "비중": st.column_config.NumberColumn(
                                        format="percent"
                                    )
                                },
                                height="content",
                            )
                with metrics:
                    with st.container(key="gauges"):
                        st.metric(
                            "누적 수익률",
                            f"{result['cumulative']:+.2%}",
                            chart_data=[p["portfolio"] for p in result["path"]],
                            delta_color="blue",
                        )
                        st.metric(
                            "Sharpe Ratio · 단순 수익률 / 연율화",
                            (
                                f"{result['sharpe']:.2f}"
                                if result["sharpe"] is not None
                                else "계산 불가"
                            ),
                        )
                        with st.container(key="mdd-gauge"):
                            st.metric(
                                "기간 최대 낙폭 · MDD",
                                f"{result['mdd']:.2%}",
                                chart_data=[p["drawdown"] for p in result["path"]],
                                chart_type="area",
                                delta_color="red",
                                help="숫자는 선택 기간의 최대 낙폭(MDD), 미니 그래프는 같은 기간의 시점별 낙폭입니다.",
                            )
                        st.caption(
                            f"현재 낙폭 {result['current_drawdown']:.2%} · 기간 내 최저 낙폭과 구분"
                        )

        st.html('<div class="lower-console" aria-hidden="true"></div>')

if data:
    with st.expander("계산 가정"):
        st.write(
            "서버에 저장된 모델 평가 결과입니다. 합성 데이터 결과는 실제 투자 성능을 뜻하지 않습니다."
        )
        st.write(
            f"연 {data['periods_per_year']} 거래일, "
            f"연 무위험 수익률 {data['annual_risk_free_rate']:.1%}. "
            "Sharpe는 비용 차감 후 단순수익률 기준이며 서버 계산값을 표시합니다."
        )
        st.write(
            "낙폭 그래프는 음수, 서버의 최대 낙폭 값은 양수입니다. "
            "모든 기준 전략에도 같은 손실 제한 규칙을 적용합니다. 보유 비중에는 현금이 포함됩니다."
        )
        st.dataframe(
            result["path"],
            hide_index=True,
            column_config={
                "date": "날짜",
                "portfolio": st.column_config.NumberColumn(
                    "포트폴리오", format="percent"
                ),
                "benchmark": st.column_config.NumberColumn(
                    "벤치마크", format="percent"
                ),
                "drawdown": st.column_config.NumberColumn("낙폭", format="percent"),
            },
        )

    with st.expander("새 포트폴리오 목표 비중 제안"):
        st.caption(
            "현금 100%로 새로 시작하는 경우의 제안입니다. 위 백테스트 종료 상태를 재투자하는 주문이 아닙니다."
        )
        if st.button("목표 비중 계산"):
            try:
                proposal = optimize()
                st.caption(
                    f"{proposal['as_of']} 피처 기준 · "
                    f"{proposal['execution_delay_trading_bars']} 거래일 지연 · 주문 전송 없음"
                )
                st.dataframe(
                    [
                        {"자산": ticker, "목표 비중": weight}
                        for ticker, weight in proposal["target_weights"].items()
                    ],
                    hide_index=True,
                )
                if proposal["cash_weight"]:
                    st.write(f"현금 비중: {proposal['cash_weight']:.1%}")
            except InvestmentAPIError as exc:
                st.error(str(exc))
