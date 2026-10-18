"""Cockpit: one return series drives the route and the instruments."""

import streamlit as st

from frontend.charts import allocation
from frontend.sample_data import load_portfolio, performance
from frontend.ui import (
    ASSETS,
    load_sample,
    page_heading,
    portfolio_navigation,
    sample_badge,
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
        sample_badge()
        data = load_sample(load_portfolio, "portfolio")
        if data:
            with st.container(key="windshield"):
                title, controls = st.columns([3, 1])
                with controls:
                    period = st.selectbox(
                        "분석 기간", ["전체 기간", "최근 63개 관측일"], key="period"
                    )
                rows = (
                    data["returns"] if period == "전체 기간" else data["returns"][-64:]
                )
                result = performance(
                    rows, data["periods_per_year"], data["annual_risk_free_rate"]
                )
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
                        st.subheader("현재 자산 비중")
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
                            "Sharpe Ratio · 로그 수익률 / 연율화",
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
    with st.expander("샘플 구성과 계산 가정"):
        st.write(
            f"포트폴리오와 {data['benchmark']} 모두 고정된 합성 시계열입니다. "
            "실제 PPO 학습·시장 데이터·백테스트 결과가 아닙니다. 자산 목록은 UI 예시입니다."
        )
        st.write(
            f"누적 수익률 = exp(일간 로그 수익률 합) − 1. "
            f"연 {data['periods_per_year']} 관측일, 연 무위험 수익률 "
            f"{data['annual_risk_free_rate']:.1%}를 가정합니다. "
            "Sharpe = 일간 초과 로그 수익률 평균 / 표본 표준편차(ddof=1) × √252. "
            "무위험 일간 로그 수익률은 ln(1 + 연 무위험 수익률) / 252입니다. "
            "기준일은 수익률 표본에서 제외하며, 기간 변경 시 모든 성과를 함께 재계산합니다."
        )
        st.write(
            "낙폭 = 누적 자산가치 / 해당 시점까지의 최고 자산가치 − 1. "
            "MDD는 선택 기간의 낙폭 최솟값입니다. 자산 비중은 마지막 기준일의 스냅샷입니다. "
            "샘플 날짜는 평일 간격이며 실제 거래소 휴장일·거래비용·환율을 반영하지 않습니다."
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
