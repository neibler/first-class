"""Portfolio overview: one return series drives the flight path and the cards."""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import allocation
from frontend.insights_client import load_rebalancing
from frontend.investment_client import InvestmentAPIError, load_portfolio, optimize
from frontend.ui import ASSETS, card, detail_link, page_heading
from frontend.weight_format import format_weight

# Register in the page's runtime, not a cached imported module. AppTest creates
# separate runtimes; each needs its own component registration.
REASONS = {
    "portfolio_weight_distance": "현재 비중이 AI 목표 비중과 기준 이상 달라져서",
    "monthly": "월간 정기 점검일이라서",
    "risk_score": "뉴스 위험 점수가 기준을 넘어서",
    "mdd_stop": "손실 제한(Safe-Guard) 규칙이 발동해서",
    "hold": "조정 조건에 해당하지 않아 유지",
}

flight_path = st.components.v2.component(
    "flight_path",
    html='<div class="flight-root"></div>',
    js=(ASSETS / "flight.js").read_text(encoding="utf-8"),
)

page_heading(
    "포트폴리오",
    "PORTFOLIO / 01",
    "지금 내 포트폴리오의 성과와 구성을 한눈에 확인하세요.",
)
period = st.selectbox("분석 기간", ["전체 기간", "최근 63개 관측일"], key="period")
try:
    data = load_portfolio(None if period == "전체 기간" else 63)
except InvestmentAPIError as exc:
    st.error(str(exc))
    data = None
    if st.button("다시 시도"):
        st.rerun()

if data:
    rows = data["returns"]
    result = data["display_metrics"]
    st.caption(
        "합성 데이터로 학습·검증한 결과"
        if data["data_source"] == "synthetic"
        else "저장된 과거 데이터 검증 결과"
    )
    if data["stopped"]:
        st.warning("손실 제한 규칙이 발동하여 현금으로 유지 중인 백테스트 결과입니다.")

    with card("returns"):
        st.subheader("누적 수익률")
        st.caption(
            f"{rows[0]['date']} — {rows[-1]['date']} · "
            f"{result['observations']}개 일간 수익률 · 비교 대상: S&P500(SPY 매수 후 보유)"
        )
        flight_path(
            data={"points": result["path"]}, key="flight_path", height="content"
        )

    weights, metrics = st.columns([1.35, 1], gap="medium")
    with weights:
        with card("allocation"):
            # Only what is actually held; zero-weight candidates just add rows.
            held = [a for a in data["assets"] if a["weight"] > 0]
            skipped = len(data["assets"]) - len(held)
            st.subheader("포트폴리오 구성")
            st.caption(
                f"{data['as_of']} 기준 보유 비중 · 합계 100%"
                + (f" · 보유하지 않은 자산 {skipped}개는 생략" if skipped else "")
            )
            donut, table = st.columns([1, 1], vertical_alignment="center")
            with donut:
                st.altair_chart(allocation(held), theme=None)
            with table:
                st.dataframe(
                    [
                        dict(자산=a["ticker"], 비중=format_weight(a["weight"]))
                        for a in sorted(held, key=lambda a: -a["weight"])
                    ],
                    hide_index=True,
                    column_config={"비중": st.column_config.TextColumn()},
                    height="content",
                )
    with metrics:
        with card("metrics"):
            st.subheader("핵심 지표")
            st.metric(
                "누적 수익률",
                f"{result['cumulative']:+.2%}",
                chart_data=[p["portfolio"] for p in result["path"]],
                delta_color="blue",
            )
            st.metric(
                "Sharpe Ratio · 위험 대비 수익",
                (
                    f"{result['sharpe']:.2f}"
                    if result["sharpe"] is not None
                    else "계산 불가"
                ),
            )
            st.metric(
                "최대 하락폭 · MDD",
                f"{result['mdd']:.2%}",
                chart_data=[p["drawdown"] for p in result["path"]],
                chart_type="area",
                delta_color="red",
                help="선택 기간에 최고점 대비 가장 많이 떨어진 폭입니다. 미니 그래프는 시점별 낙폭입니다.",
            )
            volatility = (data.get("metrics") or {}).get("annualized_volatility")
            if volatility is not None:
                st.metric(
                    "변동성 · 연환산",
                    f"{volatility:.2%}",
                    help="수익률이 얼마나 크게 오르내렸는지를 1년 기준으로 나타낸 값입니다.",
                )
            st.caption(f"현재 낙폭 {result['current_drawdown']:.2%}")

    with card("why"):
        st.subheader("왜 포트폴리오가 바뀌었나요?")
        try:
            changes = load_rebalancing(None if period == "전체 기간" else 63)
        except APIError as exc:
            st.warning(str(exc))
            changes = None
        if changes and changes["events"]:
            st.caption(
                f"선택 기간 {changes['observations']}거래일 중 "
                f"{changes['rebalanced_days']}일 비중을 조정했습니다 · "
                f"하루 평균 거래 비중(회전율) {changes['average_turnover']:.0%}"
            )
            events = {event["date"]: event for event in changes["events"]}
            day = st.selectbox(
                "날짜 (최근 순)",
                list(events),
                key="rebalance_date",
            )
            event = events[day]
            reasons = [REASONS.get(r, r) for r in event["reasons"]]
            st.markdown(f"**{day} 조정 이유:** {', '.join(reasons) or '기록 없음'}")
            if event["changes"]:
                st.dataframe(
                    [
                        {
                            "자산": item["ticker"],
                            "이전": format_weight(item["before"]),
                            "이후": format_weight(item["after"]),
                            "변화": f"{item['change'] * 100:+.0f}%p",
                        }
                        for item in event["changes"]
                    ],
                    hide_index=True,
                    height="content",
                )
            else:
                st.caption("1%p 이상 바뀐 자산이 없습니다.")
            st.caption(
                f"{event['decision_date']} 장 마감 정보로 결정하고 {day}에 반영한 비중입니다. "
                "1%p 미만 변화는 생략합니다. 이유를 쉬운 문장으로 풀어 주는 설명은 준비 중입니다."
            )
        elif changes:
            st.info("선택 기간에 비중 변화 기록이 없습니다.")
        st.caption(
            "AI가 무엇을 보고 판단했는지는 위험도·설명 메뉴의 AI 판단 요인에서 확인할 수 있습니다."
        )

# The comparison page loads its own data, so keep it reachable on failure too.
detail_link("performance", "다른 전략과 성과 비교하기")

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
            "최대 하락폭과 낙폭은 음수로 표시합니다. "
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

    with st.expander("오늘 새로 시작한다면? (참고용)"):
        st.caption(
            "빈 계좌(현금 100%)로 오늘 투자를 시작한다면 AI가 어떻게 나눌지 보여 주는 참고용 제안입니다. "
            "위 포트폴리오는 이미 자산을 들고 운용 중인 계좌라, AI가 현재 보유 비중까지 고려해 결과가 다를 수 있습니다. "
            "실제 주문은 나가지 않습니다."
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
                        {"자산": ticker, "목표 비중": format_weight(weight)}
                        for ticker, weight in proposal["target_weights"].items()
                    ],
                    hide_index=True,
                )
                if proposal["cash_weight"]:
                    st.write(f"현금 비중: {format_weight(proposal['cash_weight'])}")
            except InvestmentAPIError as exc:
                st.error(str(exc))
