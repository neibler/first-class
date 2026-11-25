"""Beginner first page: how the portfolio did, what it holds, what each ETF is.

Per-ETF analysis and the portfolio report (why weights moved, risk, comparison,
calculation notes) have their own pages, so this page stays short and easy.
"""

import streamlit as st

from frontend.charts import ALLOCATION_PICK, allocation
from frontend.etf_choice import open_on
from frontend.guide import ETF_GUIDE
from frontend.investment_client import InvestmentAPIError, load_portfolio
from frontend.periods import ALL, period_options
from frontend.shap_explanations import ASSET_NAMES
from frontend.ui import ASSETS, card, page_heading

# Register in the page's runtime, not a cached imported module. AppTest creates
# separate runtimes; each needs its own component registration.
flight_path = st.components.v2.component(
    "flight_path",
    html='<div class="flight-root"></div>',
    js=(ASSETS / "flight.js").read_text(encoding="utf-8"),
)

page_heading(
    "포트폴리오",
    "PORTFOLIO / 01",
    "AI가 굴린 내 돈이 지금 얼마나 불었는지, 어디에 나눠 담겨 있는지 보여 드려요.",
)
try:
    # The whole period first: its length decides which period choices exist.
    full = load_portfolio(None)
except InvestmentAPIError as exc:
    st.error(str(exc))
    full = None
    if st.button("다시 시도"):
        st.rerun()

data = None
if full:
    options = period_options(full["display_metrics"]["observations"])
    period = st.selectbox(
        "분석 기간", list(options), index=len(options) - 1, key="period"
    )
    if period == ALL:
        data = full
    else:
        try:
            data = load_portfolio(options[period])
        except InvestmentAPIError as exc:
            st.error(str(exc))

if data:
    rows = data["returns"]
    result = data["display_metrics"]
    st.caption(
        "합성 데이터로 학습·검증한 결과"
        if data["data_source"] == "synthetic"
        else "저장된 과거 데이터로 모의 운용한 결과"
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
            event = st.altair_chart(
                allocation(held),
                theme=None,
                on_select="rerun",
                selection_mode=[ALLOCATION_PICK],
                key="allocation_chart",
            )
            picked = [
                item["ticker"]
                for item in event.get("selection", {}).get(ALLOCATION_PICK, [])
            ]
            if picked and picked[0] not in ASSET_NAMES:
                # Cash (CASH) is shown in the composition but is not an ETF, so it
                # gets a short note and no ETF-analysis link.
                st.markdown("**현금**")
                st.write(
                    "ETF에 넣지 않고 남겨 둔 돈이에요. AI가 위험을 줄이거나 "
                    "다음 거래를 기다릴 때 일부를 현금으로 둘 수 있어요."
                )
            elif picked:
                ticker = picked[0]
                st.markdown(f"**{ticker} · {ASSET_NAMES.get(ticker, ticker)}**")
                st.write(
                    ETF_GUIDE.get(
                        ticker, "이 ETF의 소개 글은 아직 준비되지 않았습니다."
                    )
                )
                # Open ETF analysis on the same ETF (plain key only; setting the
                # other page's widget key showed SPY in its selector).
                open_on(ticker)
                st.page_link(
                    "app_pages/correlation.py",
                    label=f"{ticker} ETF 분석에서 더 보기",
                    icon=":material/query_stats:",
                )
            else:
                st.caption(
                    "도넛이나 표에 마우스를 올리면 비중이, 누르면 어떤 ETF인지 짧은 설명이 나옵니다."
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
                help="같은 위험을 졌을 때 수익을 얼마나 냈는지입니다. 높을수록 좋습니다.",
            )
            st.metric(
                "최대 하락폭 · MDD",
                f"{result['mdd']:.2%}",
                chart_data=[p["drawdown"] for p in result["path"]],
                chart_type="area",
                delta_color="red",
                help="선택 기간에 최고점 대비 가장 많이 떨어진 폭입니다. 미니 그래프는 시점별 낙폭입니다.",
            )
            # Volatility and current drawdown are in the report's risk card,
            # next to the market; this page keeps the three result numbers.

st.caption(
    "ETF별 자세한 내용은 ETF 분석에서, 비중이 바뀐 이유와 위험도는 운용 리포트에서 볼 수 있습니다."
)
