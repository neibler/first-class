"""Price, RSI and MACD of one ETF with plain-language readings."""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import asset_return, indicator_line
from frontend.etf_choice import choose_etf
from frontend.guide import INDICATOR_GUIDE, macd_example, rsi_example
from frontend.insights_client import load_asset_history
from frontend.ui import card, page_heading


def indicator_guide(item):
    """Plain-language reading of RSI or MACD, shown right under its chart."""
    for line in item["points"]:
        st.caption(f"· {line}")


def explain(what, example):
    """Beginners first: what the chart is, then what today's value means."""
    st.markdown(f"**이게 뭔가요?** {what}")
    st.markdown(f"**지금은?** {example}")


page_heading(
    "가격·지표 상세",
    "ETF ANALYSIS / PRICE",
    "가격 흐름과 함께 기술 지표 RSI·MACD를 풀이와 함께 봅니다.",
)
ticker, window = choose_etf("price")
try:
    history = load_asset_history(ticker, window)
except APIError as exc:
    st.warning(str(exc))
    history = None
if history:
    rows = history["rows"]
    with card("price-return"):
        st.subheader(f"{ticker} 누적 수익률")
        change = rows[-1]["cumulative_return"]
        explain(
            "기간 첫날 가격을 0%로 놓고, 그 뒤 가격이 얼마나 오르거나 내렸는지 보여 줘요.",
            f"{rows[0]['date']}에 샀다면 {history['as_of']}까지 "
            f"{abs(change):.1%} {'올랐' if change >= 0 else '내렸'}어요.",
        )
        st.caption(
            f"{rows[0]['date']} ~ {history['as_of']} · "
            f"최근 {history['observations']}거래일 · "
            "기간 첫날을 0%로 맞춘 가격 변화 · 모델이 학습·평가에 쓴 실제 시장 데이터"
        )
        first, last = rows[0]["close"], rows[-1]["close"]
        if first is not None and last is not None:
            # "\$" keeps Markdown from reading "$…$" as a LaTeX formula.
            st.caption(
                f"가격 \\${last:,.2f} · 기간 첫날 \\${first:,.2f} (달러, 종가 기준)"
            )
        st.altair_chart(asset_return(rows), width="stretch")
    with card("price-rsi"):
        st.subheader(f"RSI · 현재 {rows[-1]['rsi']:.0f}")
        explain(INDICATOR_GUIDE[0]["summary"], rsi_example(rows[-1]["rsi"]))
        st.altair_chart(
            indicator_line(rows, "rsi", "RSI", [30, 70], [0, 100]), width="stretch"
        )
        indicator_guide(INDICATOR_GUIDE[0])
    with card("price-macd"):
        st.subheader(f"MACD · 현재 {rows[-1]['macd']:+.2f}")
        explain(INDICATOR_GUIDE[1]["summary"], macd_example(rows[-1]["macd"]))
        st.altair_chart(indicator_line(rows, "macd", "MACD", [0]), width="stretch")
        indicator_guide(INDICATOR_GUIDE[1])
