"""Price, RSI and MACD of one ETF with plain-language readings."""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import asset_return, indicator_line
from frontend.etf_choice import choose_etf
from frontend.guide import INDICATOR_GUIDE
from frontend.insights_client import load_asset_history
from frontend.ui import card, page_heading


def indicator_guide(item):
    """Plain-language reading of RSI or MACD, shown right under its chart."""
    st.caption(item["summary"])
    for line in item["points"]:
        st.caption(f"· {line}")


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
        st.altair_chart(
            indicator_line(rows, "rsi", "RSI", [30, 70], [0, 100]), width="stretch"
        )
        indicator_guide(INDICATOR_GUIDE[0])
    with card("price-macd"):
        st.subheader(f"MACD · 현재 {rows[-1]['macd']:+.2f}")
        st.altair_chart(indicator_line(rows, "macd", "MACD", [0]), width="stretch")
        indicator_guide(INDICATOR_GUIDE[1])
