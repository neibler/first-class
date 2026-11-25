"""What happened over the evaluation period, as three plain answers.

Like ETF analysis, each card answers one question in a few lines and opens a
detail page for the rest: trade history, risk numbers, strategy comparison.
"""

import streamlit as st

from frontend.investment_client import InvestmentAPIError, load_portfolio
from frontend.portfolio_sections import (
    MARKET,
    load_changes,
    risk_card,
    trades_card,
)
from frontend.ui import card, detail_link, page_heading

page_heading(
    "운용 리포트",
    "REPORT / 03",
    "지난 기간 동안 AI가 언제, 왜 사고팔았는지와 그 사이 내 돈이 얼마나 오르내렸는지 정리했어요.",
)
try:
    portfolio = load_portfolio(None)
except InvestmentAPIError as exc:
    st.warning(str(exc))
    portfolio = None
changes = load_changes()

trades_card(portfolio, changes)
risk_card(portfolio)

with card("report-compare"):
    st.subheader("단순한 방법보다 나았나요?")
    mine = ((portfolio or {}).get("metrics") or {}).get("total_return")
    spy = ((portfolio or {}).get("benchmark_metrics") or {}).get("total_return")
    if mine is not None and spy is not None:
        st.markdown(
            f"같은 기간 AI 포트폴리오는 **{mine:+.1%}**, "
            f"{MARKET}만 들고 있었다면 **{spy:+.1%}**였어요."
        )
        st.caption(
            "같은 비용 규칙으로 계산한 다른 전략들과의 비교와, 그 차이가 우연인지 확인하는 "
            "통계 검증은 아래에서 볼 수 있어요."
        )
    # Both pages load their own data, so keep them reachable when this one fails.
    detail_link("performance", "다른 전략과 자세히 비교하기")
    detail_link("anova", "차이가 통계적으로 의미 있는지 보기 (ANOVA)")
