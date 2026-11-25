"""Every risk number next to the market, the drawdown trend and the assumptions."""

import streamlit as st

from frontend.investment_client import InvestmentAPIError, load_portfolio
from frontend.portfolio_sections import assumptions, portfolio_risk
from frontend.ui import page_heading

page_heading(
    "위험 지표 상세",
    "REPORT / RISK",
    "최대 하락폭, 변동성, 하루 예상 손실(VaR·CVaR)을 S&P500과 나란히 보고 하락폭 흐름을 확인하세요.",
)
try:
    data = load_portfolio(None)
except InvestmentAPIError as exc:
    st.warning(str(exc))
    data = None
if data:
    portfolio_risk(data)
    assumptions(data)
