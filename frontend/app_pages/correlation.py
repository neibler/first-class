"""ETF analysis for beginners: three plain answers, each with a detail page."""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import asset_return
from frontend.correlation_client import load_correlations
from frontend.etf_choice import WINDOWS, choose_asset, choose_window, etf_name
from frontend.insights_client import load_asset_history
from frontend.shap_client import load_model_shap
from frontend.shap_explanations import group_shap_features
from frontend.shap_presentation import select_factors
from frontend.ui import card, detail_link, page_heading

DIRECTIONS = {"increase": "▲ 비중을 높이는 쪽", "decrease": "▼ 비중을 낮추는 쪽"}

page_heading(
    "ETF 분석",
    "ETF ANALYSIS / 02",
    "궁금한 ETF를 골라 보세요. AI가 왜 이만큼 담았는지, 요즘 가격은 어땠는지 알려 드려요.",
)
ticker = choose_asset("correlation", where=st.columns(2)[0])

with card("factors"):
    st.subheader("AI는 왜 이 비중을 정했나요?")
    try:
        shap = load_model_shap(None)
    except APIError as exc:
        st.warning(str(exc))
        shap = None
    if shap and ticker not in shap["assets"]:
        st.info(f"{ticker}의 AI 판단 기록이 없습니다.")
    elif shap:
        row = shap["assets"][ticker]["decisions"][-1]
        # The period selector below does not apply here: SHAP is stored per
        # monthly decision, so this card always shows the latest one.
        st.caption(
            f"{row['date']} 결정 기준 (가장 최근) · AI 판단 기록은 평가 기간 중 일부 결정일만 저장됩니다."
        )
        st.markdown(
            f"AI가 계산한 {ticker} 목표 비중은 **{row['final_value']:.0%}**입니다."
        )
        if not any(row["shap_values"]):
            # Every contribution is exactly 0: the AI gave this ETF the same weight
            # whatever the inputs were, so no signal moved it.
            st.info(
                f"AI는 어떤 신호를 넣어도 {ticker}에 같은 비중"
                f"({row['final_value']:.0%})을 줘서, 비중을 움직인 신호가 없습니다."
            )
        else:
            factors = group_shap_features(
                shap["features"], row["shap_values"], ticker, row
            )
            selected, _ = select_factors(factors, "unknown")
            st.caption("비중을 가장 크게 움직인 신호 3개입니다.")
            for factor in selected:
                st.markdown(
                    f"**{factor['label']}** · "
                    f"{DIRECTIONS.get(factor['direction'], '영향 없음')}",
                    help=factor["explanation"],
                )
        st.caption(
            "자세히 보기에서는 결정일을 바꿔 저장된 "
            f"{len(shap['assets'][ticker]['decisions'])}개 결정일의 판단도 볼 수 있습니다. "
            "AI 판단에 대한 설명이며 투자 권유가 아닙니다."
        )
    detail_link("shap", "AI 판단 근거 자세히 보기 · 지난 결정일도 보기")

# Applies to the two market cards below only, so it sits right above them.
window = choose_window(
    "correlation",
    "가격·관계 분석 기간",
    where=st.columns(2)[0],
)
st.caption("아래 두 카드는 실제 시장 가격 데이터 기준입니다.")

with card("asset-history"):
    st.subheader("가격은 어떻게 움직였나요?")
    try:
        history = load_asset_history(ticker, window)
    except APIError as exc:
        st.warning(str(exc))
        history = None
    if history:
        rows = history["rows"]
        change = rows[-1]["cumulative_return"]
        st.markdown(
            f"최근 {WINDOWS[window]}({rows[0]['date']} ~ {rows[-1]['date']}) 동안 "
            f"{ticker} 가격이 **{abs(change):.1%} "
            f"{'올랐' if change >= 0 else '내렸'}습니다.**"
        )
        st.altair_chart(asset_return(rows), width="stretch")
    detail_link("price", "가격·지표(RSI·MACD) 자세히 보기")

with card("relations"):
    st.subheader("다른 자산과 같이 움직이나요?")
    try:
        data = load_correlations(ticker, window)
    except APIError as exc:
        st.warning(str(exc))
        data = None
    if data and data["relationships"]:
        st.caption(
            "같이 움직이는 자산끼리는 함께 오르고 함께 떨어져서, 서로의 손실을 덜어 주지 못합니다. "
            f"최근 {WINDOWS[window]} 동안 {ticker}와 관계가 가장 큰 3개입니다."
        )
        for item in data["relationships"][:3]:
            st.markdown(
                f"**{etf_name(item['display_name'])}** · "
                f"{item['relationship_label']} ({item['strength']})"
            )
    elif data:
        st.info(data["message"])
    detail_link("relations", "자산 관계 자세히 보기")
