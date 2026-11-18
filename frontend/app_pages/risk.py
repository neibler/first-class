"""Risk level and plain explanations of the model's decisions for general investors."""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import drawdown_trend
from frontend.guide import INDICATOR_GUIDE
from frontend.investment_client import InvestmentAPIError, load_portfolio
from frontend.shap_client import load_model_shap
from frontend.shap_explanations import ASSET_NAMES, group_shap_features
from frontend.shap_presentation import compact_state, select_factors
from frontend.ui import card, detail_link, page_heading

DIRECTIONS = {"increase": "비중을 높이는 쪽", "decrease": "비중을 낮추는 쪽"}

page_heading(
    "위험도·설명",
    "RISK & EXPLANATION / 02",
    "내 포트폴리오가 얼마나 흔들렸는지, AI가 무엇을 보고 판단했는지 확인하세요.",
)

try:
    data = load_portfolio()
except InvestmentAPIError as exc:
    st.warning(str(exc))
    data = None

if data:
    result = data["display_metrics"]
    with card("risk-now"):
        st.subheader("현재 위험도")
        if data["stopped"] or data.get("safeguard_triggered"):
            st.warning(
                "Safe-Guard가 작동했습니다. 손실 제한 규칙에 따라 현금으로 전환된 결과입니다."
            )
        else:
            st.success(
                "Safe-Guard 미작동 · 평가 기간에 손실 제한 규칙이 발동하지 않았습니다."
            )
        worst, now, swing = st.columns(3)
        worst.metric(
            "최대 하락폭 · MDD",
            f"{result['mdd']:.2%}",
            help="평가 기간에 최고점 대비 가장 많이 떨어진 폭입니다.",
        )
        now.metric(
            "현재 낙폭",
            f"{result['current_drawdown']:.2%}",
            help="기준일에 최고점 대비 얼마나 내려와 있는지입니다.",
        )
        volatility = (data.get("metrics") or {}).get("annualized_volatility")
        swing.metric(
            "변동성 · 연환산",
            f"{volatility:.2%}" if volatility is not None else "—",
            help="수익률이 얼마나 크게 오르내렸는지를 1년 기준으로 나타낸 값입니다.",
        )
        st.caption(
            f"{data['as_of']} 기준 · 서버 계산값입니다. "
            "VaR·CVaR 지표는 서버에서 제공되면 이 자리에 추가됩니다."
        )

    with card("drawdown"):
        st.subheader("하락폭 추이")
        st.caption("0%는 최고점, 아래로 내려갈수록 최고점보다 많이 떨어진 상태입니다.")
        st.altair_chart(drawdown_trend(result["path"]), theme=None, width="stretch")

with card("factors"):
    st.subheader("AI 판단 요인")
    st.caption("AI가 자산 비중을 정할 때 크게 반영한 요인입니다. 자산을 골라 보세요.")
    try:
        shap = load_model_shap(None)
    except APIError as exc:
        st.warning(str(exc))
        shap = None
    if shap:
        ticker = st.selectbox(
            "자산",
            list(shap["assets"]),
            format_func=lambda t: f"{t} · {ASSET_NAMES.get(t, t)}",
            key="risk_shap_asset",
        )
        row = shap["assets"][ticker]["decisions"][-1]
        factors = group_shap_features(shap["features"], row["shap_values"], ticker, row)
        selected, _ = select_factors(factors, "unknown")
        st.caption(f"결정일 {row['date']} · 영향이 큰 순서")
        if not selected:
            st.info(
                "요약할 주요 요인이 없습니다. 상세 화면에서 전체 요인을 확인하세요."
            )
        for factor in selected:
            st.markdown(f"**{factor['label']}**", help=factor["explanation"])
            st.caption(
                f"{compact_state(factor)} → "
                f"{DIRECTIONS.get(factor['direction'], '중립')}으로 작용 · "
                f"{factor['indicator_label']}"
            )
        st.caption("모델 판단에 대한 기여이며 경제적 인과관계나 투자 권고가 아닙니다.")
    detail_link("shap", "AI 판단 요인 자세히 보기")

with card("guide"):
    st.subheader("주요 지표 쉽게 이해하기")
    for column, item in zip(st.columns(len(INDICATOR_GUIDE)), INDICATOR_GUIDE):
        with column:
            st.markdown(f"**{item['name']}**")
            st.write(item["summary"])
            for line in item["points"]:
                st.caption(f"· {line}")

detail_link("anova", "전략 차이가 통계적으로 의미 있는지 보기 (ANOVA)")
