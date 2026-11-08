"""Display real full-input SHAP from the API."""

import pandas as pd
import numpy as np
import streamlit as st
from frontend.api_client import APIError
from frontend.charts import force_plot
from frontend.shap_client import load_model_shap
from frontend.ui import page_heading
from frontend.shap_explanations import aggregate, top_factors, category_totals

page_heading(
    "SHAP 해석",
    "DECISION INSIGHTS / 03",
    "Portfolio X-Ray · AI의 자산 배분 결정을 쉬운 요인으로 살펴보세요.",
)
model = st.selectbox(
    "해석 모델 (투자 화면의 활성 모델과 별도 선택)",
    ["ppo", "recurrent_ppo", "sac", "td3"],
    key="shap_model",
)
try:
    data = load_model_shap(model)
except APIError as exc:
    st.error(str(exc))
    if st.button("다시 시도", key="retry_shap"):
        st.rerun()
    st.stop()
ticker = st.selectbox("자산", list(data["assets"]), key="shap_asset")
records = data["assets"][ticker]["decisions"]
date = st.selectbox(
    "결정일", [r["date"] for r in records], index=len(records) - 1, key="shap_date"
)
row = next(r for r in records if r["date"] == date)
factors = aggregate(data["features"], row["shap_values"], ticker)

st.subheader(f"AI가 {ticker} 비중을 정한 주요 요인")
st.caption(
    "이번 AI 판단에서 어떤 정보가 비중을 높이거나 낮추는 방향으로 작용했는지 보여줍니다."
)
st.metric("목표 비중", f"{row['final_value']:.2%}")

for column, direction, title in zip(
    st.columns(2),
    ["increase", "decrease"],
    ["↑ 비중을 높인 주요 요인", "↓ 비중을 낮춘 주요 요인"],
):
    with column:
        st.subheader(title)
        top = top_factors(factors, direction)
        if not top:
            st.caption("뚜렷한 요인이 없습니다.")
        for factor in top:
            label, indicator = st.columns([3, 1.4])
            with label:
                st.markdown(f"**{factor['label']}**")
            with indicator:
                st.caption(
                    "▰" * factor["level"]
                    + "▱" * (4 - factor["level"])
                    + " "
                    + factor["indicator_label"]
                )
st.caption(
    "SHAP은 이번 모델 판단에 대한 기여도를 보여주며, 경제적 인과관계를 의미하지 않습니다."
)

if st.button(
    (
        "상세 분석 닫기"
        if st.session_state.get("shap_details", False)
        else "상세 분석 보기"
    ),
    key="shap_details_button",
):
    st.session_state["shap_details"] = not st.session_state.get("shap_details", False)
    st.rerun()
if st.session_state.get("shap_details", False):
    st.subheader("상세 분석")
    st.caption(
        f"{date} 관측·결정 → {row['execution_date']} 체결. 안전장치 적용 전 정책 목표 비중입니다."
    )
    st.caption(
        "방향은 signed sum, 순위는 absolute sum 기준입니다. 중립·상쇄 요인은 요약에서 제외하지만 이 표에는 보존합니다."
    )
    st.caption(
        "영향도는 전체 요인 중 최대 절댓값 합 대비 75% 이상 매우 큼,"
        "40% 이상 큼, 15% 이상 보통, 나머지는 작음입니다. 신뢰도나 통계적 유의성을 의미하지 않습니다."
    )
    table = pd.DataFrame(
        [
            {
                "요인": f["label"],
                "분류": f["category"],
                "signed contribution (%p)": f["signed_sum"] * 100,
                "absolute contribution (%p)": f["absolute_sum"] * 100,
                "방향": {
                    "increase": "증가",
                    "decrease": "감소",
                    "balanced": "상쇄·중립",
                }[f["direction"]],
            }
            for f in factors
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.dataframe(
        pd.DataFrame(category_totals(factors)), hide_index=True, width="stretch"
    )
    st.caption(f"모델 {data['model_type']} · 실행 {data['run_id']} · {data['method']}")
    st.caption(f"순열 {data['permutations']}회 · 배경 종료 {data['background_end']}")
    st.caption(data["note"])
    values = np.asarray(row["shap_values"])
    raw = pd.DataFrame(
        {
            "raw feature": data["features"],
            "model input value": row["feature_values_raw"],
            "shap value": values,
            "signed contribution (%p)": values * 100,
            "absolute contribution (%p)": np.abs(values) * 100,
        }
    )
    st.dataframe(raw, hide_index=True, width="stretch")
    st.write(
        f"기준 {row['base_value']:.2%}"
        f" + 기여도 합 {values.sum()*100:+.2f}%p = 목표 {row['final_value']:.2%}"
    )
    force_plot(data["features"], row)
    st.caption("risk_type은 모델 입력이 아니므로 유형별 기여도를 만들지 않습니다.")
    st.divider()
    st.markdown("**Rolling Correlation · 이 자산은 무엇과 같이 움직이는가?**")
    st.caption(
        "SHAP은 AI 결정 기여도이고 Correlation은 시장의 동행 관계입니다."
        "현재 이 화면에는 Rolling Correlation 결과가 연결되어 있지 않습니다."
    )
