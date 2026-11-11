"""Display real full-input SHAP from the API."""

import pandas as pd
import numpy as np
import streamlit as st
from frontend.api_client import APIError
from frontend.charts import force_plot
from frontend.shap_client import load_model_shap
from frontend.ui import page_heading
from frontend.shap_explanations import group_shap_features, category_totals

from frontend.shap_presentation import decision_direction, select_factors, compact_state

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
factors = group_shap_features(data["features"], row["shap_values"], ticker, row)

st.subheader(f"{ticker} 비중 결정과 모델 설명")
st.caption(f"해석 모델: {data['model_type']} · 실행: {data['run_id']} · 결정일: {date}")
st.caption("공유 목록의 저장된 SHAP입니다. 투자 화면의 활성 모델과 다를 수 있습니다.")
st.caption(
    data.get(
        "weight_stage", "이 SHAP의 최소 비중 필터 적용 여부는 확인되지 않았습니다."
    )
)
threshold = data.get("min_weight_threshold")
if threshold is not None:
    st.caption(f"이 학습본의 최소 목표 비중 설정: {threshold:.1%}")

execution = row.get("execution_context")
direction = decision_direction(execution)
selected, opposite = select_factors(factors, direction)
labels = {"increase": "증가", "decrease": "감소", "hold": "유지"}
if direction != "unknown":
    st.markdown(f"**{ticker} 비중 {labels[direction]} 목표**")
    before, target, after = st.columns(3)
    before.metric("거래 전 비중", f"{execution['previous_weight']:.2%}")
    target.metric("AI 목표 비중", f"{execution['target_weight']:.2%}")
    after.metric("거래 후 비중", f"{execution['current_weight']:.2%}")
    st.caption(
        f"{execution['execution_date']} 평가 기록 · "
        + (
            "해당 ETF 비중 변경 거래 있음"
            if execution["traded"]
            else "해당 ETF 거래 없음"
        )
    )
else:
    st.caption(
        "거래 기록을 확인하지 못했습니다. 증가·감소를 추정하지 않고 주요 정보만 표시합니다."
    )

titles = {
    "increase": f"{ticker} 비중을 높이는 데 크게 작용한 요인",
    "decrease": f"{ticker} 비중을 낮추는 데 크게 작용한 요인",
    "hold": f"현재 {ticker} 비중 결정에 크게 반영된 요인",
    "unknown": f"{ticker} 목표 비중 결정에 크게 반영된 요인",
}
st.subheader(titles[direction])
st.caption(
    "목표 변화와 같은 방향의 모델 기여를 보여줍니다. 실제 거래 원인을 증명하는 설명은 아닙니다."
)
if not selected:
    st.caption(
        "목표 변화와 같은 방향의 요인이 없습니다. 상세 분석에서 전체 기여를 확인하세요."
    )
for factor in selected:
    label, indicator = st.columns([3, 1.4])
    with label:
        st.markdown(f"**{factor['label']}**", help=factor["explanation"])
        st.caption(compact_state(factor))
        if direction in {"increase", "decrease"}:
            st.caption(f"→ {ticker} 비중 {labels[direction]} 방향의 기여")
    with indicator:
        st.caption(
            "▰" * factor["level"]
            + "▱" * (4 - factor["level"])
            + " "
            + factor["indicator_label"]
        )
st.caption("이 설명은 모델 판단에 대한 기여이며 경제적 인과관계를 의미하지 않습니다.")

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
    base, output = st.columns(2)
    base.metric("SHAP 기준 비중", f"{row['base_value']:.2%}")
    output.metric("SHAP이 설명하는 출력 비중", f"{row['final_value']:.2%}")
    st.caption(
        "SHAP 기준 비중은 이전 보유 비중이 아닙니다. 기여의 부호만으로 실제 매매 방향을 판단하지 않습니다."
    )
    with st.expander("반대 방향으로 작용한 요인 보기"):
        if opposite:
            st.caption(
                "목표 변화와 반대 방향으로 기여한 정보도 있습니다. 반대 기여가 더 작았다고 단정할 수는 없습니다."
            )
            for factor in opposite:
                st.markdown(f"**{factor['label']}**")
                st.caption(factor["explanation"])
        else:
            st.caption(
                "반대 방향 요인이 없거나, 유지·방향 미확인 상태입니다. 전체 요인은 아래 표에 보존됩니다."
            )

    st.caption(
        f"{date} 관측·결정 → {row['execution_date']} 평가 기록 시점. SHAP 출력과 실제 보유 비중은 구분합니다."
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
                "실제 상태": f["actual_state_summary"],
                "설명": f["explanation"],
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
        "왼쪽 자산 관계 분석 메뉴에서 실제 시장의 Rolling Correlation을 확인할 수 있습니다."
    )
