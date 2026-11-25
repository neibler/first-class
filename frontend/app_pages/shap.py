"""Display real full-input SHAP from the API."""

import pandas as pd
import numpy as np
import streamlit as st
from frontend.api_client import APIError
from frontend.charts import force_plot
from frontend.shap_client import load_model_shap
from frontend.etf_choice import choose_asset
from frontend.shap_explanations import ASSET_NAMES
from frontend.ui import card, keep_choice, page_heading
from frontend.shap_explanations import group_shap_features, category_totals

from frontend.shap_presentation import decision_direction, select_factors, compact_state

page_heading(
    "AI 판단 근거",
    "ETF ANALYSIS / AI",
    "AI가 ETF 비중을 정할 때 어떤 신호가 비중을 올리고 내렸는지 풀어서 설명합니다.",
)
try:
    data = load_model_shap(None)
except APIError as exc:
    st.error(str(exc))
    if st.button("다시 시도", key="retry_shap"):
        st.rerun()
    st.stop()
a, b = st.columns(2)
# Opens on the ETF chosen in ETF analysis, and going back keeps the one chosen here.
ticker = choose_asset("shap", data["assets"], where=a)
records = data["assets"][ticker]["decisions"]
dates = [r["date"] for r in records]
# The decision date survives leaving the page; a date this ETF lacks falls back
# to the latest one.
if st.session_state.get("shap_date_choice") not in dates:
    st.session_state["shap_date_choice"] = dates[-1]
keep_choice("shap_date", "shap_date_choice")
if st.session_state["shap_date"] not in dates:
    st.session_state["shap_date"] = dates[-1]
date = b.selectbox(
    "결정일",
    dates,
    key="shap_date",
    help="AI 판단 기록은 한 달에 한 번 저장됩니다. 처음에는 가장 최근 날짜가 선택됩니다.",
)
st.session_state["shap_date_choice"] = date
row = next(r for r in records if r["date"] == date)
factors = group_shap_features(data["features"], row["shap_values"], ticker, row)
execution = row.get("execution_context")
direction = decision_direction(execution)
selected, opposite = select_factors(factors, direction)
labels = {"increase": "증가", "decrease": "감소", "hold": "유지"}
arrows = {"increase": "▲ 높이는 쪽", "decrease": "▼ 낮추는 쪽"}
base, final = row["base_value"], row["final_value"]

st.subheader(f"{ticker} · {ASSET_NAMES.get(ticker, ticker)} — {date} 결정")

with card("shap-howto"):
    st.subheader("이 화면 읽는 법")
    st.markdown(
        f"1. AI는 평소 {ticker}에 주던 **기준 비중**에서 출발합니다.\n"
        "2. 그날의 신호(가격 흐름, 금리, 다른 ETF 움직임 등)가 비중을 조금씩 "
        "**올리거나(▲) 내립니다(▼)**.\n"
        "3. 기준 비중에 신호들의 효과를 모두 더하면 AI가 계산한 **목표 비중**이 됩니다.\n\n"
        "각 신호의 효과를 나눠 계산하는 방법을 **SHAP**이라고 합니다."
    )

with card("shap-summary"):
    st.subheader(f"{ticker} 비중은 이렇게 정해졌어요")
    start, moved, end = st.columns(3)
    start.metric(
        "기준 비중", f"{base:.1%}", help="평소(비교 기준 기간) AI가 주던 평균 비중"
    )
    moved.metric("신호들의 효과", f"{(final - base) * 100:+.1f}%p")
    end.metric("AI 목표 비중", f"{final:.1%}")
    if not any(row["shap_values"]):
        st.info(
            f"AI는 평소에도, 이날에도 {ticker}에 {final:.0%}를 줬습니다. 어떤 신호를 봐도 "
            "비중이 달라지지 않아서 비중을 움직인 신호가 없습니다."
        )
    if direction != "unknown":
        st.caption(
            f"실제 운용: 거래 전 {execution['previous_weight']:.1%} → "
            f"거래 후 {execution['current_weight']:.1%} "
            f"({execution['execution_date']}, "
            + ("거래 있음" if execution["traded"] else "거래 없음")
            + "). 거래 비용·조정 규칙 때문에 목표 비중과 다를 수 있습니다."
        )

if selected:
    with card("shap-factors"):
        st.subheader("비중을 가장 크게 움직인 신호")
        st.caption("영향이 큰 순서입니다. 막대가 길수록 영향이 큽니다.")
        for factor in selected:
            st.markdown(
                f"**{factor['label']}** · "
                f"{arrows.get(factor['direction'], '영향 없음')} "
                f"{factor['signed_sum'] * 100:+.1f}%p"
            )
            st.caption(
                "▰" * factor["level"]
                + "▱" * (4 - factor["level"])
                + f" {factor['indicator_label']} · 그날 상태: {compact_state(factor)}"
            )
            st.caption(factor["explanation"])
        if opposite:
            with st.expander(f"나머지 신호 {len(opposite)}개 보기"):
                for factor in opposite:
                    st.markdown(
                        f"**{factor['label']}** · "
                        f"{arrows.get(factor['direction'], '영향 없음')} "
                        f"{factor['signed_sum'] * 100:+.1f}%p"
                    )
                    st.caption(factor["explanation"])
st.caption(
    "AI 판단에 대한 설명입니다. 신호가 실제 가격을 움직였다는 뜻(인과관계)이나 투자 권유가 아닙니다."
)

if st.button(
    (
        "전문가용 상세 데이터 닫기"
        if st.session_state.get("shap_details", False)
        else "전문가용 상세 데이터 보기"
    ),
    key="shap_details_button",
):
    st.session_state["shap_details"] = not st.session_state.get("shap_details", False)
    st.rerun()
if st.session_state.get("shap_details", False):
    st.subheader("전문가용 상세 데이터")
    st.caption(
        f"해석 모델: {data['model_type']} · 실행: {data['run_id']} · 결정일: {date}"
    )
    st.caption(
        data.get(
            "weight_stage", "이 SHAP의 최소 비중 필터 적용 여부는 확인되지 않았습니다."
        )
    )
    threshold = data.get("min_weight_threshold")
    if threshold is not None:
        st.caption(f"이 학습본의 최소 목표 비중 설정: {threshold:.1%}")
    if direction != "unknown":
        st.caption(
            f"AI 목표 비중(거래 기록): {execution['target_weight']:.2%} · "
            f"비중 {labels[direction]} 목표"
        )
    base, output = st.columns(2)
    base.metric("SHAP 기준 비중", f"{row['base_value']:.2%}")
    output.metric("SHAP이 설명하는 출력 비중", f"{row['final_value']:.2%}")
    st.caption(
        "SHAP 기준 비중은 이전 보유 비중이 아닙니다. 기여의 부호만으로 실제 매매 방향을 판단하지 않습니다."
    )
    with st.expander("나머지 요인 보기"):
        if opposite:
            st.caption("기여 방향과 관계없이 요약에 포함되지 않은 나머지 요인입니다.")
            for factor in opposite:
                st.markdown(f"**{factor['label']}**")
                st.caption(factor["explanation"])
        else:
            st.caption(
                "추가 요인이 없습니다. 상쇄·중립을 포함한 전체 요인은 아래 표에 보존됩니다."
            )

    date_kind = "평가 기록 시점" if execution else "Replay 기준 날짜 (평가 기록 미매칭)"
    st.caption(
        f"{date} 관측·결정 → {row['execution_date']} {date_kind}. "
        "SHAP 출력과 실제 보유 비중은 구분합니다."
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
    st.caption(
        "SHAP은 AI 결정 기여도이고 상관관계는 시장의 동행 관계입니다. "
        "실제 시장의 Rolling Correlation은 ETF 분석의 '자산 관계 자세히 보기'에서 확인할 수 있습니다."
    )
