"""Full relationship analysis of one ETF (the team's correlation API, unchanged)."""

import altair as alt
import pandas as pd
import streamlit as st
from frontend.api_client import APIError
from frontend.charts import chart_style
from frontend.correlation_client import load_correlations
from frontend.etf_choice import choose_etf
from frontend.shap_explanations import ASSET_NAMES
from frontend.ui import page_heading

page_heading(
    "자산 관계 상세",
    "ETF ANALYSIS / RELATIONS",
    "고른 ETF가 다른 ETF·시장 지표와 얼마나 같이 움직였는지 숫자와 그래프로 자세히 봅니다.",
)
ticker, window = choose_etf("relations")
st.caption(
    "상관계수는 +1에 가까울수록 같이, −1에 가까울수록 반대로 움직였다는 뜻입니다. "
    "시장 데이터만으로 계산했으며 AI 판단 근거(SHAP)나 뉴스 리서치와 별개입니다."
)
try:
    data = load_correlations(ticker, window)
except APIError as exc:
    st.error(str(exc))
    if st.button("다시 시도", key="retry_correlation"):
        st.rerun()
    data = None
if data is not None:
    st.subheader(f"{ticker} 자산 관계 분석")
    st.caption(f"데이터 기준일: {data['date'] or '없음'} · 최근 {window}거래일")
    if data["window_start"]:
        st.caption(
            f"분석 구간: {data['window_start']} ~ {data['date']}"
            " · ETF 로그수익률 / WTI 단순수익률 / 금리·VIX·금리차 일별 변화량 기준"
        )
    st.caption(data["alignment_note"])
    st.caption(
        "관계 강도는 이해를 돕기 위한 구간 표현이며 절대적인 통계 기준이 아닙니다."
    )
    relationships = data["relationships"]
    if not relationships:
        st.info(data["message"])
    else:
        st.markdown("**관계가 큰 항목 Top 5**")
        for item in relationships:
            name = item["display_name"]
            if item["type"] == "ETF":
                name = f"{ASSET_NAMES.get(name, name)}({name})"
            st.markdown(f"**{name} · {item['correlation']:+.2f}**")
            st.caption(
                f"{item['strength']} · {item['relationship_label']} · 최근 {window}거래일"
            )
        selected = st.selectbox(
            "비교할 자산·지표",
            [r["name"] for r in relationships],
            format_func=lambda name: next(
                r["display_name"] for r in relationships if r["name"] == name
            ),
            key="correlation_pair",
        )
        item = next(r for r in relationships if r["name"] == selected)
        st.markdown(f"**{ticker} ↔ {item['display_name']}**")
        current, previous, difference = st.columns(3)
        current.metric("현재 구간 상관계수", f"{item['correlation']:+.2f}")
        prior = item.get("previous_correlation")
        previous.metric(
            "이전 구간 상관계수", f"{prior:+.2f}" if prior is not None else "자료 부족"
        )
        difference.metric(
            "상관계수 변화",
            f"{item['correlation']-prior:+.2f}" if prior is not None else "—",
        )
        st.caption(f"{item['strength']} · {item['relationship_label']}")
        if item.get("change_description"):
            st.caption(item["change_description"])
            st.caption(
                f"이전 구간은 {item['previous_date']}에 끝나는 겹치지 않는 {window}거래일입니다."
            )

        comparison = data.get("comparisons", {}).get(selected, {})
        kind = "가격" if item["type"] == "ETF" else "시장값"
        st.markdown(f"**최근 {window}거래일 정규화 {kind} 변화 비교**")
        if comparison.get("status") == "ok":
            records = comparison["records"]
            points = [
                {**record, "series": comparison[label], "normalized": record[key]}
                for record in records
                for key, label in [
                    ("selected", "selected_label"),
                    ("comparison", "comparison_label"),
                ]
            ]
            paths = (
                alt.Chart(pd.DataFrame(points))
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="날짜"),
                    y=alt.Y(
                        "normalized:Q",
                        title="시작값 = 100",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color(
                        "series:N",
                        title="비교 대상",
                        scale=alt.Scale(range=["#2f6fed", "#f2a33a"]),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="날짜"),
                        alt.Tooltip(
                            "selected:Q",
                            title=f"{comparison['selected_label']} (시작=100)",
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "comparison:Q",
                            title=f"{comparison['comparison_label']} (시작=100)",
                            format=".2f",
                        ),
                    ],
                )
            )
            st.altair_chart(chart_style(paths.properties(height=260)), width="stretch")
            st.caption(
                f"{comparison['start']} ~ {comparison['end']} · "
                "두 시계열의 시작값을 100으로 맞춰 상대 움직임을 비교합니다."
            )
            st.caption(comparison["source_note"])
            if comparison.get("message"):
                st.info(comparison["message"])
        else:
            st.info(
                comparison.get(
                    "message",
                    "정규화 비교 자료가 없습니다. API 서버를 재시작해 주세요.",
                )
            )
        st.caption(
            (
                "이 그래프는 상관계수 계산에 사용하지 않습니다. "
                "통계는 일별 수익률·변화량으로 계산하며, 모든 날짜가 같은 방향으로 움직인다는 뜻은 아닙니다."
            )
        )
        st.markdown(f"**{window}거래일 Rolling Correlation 변화**")
        st.caption(
            (
                f"각 날짜에서 직전 {window}거래일 데이터를 이용해 계산한 상관계수입니다. "
                "그래프는 여러 분석 구간의 변화를 보여줍니다."
            )
        )
        history = pd.DataFrame(data["rolling"][selected])
        line = (
            alt.Chart(history)
            .mark_line(color="#2f6fed")
            .encode(
                x=alt.X("date:T", title="날짜"),
                y=alt.Y(
                    "correlation:Q", title="상관계수", scale=alt.Scale(domain=[-1, 1])
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="날짜"),
                    alt.Tooltip("correlation:Q", format="+.3f", title="상관계수"),
                ],
            )
        )
        zero = (
            alt.Chart(pd.DataFrame({"zero": [0]}))
            .mark_rule(color="#8a97a8", strokeDash=[4, 4])
            .encode(y="zero:Q")
        )
        st.altair_chart(
            chart_style((line + zero).properties(height=260)), width="stretch"
        )
        st.caption(item["description"])

    with st.expander("상세 관계와 계산 기준 보기"):
        if data["all_relationships"]:
            table = pd.DataFrame(data["all_relationships"])
            st.dataframe(
                table[
                    [
                        "display_name",
                        "type",
                        "correlation",
                        "strength",
                        "transformation",
                        "observations",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
        st.caption(
            "ETF: 로그수익률 · 금리/VIX/장단기 차이: 일별 차분 · WTI: 일별 단순수익률. 각 구간 전체 관측이 필요합니다."
        )
        for excluded in data["excluded_features"] + data["unavailable_relationships"]:
            st.caption(f"{excluded['name']}: {excluded['reason']}")
        st.caption(
            "강도: |r| < 0.2 매우 약함 / < 0.4 약함 / < 0.6 보통 / < 0.8 강함 / 그 이상 매우 강함"
        )
    st.caption(data["disclaimer"])
    st.caption(
        "상관관계가 높다고 AI가 해당 정보를 사용했다는 뜻은 아닙니다. 수익률 예측이나 투자 추천이 아닙니다."
    )
