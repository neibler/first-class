import streamlit as st

from frontend.ui import page_heading

page_heading(
    "ANOVA 검증",
    "STATISTICAL ANALYSIS / 05",
    "보상 함수·전략·시장 국면별 통계 검증 결과를 확인하는 공간입니다.",
)
st.info("화면 준비 중")
with st.container(key="analysis-availability"):
    st.caption(
        "분석 데이터 없음 · 검증 표본과 통계 결과 조회 기능이 아직 연결되지 않았습니다."
    )
with st.container(key="analysis-panel-one-way"):
    st.subheader("One-way ANOVA")
    st.write(
        "보상 함수 3종 및 DRL·MVO·동일가중 전략의 비교 표본과 F-statistic, p-value, η² 결과가 없습니다."
    )
with st.container(key="analysis-panel-two-way"):
    st.subheader("시장 국면별 Two-way ANOVA")
    st.write("시장 국면별 그룹 분포·요약 통계와 요인별 검증 결과가 제공되지 않습니다.")
with st.container(key="analysis-panel-posthoc"):
    st.subheader("사후 검정")
    st.write(
        "검정 결과가 없어 적용 여부를 판단할 수 없습니다. 학교 스펙은 p < 0.05인 경우 Tukey HSD를 요구합니다."
    )
