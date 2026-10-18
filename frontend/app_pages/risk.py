import streamlit as st

from frontend.ui import page_heading

page_heading(
    "리스크 모니터링",
    "RISK MONITORING / 06",
    "위험 지표와 경보·보호 조치 이력을 확인하는 공간입니다.",
)
st.info("화면 준비 중")
with st.container(key="analysis-availability"):
    st.caption(
        "분석 데이터 없음 · 현재 위험 수준과 Safe-Guard 실행 여부를 확인할 수 없습니다."
    )
with st.container(key="analysis-panel-risk"):
    st.subheader("위험 지표")
    st.write("VaR·CVaR·MDD 추이와 변동성·상관관계 데이터가 아직 제공되지 않습니다.")
with st.container(key="analysis-panel-safeguard"):
    st.subheader("Safe-Guard 및 경보 이력")
    st.write(
        "학교 스펙의 조건은 MDD 손실 폭 15% 초과입니다. 실제 설정값과 실행 이력은 아직 제공되지 않습니다."
    )
    st.caption("임계치 초과 경보와 실제 조기 종료 실행은 별도로 확인해야 합니다.")
with st.container(key="analysis-panel-research"):
    st.subheader("리서치 리스크 태그")
    st.write("리서치에서 전달된 태그와 연동 상태를 확인할 데이터가 없습니다.")
