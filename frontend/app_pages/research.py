import streamlit as st

from frontend.ui import page_heading

page_heading(
    "에이전트 리서치",
    "AGENT RESEARCH / 04",
    "투자 질문의 조사 결과와 원문 근거를 확인하는 공간입니다.",
)
st.info("화면 준비 중")
with st.container(key="analysis-availability"):
    st.caption(
        "분석 데이터 없음 · 질문 전송 및 리서치 결과 조회 기능이 아직 연결되지 않았습니다."
    )
with st.container(key="analysis-panel-request"):
    st.subheader("리서치 요청 및 진행 상태")
    st.write(
        "질문·티커를 접수하는 기능과 실제 검색·검증 활동 로그가 아직 제공되지 않습니다."
    )
with st.container(key="analysis-panel-report"):
    st.subheader("리서치 리포트와 출처")
    st.write(
        "조회할 리포트와 원문 출처가 없습니다. 실제 결과가 연결되면 문서 길이에 맞춰 표시합니다."
    )
with st.container(key="analysis-panel-tags"):
    st.subheader("리스크 태그 연동")
    st.write("탐지 결과와 리스크 모니터링 연동 상태를 확인할 데이터가 없습니다.")
