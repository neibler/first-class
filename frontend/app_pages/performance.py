import streamlit as st

from frontend.ui import page_heading

page_heading(
    "강화학습 성과",
    "STRATEGY PERFORMANCE / 02",
    "학습 기록과 전략별 성과를 확인하는 공간입니다.",
)
st.info("화면 준비 중")
with st.container(key="analysis-availability"):
    st.caption(
        "분석 데이터 없음 · 학습·백테스트 결과 조회 기능이 아직 연결되지 않았습니다."
    )
with st.container(key="analysis-panel-training"):
    st.subheader("PPO 학습 기록")
    st.write(
        "Episode Reward, 학습 스텝, 보상 함수 및 학습 설정을 제공할 학습 기록이 없습니다."
    )
with st.container(key="analysis-panel-comparison"):
    st.subheader("전략 성과 비교")
    st.write(
        "DRL·MVO·동일가중·벤치마크의 수익률과 상세 성과표를 제공할 백테스트 결과가 없습니다."
    )
with st.container(key="analysis-panel-walk-forward"):
    st.subheader("Walk-Forward 검증")
    st.write("학습·검증 구간과 구간별 성과가 연결되면 이곳에서 확인할 수 있습니다.")
