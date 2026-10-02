"""Minimal API connectivity dashboard."""

import streamlit as st

from frontend.api_client import APIError, get_health

st.set_page_config(page_title="First Class 로보어드바이저")
st.title("First Class 로보어드바이저")
st.caption("API 연결 상태")
st.button("연결 상태 새로고침")

try:
    health = get_health()
except APIError as exc:
    st.error(str(exc))
else:
    st.success("API 연결 정상")
    if health.model_loaded:
        st.info("모델 로드됨")
    else:
        st.warning("모델이 로드되지 않았습니다.")

st.info("포트폴리오 최적화, 뉴스 리서치, SHAP 및 성과 분석은 아직 미구현입니다.")
st.caption(
    "본 시스템은 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없습니다. "
    "백테스팅 성과는 미래 수익을 보장하지 않습니다."
)
