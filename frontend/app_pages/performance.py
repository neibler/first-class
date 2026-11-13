"""Compare the active model with its benchmarks through the investment /backtest API."""

import streamlit as st

from frontend import performance_client
from frontend.charts import performance_comparison
from frontend.performance_view import chart_rows, convention_text, metric_rows
from frontend.ui import page_heading

page_heading(
    "강화학습 성과",
    "STRATEGY PERFORMANCE / 02",
    "활성 투자 모델의 백테스트로 PPO와 벤치마크의 누적수익률과 성과지표를 비교합니다.",
)
with st.container(key="analysis-panel-training"):
    st.subheader("PPO 학습 기록")
    st.write(
        "공유 모델에는 학습 상세 로그가 포함되지 않아 학습 곡선은 표시하지 않습니다."
    )
with st.container(key="analysis-panel-comparison"):
    st.subheader("전략 성과 비교")
    st.button("백테스트 결과 새로고침", key="refresh_performance")
    try:
        with st.spinner("백테스트 결과를 불러오는 중입니다…"):
            result = performance_client.get_backtest()
    except performance_client.PerformanceAPIError as exc:
        st.warning(str(exc))
    else:
        st.caption(
            f"모델: {result.model_version} · 평가 기간: {result.start_date} ~ "
            f"{result.end_date} (시작 기준일 {result.base_date}) · 기준일: {result.as_of}"
        )
        if result.data_source == "synthetic":
            st.warning(
                "합성 데이터 결과입니다. 실제 시장 데이터의 투자 성과가 아닙니다."
            )
        if result.stopped or result.safeguard_triggered:
            st.warning(
                "평가 기간 중 Safe-Guard가 작동했습니다. 이후 구간은 보호 조치가 반영된 결과입니다."
            )
        if result.missing:
            st.info("이 모델에 저장되지 않은 비교 전략: " + ", ".join(result.missing))
        st.altair_chart(performance_comparison(chart_rows(result)), width="stretch")
        st.subheader("전략별 성과지표")
        st.dataframe(
            metric_rows(result), hide_index=True, width="stretch", height="content"
        )
        st.caption(
            f"서버 계산값 · {convention_text(result)} 거래비용과 회전율은 "
            "초기 자본 대비입니다. —는 정의할 수 없는 값입니다."
        )
with st.container(key="analysis-panel-walk-forward"):
    st.subheader("평가 방식")
    st.write(
        "활성 모델의 단일 평가 기간 결과입니다. Walk-Forward 구간별 결과와 MVO 비교는 "
        "이 API에서 제공되지 않습니다."
    )
