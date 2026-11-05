import streamlit as st

from frontend import performance_client
from frontend.charts import performance_comparison
from frontend.performance_view import chart_rows, evaluation_text, metric_rows
from frontend.ui import page_heading

page_heading(
    "강화학습 성과",
    "STRATEGY PERFORMANCE / 02",
    "저장된 백테스트 결과로 전략별 누적수익률과 12개 성과지표를 비교합니다.",
)
with st.container(key="analysis-panel-training"):
    st.subheader("PPO 학습 기록")
    st.write("PPO 학습 곡선과 학습 설정 표시는 후속 작업입니다.")
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
            f"평가 기간: {result.start_date} ~ {result.end_date} · "
            f"평가 방식: {evaluation_text(result)}"
        )
        if result.smoke:
            st.warning(
                "검증용 smoke 결과입니다. 실제 학습 모델의 투자 성과가 아닙니다."
            )
        if result.window_plan:
            used = result.window_plan.used
            st.caption(
                f"실제 설정: 학습 {used.train_days}거래일 · "
                f"테스트 {used.test_days}거래일 · {used.folds}개 구간"
            )
            for message in result.window_plan.warnings:
                st.warning(message)
        if not result.strategies:
            st.info("표시할 백테스트 전략 결과가 없습니다.")
        else:
            if result.path:
                st.altair_chart(
                    performance_comparison(chart_rows(result)), width="stretch"
                )
            else:
                st.info("표시할 일별 누적수익률 경로가 없습니다.")
            st.subheader("전략별 12개 성과지표")
            st.dataframe(
                metric_rows(result), hide_index=True, width="stretch", height="content"
            )
            st.caption(
                "서버 계산값 · Alpha는 연환산, VaR·CVaR는 일별 손실 기준, "
                "MDD는 손실 크기입니다. 비율 지표는 단위가 없으며 —는 정의 불가 값입니다."
            )
with st.container(key="analysis-panel-walk-forward"):
    st.subheader("Walk-Forward 검증")
    st.write(
        "현재 평가 방식은 전략 비교에 표시합니다. 구간별 상세 성과 조회는 후속 작업입니다."
    )
