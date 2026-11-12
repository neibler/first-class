"""Read saved SHAP explanations through FastAPI; never load a model or explainer."""

import streamlit as st

from frontend import explain_client as client
from frontend import explain_view as view
from frontend.charts import contribution_plot, force_plot, summary_plot
from frontend.ui import page_heading


def keep_choice(key, label, options, default):
    """A refreshed result may drop the previous choice: switch, then say so.

    First visits have no previous choice and stay silent.
    """
    previous = st.session_state.get(key)
    if previous in options:
        return
    st.session_state[key] = default
    if previous is not None:
        st.toast(
            f"기존 선택 항목({label} {previous})이 없어 "
            f"최신 데이터({default})로 변경되었습니다.",
            icon=":material/info:",
        )


page_heading(
    "SHAP 해석",
    "DECISION INSIGHTS / 03",
    "어떤 피처가 자산 비중 결정에 영향을 주었는지 살펴보세요.",
)
with st.container(key="analysis-availability"):
    st.caption(
        "저장된 SHAP 해석 결과 · 모델이 실제로 본 관측으로 계산한 표본 결정일입니다. "
        "실시간 계산이 아니며 평가 기간 전체를 뜻하지 않습니다."
    )
    st.button("SHAP 결과 새로고침", key="refresh_explain")


def show():
    try:
        with st.spinner("SHAP 결과를 불러오는 중입니다…"):
            overview = client.get_overview()
    except client.ExplainAPIError as exc:
        st.warning(str(exc))
        return

    if not overview.assets:
        st.info("표시할 SHAP 해석 결과가 없습니다.")
        return

    asset_col, date_col = st.columns(2)
    with asset_col:
        keep_choice("shap_asset", "자산", overview.assets, overview.assets[0])
        ticker = st.selectbox("자산", overview.assets, key="shap_asset")
    try:
        with st.spinner("선택한 자산의 SHAP 결과를 불러오는 중입니다…"):
            asset = client.get_asset(ticker, overview.features)
    except client.ExplainAPIError as exc:
        st.warning(str(exc))
        return

    dates = [d.date for d in asset.decisions]
    with date_col:
        keep_choice("shap_date", "결정일", dates, dates[-1])
        date = st.selectbox("결정일", dates, key="shap_date")
    decision = next(d for d in asset.decisions if d.date == date)
    features = asset.features
    records = view.summary_records(asset)
    st.caption(
        f"{view.asset_label(asset)} · {date} · 출력 단위: 자산 목표 비중 (identity link)"
    )
    st.caption(asset.note)

    with st.container(key="shap-plots"):
        summary_column, contribution_column = st.columns([1.15, 1], gap="small")
        with summary_column:
            with st.container(key="summary"):
                st.subheader("전체 영향 분포 · Summary Plot")
                st.caption(
                    f"선택 자산의 저장된 결정일 {len(asset.decisions)}개 "
                    f"({dates[0]}~{dates[-1]}) · 평균 |SHAP| 순"
                )
                st.altair_chart(summary_plot(records, features), theme=None)
                st.caption(
                    "점 하나는 한 결정일의 피처입니다. 가로 위치는 비중에 미친 영향, "
                    "색은 피처의 0–1 정규화 값(파랑: 낮음, 금색: 높음)입니다. "
                    "색이 기여도의 부호를 뜻하지 않습니다."
                )
        with contribution_column:
            with st.container(key="contributions"):
                st.subheader("선택 결정 기여도 · 보조 차트")
                st.altair_chart(
                    contribution_plot(features, view.chart_decision(decision)),
                    theme=None,
                )
                with st.expander("SHAP 상세 표", expanded=False):
                    st.dataframe(
                        view.detail_rows(features, decision),
                        hide_index=True,
                        column_config={
                            "모델 입력 값": st.column_config.NumberColumn(
                                format="%.4f"
                            ),
                            "정규화 값 (0–1)": st.column_config.NumberColumn(
                                format="%.3f"
                            ),
                            "SHAP 값": st.column_config.NumberColumn(format="%+.4f"),
                            "비중 기여 (%p)": st.column_config.NumberColumn(
                                format="%+.2f"
                            ),
                        },
                    )
                    st.caption(
                        "예: SHAP +0.02는 비중 +2%p 기여를 뜻합니다. 모델 입력 값은 학습 구간 "
                        "통계로 전처리된 값이며 원래 RSI·수익률 단위가 아닙니다. 정규화 값은 "
                        "이 자산의 저장된 결정일 안에서 0–1로 맞춘 색 표시용 값입니다. "
                        "모델 입력 값이 빈 칸이면 서버가 값을 제공하지 않은 경우입니다."
                    )

    with st.container(key="force"):
        st.subheader("개별 결정 · Force Plot")
        st.write(view.force_summary(decision))
        force_plot(features, view.chart_decision(decision))
        st.caption(
            "금색 →: 비중 증가 (+) · 파랑 ←: 비중 감소 (−). 각 띠의 폭은 |SHAP|이며 "
            "두 방향의 기여도가 최종 비중 f(x)에서 만납니다. 점선은 기준값으로, "
            "이 자산의 저장된 결정일 평균 입력에서 모델이 낼 비중입니다. "
            "기준값 + 모든 기여도 = 최종 비중입니다. "
            "띠 위 숫자는 %p이며, 마우스를 올리면 피처와 기여도를 볼 수 있습니다."
            " 좁은 화면에서는 차트를 가로로 밀어 확인하세요."
        )

    with st.expander("전체 자산 피처 중요도"):
        rows = view.importance_rows(overview)
        if rows:
            st.dataframe(
                rows,
                hide_index=True,
                column_config={
                    "평균 |SHAP| (%p)": st.column_config.NumberColumn(format="%.2f")
                },
            )
            st.caption(
                "서버가 계산한 자산별 평균 |SHAP|의 평균입니다. SHAP은 자산마다 따로 "
                "계산되므로 여러 자산의 기여도를 더한 값은 의미가 없습니다."
            )
        else:
            st.caption("전체 피처 중요도는 이 결과에 제공되지 않았습니다.")


show()
