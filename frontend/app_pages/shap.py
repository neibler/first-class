"""Explain prepared samples; never import a model or a SHAP explainer."""

import streamlit as st

from frontend.charts import contribution_plot, force_plot, summary_plot
from frontend.sample_data import load_shap, summary_records
from frontend.ui import load_sample, page_heading, sample_badge

page_heading(
    "SHAP 해석",
    "DECISION INSIGHTS / 03",
    "어떤 피처가 자산 비중 결정에 영향을 주었는지 살펴보세요.",
)
sample_badge()
data = load_sample(load_shap, "shap")
if data:
    asset_col, date_col = st.columns(2)
    with asset_col:
        ticker = st.selectbox("자산", list(data["assets"]), key="shap_asset")
    asset = data["assets"][ticker]
    with date_col:
        dates = [d["date"] for d in asset["decisions"]]
        date = st.selectbox("결정일", dates, index=len(dates) - 1, key="shap_date")
    decision = next(d for d in asset["decisions"] if d["date"] == date)
    features = data["features"]
    records = summary_records(data, ticker, date)
    st.caption(
        f"{ticker} · {asset['name']} · {date} · 출력 단위: 자산 비중 (identity link)"
    )
    st.caption(
        "사전에 준비한 합성 SHAP 예시입니다. 실제 모델의 추론·해석 결과가 아닙니다."
    )
    with st.container(key="shap-plots"):
        summary_column, contribution_column = st.columns([1.15, 1], gap="small")
        with summary_column:
            with st.container(key="summary"):
                st.subheader("전체 영향 분포 · Summary Plot")
                st.caption(
                    f"선택 자산의 결정일까지 최근 {len(records) // len(features)}개 관측치 "
                    "· 평균 |SHAP| 순"
                )
                st.altair_chart(summary_plot(records, features), theme=None)
                st.caption(
                    "점 하나는 한 관측일의 피처입니다. 가로 위치는 비중에 미친 영향, "
                    "색은 피처 자체의 정규화 값(파랑: 낮음, 금색: 높음)입니다. 색이 기여도의 부호를 뜻하지 않습니다."
                )
        with contribution_column:
            with st.container(key="contributions"):
                st.subheader("선택 결정 기여도 · 보조 차트")
                st.altair_chart(contribution_plot(features, decision), theme=None)
                with st.expander("SHAP 상세 표", expanded=False):
                    st.dataframe(
                        [
                            {
                                "피처": f,
                                "피처 값 (0–1)": v,
                                "SHAP 값": s,
                                "비중 기여 (%p)": s * 100,
                            }
                            for f, v, s in zip(
                                features,
                                decision["feature_values"],
                                decision["shap_values"],
                            )
                        ],
                        hide_index=True,
                        column_config={
                            "피처 값 (0–1)": st.column_config.NumberColumn(
                                format="%.3f"
                            ),
                            "SHAP 값": st.column_config.NumberColumn(format="%+.4f"),
                            "비중 기여 (%p)": st.column_config.NumberColumn(
                                format="%+.2f"
                            ),
                        },
                    )
                    st.caption(
                        "예: SHAP +0.02는 비중 +2%p 기여를 뜻합니다. 피처 값은 합성 데이터의 "
                        "피처별 0–1 정규화 값이며 원래 RSI·수익률 단위가 아닙니다. "
                        "기준값은 해당 자산의 샘플 배경 출력입니다. 피처 기여도는 인과관계나 투자 권고가 아닙니다."
                    )
    with st.container(key="force"):
        st.subheader("개별 결정 · Force Plot")
        positive = sum(max(v, 0) for v in decision["shap_values"])
        negative = sum(min(v, 0) for v in decision["shap_values"])
        st.write(
            f"기준 {decision['base_value']:.2%} + 증가 {positive * 100:.2f}%p "
            f"− 감소 {abs(negative) * 100:.2f}%p = "
            f"**결정 비중 {decision['final_value']:.2%}**"
        )
        force_plot(features, decision)
        st.caption(
            "금색 →: 비중 증가 (+) · 파랑 ←: 비중 감소 (−). 각 띠의 폭은 |SHAP|이며 "
            "두 방향의 기여도가 최종 비중 f(x)에서 만납니다. 점선은 기준값입니다. "
            "띠 위 숫자는 %p이며, 마우스를 올리면 피처와 기여도를 볼 수 있습니다."
            " 좁은 화면에서는 차트를 가로로 밀어 확인하세요."
        )
