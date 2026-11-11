"""Read saved ANOVA results through FastAPI, independently of model readiness."""

import streamlit as st

from frontend import statistics_client as client
from frontend import statistics_view as view
from frontend.ui import page_heading

page_heading(
    "ANOVA 검증",
    "STATISTICAL ANALYSIS / 05",
    "보상 함수·전략·시장 국면별 통계 검증 결과를 확인하세요.",
)
with st.container(key="analysis-availability"):
    st.caption(
        "저장된 통계 결과 · 유의수준 0.05 · 비유의는 차이가 없다는 증명이 아닙니다. "
        "평가 기간과 실행 정보는 이 결과에 제공되지 않습니다."
    )
    st.button("통계 결과 새로고침", key="refresh_statistics")

try:
    with st.spinner("통계 결과를 불러오는 중입니다…"):
        result = client.get_statistics()
except client.StatisticsAPIError as exc:
    st.warning(str(exc))
else:
    if result.unsupported_count:
        st.warning(
            f"지원하지 않는 검정 유형 {result.unsupported_count}개를 제외했습니다. "
            "아래에는 해석 가능한 결과만 표시합니다."
        )
    if not result.results:
        st.info("표시할 통계 검증 결과가 없습니다.")
    else:
        one_way = [r for r in result.results if isinstance(r, client.OneWay)]
        two_way = [r for r in result.results if isinstance(r, client.TwoWay)]
        with st.container(key="analysis-panel-one-way"):
            st.subheader("One-way ANOVA")
            for group, title in (("reward", "보상 함수별"), ("strategy", "전략별")):
                if not any(r.groups_label == group for r in one_way):
                    st.info(f"{title} 검증 결과가 제공되지 않았습니다.")
            for item in one_way:
                st.markdown(f"**{item.label}**")
                st.dataframe(
                    [view.test_row(item.label, item)],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(f"표본 출처: {item.sample_source or '제공되지 않음'}")
                chart = view.mean_chart(item)
                if chart is not None:
                    st.altair_chart(chart, theme=None, width="stretch")
                st.dataframe(view.group_rows(item), hide_index=True, width="stretch")
                st.caption(
                    "평균·표준편차는 서버가 제공한 표본 수익률 기준입니다. "
                    "표본 기간이 다른 검증의 평균을 직접 비교하지 마세요."
                )
                with st.expander(f"해석·가정 점검 · {item.label}"):
                    if item.interpretation:
                        st.write(item.interpretation)
                    if item.assumptions:
                        assumptions = item.assumptions
                        rows = [
                            {
                                "가정 검정": f"Shapiro–Wilk · {view.label(g)}",
                                "p값": view.number(p),
                            }
                            for g, p in assumptions.shapiro_p_value.items()
                        ]
                        rows.append(
                            {
                                "가정 검정": "Levene (등분산성)",
                                "p값": view.number(assumptions.levene_p_value),
                            }
                        )
                        st.dataframe(rows, hide_index=True, width="stretch")
                        if assumptions.note:
                            st.caption(assumptions.note)
                    if item.nonparametric:
                        st.write(
                            "보조 검정 Kruskal–Wallis · "
                            f"H={view.number(item.nonparametric.h_statistic)}, "
                            f"p={view.number(item.nonparametric.p_value)}"
                        )
                    if item.independence:
                        independence = item.independence
                        st.write(
                            "표본 독립성 점검: "
                            + ("수행" if independence.available else "점검 불가")
                        )
                        if independence.note:
                            st.caption(independence.note)
                        if independence.available:
                            st.write(
                                "그룹 간 공유 날짜 수: "
                                f"{independence.dates_shared_across_groups} · "
                                "그룹·날짜 중복 행 수: "
                                f"{independence.duplicated_group_date_rows}"
                            )

        with st.container(key="analysis-panel-two-way"):
            st.subheader("시장 국면별 Two-way ANOVA")
            if not two_way:
                st.info("시장 국면별 검증 결과가 제공되지 않았습니다.")
            for item in two_way:
                st.markdown(f"**{item.label}**")
                st.dataframe(
                    [
                        view.test_row(view.label(term), values, partial=True)
                        for term, values in item.terms.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    f"전체 표본 수: {item.n} · η²와 partial η²는 서로 다른 효과크기 지표입니다."
                )
                if not item.interaction_included:
                    st.info("이 결과에는 상호작용 항이 포함되지 않았습니다.")
                with st.expander(f"표본 수·해석 · {item.label}"):
                    st.caption(
                        "셀 표기는 서버의 요인 순서 "
                        + " × ".join(view.label(f) for f in item.factors)
                        + "를 따릅니다. 표본 수는 거래일 수와 다를 수 있습니다."
                    )
                    if item.cell_counts:
                        st.dataframe(
                            [
                                {"요인 조합": cell, "표본 수": count}
                                for cell, count in item.cell_counts.items()
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                    if item.interpretation:
                        st.write(item.interpretation)

        with st.container(key="analysis-panel-posthoc"):
            st.subheader("Tukey HSD 사후검정")
            for item in result.results:
                st.markdown(f"**{item.label}**")
                if isinstance(item, client.OneWay):
                    blocks = [(item.groups_label, item, item.posthoc)]
                else:
                    blocks = [
                        (factor, item.terms[factor], (item.posthoc or {}).get(factor))
                        for factor in item.factors
                        if factor in item.terms
                    ]
                    st.caption(
                        "주효과 요인별 사후검정입니다. 상호작용의 사후검정은 제공되지 않습니다."
                    )
                for factor, values, posthoc in blocks:
                    st.write(
                        f"{view.label(factor)} · {view.posthoc_status(values, posthoc)}"
                    )
                    if posthoc is not None and posthoc.pairs:
                        st.dataframe(
                            view.tukey_rows(posthoc), hide_index=True, width="stretch"
                        )
                        st.caption(
                            f"신뢰수준 {1 - posthoc.alpha:.0%} · 평균 차이 = 그룹 2 − 그룹 1 · "
                            "수익률 차이와 신뢰구간은 %p · p값은 Tukey 다중비교 보정값입니다."
                        )
                if isinstance(item, client.TwoWay) and item.posthoc_note:
                    st.caption(item.posthoc_note)
        st.caption(
            "—는 계산 불가 또는 값 미제공입니다. 표와 차트는 서버 계산값을 표시합니다."
        )
