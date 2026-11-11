"""Format server statistics without recomputing tests or inventing samples."""

import altair as alt

from frontend.charts import COLORS, chart_style
from frontend.statistics_client import OneWay, TestValues, Tukey

LABELS = {
    "reward": "보상 함수",
    "return": "수익률 보상",
    "sharpe": "샤프비율 보상",
    "return_mdd": "수익률 + MDD 페널티",
    "ppo": "PPO",
    "mvo": "MVO",
    "equal_weight": "동일가중",
    "strategy": "전략",
    "regime": "시장 국면",
    "strategy:regime": "전략 × 시장 국면",
    "regime:strategy": "시장 국면 × 전략",
    "bull": "상승장",
    "sideways": "횡보장",
    "bear": "하락장",
}


def label(value: str) -> str:
    return LABELS.get(value, value)


def number(value: float | None) -> str:
    if value is None:
        return "—"
    if value != 0 and abs(value) < 0.0001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def verdict(values: TestValues) -> str:
    if values.p_value is None or values.f_statistic is None:
        return "계산 불가"
    return "유의" if values.significant else "비유의"


def test_row(name: str, values: TestValues, *, partial: bool = False) -> dict:
    row = {
        "검정 / 요인": name,
        "F값": number(values.f_statistic),
        "p값": number(values.p_value),
        "η²": number(values.eta_squared),
        "효과크기": (
            (values.effect_size_label or "—") if values.eta_squared is not None else "—"
        ),
        "판정": verdict(values),
    }
    if partial:
        row["partial η²"] = number(values.partial_eta_squared)
    return row


def group_rows(result: OneWay) -> list[dict]:
    return [
        {
            "그룹": label(group),
            "표본 수": result.n_per_group[group],
            "평균 수익률": (
                "—"
                if result.group_mean[group] is None
                else f"{result.group_mean[group]:.2%}"
            ),
            "표준편차 (%p)": (
                "—"
                if result.group_std[group] is None
                else f"{result.group_std[group] * 100:.2f}"
            ),
        }
        for group in result.groups
    ]


def mean_chart(result: OneWay):
    rows = [
        {"group": label(group), "mean": result.group_mean[group]}
        for group in result.groups
        if result.group_mean[group] is not None
    ]
    if not rows:
        return None
    return chart_style(
        alt.Chart(alt.Data(values=rows))
        .mark_bar(color=COLORS[0])
        .encode(
            x=alt.X("mean:Q", title="평균 수익률", axis=alt.Axis(format=".1%")),
            y=alt.Y("group:N", title=None, sort=[row["group"] for row in rows]),
            tooltip=[
                alt.Tooltip("group:N", title="그룹"),
                alt.Tooltip("mean:Q", title="평균 수익률", format=".2%"),
            ],
        )
        .properties(height=max(100, 36 * len(rows)))
    )


def posthoc_status(values: TestValues, posthoc: Tukey | None) -> str:
    if posthoc is not None and posthoc.pairs:
        if any(pair.significant for pair in posthoc.pairs):
            return "Tukey HSD 수행 · 유의한 비교 쌍이 있습니다."
        return "Tukey HSD 수행 · 유의한 비교 쌍이 없습니다."
    if posthoc is not None:
        return "사후검정 결과에 비교 쌍이 없습니다."
    if verdict(values) == "계산 불가":
        return "ANOVA 계산 불가 · 사후검정 적용 여부를 판단할 수 없습니다."
    if values.significant:
        return "ANOVA는 유의하지만 사후검정 결과가 제공되지 않았습니다."
    return "ANOVA 비유의 · 사후검정 미수행."


def tukey_rows(result: Tukey) -> list[dict]:
    return [
        {
            "그룹 1": label(pair.group1),
            "그룹 2": label(pair.group2),
            "평균 차이 (%p)": f"{pair.mean_difference * 100:+.2f}",
            "보정 p값": number(pair.p_adjusted),
            "신뢰구간 하한 (%p)": f"{pair.lower * 100:+.2f}",
            "신뢰구간 상한 (%p)": f"{pair.upper * 100:+.2f}",
            "판정": "유의" if pair.significant else "비유의",
        }
        for pair in result.pairs
    ]
