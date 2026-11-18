"""Native Vega charts and a small browser-only flight-path component."""

from html import escape

import altair as alt
import streamlit as st

COLORS = ["#2f6fed", "#19b8a6", "#f2a33a", "#8b6fe8", "#e86a5c", "#7d8ea3"]


def chart_style(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#44546a",
            titleColor="#5d6d84",
            gridColor="#e3e9f2",
            labelFontSize=12,
            titleFontSize=12,
            labelLimit=155,
        )
        .configure_legend(labelColor="#33435a", titleColor="#5d6d84")
    )


def performance_comparison(records):
    names = list(dict.fromkeys(row["label"] for row in records))
    days = len({row["date"] for row in records})
    chart = (
        alt.Chart(alt.Data(values=records))
        .mark_line(strokeWidth=2, point=days == 1)
        .encode(
            x=alt.X(
                "date:T",
                title="평가일",
                scale=alt.Scale(type="utc"),
                axis=alt.Axis(
                    format="%Y-%m-%d",
                    tickCount=alt.TimeIntervalStep(
                        interval="day", step=max(1, days // 4)
                    ),
                    labelOverlap=True,
                ),
            ),
            y=alt.Y(
                "cumulative_return:Q",
                title="누적수익률",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "label:N",
                title="전략",
                scale=alt.Scale(domain=names, range=COLORS),
                legend=alt.Legend(orient="bottom", columns=1, labelLimit=220),
            ),
            detail="strategy:N",
            tooltip=[
                alt.Tooltip("date:T", title="평가일", format="%Y-%m-%d"),
                alt.Tooltip("label:N", title="전략"),
                alt.Tooltip("cumulative_return:Q", title="누적수익률", format="+.2%"),
            ],
        )
    )
    return chart_style(chart.properties(height=290))


def allocation(assets):
    from frontend.weight_format import format_weight

    displayed = [{**a, "weight_label": format_weight(a["weight"])} for a in assets]
    chart = (
        alt.Chart(alt.Data(values=displayed))
        .mark_arc(innerRadius=52, outerRadius=82, stroke="#ffffff", strokeWidth=3)
        .encode(
            theta=alt.Theta("weight:Q", stack=True),
            color=alt.Color("ticker:N", scale=alt.Scale(range=COLORS), legend=None),
            tooltip=[
                alt.Tooltip("ticker:N", title="자산"),
                alt.Tooltip("weight_label:N", title="비중"),
            ],
        )
    )
    center = (
        alt.Chart(alt.Data(values=[{"label": "100%"}]))
        .mark_text(fontSize=24, color="#1b2a41")
        .encode(text="label:N")
    )
    return chart_style((chart + center).properties(height=185))


def summary_plot(records, features):
    order = sorted(
        features,
        key=lambda f: sum(abs(r["shap"]) for r in records if r["feature"] == f),
        reverse=True,
    )
    points = (
        alt.Chart(alt.Data(values=records))
        .mark_circle(size=36, opacity=0.8)
        .encode(
            x=alt.X("shap:Q", title="SHAP 값 · 비중 기여 (1.0 = 100%p)"),
            y=alt.Y("feature:N", sort=order, title=None),
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color(
                "value:Q",
                title="피처 값 (정규화)",
                scale=alt.Scale(domain=[0, 1], range=["#2f6fed", "#f2a33a"]),
                legend=alt.Legend(
                    orient="bottom",
                    gradientLength=150,
                    values=[0, 1],
                    labelExpr="datum.value == 0 ? '낮음 0' : '높음 1'",
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="피처"),
                alt.Tooltip("date:N", title="관측일"),
                alt.Tooltip("shap:Q", title="SHAP 값", format="+.4f"),
                alt.Tooltip("value:Q", title="정규화 피처 값", format=".3f"),
            ],
        )
    )
    zero = (
        alt.Chart(alt.Data(values=[{"zero": 0}]))
        .mark_rule(color="#8a97a8", strokeDash=[4, 4])
        .encode(x="zero:Q")
    )
    return chart_style((zero + points).properties(height=310))


def contribution_plot(features, decision):
    records = [
        dict(feature=f, shap=v, direction="증가 (+)" if v >= 0 else "감소 (−)")
        for f, v in zip(features, decision["shap_values"])
    ]
    chart = (
        alt.Chart(alt.Data(values=records))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("shap:Q", title="선택 결정의 비중 기여 (1.0 = 100%p)"),
            y=alt.Y("feature:N", title=None, sort=features),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(
                    domain=["증가 (+)", "감소 (−)"], range=["#ffcf65", "#259aff"]
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="피처"),
                alt.Tooltip("shap:Q", title="SHAP 값", format="+.4f"),
            ],
        )
    )
    return chart_style(chart.properties(height=310))


def force_plot(features, decision):
    """Additive force ribbons meet at f(x); width encodes |SHAP|.

    Positive ribbon: [f(x)-sum(positive), f(x)], arrows to the right.
    Negative ribbon: [f(x), f(x)+sum(abs(negative))], arrows to the left.
    Baseline remains marked on the same weight axis (identity link).
    """
    pairs = list(zip(features, decision["shap_values"]))
    positive = sum(max(v, 0) for _, v in pairs)
    negative = -sum(min(v, 0) for _, v in pairs)
    final, base = decision["final_value"], decision["base_value"]
    low, high = min(final - positive, base), max(final + negative, base)
    padding = max((high - low) * 0.13, 0.01)
    low, high = low - padding, high + padding

    def x(value):
        return 35 + (value - low) / (high - low) * 830

    elements = []
    for i in range(6):
        value = low + (high - low) * i / 5
        elements.append(
            f'<line x1="{x(value)}" y1="75" x2="{x(value)}" y2="142" stroke="#d5dde8"/>'
            f'<text x="{x(value)}" y="162" text-anchor="middle">{value:.1%}</text>'
        )
    for sign, start in [(1, final - positive), (-1, final + negative)]:
        cursor = start
        for feature, value in pairs:
            if value * sign <= 0:
                continue
            end = cursor + value
            a, b = x(cursor), x(end)
            tip = min(7, abs(b - a) / 3) * sign
            color = "#f2a33a" if sign > 0 else "#2f6fed"
            points = f"{a},87 {b-tip},87 {b},109 {b-tip},131 {a},131 {a+tip},109"
            label = escape(f"{feature}: {value * 100:+.2f}%p")
            elements.append(
                f'<polygon points="{points}" fill="{color}" stroke="#ffffff">'
                f"<title>{label}</title></polygon>"
            )
            if abs(b - a) > 48:
                elements.append(
                    f'<text x="{(a+b)/2}" y="114" fill="#ffffff" '
                    f'text-anchor="middle">{value*100:+.2f}</text>'
                )
            cursor = end
    elements.extend(
        [
            f'<line x1="{x(base)}" y1="52" x2="{x(base)}" y2="142" '
            'stroke="#7d8ea3" stroke-dasharray="4 4"/>',
            f'<text x="{x(base)}" y="43" text-anchor="middle">기준값 {base:.2%}</text>',
            f'<line x1="{x(final)}" y1="78" x2="{x(final)}" '
            'y2="190" stroke="#1b2a41"/>',
            f'<text x="{x(final)}" y="211" text-anchor="middle" '
            f'fill="#1b2a41">결정 비중 f(x) = {final:.2%}</text>',
        ]
    )
    # st.html sanitizes SVG out in Streamlit 1.60. This trusted local HTML
    # preserves SVG tooltips and horizontal scrolling on narrow screens.
    st.iframe(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        "<style>body{margin:0;background:transparent}svg{display:block}</style>"
        '<div style="overflow-x:auto" tabindex="0" role="region" '
        'aria-label="Force Plot 가로 스크롤"><svg role="img" '
        'aria-label="Force Plot: 주황색 양수 기여는 오른쪽, 파란색 음수 기여는 왼쪽으로 작용" '
        'viewBox="0 0 900 230" style="width:100%;min-width:720px;'
        'font:16px sans-serif;fill:#44546a">'
        + "".join(elements)
        + "</svg></div></html>",
        height="content",
    )


def drawdown_trend(path):
    """Daily drawdown from the server path; values are zero or negative."""
    chart = (
        alt.Chart(alt.Data(values=path))
        .mark_area(color="#e86a5c", opacity=0.25, line={"color": "#e86a5c"})
        .encode(
            x=alt.X(
                "date:T",
                title="날짜",
                scale=alt.Scale(type="utc"),
                axis=alt.Axis(format="%y-%m", labelOverlap=True),
            ),
            y=alt.Y(
                "drawdown:Q", title="최고점 대비 하락폭", axis=alt.Axis(format=".0%")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d"),
                alt.Tooltip("drawdown:Q", title="하락폭", format=".2%"),
            ],
        )
    )
    return chart_style(chart.properties(height=210))
