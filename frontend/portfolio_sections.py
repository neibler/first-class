"""Portfolio-wide sections for the report page and its detail pages.

The first page stays simple; trades and why the weights moved, the portfolio-wide
risk, the calculation notes and the "start today" proposal live here.
"""

import streamlit as st

from frontend.api_client import APIError
from frontend.charts import drawdown_trend

# Module references, not names: pages run again on every rerun but this module
# is imported once, so call through the client modules (tests replace them there).
from frontend import insights_client, investment_client
from frontend.investment_client import (
    InvestmentAPIError,
    UnknownInvestmentSessionError,
)
from frontend.shap_explanations import ASSET_NAMES
from frontend.ui import card, detail_link
from frontend.weight_format import format_weights, round_weights

MARKET = "S&P500"
REASONS = {
    # Policy-neutral: older bundles compare against the AI target, the low-churn
    # policy against the last executed weights.
    "portfolio_weight_distance": "비중 차이가 조정 기준 이상 벌어져서",
    "monthly": "월간 정기 점검일이라서",
    "risk_score": "뉴스 위험 점수가 기준을 넘어서",
    "mdd_stop": "손실 제한(Safe-Guard) 규칙이 발동해서",
    "hold": "조정 조건에 해당하지 않아 유지",
}


def change_rows(event):
    """Before/after rounded to whole percents that each add to 100; change = diff.

    Cash joins both sides so the columns balance even when the model holds cash.
    Rows whose displayed change is 0 are left out, so the shown changes add to 0.
    """
    items = event["changes"]
    if not items:
        return []
    before = [item["before"] for item in items]
    after = [item["after"] for item in items]
    names = [item["ticker"] for item in items] + ["현금"]
    before.append(max(0.0, 1.0 - sum(before)))
    after.append(event["cash_weight"])
    shown_before, shown_after = round_weights(before), round_weights(after)
    rows = [
        {
            "자산": name,
            "이전": f"{old}%",
            "이후": f"{new}%",
            "변화": f"{new - old:+d}%p",
        }
        for name, old, new in zip(names, shown_before, shown_after)
        if new != old
    ]
    return sorted(rows, key=lambda row: -abs(int(row["변화"][:-2])))


HISTORY_LIMIT = 500  # the API's largest page; FINAL2026 has 187 adjusted days


def load_changes():
    """Every asset's change, so the display rounding sees full before/after weights.

    Asks for the API's largest page so the trade history is complete; the
    default page of 30 dropped the older days.
    """
    try:
        return insights_client.load_rebalancing(None, limit=HISTORY_LIMIT, min_change=0)
    except APIError as exc:
        st.warning(str(exc))
        return None


def trades_card(data, changes):
    """Report card 1: how often and why the AI traded; the history is a detail page."""
    with card("report-trades"):
        st.subheader("AI는 언제, 왜 사고팔았나요?")
        if changes:
            st.markdown(
                f"평가 기간 {changes['observations']}거래일 중 "
                f"**{changes['rebalanced_days']}일** 비중을 조정했어요."
            )
            days, turnover, cost = st.columns(3)
            days.metric("비중을 조정한 날", f"{changes['rebalanced_days']}일")
            turnover.metric(
                "하루 평균 거래 비중",
                f"{changes['average_turnover']:.0%}",
                help=(
                    "비중을 조정한 날 하루에 포트폴리오의 몇 %를 사고팔았는지(회전율)입니다. "
                    "100%면 판 금액과 산 금액을 합쳐 포트폴리오 전체만큼 거래한 셈입니다."
                ),
            )
            spent = ((data or {}).get("metrics") or {}).get(
                "total_cost_initial_capital"
            )
            if spent is not None:
                cost.metric(
                    "거래 비용 합계",
                    f"원금의 {spent:.1%}",
                    help="평가 기간에 사고팔며 낸 비용(수수료·슬리피지 가정)을 처음 투자금 대비로 나타낸 값입니다.",
                )
            if changes["events"]:
                latest = changes["events"][0]
                reasons = [REASONS.get(r, r) for r in latest["reasons"]]
                st.caption(
                    f"가장 최근 조정 {latest['date']} · {', '.join(reasons) or '기록 없음'} · "
                    "거래가 많을수록 수수료 등 비용이 커집니다."
                )
        detail_link("trades", "날짜별 사고판 내역 보기")


def risk_card(data):
    """Report card 2: plain conclusion next to the market; numbers in a detail page."""
    with card("report-risk"):
        st.subheader("얼마나 오르내렸나요?")
        if data:
            mine = (data.get("metrics") or {}).get("annualized_volatility")
            market_value = (data.get("benchmark_metrics") or {}).get(
                "annualized_volatility"
            )
            if mine is not None and market_value:
                ratio = mine / market_value
                # Within ±15% of the market reads as "about the same".
                level = (
                    "비슷한 정도로"
                    if 0.85 <= ratio <= 1.15
                    else "더 적게" if ratio < 0.85 else "더 크게"
                )
                st.markdown(
                    f"평가 기간 동안 내 포트폴리오는 {MARKET}과 **{level}** 오르내렸어요."
                )
            if data["stopped"] or data.get("safeguard_triggered"):
                st.warning("손실 제한 규칙(Safe-Guard)이 작동해 현금으로 바꿨어요.")
            else:
                st.caption(
                    "손실이 커지면 현금으로 바꾸는 안전장치(Safe-Guard)가 있고, "
                    "이 기간에는 작동하지 않았어요."
                )
            benchmark = data.get("benchmark_metrics") or {}
            worst, swing = st.columns(2)
            worst.metric(
                "최대 하락폭",
                f"{data['display_metrics']['mdd']:.2%}",
                help="최고점에서 가장 많이 떨어졌던 폭입니다.",
            )
            if benchmark.get("max_drawdown") is not None:
                market(worst, loss(benchmark["max_drawdown"]))
            swing.metric(
                "변동성 · 연환산",
                f"{mine:.2%}" if mine is not None else "—",
                help="수익률이 얼마나 크게 오르내렸는지를 1년 기준으로 나타낸 값입니다.",
            )
            if market_value is not None:
                market(swing, f"{market_value:.2%}")
        detail_link("risk", "위험 지표 자세히 보기")


def why_changed(changes):
    with card("why"):
        st.subheader("왜 포트폴리오가 바뀌었나요?")
        if changes is None:
            return
        if not changes["events"]:
            st.info("비중 변화 기록이 없습니다.")
            return
        events = {event["date"]: event for event in changes["events"]}
        if len(events) < changes["rebalanced_days"]:
            # Only when more days exist than one page holds: say so instead of
            # presenting a cut list as the whole history.
            st.caption(
                f"비중을 조정한 {changes['rebalanced_days']}일 중 "
                f"가장 최근 {len(events)}일만 보여 줍니다."
            )
        day = st.selectbox(
            f"날짜 (최근 순 · {len(events)}일)", list(events), key="rebalance_date"
        )
        event = events[day]
        rows = change_rows(event)
        # Beginners first: why, in plain words, then what moved; numbers below.
        st.markdown(f"💬 {plain_reason(event['reasons'])}")
        for line in moved_lines(rows):
            st.markdown(line)
        with st.expander("자세히 보기 · 날짜별 비중 표"):
            if rows:
                st.dataframe(rows, hide_index=True, height="content")
            else:
                st.caption("표시 기준으로 비중이 바뀐 자산이 없습니다.")
            st.caption(
                f"{event['decision_date']} 장 마감 정보로 결정하고 {day}에 반영한 비중입니다. "
                "이전·이후 비중은 각각 합계 100%가 되도록 반올림하고, 변화는 그 차이입니다. "
                "표시상 변화가 없는 자산은 생략합니다."
            )


# The recorded rebalance reason, as one sentence a beginner can read. Same
# facts as REASONS (report card), only worded as a full sentence.
PLAIN_REASONS = {
    "monthly": "한 달에 한 번 하는 정기 점검일이라 비중을 다시 맞췄어요.",
    "risk_score": "뉴스 위험이 기준을 넘어서, 위험을 줄이려고 비중을 다시 나눴어요.",
    "portfolio_weight_distance": "AI가 정한 비중과 실제 비중이 많이 벌어져서 다시 맞췄어요.",
    "mdd_stop": "손실이 커져서 안전장치(Safe-Guard)가 작동해 현금으로 바꿨어요.",
}
MOVED_SHOWN = 2  # names per line; the rest is counted ("외 1개")


def plain_reason(reasons):
    sentences = [PLAIN_REASONS.get(r, f"기록된 조정 이유: {r}") for r in reasons]
    return " ".join(sentences) or "이날 조정 이유는 기록되지 않았어요."


def moved_lines(rows):
    """'늘린 자산' / '줄인 자산' from the displayed table rows, largest first.

    Uses the table's own rounded changes, so the lines never disagree with it.
    """
    if not rows:
        return ["표시할 만큼 바뀐 자산은 없어요."]

    def label(name):
        return name if name == "현금" else f"{ASSET_NAMES.get(name, name)}({name})"

    def line(mark, title, names):
        shown = ", ".join(label(n) for n in names[:MOVED_SHOWN])
        more = f" 외 {len(names) - MOVED_SHOWN}개" if len(names) > MOVED_SHOWN else ""
        return f"{mark} **{title}** {shown}{more}"

    change = {row["자산"]: int(row["변화"][:-2]) for row in rows}
    ups = [n for n in change if change[n] > 0]  # rows are already largest first
    downs = [n for n in change if change[n] < 0]
    lines = []
    if ups:
        lines.append(line("▲", "늘린 자산", ups))
    if downs:
        lines.append(line("▼", "줄인 자산", downs))
    return lines


def assumptions(data):
    result = data["display_metrics"]
    with st.expander("계산 기준과 원자료 (검증용)"):
        st.caption("숫자를 직접 확인하려는 분을 위한 계산 기준과 날짜별 원자료입니다.")
        st.write(
            "서버에 저장된 모델 평가 결과입니다. 합성 데이터 결과는 실제 투자 성능을 뜻하지 않습니다."
        )
        st.write(
            f"연 {data['periods_per_year']} 거래일, "
            f"연 무위험 수익률 {data['annual_risk_free_rate']:.1%}. "
            "Sharpe는 비용 차감 후 단순수익률 기준이며 서버 계산값을 표시합니다."
        )
        st.write(
            "최대 하락폭과 낙폭은 음수로 표시합니다. "
            "모든 기준 전략에도 같은 손실 제한 규칙을 적용합니다. 보유 비중에는 현금이 포함됩니다."
        )
        st.dataframe(
            result["path"],
            hide_index=True,
            column_config={
                "date": "날짜",
                "portfolio": st.column_config.NumberColumn(
                    "포트폴리오", format="percent"
                ),
                "benchmark": st.column_config.NumberColumn(
                    "벤치마크", format="percent"
                ),
                "drawdown": st.column_config.NumberColumn("낙폭", format="percent"),
            },
        )


def fresh_start():
    with st.expander("오늘 새로 시작한다면? (참고용)"):
        st.caption(
            "빈 계좌(현금 100%)로 오늘 투자를 시작한다면 AI가 어떻게 나눌지 보여 주는 참고용 제안입니다. "
            "포트폴리오 화면의 계좌는 이미 자산을 들고 운용 중이라, AI가 현재 보유 비중까지 고려해 결과가 다를 수 있습니다. "
            "실제 주문은 나가지 않습니다."
        )
        if not st.button("목표 비중 계산"):
            return
        try:
            if not st.session_state.get("investment_session_token"):
                st.session_state["investment_session_token"] = (
                    investment_client.create_inference_session()
                )
            proposal = investment_client.optimize(
                session_token=st.session_state["investment_session_token"]
            )
        except InvestmentAPIError as exc:
            if isinstance(exc, UnknownInvestmentSessionError):
                st.session_state.pop("investment_session_token", None)
            st.error(str(exc))
            return
        st.session_state["investment_session_token"] = proposal.get("session_token")
        st.caption(
            f"{proposal['as_of']} 피처 기준 · "
            f"{proposal['execution_delay_trading_bars']} 거래일 지연 · 주문 전송 없음"
        )
        # Only proposed holdings, largest first; cash joins the rounding so the
        # shown weights add up to 100%.
        targets = sorted(
            (
                (ticker, weight)
                for ticker, weight in proposal["target_weights"].items()
                if weight > 0
            ),
            key=lambda item: -item[1],
        )
        if proposal["cash_weight"]:
            targets.append(("현금", proposal["cash_weight"]))
        labels = format_weights([weight for _, weight in targets])
        st.dataframe(
            [
                {"자산": ticker, "목표 비중": label}
                for (ticker, _), label in zip(targets, labels)
            ],
            hide_index=True,
        )
        st.caption("합계가 100%가 되도록 반올림을 맞췄습니다.")


def loss(value):
    """Server loss sizes are positive; show them as negative like MDD.

    VaR/CVaR turn negative when even the worst days of the window were gains;
    that is shown as a gain, never as a loss amount.
    """
    if value is None:
        return "계산 불가"
    if value < 0:
        return f"손실 없음 · +{-value:.2%}"
    return "0.00%" if value == 0 else f"{-value:.2%}"


def market(column, text):
    """Show the market's value for the same metric right under it."""
    column.caption(f"{MARKET} {text}")


def compare(mine, market_value):
    """'less' / 'same' / 'more' than the market; within ±15% reads as the same
    (the report's risk card uses the same band). None when either is missing."""
    # A market value at or below 0 (no loss at all) gives no meaningful ratio.
    if mine is None or market_value is None or market_value <= 0:
        return None
    ratio = mine / market_value
    return "same" if 0.85 <= ratio <= 1.15 else "less" if ratio < 0.85 else "more"


def risk_summary(drop, swing):
    """One line: how far it fell and how much it swung, next to the market."""
    drops = {"less": "덜 떨어지고", "same": "비슷하게 떨어지고", "more": "더 떨어지고"}
    swings = {
        "less": "덜 오르내렸어요",
        "same": "비슷하게 오르내렸어요",
        "more": "더 크게 오르내렸어요",
    }
    if drop and swing:
        return f"{MARKET}과 비교하면 {drops[drop]}, {swings[swing]}."
    if drop:
        alone = {
            "less": "덜 떨어졌어요",
            "same": "비슷하게 떨어졌어요",
            "more": "더 떨어졌어요",
        }
        return f"{MARKET}과 비교하면 {alone[drop]}."
    if swing:
        return f"{MARKET}과 비교하면 {swings[swing]}."
    return None


EXAMPLE_WON = 10_000_000  # "1,000만 원이었다면" — display only, no recalculation


def won(rate):
    """A loss rate as money on the example amount, rounded for reading."""
    amount = EXAMPLE_WON * rate
    return (
        f"약 {amount / 10_000:,.0f}만 원" if amount >= 10_000 else f"약 {amount:,.0f}원"
    )


def meaning(column, level, less, same, more):
    """One plain line under a metric: what the number means next to the market."""
    if level:
        column.caption("👉 " + {"less": less, "same": same, "more": more}[level])


def daily_loss():
    """VaR/CVaR folded away: explained in words, values and one reading each."""
    with st.expander("자세히 보기 · 하루 예상 손실(VaR·CVaR)"):
        try:
            tail = insights_client.load_risk()
        except APIError as exc:
            st.caption(str(exc))
            tail = None
        if tail:
            # The period first, so the numbers below are read against it.
            st.caption(
                f"VaR·CVaR: {tail['start']}~{tail['as_of']} "
                f"{tail['observations']}거래일 실제 수익률 기준"
            )
        st.markdown(
            "**VaR**는 '평소 나쁜 날, 하루에 얼마까지 잃었나'이고, "
            "**CVaR**는 '그보다 더 나쁜 날에는 평균 얼마를 잃었나'입니다. "
            "둘 다 평가 기간의 실제 하루 수익률로 계산한 과거 값이에요."
        )
        if not tail:
            return
        risk_label = f"{tail['level']:.0%}"
        mine, spy = tail["portfolio"], tail["benchmark_risk"]
        var, cvar = st.columns(2)
        var.metric(
            f"하루 예상 손실 · VaR {risk_label}",
            loss(mine["var"]),
            help=(
                "평가 기간의 하루 수익률 중 나쁜 쪽 5% 경계입니다. "
                "20일 중 19일은 하루 손실이 이 값보다 작았다는 뜻입니다."
            ),
        )
        market(var, loss(spy["var"]))
        meaning(
            var,
            compare(mine["var"], spy["var"]),
            f"평소 나쁜 날의 손실이 {MARKET}보다 작았어요.",
            f"평소 나쁜 날의 손실이 {MARKET}과 비슷했어요.",
            f"평소 나쁜 날의 손실이 {MARKET}보다 컸어요.",
        )
        cvar.metric(
            f"나쁜 날 평균 손실 · CVaR {risk_label}",
            loss(mine["cvar"]),
            help="하루 손실이 VaR보다 컸던 나쁜 날들의 평균 손실입니다.",
        )
        market(cvar, loss(spy["cvar"]))
        meaning(
            cvar,
            compare(mine["cvar"], spy["cvar"]),
            f"아주 나쁜 날의 손실도 {MARKET}보다 작았어요.",
            f"아주 나쁜 날의 손실은 {MARKET}과 비슷했어요.",
            f"아주 나쁜 날의 손실은 {MARKET}보다 컸어요.",
        )
        if (mine["var"] or 0) > 0 and (mine["cvar"] or 0) > 0:
            st.caption(
                f"1,000만 원이었다면 평소 나쁜 날 하루 {won(mine['var'])}, "
                f"아주 나쁜 날에는 평균 {won(mine['cvar'])}을 잃은 셈이에요."
            )


def portfolio_risk(data):
    """Portfolio-wide risk next to the market, then the drawdown trend."""
    result = data["display_metrics"]
    benchmark = data.get("benchmark_metrics") or {}
    start, end = data["returns"][0]["date"], data["as_of"]
    volatility = (data.get("metrics") or {}).get("annualized_volatility")
    market_mdd = benchmark.get("max_drawdown")
    market_volatility = benchmark.get("annualized_volatility")
    with card("risk-now"):
        st.subheader("시장과 비교한 위험도")
        # One plain line first; the numbers below carry no extra sentences.
        summary = risk_summary(
            compare(abs(result["mdd"]), abs(market_mdd) if market_mdd else None),
            compare(volatility, market_volatility),
        )
        if summary:
            st.info(summary)
        # The portfolio page lets the user pick a shorter period; this page
        # always uses the whole evaluation, so the dates stay visible.
        st.caption(
            f"평가 기간 전체({start} ~ {end}) 기준 · 현재 낙폭만 마지막 날 · "
            f"숫자 아래 회색 글씨는 {MARKET}"
        )
        if data["stopped"] or data.get("safeguard_triggered"):
            st.warning("손실 제한 안전장치(Safe-Guard)가 작동해 현금으로 바꿨어요.")
        else:
            st.success(
                "Safe-Guard 미작동 · 손실 제한 안전장치는 이 기간에 작동하지 않았어요."
            )
        worst, now, swing = st.columns(3)
        worst.metric(
            "최대 하락폭 · MDD",
            f"{result['mdd']:.2%}",
            help=(
                "평가 기간에 최고점 대비 가장 많이 떨어진 폭입니다. "
                f"1,000만 원이었다면 {won(abs(result['mdd']))}이 줄어든 셈이에요."
            ),
        )
        if market_mdd is not None:
            market(worst, loss(market_mdd))
        now.metric(
            "현재 낙폭",
            f"{result['current_drawdown']:.2%}",
            help=f"마지막 날({end}) 최고점 대비 얼마나 내려와 있는지입니다.",
        )
        swing.metric(
            "변동성 · 연환산",
            f"{volatility:.2%}" if volatility is not None else "—",
            help="수익률이 얼마나 크게 오르내렸는지를 1년 기준으로 나타낸 값입니다.",
        )
        if market_volatility is not None:
            market(swing, f"{market_volatility:.2%}")
        daily_loss()

    with card("drawdown"):
        st.subheader("하락폭 추이")
        st.caption("하락폭은 내 돈이 가장 많았던 때보다 그날 얼마나 줄어 있었는지예요.")
        st.caption(
            "예: 최고 110만 원이었다가 그날 99만 원이면 −10%, "
            "0%는 최고 기록을 새로 쓴 날이에요."
        )
        st.altair_chart(drawdown_trend(result["path"]), theme=None, width="stretch")
