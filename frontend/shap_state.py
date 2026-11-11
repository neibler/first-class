"""Deterministic descriptions of market facts, separate from SHAP direction."""

import math
from datetime import date
from frontend.shap_explanations import asset_label, MACRO_MAPPING

MACRO_NAMES = {key: item[2] for key, item in MACRO_MAPPING.items()}
MACRO_NAMES.update(
    ust_10y="미국 10년물 국채금리",
    ust_30y="미국 30년물 국채금리",
    unemp="실업률",
    vix="시장 변동성 지표 VIX",
)


def observed_values(features, decision):
    # Legacy feature_values_raw is MODEL INPUT, not raw market data: do not invert
    # clipped standard scores or interpret their sign as a market price change.
    values = {}
    for f, v in zip(features, decision.get("feature_values_raw", [])):
        if f.startswith(
            ("rsi_div_100_", "rsi_", "risk_score_", "current_weight_")
        ) or f in {"drawdown", "mdd"}:
            if isinstance(v, (int, float)) and math.isfinite(v):
                values[f] = float(v)
    context = decision.get("market_context", {})
    if not isinstance(context, dict):
        context = {}
    previous = {}
    days = None
    if context.get("source") == "verified_market_data" and context.get(
        "as_of"
    ) == decision.get("date"):
        actual = context.get("values", {})
        for f in features:
            v = actual.get(f)
            if isinstance(v, (int, float)) and math.isfinite(v):
                values[f] = float(v)
        try:
            earlier = date.fromisoformat(
                context["comparison_date"]
            ) < date.fromisoformat(decision["date"])
        except (KeyError, TypeError, ValueError):
            earlier = False
        if earlier:
            previous = {
                f: float(v)
                for f, v in context.get("previous_values", {}).items()
                if f in features and isinstance(v, (int, float)) and math.isfinite(v)
            }
            days = context.get("comparison_trading_days")
    return values, previous, days


def rsi_summary(value):
    if not 0 <= value <= 100:
        return None
    if value < 30:
        text = "낮아 과매도 구간에 가까운 상태입니다"
    elif value < 45:
        text = "비교적 낮은 수준입니다"
    elif value <= 55:
        text = "중립적인 수준입니다"
    elif value <= 70:
        text = "비교적 높은 수준입니다"
    else:
        text = "높아 과매수 구간에 가까운 상태입니다"
    display = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"RSI는 {display}로 {text}."


def summarize_feature_state(
    group, values, previous=None, comparison_days=None, shap_values=None
):
    previous = previous or {}
    available = {f: values[f] for f in group["raw_features"] if f in values}
    sentences = []
    key = group["key"]
    if key.startswith("asset_"):
        ticker = key[6:]
        suffix = ticker.lower()
        recent = []
        for lag in range(5):
            feature = f"log_return_lag_{lag}_{suffix}"
            if feature not in available:
                break
            recent.append(available[feature])
        if recent:
            try:
                percent = math.expm1(math.fsum(recent)) * 100
                if math.isfinite(percent):
                    movement = "상승" if percent > 0 else "하락"
                    if abs(percent) < 0.005:
                        sentences.append(
                            f"최근 {len(recent)}거래일간 {asset_label(ticker)} "
                            "가격은 거의 변하지 않았습니다."
                        )
                    else:
                        sentences.append(
                            f"최근 {len(recent)}거래일간 {asset_label(ticker)} 가격이 "
                            f"{abs(percent):.2f}% {movement}했습니다."
                        )
            except OverflowError:
                pass
        rsi = available.get(f"rsi_{suffix}")
        if rsi is None and f"rsi_div_100_{suffix}" in available:
            rsi = available[f"rsi_div_100_{suffix}"] * 100
        if rsi is not None:
            text = rsi_summary(round(rsi, 6))
            if text:
                sentences.append(text)
        macd = available.get(f"macd_{suffix}")
        signal = available.get(f"macd_signal_{suffix}")
        if macd is not None:
            if signal is not None:
                if macd > signal:
                    text = "MACD가 신호선보다 높아 단기 상승 모멘텀이 나타납니다."
                elif macd < signal:
                    text = "MACD가 신호선보다 낮아 단기 하락 모멘텀이 나타납니다."
                else:
                    text = "MACD와 신호선이 같은 수준입니다."
            elif macd > 0:
                text = "MACD가 0보다 높아 최근 추세 신호가 상대적으로 강한 편입니다."
            elif macd < 0:
                text = "MACD가 0보다 낮아 최근 추세 신호가 상대적으로 약한 편입니다."
            else:
                text = "MACD가 0입니다."
            sentences.append(text)
    elif key == "news":
        ranked = sorted(
            available, key=lambda f: (-abs((shap_values or {}).get(f, 0)), f)
        )
        for f in ranked[:3]:
            if f.startswith("risk_score_") and 0 <= available[f] <= 1:
                sentences.append(
                    f"{f[len('risk_score_'):].upper()} 관련 뉴스 위험 점수는 "
                    f"{available[f]:.2f}입니다."
                )
    elif any(f.startswith("macro_") for f in group["raw_features"]):
        for f in group["raw_features"]:
            if not f.startswith("macro_"):
                continue
            name = MACRO_NAMES.get(f[6:], group["label"])
            if f not in available:
                continue
            v = available[f]
            if f in previous:
                delta = v - previous[f]
                span = (
                    f"{comparison_days}거래일 전보다"
                    if isinstance(comparison_days, int) and comparison_days > 0
                    else "이전 관측보다"
                )
                direction = (
                    "상승" if delta > 0 else "하락" if delta < 0 else "같은 수준을 유지"
                )
                sentences.append(
                    f"{name} 입력값은 {v:.4g}이며, {span} {direction}했습니다."
                )
            else:
                sentences.append(
                    f"현재 {name} 입력값은 {v:.4g}입니다. "
                    "최근 변화 방향은 확인되지 않았습니다."
                )
    elif key == "allocation":
        for f in sorted(available, key=lambda f: (-available[f], f))[:3]:
            sentences.append(
                f"{f[len('current_weight_'):].upper()} 현재 비중은 "
                f"{available[f]*100:.1f}%입니다."
            )
    elif key == "loss":
        for f, name in [("drawdown", "현재 낙폭"), ("mdd", "누적 최대낙폭")]:
            if f in available:
                sentences.append(f"{name}은 {available[f]*100:.1f}%입니다.")
    if not sentences:
        sentences = ["해당 요인의 실제 시장 상태를 확인할 데이터가 부족합니다."]
    return " ".join(sentences[:3])


def generate_shap_explanation(group, selected, summary):
    direction = group["direction"]
    if direction == "balanced":
        contribution = f"{group['label']}의 순 기여는 상쇄되어 뚜렷한 방향이 없습니다."
    else:
        verb = "높이는" if direction == "increase" else "낮추는"
        contribution = (
            f"{group['label']}의 합산 기여는 SHAP 기준 대비 "
            f"{selected.upper()} 비중을 {verb} 방향으로 작용했습니다."
        )
    return {
        **group,
        "group_name": group["label"],
        "asset": group["key"][6:] if group["key"].startswith("asset_") else None,
        "factor": group["key"],
        "actual_state_summary": summary,
        "shap_direction": direction,
        "shap_importance": group["absolute_sum"],
        "explanation": summary + " " + contribution,
    }
