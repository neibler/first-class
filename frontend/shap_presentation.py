"""Decision-focused presentation; never alter SHAP contributions."""

import math
import re

WEIGHT_EPSILON = 1e-8


def decision_direction(execution):
    if not execution:
        return "unknown"
    before = execution.get("previous_weight")
    target = execution.get("target_weight")
    if not all(
        isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1
        for x in (before, target)
    ):
        return "unknown"
    delta = target - before
    return (
        "increase"
        if delta > WEIGHT_EPSILON
        else "decrease" if delta < -WEIGHT_EPSILON else "hold"
    )


def select_factors(factors, direction):
    ordered = sorted(factors, key=lambda f: (-f["absolute_sum"], f["key"]))
    if direction in {"hold", "unknown"}:
        return ordered[:3], []
    opposite = "decrease" if direction == "increase" else "increase"
    return (
        [f for f in ordered if f["direction"] == direction][:3],
        [f for f in ordered if f["direction"] == opposite],
    )


def compact_state(factor):
    text = factor["actual_state_summary"]
    if not factor["key"].startswith("asset_"):
        return text
    parts = []
    price = re.search(r"최근 (\d+)거래일간 .+? 가격이 ([\d.]+)% (상승|하락)", text)
    if price:
        sign = "+" if price[3] == "상승" else "-"
        parts.append(f"{price[1]}거래일 수익률 {sign}{price[2]}%")
    rsi = re.search(r"RSI는 ([\d.]+)로 ([^.]+)\.", text)
    if rsi:
        state = rsi[2].replace("입니다", "").replace("중립적인 수준", "중립")
        parts.append(f"RSI {rsi[1]} · {state}")
    if "MACD가 0보다 높아" in text:
        parts.append("MACD 양수")
    elif "MACD가 0보다 낮아" in text:
        parts.append("MACD 음수")
    elif "MACD가 신호선보다 높아" in text:
        parts.append("MACD > 신호선")
    elif "MACD가 신호선보다 낮아" in text:
        parts.append("MACD < 신호선")
    return " · ".join(parts) if parts else text
