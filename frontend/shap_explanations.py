"""Presentation-only grouping. Preserve every raw SHAP contribution exactly."""

import re
import math

ASSET_NAMES = {
    "SPY": "미국 대형주",
    "QQQ": "기술주",
    "SMH": "반도체",
    "SCHD": "미국 배당주",
    "XLE": "에너지주",
    "EFA": "미국 외 선진국 주식",
    "SHY": "미국 단기국채",
    "IEF": "미국 중기국채",
    "TLT": "미국 장기국채",
    "TIP": "물가연동국채",
    "LQD": "투자등급 회사채",
    "GLD": "금",
    "DBB": "산업금속",
    "DBA": "농산물",
    "VNQ": "미국 부동산",
}
# Raw suffix -> category, aggregation key, beginner label.
MACRO_MAPPING = {
    "effr": ("금리 환경", "policy_rate", "미국 기준금리"),
    "ust_2y": ("금리 환경", "short_rate", "미국 단기금리"),
    "ust_10y": ("금리 환경", "long_rate", "미국 장기금리"),
    "ust_30y": ("금리 환경", "long_rate", "미국 장기금리"),
    "tips_10y": ("금리 환경", "real_rate", "미국 실질금리"),
    "yield_spread": ("금리 환경", "rate_spread", "장기·단기 금리 차이"),
    "core_pce_yoy": ("경기/물가", "inflation", "미국 물가 흐름"),
    "real_pce_mom": ("경기/물가", "consumption", "미국 소비 흐름"),
    "unemp": ("경기/물가", "employment", "미국 고용 상황"),
    "ism_pmi": ("경기/물가", "manufacturing", "미국 제조업 경기"),
    "vix": ("시장 위험", "volatility", "시장 변동성"),
    "usd_ret_20d": ("환율/원자재", "dollar", "달러 가치의 최근 움직임"),
    "wti": ("환율/원자재", "oil", "국제 유가"),
}


def asset_label(ticker):
    ticker = ticker.upper()
    return f"{ASSET_NAMES.get(ticker, '자산')}({ticker})"


def signal_label(ticker):
    if ticker == "GLD":
        return "금(GLD) 관련 신호"
    return f"{asset_label(ticker)}의 최근 움직임"


def classify(feature, selected):
    selected = selected.upper()
    match = re.fullmatch(r"log_return_lag_\d+_([a-z0-9]+)", feature)
    if match:
        ticker = match[1].upper()
        category = "자산 자체 가격/추세" if ticker == selected else "다른 자산 움직임"
        return category, f"asset_{ticker}", signal_label(ticker)
    match = re.fullmatch(
        r"(?:rsi_div_100|rsi|macd_signal|macd_histogram|macd_hist|macd)_([a-z0-9]+)",
        feature,
    )
    if match:
        ticker = match[1].upper()
        category = "자산 자체 가격/추세" if ticker == selected else "다른 자산 움직임"
        return category, f"asset_{ticker}", signal_label(ticker)
    if feature.startswith("macro_") and feature[6:] in MACRO_MAPPING:
        return MACRO_MAPPING[feature[6:]]
    if re.fullmatch(r"risk_score_([a-z0-9]+)", feature):
        return "뉴스 위험", "news", "최근 뉴스 위험"
    if feature.startswith("current_weight_"):
        return "현재 포트폴리오", "allocation", "현재 자산 배분 상태"
    if feature in {"drawdown", "mdd"}:
        return "운용 상태", "loss", "현재 손실과 누적 낙폭"
    if feature == "rebalance_phase":
        return "운용 상태", "rebalance", "비중 조정 주기"
    return "기타 정보", "other", "기타 모델 입력 정보"


def aggregate(features, values, selected):
    if len(features) != len(values):
        raise ValueError("Feature/contribution length mismatch")
    groups = {}
    for feature, value in zip(features, values):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Nonfinite contribution")
        category, key, label = classify(feature, selected)
        row = groups.setdefault(
            key,
            {
                "key": key,
                "category": category,
                "label": label,
                "signed_sum": 0.0,
                "absolute_sum": 0.0,
                "raw_features": [],
            },
        )
        row["signed_sum"] += value
        row["absolute_sum"] += abs(value)
        row["raw_features"].append(feature)
    rows = sorted(groups.values(), key=lambda r: (-r["absolute_sum"], r["key"]))
    maximum = max((r["absolute_sum"] for r in rows), default=0.0)
    for row in rows:
        net = row["signed_sum"]
        row["direction"] = (
            "increase" if net > 1e-12 else "decrease" if net < -1e-12 else "balanced"
        )
        row["relative_strength"] = row["absolute_sum"] / maximum if maximum else 0.0
        ratio = row["relative_strength"]
        row["impact"] = (
            "매우 큰 영향"
            if ratio >= 0.75
            else (
                "큰 영향"
                if ratio >= 0.4
                else "보통 영향" if ratio >= 0.15 else "작은 영향"
            )
        )
        row["level"] = (
            4 if ratio >= 0.75 else 3 if ratio >= 0.4 else 2 if ratio >= 0.15 else 1
        )
        row["indicator_label"] = {4: "매우 큼", 3: "큼", 2: "보통", 1: "작음"}[
            row["level"]
        ]
        row["mixed"] = row["absolute_sum"] - abs(net) > 1e-12
    return rows


def top_factors(rows, direction, limit=3):
    return [r for r in rows if r["direction"] == direction][:limit]


def category_totals(rows):
    groups = {}
    for row in rows:
        total = groups.setdefault(
            row["category"],
            {"분류": row["category"], "순 기여 (%p)": 0.0, "전체 영향력 (%p)": 0.0},
        )
        total["순 기여 (%p)"] += row["signed_sum"] * 100
        total["전체 영향력 (%p)"] += row["absolute_sum"] * 100
    return sorted(groups.values(), key=lambda r: -r["전체 영향력 (%p)"])


def group_shap_features(features, values, selected, decision):
    """Enrich unchanged groups with verified state and non-causal explanations."""
    from frontend.shap_state import (
        observed_values,
        summarize_feature_state,
        generate_shap_explanation,
    )

    groups = aggregate(features, values, selected)
    actual, previous, days = observed_values(features, decision)
    contributions = dict(zip(features, values))
    return [
        generate_shap_explanation(
            group,
            selected,
            summarize_feature_state(group, actual, previous, days, contributions),
        )
        for group in groups
    ]
