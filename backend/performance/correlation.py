"""Read-only correlations of aligned market observations, never model outputs."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationConfig:
    default_window: int = 60
    windows: tuple = (20, 60, 120, 252)
    top_count: int = 5
    strength_edges: tuple = (0.2, 0.4, 0.6, 0.8)


CONFIG = CorrelationConfig()
MARKETS = {
    "ust_2y": ("미국 2년물 금리 변화", "diff"),
    "ust_10y": ("미국 10년물 금리 변화", "diff"),
    "ust_30y": ("미국 30년물 금리 변화", "diff"),
    "tips_10y": ("미국 10년물 실질금리 변화", "diff"),
    "yield_spread": ("장단기 금리 차이의 변화", "diff"),
    "vix": ("시장 변동성(VIX) 변화", "diff"),
    "wti": ("국제 유가(WTI) 일별 수익률", "pct_change"),
}
# Monthly/release-driven values are not observations of daily economic changes.
EXCLUSIONS = {
    "core_pce_yoy": "월별 물가 지표: 반복된 일별 수준으로 상관을 계산하지 않습니다.",
    "real_pce_mom": "월별 소비 지표: 반복된 일별 수준으로 상관을 계산하지 않습니다.",
    "unemp": "월별 실업률: 저빈도 발표 지표입니다.",
    "ism_pmi": "월별 제조업 지표: 저빈도 발표 지표입니다.",
    "effr": "정책 변화 중심으로 장기간 일정한 금리이므로 초기 일별 분석에서 제외합니다.",
    "usd_ret_20d": (
        "정렬된 입력은 20일 달러 수익률입니다. "
        "수정 이력 원본의 일별 시점 정렬을 별도 검증하기 전에는 일별 수익률로 대체하지 않습니다."
    ),
}


def calculate_rolling_correlation(left, right, window):
    if window not in CONFIG.windows:
        raise ValueError("window must be 20, 60, 120 or 252")
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan)
    x, y = pair.iloc[:, 0], pair.iloc[:, 1]
    full = pair.notna().all(axis=1).rolling(window, min_periods=window).sum().eq(window)
    variance = x.rolling(window).std().gt(1e-12) & y.rolling(window).std().gt(1e-12)
    return (
        x.rolling(window, min_periods=window).corr(y).where(full & variance).clip(-1, 1)
    )


def describe_correlation(value):
    levels = ["매우 약한 관계", "약한 관계", "보통 관계", "강한 관계", "매우 강한 관계"]
    strength = levels[sum(abs(value) >= edge for edge in CONFIG.strength_edges)]
    direction = "positive" if value > 0 else "negative" if value < 0 else "neutral"
    label = {
        "positive": "같이 움직이는 경향",
        "negative": "반대로 움직이는 경향",
        "neutral": "뚜렷한 선형 관계 없음",
    }[direction]
    return direction, strength, label


def describe_correlation_change(current, previous):
    if not np.isfinite([current, previous]).all():
        return None
    if current * previous < 0:
        return "이전 구간과 비교해 관측된 관계의 방향이 바뀌었습니다."
    difference = abs(current) - abs(previous)
    if abs(difference) < 1e-8:
        return "이전 구간과 관계의 크기가 비슷합니다."
    direction = "같은" if current > 0 else "반대" if current < 0 else "뚜렷하지 않은"
    change = "강해졌습니다" if difference > 0 else "약해졌습니다"
    return f"이전 구간보다 {direction} 방향 관계가 {change}."


def build_series(data):
    index = pd.DatetimeIndex(data.dates)
    series, metadata = {}, {}
    for i, ticker in enumerate(data.assets):
        series[ticker] = pd.Series(data.returns[:, i], index=index, dtype=float)
        metadata[ticker] = {
            "name": ticker,
            "display_name": ticker,
            "type": "ETF",
            "transformation": "daily_log_return",
        }
    excluded = []
    for i, name in enumerate(data.macros):
        if name not in MARKETS:
            excluded.append(
                {
                    "name": name,
                    "reason": EXCLUSIONS.get(
                        name, "일별 변환 의미가 검증되지 않은 지표입니다."
                    ),
                }
            )
            continue
        label, transform = MARKETS[name]
        value = pd.Series(data.macro[:, i], index=index, dtype=float)
        key = name + ("_change" if transform == "diff" else "_return")
        # No fill: a missing/zero/negative oil price does not generate a return.
        transformed = (
            value.diff()
            if transform == "diff"
            else (value / value.shift(1) - 1).where(value.gt(0) & value.shift(1).gt(0))
        )
        series[key] = transformed
        metadata[key] = {
            "name": key,
            "display_name": label,
            "type": "macro",
            "transformation": transform,
        }
    return series, metadata, excluded


def rank_correlations(rows, top_k):
    return sorted(rows, key=lambda r: (-abs(r["correlation"]), r["name"]))[:top_k]


def analyze_correlations(data, ticker, window=60, top_k=5, as_of=None):
    if ticker not in data.assets:
        raise KeyError(ticker)
    if window not in CONFIG.windows or not 1 <= top_k <= 30:
        raise ValueError("Invalid window/top_k")
    index = pd.DatetimeIndex(data.dates)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Market dates must be sorted and unique")
    series, metadata, excluded = build_series(data)
    cutoff = pd.Timestamp(as_of) if as_of else index[-1]
    series = {k: s.loc[:cutoff] for k, s in series.items()}
    selected = series[ticker]
    rows, histories, unavailable = [], {}, []
    for name, other in series.items():
        if name == ticker:
            continue
        rolling = calculate_rolling_correlation(selected, other, window)
        if rolling.empty or not np.isfinite(rolling.iloc[-1]):
            unavailable.append(
                {
                    "name": name,
                    "reason": "기간 내 유효 관측 부족 또는 값이 일정하여 계산할 수 없습니다.",
                }
            )
            continue
        value = float(rolling.iloc[-1])
        direction, strength, label = describe_correlation(value)
        row = {
            **metadata[name],
            "correlation": value,
            "direction": direction,
            "strength": strength,
            "relationship_label": label,
            "observations": window,
            "description": (
                f"최근 {window}거래일 동안 {ticker}와 "
                f"{metadata[name]['display_name']} 사이에 {label}이 관측되었습니다."
            ),
        }
        # Compare adjacent non-overlapping windows, not yesterday's window.
        if len(rolling) > window and np.isfinite(rolling.iloc[-1 - window]):
            previous = float(rolling.iloc[-1 - window])
            row.update(
                previous_correlation=previous,
                previous_date=str(rolling.index[-1 - window].date()),
                change_description=describe_correlation_change(value, previous),
            )
        histories[name] = [
            {"date": str(d.date()), "correlation": float(v) if np.isfinite(v) else None}
            for d, v in rolling.items()
        ]
        rows.append(row)
    ranked = rank_correlations(rows, len(rows))
    top = ranked[:top_k]
    from .correlation_visuals import build_comparison

    comparisons = {
        r["name"]: build_comparison(data, ticker, r, window, as_of) for r in top
    }
    return {
        "ticker": ticker,
        "assets": data.assets,
        "window": window,
        "window_options": list(CONFIG.windows),
        "date": str(selected.index[-1].date()) if len(selected) else None,
        "window_start": (
            str(selected.index[-window].date()) if len(selected) >= window else None
        ),
        "status": "ok" if top else "insufficient_data",
        "message": (
            None
            if top
            else "해당 기간의 관계를 계산하기 위한 데이터가 충분하지 않거나 값의 변화가 없습니다."
        ),
        "relationships": top,
        "comparisons": comparisons,
        "all_relationships": ranked,
        "rolling": {r["name"]: histories[r["name"]] for r in top},
        "excluded_features": excluded,
        "unavailable_relationships": unavailable,
        "alignment_note": (
            "거시지표는 기존 전처리의 이용 가능 시점 정렬을 유지합니다. "
            "발표 지연·수정에 따른 변화도 포함될 수 있습니다."
        ),
        "disclaimer": (
            "Correlation은 선택한 기간의 실제 시장 데이터에서 관측된 움직임의 관계를 보여주며, "
            "인과관계나 미래 움직임을 의미하지 않습니다."
        ),
    }


def load_correlation_data():
    from pathlib import Path
    from backend.investment_ai.settings import ROOT, read_config
    from backend.investment_ai.data_loader import load_data

    config = read_config(
        Path(__file__).resolve().parents[2] / "config/multi_model.json"
    )
    if not config.get("data_path"):
        raise ValueError("Correlation requires real market data")
    data, _ = load_data(config, ROOT)
    return data
