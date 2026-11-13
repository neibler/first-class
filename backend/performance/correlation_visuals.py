"""Visualization-only market paths. Never feed these into correlations."""

import numpy as np
import pandas as pd


def normalize_series_to_100(series):
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = clean.dropna()
    if valid.empty or valid.iloc[0] <= 0:
        raise ValueError("A positive finite starting value is required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = clean / valid.iloc[0] * 100
    if not np.isfinite(normalized.dropna()).all():
        raise ValueError("Nonfinite normalized values")
    normalized.loc[valid.index[0]] = 100.0
    return normalized


def normalize_pair(left, right):
    # Join BEFORE choosing a baseline; both series must start on the same date.
    pair = pd.concat(
        [left.rename("selected"), right.rename("comparison")], axis=1, join="inner"
    )
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(pair) < 2 or pair.index.has_duplicates:
        raise ValueError("Not enough distinct common observations")
    return pd.DataFrame({name: normalize_series_to_100(pair[name]) for name in pair})


def etf_relative_path(data, ticker, dates):
    daily = pd.Series(
        data.returns[:, data.assets.index(ticker)], index=pd.DatetimeIndex(data.dates)
    ).reindex(dates)
    # Close at the first displayed date is 100. Its return from the preceding
    # date is part of correlation, but not a movement AFTER this graph's baseline.
    daily = daily.copy()
    if len(daily):
        daily.iloc[0] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(daily.cumsum(skipna=False)) * 100


def build_comparison(data, ticker, relationship, window, as_of=None):
    index = pd.DatetimeIndex(data.dates)
    cutoff = pd.Timestamp(as_of) if as_of else index[-1]
    dates = index[index <= cutoff][-window:]
    is_etf = relationship["type"] == "ETF"
    result = {
        "status": "unavailable",
        "records": [],
        "kind": "price" if is_etf else "market_value",
        "selected_label": ticker,
        "comparison_label": relationship["display_name"],
        "source_note": "ETF는 기존 일별 로그수익률로 재구성한 상대 가격입니다. 원본 가격 금액을 표시하지 않습니다.",
        "message": "동일 기간의 유효 데이터가 부족하거나 시작값이 0·음수여서 정규화 비교를 제공하지 않습니다.",
    }
    if len(dates) < window:
        return result
    try:
        left = etf_relative_path(data, ticker, dates)
        if is_etf:
            right = etf_relative_path(data, relationship["name"], dates)
        else:
            suffix = (
                "_change" if relationship["transformation"] == "diff" else "_return"
            )
            macro = relationship["name"].removesuffix(suffix)
            right = pd.Series(
                data.macro[:, data.macros.index(macro)], index=index
            ).reindex(dates)
            result["comparison_label"] = {
                "ust_2y": "미국 2년물 금리",
                "ust_10y": "미국 10년물 금리",
                "ust_30y": "미국 30년물 금리",
                "tips_10y": "미국 10년물 실질금리",
                "yield_spread": "장단기 금리 차이",
                "vix": "VIX",
                "wti": "WTI",
            }.get(macro, macro)
        pair = normalize_pair(left, right)
        result.update(
            status="ok",
            message=None,
            start=str(pair.index[0].date()),
            end=str(pair.index[-1].date()),
            observations=len(pair),
            requested_observations=window,
            records=[
                {
                    "date": str(d.date()),
                    "selected": float(r.selected),
                    "comparison": float(r.comparison),
                }
                for d, r in pair.iterrows()
            ],
        )
        if len(pair) != window:
            result["message"] = (
                f"최근 {window}거래일 중 공통 유효 날짜 {len(pair)}개만 표시합니다. 시작점은 첫 공통 유효일입니다."
            )
    except (ValueError, KeyError, IndexError):
        pass
    return result
