"""ETF 기술적 지표와 동일 시점의 ALFRED Vintage 기반 macro 파생변수 계산."""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TICKERS = "SPY QQQ SMH SCHD XLE EFA SHY IEF TLT TIP LQD GLD DBB DBA VNQ".split()
LEVELS = "unemp ism_pmi effr ust_2y ust_10y ust_30y tips_10y vix wti".split()
DATE_COLUMNS = ["observation_date", "realtime_start", "realtime_end", "release_date"]
INPUT_COLUMNS = [
    "calculation_date", "input_value", "input_realtime_start", "input_realtime_end",
    "base_observation_date", "base_value", "base_realtime_start", "base_realtime_end",
    "lag_reference_date",
]


def calculate_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """최초 14개 가격 변동의 단순평균으로 초기화한 뒤 Wilder 평활화."""
    change = price.diff()
    averages = []
    for moves in (change.clip(lower=0), -change.clip(upper=0)):
        seeded = moves.copy()
        seeded.iloc[:period] = np.nan
        if len(seeded) > period:
            seeded.iloc[period] = moves.iloc[1:period + 1].mean()
        averages.append(seeded.ewm(alpha=1 / period, adjust=False).mean())
    gain, loss = averages
    total = gain + loss
    # 변화가 전혀 없으면 중립값 50. 계산 준비 구간의 NaN은 유지.
    return (100 * gain / total).mask(total.eq(0), 50)


def calculate_etf(etf: pd.DataFrame) -> pd.DataFrame:
    if set(etf["ticker"]) != set(TICKERS):
        raise ValueError("ETF 15종목 구성이 일치하지 않습니다.")
    if etf.duplicated(["date", "ticker"]).any():
        raise ValueError("ETF 날짜·종목 중복이 있습니다.")
    if (~np.isfinite(etf["adj_close"]) | etf["adj_close"].le(0)).any():
        raise ValueError("수정종가에 결측·무한대·0 이하 가격이 있습니다.")

    frames = []
    for ticker in TICKERS:
        data = etf.loc[etf["ticker"].eq(ticker), ["date", "ticker", "adj_close"]]
        data = data.sort_values("date").reset_index(drop=True)
        price = data["adj_close"]
        data["log_return"] = np.log(price / price.shift(1))
        data["rsi_14"] = calculate_rsi(price)
        fast = price.ewm(span=12, adjust=False, min_periods=12).mean()
        slow = price.ewm(span=26, adjust=False, min_periods=26).mean()
        data["macd"] = fast - slow
        data["macd_signal"] = data["macd"].ewm(
            span=9, adjust=False, min_periods=9
        ).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        frames.append(data)
    result = pd.concat(frames, ignore_index=True).sort_values("date", kind="stable")
    result = result.reset_index(drop=True)
    result.attrs = {
        **etf.attrs,
        "price_basis": "adj_close",
        "log_return": "ln(adj_close_t / adj_close_previous_session), unscaled",
        "rsi_14": "Wilder alpha=1/14, initial 14-change SMA, flat series=50",
        "macd": "EMA12 - EMA26; adjust=False; min_periods=span",
        "macd_signal": "EMA9 of valid MACD; adjust=False; min_periods=9",
        "macd_hist": "macd - macd_signal",
    }
    return result


def vintage_snapshots(data: pd.DataFrame, cutoff: pd.Timestamp):
    """변경일마다 복원. 입력 유효기간이 끝나는 경우에도 재평가."""
    changes = pd.DatetimeIndex(data["realtime_start"].unique()).union(
        pd.DatetimeIndex(data["realtime_end"].unique()) + timedelta(days=1)
    ).sort_values()
    changes = changes[changes <= cutoff]
    for position, asof in enumerate(changes):
        end = changes[position + 1] - timedelta(days=1) if position + 1 < len(changes) else cutoff
        active = data.loc[
            data["realtime_start"].le(asof) & data["realtime_end"].ge(asof)
            & data["observation_date"].le(asof)
        ].sort_values("observation_date")
        yield asof, end, active


def derived_row(name, series_id, observed, value, asof, end, current, base=None, reference=pd.NaT):
    """계산 결과의 유효기간과 실제 사용한 두 입력의 원본 날짜를 구분."""
    return {
        "series_id": series_id, "variable_name": name,
        "observation_date": observed, "value": value,
        "realtime_start": asof, "realtime_end": end,
        "release_date": pd.NaT, "source": "alfred", "calculation_date": asof,
        "input_value": current["value"] if current is not None else np.nan,
        "input_realtime_start": current["realtime_start"] if current is not None else pd.NaT,
        "input_realtime_end": current["realtime_end"] if current is not None else pd.NaT,
        "base_observation_date": base["observation_date"] if base is not None else pd.NaT,
        "base_value": base["value"] if base is not None else np.nan,
        "base_realtime_start": base["realtime_start"] if base is not None else pd.NaT,
        "base_realtime_end": base["realtime_end"] if base is not None else pd.NaT,
        "lag_reference_date": reference,
    }


def calculate_pce(data: pd.DataFrame, months: int, name: str, cutoff: pd.Timestamp) -> list[dict]:
    rows = []
    for asof, end, active in vintage_snapshots(data, cutoff):
        snapshot = active.set_index("observation_date", drop=False, verify_integrity=True)
        current = snapshot.iloc[-1] if len(snapshot) else None
        observed = current["observation_date"] if current is not None else pd.NaT
        reference = observed - pd.DateOffset(months=months) if pd.notna(observed) else pd.NaT
        base = snapshot.loc[reference] if reference in snapshot.index else None
        value = np.nan
        if current is not None and base is not None and base["value"] > 0:
            value = (current["value"] / base["value"] - 1) * 100
        rows.append(derived_row(
            name, data["series_id"].iloc[0], observed, value,
            asof, end, current, base, reference,
        ))
    return rows


def calculate_usd(data: pd.DataFrame, sessions: pd.DatetimeIndex, cutoff: pd.Timestamp) -> list[dict]:
    rows = []
    for asof, end, active in vintage_snapshots(data, cutoff):
        snapshot = active.set_index("observation_date", drop=False, verify_integrity=True)
        current = snapshot.iloc[-1] if len(snapshot) else None
        observed = current["observation_date"] if current is not None else pd.NaT
        reference, base, value = pd.NaT, None, np.nan
        if pd.notna(observed):
            # 거래일이면 20칸 전, 비거래일이면 그 날짜보다 앞선 20번째 SPY 거래일.
            position = sessions.searchsorted(observed, side="left")
            if position >= 20:
                reference = sessions[position - 20]
                candidates = snapshot.loc[
                    (snapshot.index <= reference) & snapshot["value"].notna()
                ]
                base = candidates.iloc[-1] if len(candidates) else None
        if current is not None and base is not None and current["value"] > 0 and base["value"] > 0:
            value = np.log(current["value"] / base["value"])
        rows.append(derived_row(
            "usd_ret_20d", "DTWEXBGS", observed, value,
            asof, end, current, base, reference,
        ))
    return rows


def calculate_spread(rates: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict]:
    rows = []
    for observed, data in rates.groupby("observation_date", sort=True):
        for asof, end, active in vintage_snapshots(data, cutoff):
            snapshot = active.set_index("variable_name", verify_integrity=True)
            ten = snapshot.loc["ust_10y"] if "ust_10y" in snapshot.index else None
            two = snapshot.loc["ust_2y"] if "ust_2y" in snapshot.index else None
            value = ten["value"] - two["value"] if ten is not None and two is not None else np.nan
            rows.append(derived_row(
                "yield_spread", "DGS10,DGS2", observed, value, asof, end, ten, two,
            ))
    return rows


def calculate_macro(macro: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    required = set(LEVELS) | {"core_pce", "real_pce", "broad_usd"}
    if set(macro["variable_name"]) != required:
        raise ValueError("원본 macro 12개 지표 구성이 일치하지 않습니다.")
    fred = macro.loc[macro["source"].eq("alfred")].sort_values(
        ["variable_name", "observation_date", "realtime_start"]
    )
    previous_end = fred.groupby(["variable_name", "observation_date"])["realtime_end"].shift()
    if (
        fred[["observation_date", "realtime_start", "realtime_end"]].isna().any().any()
        or fred["realtime_start"].gt(fred["realtime_end"]).any()
        or fred["realtime_start"].le(previous_end).any()
    ):
        raise ValueError("ALFRED 날짜 결측 또는 Vintage 유효기간 중복이 있습니다.")
    cutoff = pd.Timestamp(macro.attrs["requested_end"])
    if fred[["realtime_start", "realtime_end"]].gt(cutoff).any().any():
        raise ValueError("수집 종료일 이후의 Vintage가 포함되어 있습니다.")

    # 원지수 9개는 관측일·수정값·결측값·PMI 발표일을 그대로 복사.
    rows = macro.loc[macro["variable_name"].isin(LEVELS)].to_dict("records")
    for original, months, name in [
        ("core_pce", 12, "core_pce_yoy"), ("real_pce", 1, "real_pce_mom")
    ]:
        rows.extend(calculate_pce(fred.loc[fred["variable_name"].eq(original)], months, name, cutoff))
    rows.extend(calculate_usd(fred.loc[fred["variable_name"].eq("broad_usd")], sessions, cutoff))
    rows.extend(calculate_spread(fred.loc[fred["variable_name"].isin(["ust_10y", "ust_2y"])], cutoff))

    result = pd.DataFrame(rows, columns=[*macro.columns, *INPUT_COLUMNS])
    for column in [*DATE_COLUMNS, *[c for c in INPUT_COLUMNS if "date" in c or "realtime" in c]]:
        result[column] = pd.to_datetime(result[column], errors="raise")
    result = result.sort_values(
        ["variable_name", "observation_date", "realtime_start", "release_date"], kind="stable"
    ).reset_index(drop=True)
    result.attrs = {
        **macro.attrs,
        "core_pce_yoy": "(latest_month / exact_12_months_prior - 1) * 100; same vintage",
        "real_pce_mom": "(latest_month / exact_previous_month - 1) * 100; same vintage",
        "usd_ret_20d": "ln(latest_observation / base); same vintage; no multiplication by 100",
        "usd_anchor": "latest USD observation_date, not calculation_date",
        "usd_calendar": "SPY raw sessions; 20 sessions earlier; non-session: 20th preceding session",
        "usd_base": "latest nonmissing observation on or before lag_reference_date; no old-vintage revival",
        "yield_spread": "ust_10y - ust_2y, percentage points; same observation_date and vintage",
        "input_columns": "input_*: numerator/10Y; base_*: denominator/2Y; raw nine levels have no calculation fields",
        "derived_validity": "realtime_start/end: calculation-state interval; input/base realtime fields: raw intervals",
        "availability_note": "ALFRED dates only; official publication time and ETF alignment deferred to 05",
        "missing_policy": "NaN preserved; no global forward fill, interpolation, scaling, or WTI transformation",
    }
    return result


def save_parquet(data: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    try:
        data.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="기존 04번 결과 교체")
    args = parser.parse_args()
    output = ROOT / "processed"
    paths = [output / name for name in ("etf_features.parquet", "log_returns.parquet", "macro_features.parquet")]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error("04번 결과가 이미 있습니다. 교체하려면 --overwrite를 사용하세요.")

    etf = pd.read_parquet(ROOT / "raw/etf_prices.parquet")
    macro = pd.read_parquet(ROOT / "raw/macro_observations.parquet")
    etf["date"] = pd.to_datetime(etf["date"], errors="raise")
    for column in DATE_COLUMNS:
        macro[column] = pd.to_datetime(macro[column], errors="raise")
    sessions = pd.DatetimeIndex(etf.loc[etf["ticker"].eq("SPY"), "date"].unique()).sort_values()
    features = calculate_etf(etf)
    returns = features[["date", "ticker", "log_return"]].copy()
    returns.attrs = {**etf.attrs, "formula": "ln(adj_close_t / adj_close_previous_session), unscaled"}
    macro_features = calculate_macro(macro, sessions)
    output.mkdir(parents=True, exist_ok=True)
    for data, path in zip((features, returns, macro_features), paths):
        save_parquet(data, path)
        print(f"저장: {path.name} / {len(data):,}행 / {len(data.columns)}컬럼")
    print("macro 값 결측:", macro_features.groupby("variable_name")["value"].apply(lambda x: int(x.isna().sum())).to_dict())
    print("정규화·ETF/macro 최종 결합·원본 변경 없음")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"계산 실패: {error}") from None
