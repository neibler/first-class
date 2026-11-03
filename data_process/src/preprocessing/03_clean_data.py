"""원본을 보존하면서 날짜·중복·결측·거래일·가격·Vintage 이력 점검."""

from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MONTHLY = {"core_pce", "real_pce", "unemp", "ism_pmi"}
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """메모리에서 날짜 형식과 순서만 통일. 행·가격·결측값은 유지."""
    etf = pd.read_parquet(ROOT / "raw/etf_prices.parquet")
    macro = pd.read_parquet(ROOT / "raw/macro_observations.parquet")
    etf["date"] = pd.to_datetime(etf["date"], errors="raise")
    for column in ("observation_date", "realtime_start", "realtime_end", "release_date"):
        macro[column] = pd.to_datetime(macro[column], errors="raise")

    # 같은 날짜의 기존 ETF 순서를 유지하며 날짜만 정렬.
    etf = etf.sort_values("date", kind="stable").reset_index(drop=True)
    macro = macro.sort_values(
        ["variable_name", "observation_date", "realtime_start", "release_date"],
        kind="stable",
    ).reset_index(drop=True)
    return etf, macro


def check_etf(etf: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """실제 거래일 누락과 휴장일을 구분하고, 가격 오류를 종목별 집계."""
    tickers = list(dict.fromkeys([
        *etf.attrs["ticker_order"], *etf["ticker"].dropna().unique()
    ]))
    results = []
    for ticker in tickers:
        data = etf.loc[etf["ticker"].eq(ticker)]
        dates = pd.DatetimeIndex(data["date"].dropna().unique())
        prices = data[PRICE_COLUMNS].apply(pd.to_numeric, errors="raise")
        bad_price = prices.notna() & (~np.isfinite(prices) | prices.le(0))
        bad_ohlc = data["high"].lt(data[["open", "low", "close"]].max(axis=1)) | data[
            "low"
        ].gt(data[["open", "high", "close"]].min(axis=1))
        results.append({
            "ticker": ticker,
            "행": len(data),
            "중복": int(data.duplicated(["date", "ticker"]).sum()),
            "결측": int(data.isna().sum().sum()),
            "거래일누락": len(sessions.difference(dates)),
            "휴장일행": int((data["date"].notna() & ~data["date"].isin(sessions)).sum()),
            "수정종가결측": int(data["adj_close"].isna().sum()),
            "비정상가격": int(bad_price.any(axis=1).sum()),
            "OHLC오류": int(bad_ohlc.sum()),
        })
    return pd.DataFrame(results)


def check_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """관측일이 같은 수정 이력을 중복과 구분. 미확인 날짜는 오류로 간주하지 않음."""
    start = pd.Timestamp(macro.attrs["requested_start"])
    end = pd.Timestamp(macro.attrs["requested_end"])
    results = []
    for variable, data in macro.groupby("variable_name", sort=False):
        fred = data.loc[data["source"].eq("alfred")].sort_values(
            ["series_id", "observation_date", "realtime_start"]
        )
        pmi = data.loc[data["source"].eq("investing")]
        previous_end = fred.groupby(["series_id", "observation_date"])[
            "realtime_end"
        ].shift()
        bad_vintage = (
            fred[["observation_date", "realtime_start", "realtime_end"]].isna().any(axis=1)
            | ~fred["observation_date"].between(start, end)
            | fred["realtime_start"].gt(fred["realtime_end"])
            | fred[["realtime_start", "realtime_end"]].gt(end).any(axis=1)
            | fred["realtime_start"].le(previous_end)
        )
        bad_release = ~pmi["release_date"].dt.normalize().between(start, end)
        duplicates = fred.duplicated(["series_id", "observation_date", "realtime_start"]).sum()
        duplicates += pmi.duplicated(["series_id", "release_date"]).sum()
        versions = fred.groupby(["series_id", "observation_date"])["realtime_start"].nunique()

        dates = data["release_date"] if variable == "ism_pmi" else data["observation_date"]
        dates = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
        gaps = 0
        if len(dates):
            # 확보한 관측 범위 안의 공백만 검사. 발표 전 마지막 달은 누락으로 보지 않음.
            if variable in MONTHLY:
                months = dates.to_period("M").unique()
                gaps = len(pd.period_range(months.min(), months.max()).difference(months))
            else:
                gaps = len(pd.bdate_range(dates.min(), dates.max()).difference(dates))

        first_vintage = fred["realtime_start"].min()
        values = pd.to_numeric(data["value"], errors="raise")
        results.append({
            "지표": variable,
            "행": len(data),
            "값결측": int(values.isna().sum()),
            "중복키": int(duplicates),
            "관측공백": gaps,
            "비정상값": int((values.notna() & ~np.isfinite(values)).sum()),
            "추가버전": int((versions - 1).clip(lower=0).sum()),
            "날짜·Vintage오류": int(bad_vintage.sum() + bad_release.sum()),
            "최초Vintage": first_vintage.strftime("%Y-%m-%d") if pd.notna(first_vintage) else "-",
        })
    return pd.DataFrame(results)


def main() -> None:
    etf, macro = load_data()
    start, end = etf.attrs["requested_start"], etf.attrs["requested_end"]
    sessions = xcals.get_calendar("XNYS").sessions_in_range(start, end)
    days = pd.date_range(start, end)
    weekdays = days[days.dayofweek < 5]
    etf_report = check_etf(etf, sessions)
    macro_report = check_macro(macro)
    with pd.option_context("display.unicode.east_asian_width", True):
        print(f"[ETF] {etf['ticker'].nunique()}개 / {len(etf):,}행 / 정상 거래일 {len(sessions):,}일")
        print(f"기간: {start} ~ {end} / 날짜: {etf['date'].dtype}, 오름차순")
        print(etf_report.to_string(index=False))
        print(
            f"정상 제외 날짜: 주말 {int((days.dayofweek >= 5).sum()):,}일 / "
            f"평일 휴장 {len(weekdays.difference(sessions)):,}일"
        )
        print(f"\n[거시지표] {macro['variable_name'].nunique()}개 / {len(macro):,}행")
        print("날짜: datetime / 지표별 관측일·Vintage 순서 정렬")
        print(macro_report.to_string(index=False))
    print("관측공백: 확보된 첫~마지막 날짜 안에서 월별은 관측월, 일별은 평일 기준 참고값")

    pmi = macro.loc[macro["source"].eq("investing")]
    fred = macro.loc[macro["source"].eq("alfred")]
    print(
        f"미확인 날짜 유지: PMI 관측일·Vintage {len(pmi):,}행 / "
        f"ALFRED 실제 발표일 {int(fred['release_date'].isna().sum()):,}행"
    )
    print(f"PMI 주말 발표일: {int(pmi['release_date'].dt.dayofweek.ge(5).sum())}건, 원본 유지")
    print("값 채우기·행 제거·원본 파일 변경 없음")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"점검 실패: {error}") from None
