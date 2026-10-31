"""ETF 15개와 벤치마크의 원본 일봉을 Parquet으로 저장."""

import argparse
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
TICKERS = (
    "SPY",
    "QQQ",
    "SMH",
    "SCHD",
    "XLE",
    "EFA",
    "SHY",
    "IEF",
    "TLT",
    "TIP",
    "LQD",
    "GLD",
    "DBB",
    "DBA",
    "VNQ",
)
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


def collect_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """가격은 보정하거나 채우지 않고, 날짜와 컬럼명만 통일."""
    frames = []
    for ticker in tickers:
        prices = yf.Ticker(ticker).history(
            start=start.isoformat(),
            # yfinance의 end는 미포함이므로 요청한 마지막 날의 다음 날 지정.
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            back_adjust=False,
            actions=True,
            repair=False,
            keepna=True,
            rounding=False,
            timeout=20,
        )
        if prices.empty:
            raise ValueError(f"{ticker}: 조회된 데이터가 없습니다.")

        prices.columns = prices.columns.str.lower().str.replace(" ", "_", regex=False)
        missing = set(PRICE_COLUMNS) - set(prices.columns)
        if missing:
            raise ValueError(f"{ticker}: 필수 컬럼 누락 {sorted(missing)}")
        if prices[["close", "adj_close"]].isna().all().any():
            raise ValueError(f"{ticker}: 종가 또는 수정종가가 모두 결측입니다.")

        # 거래소 현지 날짜 유지. UTC 변환 시 거래일이 바뀔 수 있으므로 시간대만 제거.
        prices.index = prices.index.tz_localize(None).normalize().rename("date")
        prices = prices.reset_index()
        if prices["date"].duplicated().any():
            raise ValueError(f"{ticker}: 거래일이 중복됩니다.")
        if not prices["date"].between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError(f"{ticker}: 요청 기간 밖의 데이터가 있습니다.")
        prices.insert(1, "ticker", ticker)
        frames.append(prices)

    data = pd.concat(frames, ignore_index=True)
    data["ticker"] = pd.Categorical(data["ticker"], categories=tickers, ordered=True)
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    data["ticker"] = data["ticker"].astype(str)
    data.attrs = {
        "source": "Yahoo Finance / yfinance",
        "yfinance_version": yf.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": version("pyarrow"),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "collected_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "ticker_order": list(tickers),
    }
    return data


def save_raw(data: pd.DataFrame, path: Path) -> None:
    """저장이 완료된 파일만 최종 원본 경로에 반영."""
    temporary = path.with_suffix(".parquet.tmp")
    try:
        data.to_parquet(temporary, engine="pyarrow", index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 9, 30),
        help="수집 마지막 날짜, 포함",
    )
    parser.add_argument("--benchmark", default="^SP500TR", help="S&P500 총수익지수")
    parser.add_argument("--overwrite", action="store_true", help="기존 원본 교체")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("start는 end보다 이전이어야 합니다.")

    raw_folder = ROOT / "raw"
    etf_path = raw_folder / "etf_prices.parquet"
    benchmark_path = raw_folder / "benchmark.parquet"
    if not args.overwrite and (etf_path.exists() or benchmark_path.exists()):
        parser.error("원본 파일이 이미 있습니다. 교체하려면 --overwrite를 사용하세요.")

    yf.set_tz_cache_location(str(ROOT / ".cache/yfinance"))
    yf.config.debug.hide_exceptions = False
    etf = collect_prices(TICKERS, args.start, args.end)
    benchmark = collect_prices((args.benchmark,), args.start, args.end)

    raw_folder.mkdir(parents=True, exist_ok=True)
    save_raw(etf, etf_path)
    save_raw(benchmark, benchmark_path)
    print(f"ETF 수집 완료: {len(TICKERS)}개 / {len(etf):,}행")
    print(f"벤치마크 수집 완료: {args.benchmark} / {len(benchmark):,}행")
    print(f"기간: {etf['date'].min():%Y-%m-%d} ~ {etf['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()


