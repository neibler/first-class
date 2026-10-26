"""
ETF 15개의 yfinance 원본 일봉을 하나의 CSV로 저장
가격 보정, 결측치 채우기, 수익률/RSI/MACD 계산은 수행 X
"""

import argparse
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_DIR = Path(__file__).resolve().parent
TICKERS = (
    "SPY", "QQQ", "SMH", "SCHD", "XLE", "EFA", "SHY", "IEF",
    "TLT", "TIP", "LQD", "GLD", "DBB", "DBA", "VNQ",
)
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """실패하면 최대 세 번 재시도하고, 제공받은 값은 그대로 유지한다."""
    for attempt in range(1, 4):
        try:
            prices = yf.Ticker(ticker).history(
                start=start,
                end=end,
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
                raise ValueError("조회 결과가 비어 있습니다.")
            missing_columns = set(PRICE_COLUMNS) - set(prices.columns)
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {sorted(missing_columns)}")
            if prices.index.has_duplicates:
                raise ValueError("원본에 중복 날짜가 있습니다.")
            if prices["Close"].notna().sum() == 0:
                raise ValueError("종가가 모두 결측입니다.")

            # 미국 거래소 현지 날짜를 유지한다. UTC로 변환하지 않는다.
            prices = prices.reset_index()
            prices["Date"] = pd.to_datetime(prices["Date"]).dt.strftime("%Y-%m-%d")
            prices.insert(1, "Ticker", ticker)
            return prices
        except Exception as error:
            print(f"  {ticker}: {attempt}/3 실패: {error}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"{ticker} 수집에 실패했습니다.") from error
            time.sleep(attempt * 2)
    raise RuntimeError("도달할 수 없는 분기입니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-07-01", help="수집 시작일, 포함")
    parser.add_argument("--end", default="2026-01-01", help="수집 종료일, 미포함")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "data/raw/etf_prices.csv"
    )
    parser.add_argument("--overwrite", action="store_true", help="기존 CSV 덮어쓰기")
    args = parser.parse_args()

    if date.fromisoformat(args.start) >= date.fromisoformat(args.end):
        parser.error("start는 end보다 이전이어야 합니다.")
    if args.output.exists() and not args.overwrite:
        parser.error("출력 파일이 이미 있습니다. 다른 경로 또는 --overwrite를 사용하세요.")

    # yfinance의 시간대/쿠키 캐시도 프로젝트 내부에 둔다.
    yf.set_tz_cache_location(str(PROJECT_DIR / ".cache/yfinance"))
    yf.config.debug.hide_exceptions = False
    fetched_at = datetime.now(timezone.utc).isoformat()
    print(f"수집 시각 (UTC): {fetched_at}", flush=True)
    print(f"기간: {args.start} 이상, {args.end} 미만 / USD / 일봉", flush=True)

    frames = []
    for number, ticker in enumerate(TICKERS, start=1):
        print(f"[{number}/{len(TICKERS)}] {ticker} 수집 중", flush=True)
        prices = fetch_ticker(ticker, args.start, args.end)
        print(
            f"  {len(prices):,}행 / {prices['Date'].min()} ~ {prices['Date'].max()}",
            flush=True,
        )
        frames.append(prices)

    # 행 하나 = 거래일 하나의 ETF 하나. 가격값은 수정하거나 반올림하지 않는다.
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.sort_values(["Date", "Ticker"], kind="stable").reset_index(drop=True)
    if raw.duplicated(["Date", "Ticker"]).any():
        raise ValueError("날짜/티커가 중복됩니다. 파일을 저장하지 않습니다.")
    if set(raw["Ticker"].unique()) != set(TICKERS):
        raise ValueError("ETF 15개가 모두 수집되지 않았습니다.")
    if not raw["Date"].between(args.start, args.end, inclusive="left").all():
        raise ValueError("조회 범위를 벗어나는 날짜가 있습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 15개 모두 성공했을 때 저장한다. 실패한 수집으로 기존 CSV를 대체하지 않는다.
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        raw.to_csv(temporary, index=False, encoding="utf-8", date_format="%Y-%m-%d")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"\n저장 완료: {args.output.resolve()}", flush=True)
    print(f"{len(raw):,}행 / ETF {raw['Ticker'].nunique()}개", flush=True)
    print(f"컬럼: {', '.join(raw.columns)}", flush=True)
    print("가격/거래량 결측 개수 (채우지 않고 원본 유지):", flush=True)
    print(raw[list(PRICE_COLUMNS)].isna().sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
