"""Close 기준 로그수익률, RSI, MACD를 계산해 날짜별 CSV로 저장"""

from pathlib import Path

import numpy as np
import pandas as pd


FOLDER = Path(__file__).resolve().parent
TICKERS = (
    "SPY", "QQQ", "SMH", "SCHD", "XLE", "EFA", "SHY", "IEF",
    "TLT", "TIP", "LQD", "GLD", "DBB", "DBA", "VNQ",
)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()
    movements = pd.DataFrame({
        "gain": change.clip(lower=0),
        "loss": -change.clip(upper=0),
    })
    # Wilder 초기값: 첫 14개 가격 변화의 평균. 이후에는 전날 평균을 이어 쓴다.
    movements.iloc[period] = movements.iloc[1:period + 1].mean()
    averages = movements.iloc[period:].ewm(alpha=1 / period, adjust=False).mean()
    total = averages["gain"] + averages["loss"]
    return (100 * averages["gain"] / total).where(total.ne(0), 50)


def preprocess() -> pd.DataFrame:
    raw = pd.read_csv(
        FOLDER.parent / "raw/etf_prices.csv",
        usecols=["Date", "Ticker", "Close"],
        parse_dates=["Date"],
    )
    if set(raw["Ticker"]) != set(TICKERS):
        raise ValueError("원본의 ETF 목록이 지정한 15개 종목과 다릅니다.")

    prices = raw.pivot(index="Date", columns="Ticker", values="Close")
    prices = prices.reindex(columns=TICKERS).sort_index()
    if not np.isfinite(prices.to_numpy()).all() or prices.le(0).any().any():
        raise ValueError("원본 종가에 결측, 무한대 또는 0 이하 값이 있습니다.")

    features = {}
    for ticker in TICKERS:
        close = prices[ticker]
        features[f"{ticker}_CLOSE"] = close
        features[f"{ticker}_LOG_RETURN"] = np.log(close / close.shift(1))
        features[f"{ticker}_RSI"] = calculate_rsi(close)
        features[f"{ticker}_MACD"] = (
            close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean()
        )

    # 준비 구간까지 계산한 뒤, 전달할 3개년만 추출한다.
    dataset = pd.DataFrame(features).loc["2023-01-01":"2025-12-31"]
    if dataset.empty or not np.isfinite(dataset.to_numpy()).all():
        raise ValueError("최종 데이터가 비어 있거나 결측/무한대가 있습니다.")
    return dataset


if __name__ == "__main__":
    dataset = preprocess()
    output = FOLDER / "etf_features.csv"
    dataset.to_csv(output, index_label="Date", date_format="%Y-%m-%d")
    print(f"저장 완료: {output}")
    print(f"{len(dataset):,}행 × {len(dataset.columns) + 1}컬럼 (Date 포함)")
