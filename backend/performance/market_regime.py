"""Market regime labelling: 상승장 / 횡보장 / 하락장 (과제 12번).

분류 규칙 (보고서에 그대로 적을 수 있도록 단순·결정적으로 둔다)
- 기준: 벤치마크(S&P500) 일별 수익률.
- 각 날짜 t 의 라벨은 t 를 포함한 과거 `window` 거래일 누적수익률로 정한다.
  과거만 쓰므로 미래 정보가 들어가지 않는다 (전략 수익률과 같은 시점 비교).
- 누적수익률 >= +threshold -> 상승장, <= -threshold -> 하락장, 그 사이 -> 횡보장.
- 창이 아직 다 차지 않은 초반 구간은 가용한 구간으로 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import METRIC_ORDER, performance_metrics

REGIME_LABELS = {"bull": "상승장", "sideways": "횡보장", "bear": "하락장"}
DEFAULT_WINDOW = 63  # 약 3개월
DEFAULT_THRESHOLD = 0.05  # ±5%


def classify(
    benchmark_returns,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    index=None,
) -> pd.Series:
    """일별 벤치마크 수익률 -> 날짜별 국면 라벨 Series."""
    if window < 2 or threshold <= 0:
        raise ValueError("window >= 2, threshold > 0 이어야 합니다")
    series = pd.Series(
        np.asarray(benchmark_returns, dtype=float),
        index=index if index is not None else None,
    )
    trailing = (1 + series).rolling(window, min_periods=2).apply(np.prod, raw=True) - 1
    # 창이 덜 찬 맨 앞 구간은 지금까지의 누적으로 대체한다.
    trailing = trailing.fillna((1 + series).cumprod() - 1)
    labels = pd.Series("sideways", index=series.index, name="regime")
    labels[trailing >= threshold] = "bull"
    labels[trailing <= -threshold] = "bear"
    return labels


def regime_frame(returns: pd.DataFrame, benchmark: str | None = None, **options):
    """전략 수익률 표에 regime 열을 붙인다 (기준: 순수 시장 수익률)."""
    if benchmark is None:
        benchmark = next(
            (n for n in ("benchmark_raw", "benchmark") if n in returns.columns),
            "benchmark",
        )
    if benchmark not in returns.columns:
        raise ValueError(f"벤치마크 열 '{benchmark}' 이 없습니다")
    frame = returns.copy()
    frame["regime"] = classify(
        frame[benchmark].to_numpy(dtype=float), index=frame.index, **options
    )
    return frame


def regime_summary(
    frame: pd.DataFrame, annual_rf: float = 0.0, market: str | None = None
) -> pd.DataFrame:
    """국면 × 전략별 성과표 (과제 12번 산출물).

    Args:
        market: Alpha/Beta 기준 열. 기본은 benchmark_raw(순수 시장) -> benchmark.
    """
    strategies = [c for c in frame.columns if c != "regime"]
    if market is None:
        market = next(
            (n for n in ("benchmark_raw", "benchmark") if n in frame.columns), None
        )
    rows = []
    for regime, block in frame.groupby("regime", sort=False):
        benchmark = (
            block[market].to_numpy(dtype=float) if market in block.columns else None
        )
        for strategy in strategies:
            values = performance_metrics(
                block[strategy].to_numpy(dtype=float),
                None if strategy == market else benchmark,
                annual_rf,
            )
            rows.append(
                {
                    "regime": regime,
                    "regime_label": REGIME_LABELS.get(regime, regime),
                    "strategy": strategy,
                    **{key: values[key] for key in ["days", *METRIC_ORDER]},
                }
            )
    return pd.DataFrame(rows)


def regime_segments(labels: pd.Series) -> pd.DataFrame:
    """연속된 같은 국면을 구간으로 묶는다 (그래프 음영 표시용)."""
    changed = labels.ne(labels.shift()).cumsum()
    rows = []
    for _, block in labels.groupby(changed):
        rows.append(
            {
                "regime": block.iloc[0],
                "regime_label": REGIME_LABELS.get(block.iloc[0], block.iloc[0]),
                "start": block.index[0],
                "end": block.index[-1],
                "days": int(len(block)),
            }
        )
    return pd.DataFrame(rows)


def regime_counts(labels: pd.Series) -> pd.DataFrame:
    """국면별 거래일 수 — ANOVA 표본 균형을 설명할 때 쓴다."""
    counts = labels.value_counts().rename_axis("regime").reset_index(name="days")
    counts["regime_label"] = counts.regime.map(REGIME_LABELS).fillna(counts.regime)
    counts["share"] = counts.days / counts.days.sum()
    return counts
