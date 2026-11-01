"""Mean-Variance Optimization benchmark (과제 4번).

제약 (과제 명세)
- 공분산: 과거 252 거래일
- 비중 합 = 1
- 공매도 금지 (w >= 0)
- 개별 자산 최대 비중 <= 40%
목적함수는 max_sharpe 또는 min_variance 중 선택한다.

회계 일관성: 관측은 log_return 이지만 포트폴리오 회계는 simple return 이므로
(CONTRACT_V3.md "회계와 Reward") 최적화 입력도 expm1(log_return) 으로 변환한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import PERIODS_PER_YEAR, daily_risk_free

COVARIANCE_WINDOW = 252
MAX_WEIGHT = 0.4
#: 근사 주기 (거래일 수). 실제 달력 말일과는 다르다.
REBALANCE_DAYS = {"monthly": 21, "quarterly": 63}
#: 실제 달력 기준 리밸런싱.
#:   month_end/quarter_end   — 각 월(분기)의 **마지막** 거래일
#:   month_start/quarter_start — 각 월(분기)의 **첫** 거래일
#: 과제 문구가 "매월 첫 거래일"인지 "월 1회"인지에 따라 골라 쓴다.
CALENDAR_FREQUENCIES = ("month_end", "quarter_end", "month_start", "quarter_start")


def calendar_rebalance_dates(dates, frequency: str) -> set:
    """달력 기준 리밸런싱 날짜의 인덱스 집합.

    REBALANCE_DAYS 의 21·63일은 근사치여서 실제 월 경계와 어긋난다.
    "월 1회 / 분기 1회"를 달력 기준으로 요구하면 이쪽을 쓴다.
    """
    if frequency not in CALENDAR_FREQUENCIES:
        raise ValueError(f"frequency must be one of {CALENDAR_FREQUENCIES}")
    stamps = pd.DatetimeIndex([pd.Timestamp(str(value)[:10]) for value in dates])
    period = "M" if frequency.startswith("month") else "Q"
    frame = pd.DataFrame(
        {"period": stamps.to_period(period)}, index=np.arange(len(stamps))
    )
    pick = (
        (lambda block: block.index[0])
        if frequency.endswith("start")
        else (lambda block: block.index[-1])
    )
    return set(frame.groupby("period", observed=True).apply(pick))


def simple_returns(log_returns) -> np.ndarray:
    """로그수익률 패널 -> simple 수익률 패널."""
    return np.expm1(np.asarray(log_returns, dtype=float))


def equal_weights(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def _feasible(n_assets: int, max_weight: float) -> bool:
    """상한 때문에 합=1 이 불가능한 경우를 걸러낸다 (예: 2자산 40% 상한)."""
    return n_assets * max_weight >= 1.0 - 1e-12


def optimize_weights(
    window_simple_returns,
    objective: str = "max_sharpe",
    max_weight: float = MAX_WEIGHT,
    annual_rf: float = 0.0,
    ridge: float = 1e-8,
) -> np.ndarray:
    """SLSQP 로 제약 하의 MVO 비중을 계산한다.

    Args:
        window_simple_returns: (days, n_assets) simple 수익률 행렬.
        objective: "max_sharpe" 또는 "min_variance".
        max_weight: 개별 자산 상한.
        annual_rf: 연 무위험수익률.
        ridge: 공분산 대각 보정 (특이행렬 방지).

    Returns:
        합이 1 이고 [0, max_weight] 범위인 비중 벡터.
        최적화 실패/표본 부족 시 동일가중으로 안전하게 되돌린다.
    """
    matrix = np.asarray(window_simple_returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("window_simple_returns must be (days, assets)")
    days, n_assets = matrix.shape
    if objective not in ("max_sharpe", "min_variance"):
        raise ValueError("objective must be max_sharpe or min_variance")
    if not _feasible(n_assets, max_weight):
        raise ValueError("max_weight too small for a fully invested portfolio")
    if days < 2 or not np.isfinite(matrix).all():
        return equal_weights(n_assets)

    mean = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance) + ridge * np.eye(n_assets)
    rf = daily_risk_free(annual_rf)

    def variance(w):
        return float(w @ covariance @ w)

    def negative_sharpe(w):
        risk = np.sqrt(max(variance(w), 1e-18))
        return float(-((w @ mean - rf) / risk) * np.sqrt(PERIODS_PER_YEAR))

    target = variance if objective == "min_variance" else negative_sharpe
    result = minimize(
        target,
        equal_weights(n_assets),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        return equal_weights(n_assets)
    weights = np.clip(result.x, 0.0, max_weight)
    total = weights.sum()
    if total <= 1e-12:
        return equal_weights(n_assets)
    weights = weights / total
    # 정규화로 상한을 다시 넘길 수 있으므로 한 번 더 투영한다.
    for _ in range(50):
        excess = weights - max_weight
        if excess.max() <= 1e-9:
            break
        spill = excess.clip(min=0).sum()
        weights = np.minimum(weights, max_weight)
        room = (max_weight - weights).clip(min=0)
        if room.sum() <= 1e-12:
            return equal_weights(n_assets)
        weights = weights + spill * room / room.sum()
    return weights / weights.sum()


def weights_at(
    data,
    index: int,
    objective: str = "max_sharpe",
    window: int = COVARIANCE_WINDOW,
    max_weight: float = MAX_WEIGHT,
    annual_rf: float = 0.0,
    strict_window: bool = False,
) -> np.ndarray:
    """결정일 index 의 종가까지만 사용해 MVO 비중을 계산한다 (미래 정보 차단).

    PortfolioEnv 는 t 종가를 관측해 t+1 종가에 체결하므로, t 를 포함한
    과거 252 거래일을 쓰는 것이 계약상 허용되는 최대 정보량이다.

    Args:
        strict_window: True 면 과거 252일이 다 모이기 전(워밍업 구간)에는
            최적화를 하지 않고 동일가중을 쓴다. 과제가 "공분산: 과거 252 거래일"
            을 엄격히 요구할 때 쓴다. False(기본)면 가용한 기간으로 계산하되
            표본이 짧아 공분산 추정이 불안정하다는 점을 보고서에 적어야 한다.
    """
    if not 0 <= index < len(data.dates):
        raise ValueError("MVO decision index outside data")
    start = max(0, index - window + 1)
    available = index + 1 - start
    if strict_window and available < window:
        return equal_weights(len(data.assets))
    return optimize_weights(
        simple_returns(data.returns[start : index + 1]),
        objective=objective,
        max_weight=max_weight,
        annual_rf=annual_rf,
    )
