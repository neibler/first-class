"""12 performance metrics on the investment-AI accounting convention.

계약(CONTRACT_V3.md) 준수 사항
- 포트폴리오 수익률은 비용 차감 후 simple return을 그대로 사용한다.
- Sharpe 는 투자 AI 담당의 accounting.sharpe_ratio 를 그대로 재사용한다
  (ddof=1, sqrt(252), 연 rf -> 일 rf 는 expm1(log1p(rf)/252)).
- 정의할 수 없는 지표는 0 으로 메우지 않고 None 을 반환한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.investment_ai.accounting import equity_curve, sharpe_ratio

PERIODS_PER_YEAR = 252

#: 과제에서 요구하는 12개 지표의 표준 순서와 한국어 라벨.
METRIC_LABELS = {
    "cumulative_return": "누적 수익률",
    "cagr": "CAGR",
    "annualized_volatility": "연환산 변동성",
    "var_95": "VaR 95%",
    "cvar_95": "CVaR 95%",
    "max_drawdown": "MDD",
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "alpha": "Alpha",
    "beta": "Beta",
    "information_ratio": "Information Ratio",
}
METRIC_ORDER = list(METRIC_LABELS)


def as_returns(values) -> np.ndarray:
    """Coerce a return series to a finite float array."""
    array = np.asarray(
        values.to_numpy() if isinstance(values, (pd.Series, pd.Index)) else values,
        dtype=float,
    )
    if array.ndim != 1:
        raise ValueError("Returns must be one-dimensional")
    if len(array) and not np.isfinite(array).all():
        raise ValueError("Returns contain nonfinite values")
    return array


def daily_risk_free(annual_rf: float = 0.0) -> float:
    """Annual rate -> daily simple rate, identical to accounting.sharpe_ratio."""
    return float(np.expm1(np.log1p(annual_rf) / PERIODS_PER_YEAR))


def drawdown_curve(returns) -> tuple[np.ndarray, np.ndarray]:
    """Return (equity curve incl. 1.0 start, drawdown series)."""
    equity = equity_curve(as_returns(returns))
    return equity, 1 - equity / np.maximum.accumulate(equity)


# --------------------------------------------------------------------------- #
# 수익률 지표
# --------------------------------------------------------------------------- #
def cumulative_return(returns) -> float | None:
    r = as_returns(returns)
    if not len(r):
        return None
    return float(equity_curve(r)[-1] - 1)


def cagr(returns) -> float | None:
    """Compound annual growth rate. 전액 손실(에쿼티<=0)이면 정의 불가."""
    r = as_returns(returns)
    if not len(r):
        return None
    final = equity_curve(r)[-1]
    if final <= 0:
        return None
    return float(final ** (PERIODS_PER_YEAR / len(r)) - 1)


# --------------------------------------------------------------------------- #
# 리스크 지표
# --------------------------------------------------------------------------- #
def annualized_volatility(returns) -> float | None:
    r = as_returns(returns)
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def value_at_risk(returns, level: float = 0.95) -> float | None:
    """Historical 1-day VaR, 손실 크기를 양수로 반환한다 (5% 분위수의 부호 반전).

    예: 0.021 이면 "하루 손실이 2.1% 를 넘는 경우가 전체의 5%".
    """
    r = as_returns(returns)
    if not 0 < level < 1 or len(r) < 2:
        return None
    return float(-np.quantile(r, 1 - level))


def conditional_value_at_risk(returns, level: float = 0.95) -> float | None:
    """VaR 를 초과하는 꼬리 손실의 평균 (양수)."""
    r = as_returns(returns)
    if not 0 < level < 1 or len(r) < 2:
        return None
    threshold = np.quantile(r, 1 - level)
    tail = r[r <= threshold]
    if not len(tail):
        return None
    return float(-tail.mean())


def max_drawdown(returns) -> float | None:
    """Maximum drawdown, 양수 (0.12 == -12% 낙폭). investment_ai 와 동일 부호."""
    r = as_returns(returns)
    if not len(r):
        return None
    return float(drawdown_curve(r)[1].max())


# --------------------------------------------------------------------------- #
# 위험조정 성과
# --------------------------------------------------------------------------- #
def sharpe(returns, annual_rf: float = 0.0) -> float | None:
    """투자 AI 담당 구현을 그대로 호출해 전략 간 정의를 일치시킨다."""
    return sharpe_ratio(as_returns(returns), annual_rf)


def sortino(returns, annual_rf: float = 0.0) -> float | None:
    """하방 편차만 위험으로 보는 Sharpe 변형. 하방 관측이 없으면 None."""
    r = as_returns(returns)
    if len(r) < 2:
        return None
    excess = r - daily_risk_free(annual_rf)
    downside = np.minimum(excess, 0.0)
    # 전체 표본 기준 하방 편차 (표준 Sortino 정의).
    deviation = np.sqrt((downside**2).sum() / (len(r) - 1))
    if deviation <= 1e-12:
        return None
    return float(excess.mean() / deviation * np.sqrt(PERIODS_PER_YEAR))


def calmar(returns) -> float | None:
    """CAGR / MDD. MDD 가 0 이면 정의 불가."""
    growth, mdd = cagr(returns), max_drawdown(returns)
    if growth is None or mdd is None or mdd <= 1e-12:
        return None
    return float(growth / mdd)


# --------------------------------------------------------------------------- #
# 상대성과 지표 (벤치마크 필요)
# --------------------------------------------------------------------------- #
def align(returns, benchmark) -> tuple[np.ndarray, np.ndarray]:
    r, b = as_returns(returns), as_returns(benchmark)
    if len(r) != len(b):
        raise ValueError("Strategy and benchmark return lengths differ")
    return r, b


def beta(returns, benchmark) -> float | None:
    r, b = align(returns, benchmark)
    if len(r) < 2:
        return None
    variance = b.var(ddof=1)
    if variance <= 1e-18:
        return None
    return float(np.cov(r, b, ddof=1)[0, 1] / variance)


def alpha(returns, benchmark, annual_rf: float = 0.0) -> float | None:
    """연환산 Jensen's alpha: mean(rp-rf) - beta * mean(rb-rf), * 252."""
    r, b = align(returns, benchmark)
    slope = beta(r, b)
    if slope is None:
        return None
    rf = daily_risk_free(annual_rf)
    daily = (r - rf).mean() - slope * (b - rf).mean()
    return float(daily * PERIODS_PER_YEAR)


def information_ratio(returns, benchmark) -> float | None:
    """초과수익의 평균 / 추적오차, 연환산."""
    r, b = align(returns, benchmark)
    if len(r) < 2:
        return None
    active = r - b
    tracking = active.std(ddof=1)
    if tracking <= 1e-12:
        return None
    return float(active.mean() / tracking * np.sqrt(PERIODS_PER_YEAR))


# --------------------------------------------------------------------------- #
# 묶음 계산
# --------------------------------------------------------------------------- #
def performance_metrics(returns, benchmark=None, annual_rf: float = 0.0) -> dict:
    """12개 지표를 한 번에 계산한다. 벤치마크가 없으면 상대지표는 None."""
    r = as_returns(returns)
    values = {
        "cumulative_return": cumulative_return(r),
        "cagr": cagr(r),
        "annualized_volatility": annualized_volatility(r),
        "var_95": value_at_risk(r, 0.95),
        "cvar_95": conditional_value_at_risk(r, 0.95),
        "max_drawdown": max_drawdown(r),
        "sharpe": sharpe(r, annual_rf),
        "sortino": sortino(r, annual_rf),
        "calmar": calmar(r),
        "alpha": None,
        "beta": None,
        "information_ratio": None,
    }
    if benchmark is not None:
        values["alpha"] = alpha(r, benchmark, annual_rf)
        values["beta"] = beta(r, benchmark)
        values["information_ratio"] = information_ratio(r, benchmark)
    values["days"] = int(len(r))
    return values


def metrics_table(returns_by_strategy: dict, benchmark_key=None, annual_rf=0.0):
    """전략별 수익률 dict -> 성과 비교 테이블 (과제 9번 산출물)."""
    benchmark = (
        returns_by_strategy.get(benchmark_key) if benchmark_key is not None else None
    )
    rows = {
        name: performance_metrics(
            values, None if name == benchmark_key else benchmark, annual_rf
        )
        for name, values in returns_by_strategy.items()
    }
    frame = pd.DataFrame(rows).T
    return frame[["days", *METRIC_ORDER]]
