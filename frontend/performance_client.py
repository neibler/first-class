"""Compare the active model with its benchmarks through the investment /backtest API.

The investment API returns PPO plus one benchmark per request, so the page asks once
per benchmark and checks that every answer describes the same model and period.
"""

from dataclasses import dataclass
from datetime import date
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from frontend.api_client import get_api_base_url

# Benchmarks the investment API keeps for the active model, in display order.
BENCHMARKS = (
    ("spy_buy_hold", "S&P500 (SPY 매수 후 보유)"),
    ("equal_rebalanced", "동일가중 (리밸런싱)"),
    ("equal_buy_hold", "동일가중 (매수 후 보유)"),
)
# Server metrics shown in this order; "%" values are fractions from the server.
METRICS = (
    ("total_return", "누적 수익률", "%"),
    ("annualized_return", "연환산 수익률", "%"),
    ("annualized_volatility", "연환산 변동성", "%"),
    ("sharpe", "Sharpe Ratio", ""),
    ("max_drawdown", "MDD (최대 낙폭)", "%"),
    ("total_turnover", "총 회전율", "x"),
    ("total_cost_initial_capital", "거래비용 (초기자본 대비)", "%"),
    ("cash_days", "현금 보유일", "일"),
    ("days", "평가 거래일", "일"),
)
Number = Annotated[float, Field(allow_inf_nan=False)]
TIMEOUT = httpx.Timeout(20.0, connect=3.0)


class Contract(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Metrics(Contract):
    days: int = Field(ge=0)
    total_return: Number
    annualized_return: Number
    annualized_volatility: Number
    sharpe: Number | None
    max_drawdown: Number
    total_turnover: Number
    total_cost_initial_capital: Number
    cash_days: int = Field(ge=0)
    stop_triggered: bool


class PathRow(Contract):
    date: date
    portfolio: Number
    benchmark: Number


class Backtest(Contract):
    schema_version: str
    model_version: str = Field(min_length=1)
    data_source: str
    as_of: date
    benchmark: str
    metric_convention: str
    stopped: bool
    safeguard_triggered: bool
    path: list[PathRow] = Field(min_length=1)
    metrics: Metrics
    benchmark_metrics: Metrics


@dataclass
class Strategy:
    strategy: str
    label: str
    metrics: Metrics
    path: list[tuple[date, float]]


@dataclass
class Comparison:
    model_version: str
    data_source: str
    metric_convention: str
    as_of: date
    base_date: date
    start_date: date
    end_date: date
    stopped: bool
    safeguard_triggered: bool
    strategies: list[Strategy]
    missing: list[str]


class PerformanceAPIError(Exception):
    """Only fixed, user-readable messages are exposed to the page."""


def _get(benchmark):
    try:
        response = httpx.get(
            f"{get_api_base_url()}/backtest",
            params={"benchmark": benchmark},
            timeout=TIMEOUT,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        raise PerformanceAPIError(
            "백테스트 API 응답 시간이 초과되었습니다. 다시 시도해 주세요."
        ) from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise PerformanceAPIError(
            "백테스트 API에 연결할 수 없습니다. 서버 연결 상태를 확인해 주세요."
        ) from None
    if response.status_code == 503:
        raise PerformanceAPIError(
            "백테스트 분석 결과가 아직 준비되지 않았습니다. "
            "투자 모델이 준비된 뒤 다시 시도해 주세요."
        )
    if response.status_code == 422:
        return None  # This model has no saved history for that benchmark.
    if response.status_code != 200:
        raise PerformanceAPIError(
            "백테스트 결과를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
        )
    try:
        return response.json()
    except ValueError:
        raise _invalid() from None


def _invalid():
    return PerformanceAPIError(
        "백테스트 응답 형식이 올바르지 않습니다. 관리자에게 확인을 요청해 주세요."
    )


def build_comparison(responses: dict) -> Comparison:
    """Combine one validated /backtest answer per benchmark into one comparison."""
    answers = {
        key: Backtest.model_validate(body)
        for key, body in responses.items()
        if body is not None
    }
    if not answers:
        raise ValueError("No benchmark answered")
    first = next(iter(answers.values()))
    dates = [row.date for row in first.path]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("Path dates must increase")
    for key, item in answers.items():
        if (
            item.schema_version != "3.0"
            or item.benchmark != key
            or item.model_version != first.model_version
            or item.as_of != first.as_of
            or [row.date for row in item.path] != dates
            or item.metrics != first.metrics
            or [row.portfolio for row in item.path]
            != [row.portfolio for row in first.path]
        ):
            raise ValueError("Answers describe different models or periods")
    strategies = [
        Strategy(
            "ppo",
            f"PPO ({first.model_version})",
            first.metrics,
            [(row.date, row.portfolio) for row in first.path],
        )
    ]
    for key, label in BENCHMARKS:
        if key in answers:
            item = answers[key]
            strategies.append(
                Strategy(
                    key,
                    label,
                    item.benchmark_metrics,
                    [(row.date, row.benchmark) for row in item.path],
                )
            )
    return Comparison(
        model_version=first.model_version,
        data_source=first.data_source,
        metric_convention=first.metric_convention,
        as_of=first.as_of,
        # The first path row is the 0% starting point, not an evaluation day.
        base_date=dates[0],
        start_date=dates[1] if len(dates) > 1 else dates[0],
        end_date=dates[-1],
        stopped=first.stopped,
        safeguard_triggered=first.safeguard_triggered,
        strategies=strategies,
        missing=[label for key, label in BENCHMARKS if key not in answers],
    )


def get_backtest() -> Comparison:
    """One GET per benchmark per page run; no cache, retries or fallback data."""
    responses = {key: _get(key) for key, _ in BENCHMARKS}
    try:
        return build_comparison(responses)
    except (ValidationError, ValueError, TypeError):
        raise _invalid() from None
