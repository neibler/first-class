"""Read and validate saved backtests over HTTP; independent of investment v3."""

from datetime import date
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, Field, StrictBool, model_validator

from frontend.api_client import get_api_base_url

# Order and units follow backend/performance/metrics.py. No backend imports.
METRICS = (
    ("cumulative_return", "누적 수익률", "%"),
    ("cagr", "CAGR", "%"),
    ("annualized_volatility", "연환산 변동성", "%"),
    ("var_95", "VaR 95%", "%"),
    ("cvar_95", "CVaR 95%", "%"),
    ("max_drawdown", "MDD", "%"),
    ("sharpe", "Sharpe Ratio", ""),
    ("sortino", "Sortino Ratio", ""),
    ("calmar", "Calmar Ratio", ""),
    ("alpha", "Alpha", "%"),
    ("beta", "Beta", ""),
    ("information_ratio", "Information Ratio", ""),
)
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PerformanceAPIError(Exception):
    """Only fixed, user-readable messages are exposed to the page."""


class Strategy(BaseModel):
    strategy: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metrics: dict[str, Number | None]

    @model_validator(mode="after")
    def check_metrics(self):
        if any(key not in self.metrics for key, _, _ in METRICS):
            raise ValueError("Incomplete metrics")
        return self


class PathValue(BaseModel):
    value: Number


class WindowSettings(BaseModel):
    train_days: int = Field(gt=0, strict=True)
    test_days: int = Field(gt=0, strict=True)
    folds: int = Field(gt=0, strict=True)


class WindowPlan(BaseModel):
    used: WindowSettings
    warnings: list[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    schema_version: Literal["performance-1"]
    start_date: date
    end_date: date
    strategies: list[Strategy]
    path: list[dict[str, Any]]
    metric_labels: dict[str, str] = Field(default_factory=dict)
    evaluation: str | None = None
    window_plan: WindowPlan | None = None
    smoke: StrictBool | None = None

    @model_validator(mode="after")
    def check_path(self):
        if self.start_date > self.end_date:
            raise ValueError("Invalid period")
        names = [item.strategy for item in self.strategies]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate strategies")
        previous = None
        for row in self.path:
            day = date.fromisoformat(row["date"])
            if not self.start_date <= day <= self.end_date:
                raise ValueError("Date outside period")
            if previous is not None and day <= previous:
                raise ValueError("Dates must be unique and increasing")
            for name in names:
                PathValue.model_validate(row[name])
            previous = day
        return self


def get_backtest() -> BacktestResult:
    """One GET; defaults include the daily path and all strategies."""
    try:
        response = httpx.get(
            f"{get_api_base_url()}/performance/backtest",
            timeout=20.0,
            follow_redirects=False,
        )
        if response.status_code == 503:
            raise PerformanceAPIError(
                "백테스트 분석 결과가 아직 준비되지 않았습니다. "
                "저장된 분석 결과가 준비된 뒤 다시 시도해 주세요."
            )
        if response.status_code != 200:
            raise PerformanceAPIError(
                "백테스트 결과를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
            )
        return BacktestResult.model_validate(response.json())
    except httpx.TimeoutException:
        raise PerformanceAPIError(
            "백테스트 API 응답 시간이 초과되었습니다. 다시 시도해 주세요."
        ) from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise PerformanceAPIError(
            "백테스트 API에 연결할 수 없습니다. 서버 연결 상태를 확인해 주세요."
        ) from None
    except (ValueError, TypeError, KeyError):
        raise PerformanceAPIError(
            "백테스트 응답 형식이 올바르지 않습니다. 관리자에게 확인을 요청해 주세요."
        ) from None
