"""FastAPI adapter shared by standalone and First Class deployments."""

import logging
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from .serving import InvestmentService


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PortfolioState(Contract):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    weights: dict[str, float]
    portfolio_value: float = Field(gt=0)
    peak_value: float = Field(gt=0)
    max_drawdown: float = Field(ge=0, le=1)
    days_since_rebalance: int = Field(ge=0, strict=True)
    stopped: StrictBool = False
    safeguard_triggered: StrictBool = False
    episode_id: str = Field(default="default", min_length=1, max_length=128)
    reset_episode: StrictBool = False


class OptimizeRequest(Contract):
    market: list[dict] | None = Field(default=None, min_length=1, max_length=100000)
    news: list[dict] | None = Field(default=None, max_length=100000)
    state: PortfolioState | None = None


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    ready: bool
    model_version: str | None = None


class OptimizeResponse(Contract):
    schema_version: str
    model_version: str
    data_source: str
    model_trained_on_synthetic: bool
    as_of: str
    status: Literal["cash_safeguard", "hold_until_rebalance", "target_proposal"]
    mode: str
    safeguard_triggered: bool
    episode_id: str
    observation_date: str
    decision_date: str
    execution_date: str | None
    first_return_end_date: str | None
    first_return_end_delay_trading_bars: int
    target_weights: dict[str, float]
    current_weights: dict[str, float]
    cash_weight: float
    current_cash_weight: float
    execution_delay_trading_bars: int
    is_order: Literal[False]


class Asset(Contract):
    ticker: str
    name: str
    weight: float = Field(ge=0, le=1)


class ReturnRow(Contract):
    date: str
    portfolio_log: float
    benchmark_log: float


class PathRow(Contract):
    date: str
    portfolio: float
    benchmark: float
    drawdown: float


class Metrics(Contract):
    days: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float | None
    max_drawdown: float
    total_turnover: float
    total_cost_initial_capital: float
    cash_days: int
    stop_triggered: bool


class BacktestResponse(Contract):
    schema_version: str
    model_version: str
    data_source: str
    as_of: str
    benchmark: str
    periods_per_year: int
    annual_risk_free_rate: float
    metric_convention: str
    assets: list[Asset]
    cash_weight: float
    stopped: bool
    safeguard_triggered: bool
    allocation_kind: str
    returns: list[ReturnRow]
    path: list[PathRow]
    metrics: Metrics
    benchmark_metrics: Metrics
    safeguard_applied_to_benchmarks: bool


def create_app(run_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        try:
            app.state.service = InvestmentService(run_dir)
        except Exception:
            logging.getLogger(__name__).exception(
                "Investment model service is unavailable"
            )
        yield
        app.state.service = None

    app = FastAPI(title="First Class Investment AI", version="1.0", lifespan=lifespan)

    def service():
        value = getattr(app.state, "service", None)
        if value is None:
            raise HTTPException(
                503, "Investment model is not ready; check server configuration"
            )
        return value

    @app.get("/health", response_model=HealthResponse)
    def health():
        value = getattr(app.state, "service", None)
        return HealthResponse(
            model_loaded=value is not None,
            ready=value is not None,
            model_version=value.run_id if value else None,
        )

    @app.get("/ready", response_model=HealthResponse)
    def ready():
        service()
        return health()

    @app.post("/optimize", response_model=OptimizeResponse)
    def optimize(request: OptimizeRequest):
        try:
            return service().optimize(
                request.market,
                request.news,
                request.state.model_dump() if request.state else None,
            )
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/backtest", response_model=BacktestResponse)
    def backtest(
        benchmark: str = "spy_buy_hold",
        observations: int | None = Query(default=None, ge=2),
    ):
        try:
            return service().backtest(benchmark, observations)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
