"""Strategy backtests on one shared accounting engine (과제 3·4·5·6번).

PPO / MVO / 동일가중 / 벤치마크를 모두 investment_ai.PortfolioEnv 로 돌린다.
같은 거래비용·슬리피지·Safe-Guard·체결 지연(t 관측 -> t+1 체결)을 적용해야
전략 간 비교가 공정하기 때문이다. PPO·동일가중·SPY 는 투자 AI 담당의
run_backtest 를 그대로 호출하고, MVO 만 동일한 규약으로 여기서 구동한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.investment_ai.backtest import run_backtest
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.rewards import calculate_reward

from . import mvo

#: 분석에서 쓰는 표준 전략 이름 -> 한국어 라벨.
STRATEGY_LABELS = {
    "ppo": "PPO (강화학습)",
    "mvo": "MVO (평균-분산)",
    "equal_weight": "동일가중",
    "benchmark": "벤치마크 (S&P500 Buy & Hold, Safe-Guard 적용)",
    "benchmark_raw": "S&P500 Buy & Hold (Safe-Guard 없음, 비용·체결지연 포함)",
    "market_index": "S&P500 지수 수익률 (비용·체결지연 없음)",
}
#: 통계 검증에서 "전략"으로 취급하는 3종 (벤치마크는 시장 기준이라 제외).
COMPARED_STRATEGIES = ("ppo", "mvo", "equal_weight")
#: Alpha/Beta 와 시장 국면 분류의 기준이 되는 열 (지수 수익률 그 자체).
MARKET_REFERENCE = "market_index"
#: 매매 규칙(비용·체결지연)을 적용한, 실제로 투자 가능한 SPY Buy & Hold.
INVESTABLE_BENCHMARK = "benchmark_raw"
#: Safe-Guard 를 사실상 끄기 위한 값. PortfolioEnv 는 0 < max_drawdown < 1 만
#: 허용하므로 비활성화 플래그 대신 도달 불가능한 한계값을 쓴다. 거래비용·체결
#: 지연 등 다른 조건은 전부 동일하게 유지된다.
DISABLED_SAFEGUARD = 0.999999


def _cash_tail(
    rows: list[dict], data, config, first_index: int, end: int
) -> list[dict]:
    """Safe-Guard 발동 후 남은 기간을 무이자 현금으로 채운다.

    investment_ai.backtest.run_backtest 의 처리와 동일하다
    (CONTRACT_V3.md "Safe-Guard": 백테스트 잔여 기간은 현금).
    """
    for t in range(first_index, end + 1):
        row = rows[-1].copy()
        row.update(
            date=str(data.dates[t])[:10],
            execution_date=str(data.dates[t])[:10],
            return_start_date=str(data.dates[t - 1])[:10],
            new_weights_return_start_date=str(data.dates[t])[:10],
            new_weights_return_end_date=(
                str(data.dates[t + 1])[:10] if t + 1 < len(data.dates) else ""
            ),
            safeguard_triggered=True,
            decision_date=str(data.dates[t - 1])[:10],
            observation_date=str(data.dates[t - 1])[:10],
            portfolio_return=0.0,
            gross_return=0.0,
            turnover=0.0,
            exit_turnover=0.0,
            cost=0.0,
            exit_cost=0.0,
            cost_amount=0.0,
            rebalanced=False,
            cash_weight=1.0,
            stop_reason="cash_after_stop",
            reward=0.0,
        )
        for asset in data.assets:
            for prefix in (
                "weight",
                "held",
                "previous_weight",
                "current_weight",
                "target_weight",
                "weight_change",
            ):
                row[f"{prefix}_{asset.lower()}"] = 0.0
        row["reward"] = calculate_reward(
            config["reward"],
            0.0,
            drawdown=row["drawdown"],
            mdd=row["max_drawdown"],
            history=rows + [row],
            window=config.get("reward_window", 20),
            mdd_lambda=config.get("mdd_lambda", 0.1),
            annual_rf=config.get("risk_free_annual", 0.0),
        )
        rows.append(row)
    return rows


def run_mvo(
    data,
    config: dict,
    start: int,
    end: int,
    objective: str = "max_sharpe",
    frequency: str = "monthly",
    max_weight: float = mvo.MAX_WEIGHT,
    covariance_window: int = mvo.COVARIANCE_WINDOW,
    strict_window: bool = False,
) -> pd.DataFrame:
    """MVO 전략을 PPO 와 동일한 환경·비용·안전장치로 백테스트한다.

    strict_window=True 면 과거 252 거래일이 모이기 전에는 동일가중을 유지한다
    (과제가 252일 공분산을 엄격히 요구하는 경우).
    """
    calendar = frequency in mvo.CALENDAR_FREQUENCIES
    if not calendar and frequency not in mvo.REBALANCE_DAYS:
        allowed = sorted(mvo.REBALANCE_DAYS) + list(mvo.CALENDAR_FREQUENCIES)
        raise ValueError(f"frequency must be one of {allowed}")
    rebalance_dates = (
        mvo.calendar_rebalance_dates(data.dates, frequency) if calendar else set()
    )
    strategy_config = {
        **config,
        # 달력 기준이면 매일 체결 가능 상태로 두고, 리밸런싱 여부는
        # step_weights(execute=...) 로 직접 지정한다 (execute=False 면 비중이
        # 그대로 흘러가며 거래도 비용도 발생하지 않는다).
        "rebalance_days": 1 if calendar else mvo.REBALANCE_DAYS[frequency],
        "window_id": f"{config.get('window_id', 'standalone')}_mvo",
    }
    env = PortfolioEnv(data, strategy_config, start=start, end=end)
    env.reset(seed=config["seed"])
    first = True
    while True:
        # 결정일 = env.t 종가. 체결은 환경이 t+1 종가에 처리한다.
        weights = mvo.weights_at(
            data,
            env.t,
            objective=objective,
            window=covariance_window,
            max_weight=max_weight,
            annual_rf=config.get("risk_free_annual", 0.0),
            strict_window=strict_window,
        )
        # 첫날은 현금에서 진입해야 하므로 반드시 체결한다.
        execute = True if not calendar else (first or env.t in rebalance_dates)
        first = False
        _, _, terminated, truncated, _ = env.step_weights(weights, execute=execute)
        if terminated or truncated:
            break
    rows = _cash_tail(env.history.copy(), data, strategy_config, env.t + 1, end)
    return pd.DataFrame(rows)


def run_strategy(
    data,
    config: dict,
    start: int,
    end: int,
    strategy: str,
    models=None,
    scaler=None,
    **options,
) -> pd.DataFrame:
    """표준 이름으로 전략 하나를 백테스트한다."""
    if strategy == "ppo":
        return run_backtest(data, config, start, end, models, scaler, "ppo")
    if strategy == "equal_weight":
        return run_backtest(data, config, start, end, None, scaler, "equal_rebalanced")
    if strategy == "benchmark":
        if "SPY" not in data.assets:
            raise ValueError("벤치마크(S&P500)는 자산 목록에 SPY 가 있어야 합니다")
        return run_backtest(data, config, start, end, None, scaler, "spy_buy_hold")
    if strategy == "benchmark_raw":
        # 전략 안전장치를 적용하지 않은 순수 시장 수익률.
        # Safe-Guard 가 걸린 벤치마크는 15% 낙폭 이후 현금이 되어 시장을 더 이상
        # 대표하지 못한다. Alpha/Beta 와 국면 분류는 이 열을 기준으로 한다.
        if "SPY" not in data.assets:
            raise ValueError("시장 기준(S&P500)은 자산 목록에 SPY 가 있어야 합니다")
        return run_backtest(
            data,
            {**config, "max_drawdown": DISABLED_SAFEGUARD},
            start,
            end,
            None,
            scaler,
            "spy_buy_hold",
        )
    if strategy == "mvo":
        return run_mvo(data, config, start, end, **options)
    raise ValueError(f"Unknown strategy: {strategy}")


def run_all(
    data,
    config: dict,
    start: int,
    end: int,
    models=None,
    scaler=None,
    strategies=("ppo", "mvo", "equal_weight", "benchmark", "benchmark_raw"),
    mvo_options: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """과제 7번 비교 대상(PPO/MVO/동일가중/벤치마크) 히스토리를 모두 만든다."""
    histories = {}
    for name in strategies:
        options = mvo_options or {} if name == "mvo" else {}
        histories[name] = run_strategy(
            data, config, start, end, name, models, scaler, **options
        )
    return histories


def market_index_returns(data, dates, ticker: str = "SPY") -> pd.Series:
    """지수 수익률 그 자체 — 거래비용·체결지연·초기 현금 상태가 없는 기준선.

    `benchmark_raw` 는 Safe-Guard 만 끈 Buy & Hold 라서 여전히 첫 진입 수수료와
    t 관측 -> t+1 체결 규칙의 영향을 받는다. "S&P500 을 이겼는가"를 말하려면
    지수 자체와도 비교할 수 있어야 하므로, 같은 날짜에 대응하는 SPY 의 일별
    단순수익률(expm1(log_return))을 그대로 돌려준다.

    Args:
        dates: 비교 대상 수익률 표의 날짜 인덱스 (PortfolioEnv 기록의 체결일).

    Returns:
        같은 날짜로 정렬된 Series. 포트폴리오 회계와 같은 날짜 정렬을 쓰므로
        Alpha/Beta 계산에 바로 쓸 수 있다.
    """
    if ticker not in data.assets:
        raise ValueError(f"지수 기준 자산 '{ticker}' 이 자산 목록에 없습니다")
    column = data.assets.index(ticker)
    lookup = {
        pd.Timestamp(str(value)[:10]): index for index, value in enumerate(data.dates)
    }
    wanted = pd.DatetimeIndex(dates)
    missing = [date for date in wanted if date not in lookup]
    if missing:
        raise ValueError(f"지수 수익률에 없는 날짜가 있습니다: {missing[:3]}")
    values = [float(np.expm1(data.returns[lookup[date], column])) for date in wanted]
    return pd.Series(values, index=wanted, name=MARKET_REFERENCE)


def stitch_histories(frames) -> pd.DataFrame:
    """Walk-forward 구간별 히스토리를 하나의 연속 OOS 경로로 이어붙인다.

    각 window 의 환경은 자본 1.0 으로 다시 시작하므로 portfolio_value 를 그대로
    이으면 구간 경계에서 값이 1.0 으로 되돌아간다. 그래서 일별 수익률만 이어붙인
    뒤 누적 가치·drawdown·MDD 를 전체 구간에서 다시 계산한다.

    주의: 이렇게 계산한 MDD 는 구간별 MDD(walk_forward_metrics.csv)와 다르다.
    구간을 넘는 낙폭까지 반영하기 때문이며, 보고서에는 둘을 구분해 적어야 한다.
    또한 Safe-Guard 는 window 마다 초기화된다 (재학습 때 새 에피소드가 시작되므로).
    """
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        raise ValueError("이어붙일 히스토리가 없습니다")
    joined = pd.concat(frames, ignore_index=True)
    joined = joined.sort_values("date", kind="stable").reset_index(drop=True)
    duplicated = joined.date.duplicated()
    if duplicated.any():
        raise ValueError(
            "테스트 구간이 겹칩니다: "
            f"{sorted(joined.loc[duplicated, 'date'].unique())[:5]} "
            "(walk-forward window 는 서로 겹치지 않아야 합니다)"
        )
    returns = joined.portfolio_return.to_numpy(dtype=float)
    equity = np.cumprod(1 + returns)
    # 고점 후보에 최초 투자원금 1.0 을 포함한다 (metrics.drawdown_curve 와 동일).
    peak = np.maximum(1.0, np.maximum.accumulate(equity))
    joined["portfolio_value"] = equity
    joined["drawdown"] = 1 - equity / peak
    joined["max_drawdown"] = np.maximum.accumulate(joined.drawdown.to_numpy(float))
    return joined


def returns_frame(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """전략별 일별 수익률을 date 인덱스로 정렬해 하나의 표로 만든다.

    join="inner" 이므로 모든 전략이 공통으로 가진 날짜만 남는다. 전략별 날짜 수가
    다르면(구간 설정 실수 등) 조용히 잘리지 않도록 경고 대신 예외를 낸다.
    """
    columns = {
        name: pd.Series(
            history.portfolio_return.to_numpy(dtype=float),
            index=pd.to_datetime(history.date),
            name=name,
        )
        for name, history in histories.items()
    }
    frame = pd.concat(columns.values(), axis=1, join="inner")
    frame.index.name = "date"
    lengths = {name: len(series) for name, series in columns.items()}
    # 공통 날짜가 어느 전략의 날짜 수와도 다르면 조용히 잘렸다는 뜻이다.
    if frame.empty or max(lengths.values()) != len(frame):
        raise ValueError(
            "전략별 날짜가 일치하지 않습니다 "
            f"(전략별 일수 {lengths}, 공통 {len(frame)}일). "
            "모든 전략을 같은 start/end 로 백테스트했는지 확인하세요."
        )
    return frame


def value_frame(returns: pd.DataFrame, initial: float = 1.0) -> pd.DataFrame:
    """누적 자산 가치 (과제 10번 그래프용)."""
    return initial * (1 + returns).cumprod()


def drawdown_frame(returns: pd.DataFrame, initial: float = 1.0) -> pd.DataFrame:
    """전략별 drawdown 시계열 (과제 11번).

    고점에는 **최초 투자원금도 포함**해야 한다. 그러지 않으면 첫 거래일에 손실이
    나도 그날의 종가가 그때까지의 최고값이 되어 낙폭이 0 으로 계산된다
    (metrics.max_drawdown 은 원금을 포함하므로 두 값이 어긋나게 된다).
    """
    values = value_frame(returns, initial)
    return 1 - values / values.cummax().clip(lower=initial)


def weight_frame(history: pd.DataFrame, assets) -> pd.DataFrame:
    """체결 후 비중 시계열 (SHAP·회전율 해석용)."""
    columns = {asset: history[f"current_weight_{asset.lower()}"] for asset in assets}
    frame = pd.DataFrame(columns)
    frame.index = pd.to_datetime(history.date)
    frame.index.name = "date"
    return frame


def turnover_summary(history: pd.DataFrame) -> dict:
    """비용·회전율 요약. 전략 간 비용 공정성을 보고서에 적을 때 쓴다."""
    return {
        "total_turnover": float((history.turnover + history.exit_turnover).sum()),
        "total_cost_initial_capital": float(history.cost_amount.sum()),
        "rebalance_count": int(history.rebalanced.sum()),
        "cash_days": int((history.cash_weight == 1).sum()),
        "safeguard_triggered": bool(history.safeguard_triggered.any()),
    }


def ensemble_weights(env: PortfolioEnv, models, observation) -> np.ndarray:
    """여러 시드 모델의 목표 비중 평균 (run_backtest 와 동일한 앙상블 규칙)."""
    return np.mean(
        [
            env.action_weights(model.predict(observation, deterministic=True)[0])
            for model in models
        ],
        axis=0,
    )
