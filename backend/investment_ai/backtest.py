"""Same accounting and full evaluation horizon for agents and benchmarks."""

import numpy as np
import pandas as pd
from .portfolio_env import PortfolioEnv
from .accounting import equity_curve, sharpe_ratio
from .records import save_asset_history
from .rewards import calculate_reward


def performance(frame, annual_rf=0.0):
    r = frame.portfolio_return.to_numpy(dtype=float)
    equity = equity_curve(r)
    dd = 1 - equity / np.maximum.accumulate(equity)
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    return {
        "days": len(r),
        "total_return": float(equity[-1] - 1),
        "annualized_return": float(equity[-1] ** (252 / len(r)) - 1),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": sharpe_ratio(r, annual_rf),
        "max_drawdown": float(dd.max()),
        "total_turnover": float((frame.turnover + frame.exit_turnover).sum()),
        "total_cost_initial_capital": float(frame.cost_amount.sum()),
        "cash_days": int((frame.cash_weight == 1).sum()),
        "stop_triggered": bool((frame.stop_reason == "max_drawdown").any()),
    }


def run_backtest(data, config, start, end, models=None, scaler=None, strategy="ppo"):
    env = PortfolioEnv(data, config, start=start, end=end, scaler=scaler)
    obs, _ = env.reset(seed=config["seed"])
    initial_target = np.full(len(data.assets), 1 / len(data.assets))
    if strategy == "spy_buy_hold":
        if "SPY" not in data.assets:
            raise ValueError("SPY benchmark requires SPY in universe")
        initial_target[:] = 0
        initial_target[data.assets.index("SPY")] = 1
    while True:
        if strategy == "ppo":
            if not models:
                raise ValueError("PPO models required")
            weights = np.mean(
                [
                    env.action_weights(model.predict(obs, deterministic=True)[0])
                    for model in models
                ],
                axis=0,
            )
            obs, _, terminated, truncated, _ = env.step_weights(weights)
        elif strategy in ("equal_rebalanced", "equal_buy_hold", "spy_buy_hold"):
            weights = initial_target
            obs, _, terminated, truncated, _ = env.step_weights(
                weights, execute=not (strategy.endswith("buy_hold") and env.history)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        if terminated or truncated:
            break
    rows = env.history.copy()
    # Safeguard liquidates once and keeps cash (0 interest) for the remaining horizon.
    for t in range(env.t + 1, end + 1):
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
            observation_date=str(data.dates[t - 1 - env.lag])[:10],
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
            for prefix in [
                "weight",
                "held",
                "previous_weight",
                "current_weight",
                "target_weight",
                "weight_change",
            ]:
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
    return pd.DataFrame(rows)


def compare(data, config, start, end, models, scaler, output):
    output.mkdir(parents=True, exist_ok=True)
    metrics, histories = {}, {}
    strategies = ["ppo", "equal_rebalanced", "equal_buy_hold"]
    if "SPY" in data.assets:
        strategies.append("spy_buy_hold")
    for strategy in strategies:
        history = run_backtest(data, config, start, end, models, scaler, strategy)
        history.to_csv(output / f"{strategy}_history.csv", index=False)
        save_asset_history(
            history, data.assets, output / f"{strategy}_asset_history.csv"
        )
        histories[strategy] = history
        metrics[strategy] = performance(history, config["risk_free_annual"])
    pd.DataFrame(metrics).T.to_csv(output / "comparison.csv", index_label="strategy")
    return metrics, histories
