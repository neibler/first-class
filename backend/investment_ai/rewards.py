from .accounting import sharpe_ratio


def canonical_reward(name):
    return {"portfolio_return": "return", "rolling_sharpe": "sharpe"}.get(name, name)


def calculate_reward(
    name,
    portfolio_return,
    *,
    drawdown,
    history,
    window=20,
    mdd_lambda=0.1,
    annual_rf=0.0,
    mdd=None,
    executed_turnover=0.0,
    turnover_penalty_lambda=0.0,
):
    penalty = turnover_penalty_lambda * executed_turnover
    name = canonical_reward(name)
    if name == "return":
        return float(portfolio_return - penalty)
    if name == "return_mdd":
        cumulative = mdd if mdd is not None else max(row["drawdown"] for row in history)
        return float(portfolio_return - mdd_lambda * cumulative - penalty)
    if name == "sharpe":
        value = sharpe_ratio(
            [row["portfolio_return"] for row in history[-window:]],
            annual_rf,
            minimum=window,
        )
        return (0.0 if value is None else value) - penalty
    raise ValueError(f"Unsupported reward: {name}")
