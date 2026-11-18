"""Low-churn execution baseline; no model/observation changes."""

import numpy as np

POLICY = "executed_drift_capped_hysteresis_v1"
EPS = 1e-12


def enabled(config):
    return config.get("execution_policy") == POLICY


def policy_for(config):
    from .rebalancing import REBALANCE_POLICY

    policy = config.get("execution_policy", REBALANCE_POLICY)
    if policy not in (REBALANCE_POLICY, POLICY):
        raise ValueError("Unsupported execution_policy")
    return policy


def validate(config):
    policy_for(config)
    for key, default in (
        ("portfolio_distance_threshold", 0.10),
        ("max_rebalance_turnover", 0.15),
        ("buy_threshold", 0.05),
        ("sell_threshold", 0.03),
        ("turnover_penalty_lambda", 0.003),
    ):
        v = config.get(key, default)
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not np.isfinite(v)
            or not 0 <= v <= 1
        ):
            raise ValueError(f"Invalid {key}")
        if key in ("portfolio_distance_threshold", "max_rebalance_turnover") and v == 0:
            raise ValueError(f"Invalid {key}")
    if config.get("sell_threshold", 0.03) > config.get("buy_threshold", 0.05):
        raise ValueError("sell_threshold must not exceed buy_threshold")


def weights(value):
    a = np.asarray(value, dtype=float)
    if (
        a.ndim != 1
        or not a.size
        or not np.isfinite(a).all()
        or np.any(a < 0)
        or a.sum() > 1 + EPS
    ):
        raise ValueError("Invalid actual weights")
    return a.copy()


def reasons(date, previous_date, current, last, risk, mdd, config, stopped=False):
    from .accounting import safeguard_breached
    from .rebalancing import is_monthly_rebalance_day, portfolio_weight_distance

    if stopped or safeguard_breached(mdd, config["max_drawdown"]):
        return ["mdd_stop"]
    current, last = weights(current), weights(last)
    risk = np.asarray(risk, dtype=float)
    if (
        current.shape != last.shape
        or risk.shape != current.shape
        or not np.isfinite(risk).all()
        or np.any((risk < 0) | (risk > 1))
    ):
        raise ValueError("Invalid rebalance state")
    result = []
    if is_monthly_rebalance_day(date, previous_date):
        result.append("monthly")
    if risk.max() >= config.get("risk_rebalance_threshold", 0.7) - EPS:
        result.append("risk_score")
    if (
        portfolio_weight_distance(current, last)
        >= config.get("portfolio_distance_threshold", 0.10) - EPS
    ):
        result.append("portfolio_weight_distance")
    return result or ["hold"]


def executable_target(raw, current, config):
    """Retain eligible incumbents; admit new assets only into available slots.

    If proportional partial exits still occupy slots, defer new entries. There
    is no order queue: the next trigger recalculates everything from scratch.
    """
    from .rebalancing import portfolio_weight_distance

    raw, current = weights(raw), weights(current)
    if raw.shape != current.shape or not np.isclose(raw.sum(), 1):
        raise ValueError("Invalid raw target")
    limit = config.get("max_holdings", 7)
    if np.count_nonzero(current > EPS) > limit:
        raise ValueError("Existing holdings exceed max_holdings")
    held = current > EPS
    keep = held & (raw >= config.get("sell_threshold", 0.03))
    new = np.flatnonzero(~held & (raw >= config.get("buy_threshold", 0.05)))
    new = new[np.argsort(-raw[new], kind="stable")]
    keep[new[: max(0, limit - int(keep.sum()))]] = True
    desired = np.where(keep, raw, 0.0)
    if desired.sum() > 0:
        desired /= desired.sum()
    desired_turnover = portfolio_weight_distance(current, desired)
    cap = config.get("max_rebalance_turnover", 0.15)

    def adjust(target):
        distance = portfolio_weight_distance(current, target)
        alpha = min(1.0, cap / distance) if distance > EPS else 0.0
        return current + alpha * (target - current)

    adjusted = adjust(desired)
    if np.count_nonzero(adjusted > EPS) > limit:
        # An exit cannot free a slot until fully filled. Keep residual cash
        # rather than opening an eighth asset or exceeding the turnover cap.
        deferred = desired.copy()
        deferred[~held] = 0.0
        adjusted = adjust(deferred)
    adjusted[np.abs(adjusted) < EPS] = 0.0
    return desired, adjusted, desired_turnover


def execute(current, target, rate):
    """Self-financing fill: solve fees on actual bought/sold notional.

    Returned target is post-fee weight; cash funds fees. q is traded ETF
    notional / pre-trade equity, not the one-way weight-distance metric.
    """
    current, target = weights(current), weights(target)
    if np.array_equal(current, target):
        return 0.0, 0.0
    lo, hi = 0.0, 2.0
    for _ in range(70):
        q = (lo + hi) / 2
        required = float(np.abs((1 - rate * q) * target - current).sum())
        if q < required:
            lo = q
        else:
            hi = q
    q = (lo + hi) / 2
    return q, rate * q
