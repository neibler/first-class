"""Shared close-t rebalance decisions; normal fills remain at close t+1."""

import numpy as np
import pandas as pd
from .accounting import safeguard_breached

REBALANCE_POLICY = "monthly_risk_drift_v1"
RISK_REBALANCE_THRESHOLD = 0.7
WEIGHT_DRIFT_THRESHOLD = 0.05
MDD_STOP_THRESHOLD = 0.15
ROUNDING_TOLERANCE = 1e-12


def validate_rebalancing(config):
    for key, default in (
        ("risk_rebalance_threshold", RISK_REBALANCE_THRESHOLD),
        ("weight_drift_threshold", WEIGHT_DRIFT_THRESHOLD),
        ("max_drawdown", MDD_STOP_THRESHOLD),
    ):
        value = config.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 < value <= 1
            or (key == "max_drawdown" and value == 1)
        ):
            raise ValueError(f"Invalid {key}")


def is_monthly_rebalance_day(date, previous_date=None):
    """First observed trading close of a month, not every N trading days.

    A prior trading date establishes the month transition (weekends/holidays
    included). Without one, only calendar day 1 is known to be the first day;
    a truncated mid-month input must not manufacture a monthly trigger.
    This is the decision date: normal execution remains next trading close.
    """
    date = pd.Timestamp(date)
    if previous_date is None:
        return date.day == 1
    previous = pd.Timestamp(previous_date)
    if previous >= date:
        raise ValueError("Previous trading date must precede decision date")
    return (date.year, date.month) != (previous.year, previous.month)


def get_rebalance_reasons(
    date,
    previous_date,
    current_weights,
    target_weights,
    risk_scores,
    mdd,
    config,
    stopped=False,
):
    """Return every normal trigger, or the overriding latched cash stop."""
    validate_rebalancing(config)
    if stopped or safeguard_breached(
        mdd, config.get("max_drawdown", MDD_STOP_THRESHOLD)
    ):
        return ["mdd_stop"]
    current, target, risk = [
        np.asarray(v, dtype=float)
        for v in (current_weights, target_weights, risk_scores)
    ]
    if (
        current.ndim != 1
        or not current.size
        or target.shape != current.shape
        or risk.shape != current.shape
        or not np.isfinite([current, target, risk]).all()
        or np.any(current < 0)
        or np.any(target < 0)
        or not np.isclose(target.sum(), 1)
        or not (np.isclose(current.sum(), 0) or np.isclose(current.sum(), 1))
        or np.any((risk < 0) | (risk > 1))
    ):
        raise ValueError("Invalid rebalance weights or risk scores")
    reasons = []
    if is_monthly_rebalance_day(date, previous_date):
        reasons.append("monthly")
    # Tolerance only absorbs binary roundoff, e.g. 0.55 - 0.50.
    if (
        risk.max()
        >= config.get("risk_rebalance_threshold", RISK_REBALANCE_THRESHOLD)
        - ROUNDING_TOLERANCE
    ):
        reasons.append("risk_score")
    if (
        np.abs(target - current).max()
        >= config.get("weight_drift_threshold", WEIGHT_DRIFT_THRESHOLD)
        - ROUNDING_TOLERANCE
    ):
        reasons.append("weight_drift")
    return reasons or ["hold"]
