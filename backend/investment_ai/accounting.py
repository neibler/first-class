"""Shared net simple-return metrics for rewards and reporting."""

import numpy as np


def equity_curve(returns):
    return np.r_[1.0, np.cumprod(1 + np.asarray(returns, dtype=float))]


def sharpe_ratio(returns, annual_rf=0.0, minimum=2):
    values = np.asarray(returns, dtype=float)
    if len(values) < minimum:
        return None
    std = values.std(ddof=1)
    if std <= 1e-12:
        return None
    excess = values - np.expm1(np.log1p(annual_rf) / 252)
    return float(excess.mean() / std * np.sqrt(252))


def safeguard_breached(mdd, limit):
    """Ignore roundoff at the strict threshold (e.g. 1 - 0.85)."""
    return bool(mdd > limit + 1e-12)
