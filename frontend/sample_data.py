"""Versioned UI fixtures, not model inference or an API-error fallback.

Replace load_portfolio/load_shap with HTTP adapters when the API contract exists.
The view and chart modules need not know where the response originated.
"""

import json
import math
from pathlib import Path
from statistics import mean, stdev

FIXTURES = Path(__file__).parent / "fixtures"


class SampleDataError(ValueError):
    """A malformed UI sample must be visible rather than silently repaired."""


def load_portfolio():
    data = json.loads((FIXTURES / "portfolio.json").read_text(encoding="utf-8"))
    if not data or not data.get("assets"):
        return None
    if len(data.get("returns") or []) < 3:
        raise SampleDataError(
            "성과 계산에 기준일과 수익률 관측치 2개 이상이 필요합니다."
        )
    weights = [asset["weight"] for asset in data["assets"]]
    if not all(math.isfinite(w) and w >= 0 for w in weights) or not math.isclose(
        sum(weights), 1.0, abs_tol=1e-8
    ):
        raise SampleDataError("자산 비중 합계가 100%여야 합니다.")
    return data


def load_shap():
    data = json.loads((FIXTURES / "shap.json").read_text(encoding="utf-8"))
    if not data or not data.get("assets") or not data.get("features"):
        return None
    count = len(data["features"])
    for asset in data["assets"].values():
        if not asset.get("decisions") or not asset.get("observations"):
            raise SampleDataError("자산의 결정 또는 관측치가 없습니다.")
        for record in asset["observations"] + asset["decisions"]:
            values, phis = record["feature_values"], record["shap_values"]
            if len(values) != count or len(phis) != count:
                raise SampleDataError("피처 수와 해석 값 수가 다릅니다.")
            if not all(math.isfinite(x) for x in values + phis):
                raise SampleDataError("유한한 해석 값이 필요합니다.")
        for decision in asset["decisions"]:
            if not math.isclose(
                decision["base_value"] + sum(decision["shap_values"]),
                decision["final_value"],
                abs_tol=1e-8,
            ):
                raise SampleDataError("기준값과 기여도 합계가 최종값과 다릅니다.")
    return data


def performance(rows, periods_per_year=252, annual_risk_free_rate=0.02):
    """Log-return Sharpe; initial row is the period's baseline (not a return)."""
    if len(rows) < 3:
        raise SampleDataError(
            "성과 계산에 기준일과 수익률 관측치 2개 이상이 필요합니다."
        )
    if periods_per_year <= 0 or annual_risk_free_rate <= -1:
        raise SampleDataError("연율화 또는 무위험 수익률 가정이 올바르지 않습니다.")
    path = []
    p_log = b_log = 0.0
    peak = 1.0
    daily = []
    for i, row in enumerate(rows):
        if i:
            p, b = row["portfolio_log"], row["benchmark_log"]
            if not math.isfinite(p) or not math.isfinite(b):
                raise SampleDataError("수익률에 유한하지 않은 값이 있습니다.")
            p_log += p
            b_log += b
            daily.append(p)
        wealth = math.exp(p_log)
        peak = max(peak, wealth)
        path.append(
            dict(
                date=row["date"],
                portfolio=math.expm1(p_log),
                benchmark=math.expm1(b_log),
                drawdown=wealth / peak - 1,
            )
        )
    sigma = stdev(daily)
    rf_daily = math.log1p(annual_risk_free_rate) / periods_per_year
    sharpe = (
        (mean(daily) - rf_daily) / sigma * math.sqrt(periods_per_year)
        if sigma > 1e-12
        else None
    )
    return dict(
        path=path,
        cumulative=path[-1]["portfolio"],
        benchmark=path[-1]["benchmark"],
        sharpe=sharpe,
        mdd=min(point["drawdown"] for point in path),
        current_drawdown=path[-1]["drawdown"],
        observations=len(daily),
    )


def summary_records(data, ticker, decision_date):
    observations = [
        row
        for row in data["assets"][ticker]["observations"]
        if row["date"] <= decision_date
    ][-60:]
    records = []
    for i, row in enumerate(observations):
        for j, feature in enumerate(data["features"]):
            records.append(
                dict(
                    feature=feature,
                    date=row["date"],
                    shap=row["shap_values"][j],
                    value=row["feature_values"][j],
                    # Stable vertical jitter separates overlapping observations.
                    jitter=((i * 7 + j * 11) % 19 - 9) / 24,
                )
            )
    return records
