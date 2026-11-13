"""Format server backtest values for the page, without recalculating metrics."""

from frontend.performance_client import METRICS


def chart_rows(result):
    return [
        {
            "date": day.isoformat(),
            "strategy": item.strategy,
            "label": item.label,
            "cumulative_return": value,
        }
        for item in result.strategies
        for day, value in item.path
    ]


def _format(value, unit):
    if value is None:
        return "—"
    if unit == "%":
        return f"{value:.2%}"
    if unit == "x":
        return f"{value:.2f}배"
    if unit == "일":
        return f"{value}일"
    return f"{value:.3f}"


def metric_rows(result):
    rows = []
    for key, label, unit in METRICS:
        row = {"성과지표": label}
        for item in result.strategies:
            row[item.label] = _format(getattr(item.metrics, key), unit)
        rows.append(row)
    return rows


def convention_text(result):
    if result.metric_convention == (
        "simple_daily_returns_annualized_sharpe_positive_mdd"
    ):
        return "일별 단순수익률 기준 · Sharpe는 연환산 · MDD는 손실 크기(양수)로 표시합니다."
    return f"지표 계산 기준: {result.metric_convention}"
