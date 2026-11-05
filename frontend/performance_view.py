"""Format validated backtest values for the page, without recalculating metrics."""

from collections import Counter

from frontend.performance_client import METRICS


def strategy_names(result):
    counts = Counter(item.label for item in result.strategies)
    return {
        item.strategy: (
            item.label if counts[item.label] == 1 else f"{item.label} ({item.strategy})"
        )
        for item in result.strategies
    }


def chart_rows(result):
    names = strategy_names(result)
    return [
        {
            "date": row["date"],
            "strategy": strategy,
            "label": label,
            "cumulative_return": row[strategy]["value"] - 1,
        }
        for row in result.path
        for strategy, label in names.items()
    ]


def metric_rows(result):
    names = strategy_names(result)
    rows = []
    for key, fallback, unit in METRICS:
        label = result.metric_labels.get(key, fallback)
        row = {"성과지표": f"{label} (%)" if unit else label}
        for strategy in result.strategies:
            number = strategy.metrics[key]
            row[names[strategy.strategy]] = (
                "—" if number is None else f"{number:.2%}" if unit else f"{number:.3f}"
            )
        rows.append(row)
    return rows


def evaluation_text(result):
    if result.evaluation == "single_holdout":
        return "단일 홀드아웃"
    if result.evaluation == "walk_forward_stitched_out_of_sample":
        if result.window_plan and result.window_plan.used.folds == 1:
            return "단일 홀드아웃 (Walk-Forward 설정의 실제 평가 구간 1개)"
        return "Walk-Forward (표본 외 평가 구간 연결)"
    return "평가 방식 확인 필요" if result.evaluation else "평가 방식 미제공"
