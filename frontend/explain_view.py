"""Shape server SHAP values for the existing charts; never recompute SHAP."""

from frontend.explain_client import AssetExplanation, Decision, Overview


def asset_label(result: AssetExplanation) -> str:
    if result.name and result.name != result.ticker:
        return f"{result.ticker} · {result.name}"
    return result.ticker


def summary_records(result: AssetExplanation) -> list[dict]:
    """Every saved decision of the asset, one point per feature (Summary Plot)."""
    records = []
    for i, decision in enumerate(result.decisions):
        for j, feature in enumerate(result.features):
            records.append(
                dict(
                    feature=feature,
                    date=decision.date,
                    shap=decision.shap_values[j],
                    value=decision.feature_values[j],
                    # Stable vertical jitter separates overlapping observations.
                    jitter=((i * 7 + j * 11) % 19 - 9) / 24,
                )
            )
    return records


def chart_decision(decision: Decision) -> dict:
    """The chart helpers take the same keys as the former sample fixture."""
    return decision.model_dump()


def detail_rows(features: list[str], decision: Decision) -> list[dict]:
    raw = decision.feature_values_raw or [None] * len(features)
    return [
        {
            "피처": feature,
            "모델 입력 값": value_raw,
            "정규화 값 (0–1)": value,
            "SHAP 값": shap,
            "비중 기여 (%p)": shap * 100,
        }
        for feature, value_raw, value, shap in zip(
            features, raw, decision.feature_values, decision.shap_values
        )
    ]


def force_summary(decision: Decision) -> str:
    positive = sum(max(v, 0) for v in decision.shap_values)
    negative = sum(min(v, 0) for v in decision.shap_values)
    return (
        f"기준 {decision.base_value:.2%} + 증가 {positive * 100:.2f}%p "
        f"− 감소 {abs(negative) * 100:.2f}%p = "
        f"**결정 비중 {decision.final_value:.2%}**"
    )


def importance_rows(overview: Overview) -> list[dict]:
    return [
        {"피처": item.feature, "평균 |SHAP| (%p)": item.mean_abs_shap * 100}
        for item in sorted(
            overview.feature_importance or [],
            key=lambda item: item.mean_abs_shap,
            reverse=True,
        )
    ]
