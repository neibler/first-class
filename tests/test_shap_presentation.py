from copy import deepcopy
import pytest
from frontend.shap_presentation import decision_direction, select_factors
from frontend.shap_explanations import aggregate


@pytest.mark.parametrize(
    "before,target,expected",
    [(0.1, 0.2, "increase"), (0.2, 0.1, "decrease"), (0.1, 0.1 + 1e-10, "hold")],
)
def test_direction_ignores_baseline(before, target, expected):
    assert (
        decision_direction(
            {"previous_weight": before, "target_weight": target, "base_value": 0.99}
        )
        == expected
    )


def test_filter_preserves_values_ranking_and_opposite():
    groups = aggregate(
        ["macd_gld", "macd_spy", "macd_qqq", "macd_smh", "macro_vix"],
        [0.1, 0.3, 0.2, 0.4, -0.5],
        "GLD",
    )
    original = deepcopy(groups)
    positive, negative = select_factors(groups, "increase")
    assert [g["absolute_sum"] for g in positive] == [0.4, 0.3, 0.2]
    assert all(g["direction"] == "increase" for g in positive)
    assert len(negative) == 1
    negative, positive = select_factors(groups, "decrease")
    assert len(negative) == 1 and len(positive) == 4
    neutral, _ = select_factors(groups, "hold")
    assert [g["absolute_sum"] for g in neutral] == [0.5, 0.4, 0.3]
    assert groups == original


def test_missing_execution_does_not_guess():
    assert decision_direction(None) == "unknown"
    assert (
        decision_direction({"previous_weight": float("nan"), "target_weight": 0.1})
        == "unknown"
    )
