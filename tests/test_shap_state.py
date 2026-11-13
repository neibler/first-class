import math
from copy import deepcopy

import pytest
from frontend.shap_explanations import aggregate, group_shap_features, top_factors
from frontend.shap_state import observed_values, rsi_summary


def explain(values, contributions=None, previous=None, days=5, selected="TLT"):
    features = list(values)
    row = {
        "date": "2025-01-10",
        "feature_values_raw": [-9.0] * len(features),
        "market_context": {
            "source": "verified_market_data",
            "as_of": "2025-01-10",
            "values": values,
            "previous_values": previous or {},
            "comparison_date": "2025-01-03",
            "comparison_trading_days": days,
        },
    }
    before = deepcopy(row)
    groups = group_shap_features(
        features, contributions or [0.1] * len(features), selected, row
    )
    assert row == before
    return groups


@pytest.mark.parametrize("daily,word", [(0.01, "상승"), (-0.01, "하락")])
def test_five_day_return_uses_compounded_raw_logs(daily, word):
    values = {f"log_return_lag_{i}_gld": math.log1p(daily) for i in range(5)}
    text = explain(values)[0]["actual_state_summary"]
    assert "최근 5거래일" in text
    assert f"{abs((1+daily)**5-1)*100:.2f}% {word}" in text


@pytest.mark.parametrize(
    "value,word",
    [
        (29, "과매도"),
        (30, "비교적 낮은"),
        (44.9, "비교적 낮은"),
        (45, "중립"),
        (55, "중립"),
        (55.1, "비교적 높은"),
        (70, "비교적 높은"),
        (71, "과매수"),
    ],
)
def test_rsi_boundaries(value, word):
    assert word in rsi_summary(value)
    assert word in explain({"rsi_div_100_gld": value / 100})[0]["actual_state_summary"]


@pytest.mark.parametrize(
    "macd,signal,word", [(2, 1, "상승 모멘텀"), (1, 2, "하락 모멘텀")]
)
def test_macd_signal_relation(macd, signal, word):
    assert (
        word in explain({"macd_gld": macd, "macd_signal_gld": signal})[0]["explanation"]
    )


@pytest.mark.parametrize("macd,word", [(2, "0보다 높아"), (-2, "0보다 낮아")])
def test_macd_without_signal_only_uses_zero(macd, word):
    text = explain({"macd_gld": macd})[0]["explanation"]
    assert word in text
    assert "신호선" not in text


@pytest.mark.parametrize(
    "contribution,direction,word",
    [(0.1, "increase", "높이는"), (-0.1, "decrease", "낮추는")],
)
def test_direction_describes_selected_output_not_input_asset(
    contribution, direction, word
):
    group = explain({"rsi_gld": 31}, [contribution])[0]
    assert group["asset"] == "GLD"
    assert group["shap_direction"] == direction
    assert f"TLT 비중을 {word}" in group["explanation"]
    assert group["explanation"].startswith(group["actual_state_summary"])
    assert "때문" not in group["explanation"]


def test_news_actual_score_without_invented_risk_level():
    text = explain({"risk_score_smh": 0.72}, [-0.1])[0]["explanation"]
    assert "SMH 관련 뉴스 위험 점수는 0.72" in text
    assert "높은 편" not in text


def test_macro_without_prior_does_not_invent_direction_or_level():
    for feature in ["macro_vix", "macro_ust_30y", "macro_wti"]:
        text = explain({feature: 4.5})[0]["actual_state_summary"]
        assert "4.5" in text
        assert "상승" not in text and "하락" not in text and "높은" not in text
        assert "확인되지" in text


def test_macro_with_prior_and_distinct_mixed_rate_directions():
    groups = explain(
        {"macro_ust_10y": 4.5, "macro_ust_30y": 4.6},
        previous={"macro_ust_10y": 4.3, "macro_ust_30y": 4.9},
    )
    assert len(groups) == 1
    text = groups[0]["actual_state_summary"]
    assert "10년물" in text and "30년물" in text
    assert "5거래일 전보다 상승" in text and "5거래일 전보다 하락" in text


def test_scaled_inputs_never_become_market_returns_macd_or_macro():
    features = [
        "log_return_lag_0_gld",
        "macd_gld",
        "macro_vix",
        "rsi_div_100_gld",
        "risk_score_gld",
    ]
    row = {"date": "2025-01-10", "feature_values_raw": [2, -2, 5, 0.31, 0.72]}
    values, _, _ = observed_values(features, row)
    assert values == {"rsi_div_100_gld": 0.31, "risk_score_gld": 0.72}
    groups = group_shap_features(features, [0.1] * 5, "GLD", row)
    asset = next(g for g in groups if g["asset"] == "GLD")
    assert "RSI는 31" in asset["actual_state_summary"]
    assert (
        "상승" not in asset["actual_state_summary"]
        and "MACD" not in asset["actual_state_summary"]
    )


def test_future_or_mismatched_context_not_used():
    row = {
        "date": "2025-01-10",
        "feature_values_raw": [9],
        "market_context": {
            "source": "verified_market_data",
            "as_of": "2025-01-10",
            "values": {"macro_vix": 20},
            "previous_values": {"macro_vix": 30},
            "comparison_date": "2025-01-11",
        },
    }
    assert observed_values(["macro_vix"], row) == ({"macro_vix": 20}, {}, None)
    row["market_context"]["as_of"] = "2025-01-11"
    assert observed_values(["macro_vix"], row) == ({}, {}, None)


def test_raw_and_ranking_preserved_and_top_three():
    features = [f"rsi_{a}" for a in ["gld", "spy", "qqq", "smh", "efa"]] + ["macd_gld"]
    contributions = [0.3, 0.2, 0.1, 0.05, -0.1, -0.25]
    row = {"feature_values_raw": [40] * 6}
    groups = group_shap_features(features, contributions, "TLT", row)
    original = aggregate(features, contributions, "TLT")
    for new, old in zip(groups, original):
        for key, value in old.items():
            assert new[key] == value
    assert len(top_factors(groups, "increase")) == 3
    assert sum(g["key"] == "asset_GLD" for g in groups) == 1
    assert sorted(f for g in groups for f in g["raw_features"]) == sorted(features)
