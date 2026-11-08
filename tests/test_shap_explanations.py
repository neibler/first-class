import numpy as np
from frontend.shap_explanations import aggregate, classify, top_factors, MACRO_MAPPING


def test_lags_aggregate_without_losing_cancellation():
    rows = aggregate(
        ["log_return_lag_1_gld", "log_return_lag_5_gld", "macro_ust_30y"],
        [0.08, -0.07, 0.02],
        "GLD",
    )
    assert len(rows) == 2
    assert rows[0]["label"] == "금(GLD) 관련 신호"
    assert rows[0]["direction"] == "increase" and rows[0]["mixed"]
    np.testing.assert_allclose(rows[0]["signed_sum"], 0.01)
    np.testing.assert_allclose(rows[0]["absolute_sum"], 0.15)
    assert top_factors(rows, "increase")[0] == rows[0]
    np.testing.assert_allclose(sum(r["signed_sum"] for r in rows), 0.03)
    np.testing.assert_allclose(sum(r["absolute_sum"] for r in rows), 0.17)


def test_neutral_is_not_mislabelled_and_unknown_is_retained():
    rows = aggregate(
        ["log_return_lag_1_gld", "log_return_lag_2_gld", "new_raw_input"],
        [0.1, -0.1, 0.01],
        "GLD",
    )
    assert rows[0]["direction"] == "balanced"
    assert len(top_factors(rows, "increase")) == 1
    assert rows[1]["label"] == "기타 모델 입력 정보"
    assert "new_raw_input" not in rows[1]["label"]


def test_context_and_all_macro_labels():
    assert classify("macd_gld", "GLD")[0] == "자산 자체 가격/추세"
    assert classify("macd_gld", "TLT")[0] == "다른 자산 움직임"
    assert classify("log_return_lag_8_qqq", "TLT")[2] == "기술주(QQQ)의 최근 움직임"
    assert classify("risk_score_tlt", "TLT")[0] == "뉴스 위험"
    assert len(MACRO_MAPPING) == 13
    for key in MACRO_MAPPING:
        assert classify("macro_" + key, "TLT")[0] != "기타 정보"


def test_top_three_and_shared_relative_scale():
    features = ["macd_" + a for a in ["spy", "qqq", "smh", "gld", "tlt", "ief", "shy"]]
    rows = aggregate(features, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, -0.12], "TLT")
    assert len(top_factors(rows, "increase")) == 3
    assert top_factors(rows, "decrease")[0]["relative_strength"] == 1
    assert top_factors(rows, "increase")[0]["relative_strength"] == 0.5


def test_asset_technicals_and_returns_have_one_net_direction():
    rows = aggregate(
        [
            "macd_gld",
            "rsi_div_100_gld",
            "log_return_lag_5_gld",
            "risk_score_gld",
            "risk_score_tlt",
        ],
        [0.03, 0.01, -0.08, 0.02, -0.01],
        "SPY",
    )
    assert len(rows) == 2
    asset = next(r for r in rows if r["key"] == "asset_GLD")
    assert asset["direction"] == "decrease"
    np.testing.assert_allclose(asset["signed_sum"], -0.04)
    np.testing.assert_allclose(asset["absolute_sum"], 0.12)
    assert top_factors(rows, "increase")[0]["label"] == "최근 뉴스 위험"
    assert not any("GLD" in r["label"] for r in top_factors(rows, "increase"))
