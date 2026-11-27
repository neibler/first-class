"""ETF relationship words (beginner card) and the server's correlation map."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.performance.correlation import correlation_matrix
from frontend.relations_text import (
    conclusion,
    has_final,
    josa,
    mix_sentence,
    opposite_title,
    partners,
)


def test_korean_particles_follow_the_last_syllable():
    assert josa("금", "과", "와") == "금과"
    assert josa("미국 대형주", "과", "와") == "미국 대형주와"
    assert josa("반도체", "은", "는") == "반도체는"
    assert has_final("S&P500")  # 오백: '백' has a final consonant


def row(name, value, kind="ETF"):
    return {"name": name, "type": kind, "correlation": value, "strength": "s"}


@pytest.mark.parametrize(
    "opposite,title,ending",
    [
        (-0.5, "반대로 움직여요", "서로 버텨 줘서 흔들림이 줄어요."),
        (0.1, "따로 움직여요", "따로 움직이는 자산을 함께 담으면 흔들림이 줄어요."),
        (0.4, "그나마 덜 같이 움직여요", "흔들림이 크게 줄지는 않아요."),
    ],
)
def test_conclusion_depends_on_the_least_related_etf(opposite, title, ending):
    together, other = partners([row("SPY", 0.9), row("GLD", opposite)])
    assert (together["name"], other["name"]) == ("SPY", "GLD")
    assert opposite_title(other) == title
    assert conclusion("QQQ", together, other).endswith(ending)


def test_no_etf_partner_means_no_conclusion():
    assert partners([row("vix_change", -0.8, "macro")]) == (None, None)
    assert conclusion("QQQ", None, None) is None


def test_mix_sentence_bands():
    assert "섞은 효과가 작은" in mix_sentence(0.7, "강한 관계")
    assert "섞은 효과가 보통" in mix_sentence(0.5, "보통 관계")
    assert "섞은 효과가 큰" in mix_sentence(0.1, "매우 약한 관계")


def market(window=60):
    rng = np.random.default_rng(3)
    base = rng.normal(size=window + 5)
    returns = np.column_stack(
        [
            base,
            base * 0.9 + rng.normal(scale=0.1, size=window + 5),
            -base,
            np.zeros(window + 5),
        ]
    )
    return SimpleNamespace(
        dates=np.datetime64("2026-01-01") + np.arange(window + 5),
        assets=["AAA", "BBB", "CCC", "FLAT"],
        returns=returns,
    )


def test_matrix_and_portfolio_average():
    result = correlation_matrix(market(), 60, None, {"AAA": 0.5, "BBB": 0.5, "XYZ": 1})
    matrix = np.array(
        [[np.nan if v is None else v for v in r] for r in result["matrix"]]
    )
    assert matrix[0, 1] > 0.9 and matrix[0, 2] == pytest.approx(-1)
    # A flat series has no correlation at all.
    assert np.isnan(matrix[3]).all() and np.isnan(matrix[:, 3]).all()
    mix = result["portfolio"]
    # Unknown tickers are ignored; two held ETFs give one pair.
    assert mix["holdings"] == ["AAA", "BBB"] and mix["pairs"] == 1
    assert mix["average_correlation"] == pytest.approx(matrix[0, 1])


def test_one_holding_or_flat_pairs_give_no_portfolio_average():
    assert correlation_matrix(market(), 60, None, {"AAA": 1})["portfolio"] is None
    assert (
        correlation_matrix(market(), 60, None, {"AAA": 1, "FLAT": 1})["portfolio"]
        is None
    )


def test_invalid_window_or_short_history_is_refused():
    with pytest.raises(ValueError):
        correlation_matrix(market(), 61)
    with pytest.raises(ValueError):
        correlation_matrix(market(window=10), 60)
