"""Plain-language reason and moved-asset lines on the trades page."""

from frontend.portfolio_sections import (
    compare,
    moved_lines,
    plain_reason,
    risk_summary,
    won,
)


def test_risk_summary_is_one_line_in_any_combination():
    assert (
        risk_summary("less", "less")
        == "S&P500과 비교하면 덜 떨어지고, 덜 오르내렸어요."
    )
    assert risk_summary("more", "same") == (
        "S&P500과 비교하면 더 떨어지고, 비슷하게 오르내렸어요."
    )
    assert risk_summary("less", None) == "S&P500과 비교하면 덜 떨어졌어요."
    assert risk_summary(None, "more") == "S&P500과 비교하면 더 크게 오르내렸어요."
    assert risk_summary(None, None) is None


def test_example_amount_is_rounded_for_reading():
    assert won(0.0183) == "약 18만 원"
    assert won(0.0005) == "약 5,000원"


def test_market_comparison_bands():
    # Within ±15% of the market reads as the same, as on the report card.
    assert compare(0.0183, 0.0888) == "less"
    assert compare(0.13, 0.1222) == "same"
    assert compare(0.20, 0.10) == "more"
    # No ratio without a market loss (or a missing value).
    assert compare(0.01, 0) is None and compare(0.01, -0.002) is None
    assert compare(None, 0.1) is None


def rows(*changes):
    return [{"자산": name, "변화": f"{value:+d}%p"} for name, value in changes]


def test_each_recorded_reason_reads_as_one_sentence():
    assert plain_reason(["risk_score"]).startswith("뉴스 위험이 기준을 넘어서")
    both = plain_reason(["monthly", "risk_score"])
    assert both.startswith("한 달에 한 번") and "뉴스 위험이" in both


def test_unknown_or_missing_reason_is_said_plainly():
    assert plain_reason(["new_rule"]) == "기록된 조정 이유: new_rule"
    assert plain_reason([]) == "이날 조정 이유는 기록되지 않았어요."


def test_two_names_per_line_then_a_count():
    lines = moved_lines(rows(("GLD", 30), ("SHY", -20), ("EFA", 10), ("현금", -20)))
    assert lines == [
        "▲ **늘린 자산** 금(GLD), 미국 외 선진국 주식(EFA)",
        "▼ **줄인 자산** 미국 단기국채(SHY), 현금",
    ]


def test_only_one_direction_and_no_change():
    assert moved_lines(rows(("SPY", 5), ("QQQ", 3), ("SMH", 1))) == [
        "▲ **늘린 자산** 미국 대형주(SPY), 기술주(QQQ) 외 1개"
    ]
    assert moved_lines([]) == ["표시할 만큼 바뀐 자산은 없어요."]
