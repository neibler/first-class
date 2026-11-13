import numpy as np
import pandas as pd
import pytest
from tests.test_correlation import market as market_fixture
from backend.performance.correlation import (
    analyze_correlations,
    calculate_rolling_correlation,
)
from backend.performance.correlation_visuals import (
    normalize_series_to_100,
    normalize_pair,
    build_comparison,
)


@pytest.fixture
def market():
    return market_fixture.__wrapped__()


def test_normalization_ratio_and_invalid_baseline():
    result = normalize_series_to_100(pd.Series([10, 12, 9]))
    assert result.tolist() == [100, 120, 90]
    assert normalize_series_to_100(pd.Series([np.nan, 10, 20])).iloc[1:].tolist() == [
        100,
        200,
    ]
    for values in [[0, 1], [-1, -2], [np.nan, np.inf]]:
        with pytest.raises(ValueError):
            normalize_series_to_100(pd.Series(values))


def test_alignment_uses_same_first_common_date():
    a = pd.Series(
        [10, 20, 30], index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    )
    b = pd.Series(
        [40, 80, 90], index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04"])
    )
    p = normalize_pair(a, b)
    assert p.index.tolist() == a.index[1:].tolist()
    assert p.iloc[0].tolist() == [100, 100]
    assert p.iloc[1].tolist() == [150, 200]
    b.iloc[0] = np.nan
    with pytest.raises(ValueError):
        normalize_pair(a, b)


@pytest.mark.parametrize("window", [20, 60, 120, 252])
def test_etf_paths_and_period_match_correlation(market, window):
    response = analyze_correlations(market, "SPY", window)
    path = response["comparisons"]["TLT"]
    rows = path["records"]
    assert len(rows) == window
    assert rows[0]["date"] == response["window_start"]
    assert rows[-1]["date"] == response["date"]
    assert rows[0]["selected"] == rows[0]["comparison"] == 100
    expected = 100 * np.exp(market.returns[-window + 1 :, 0].sum())
    assert rows[-1]["selected"] == pytest.approx(expected)
    # Negatively correlated return series is NOT flipped for display.
    assert rows[-1]["comparison"] == pytest.approx(
        100 * np.exp(market.returns[-window + 1 :, 1].sum())
    )
    original = calculate_rolling_correlation(
        pd.Series(market.returns[:, 0]), pd.Series(market.returns[:, 1]), window
    )
    assert (
        next(r for r in response["relationships"] if r["name"] == "TLT")["correlation"]
        == original.iloc[-1]
    )


@pytest.mark.parametrize("name,position", [("vix", 1), ("ust_10y", 0), ("wti", 2)])
def test_market_graph_uses_original_level(market, name, position):
    row = {
        "name": name + ("_return" if name == "wti" else "_change"),
        "type": "macro",
        "display_name": name,
        "transformation": "pct_change" if name == "wti" else "diff",
    }
    p = build_comparison(market, "SPY", row, 60)
    assert p["kind"] == "market_value" and p["status"] == "ok"
    assert p["records"][0]["comparison"] == 100
    assert p["records"][-1]["comparison"] == pytest.approx(
        market.macro[-1, position] / market.macro[-60, position] * 100
    )


def test_unavailable_graph_does_not_remove_statistic(market):
    market.macro[:, 0] -= 10
    result = analyze_correlations(market, "SPY", 60)
    assert next(r for r in result["relationships"] if r["name"] == "ust_10y_change")[
        "correlation"
    ] == pytest.approx(-1)
    assert result["comparisons"]["ust_10y_change"]["status"] == "unavailable"


def test_not_enough_data_and_future_exclusion(market):
    rel = {"name": "TLT", "type": "ETF", "display_name": "TLT"}
    assert (
        build_comparison(market, "SPY", rel, 60, str(market.dates[19].date()))["status"]
        == "unavailable"
    )
    cutoff = str(market.dates[300].date())
    original = build_comparison(market, "SPY", rel, 60, cutoff)
    market.returns[301:] = 999
    assert build_comparison(market, "SPY", rel, 60, cutoff) == original
