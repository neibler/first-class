"""Financial identities and chart semantics, independent of screenshot styling."""

import json
import math
from statistics import mean, stdev

import pytest

from frontend import sample_data
from frontend.sample_data import (
    SampleDataError,
    load_portfolio,
    performance,
)


def test_weights_reproducible_and_not_mutable_across_loads():
    data = load_portfolio()
    assert sum(a["weight"] for a in data["assets"]) == pytest.approx(1)
    assert all(a["weight"] >= 0 for a in data["assets"])
    assert data == load_portfolio()
    data["assets"].clear()
    assert load_portfolio()["assets"]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_portfolio_loader_rejects_insufficient_returns(monkeypatch, tmp_path, count):
    data = load_portfolio()
    data["returns"] = data["returns"][:count]
    (tmp_path / "portfolio.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sample_data, "FIXTURES", tmp_path)

    with pytest.raises(SampleDataError, match="기준일과 수익률 관측치 2개"):
        load_portfolio()


@pytest.mark.parametrize("recent", [False, True])
def test_minimum_portfolio_sample_supports_both_periods(monkeypatch, tmp_path, recent):
    data = load_portfolio()
    data["returns"] = data["returns"][:3]
    (tmp_path / "portfolio.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sample_data, "FIXTURES", tmp_path)

    loaded = load_portfolio()
    assert loaded == data
    rows = loaded["returns"][-64:] if recent else loaded["returns"]
    result = performance(
        rows, loaded["periods_per_year"], loaded["annual_risk_free_rate"]
    )
    assert result["observations"] == 2
    assert len(result["path"]) == 3
    assert result["cumulative"] == pytest.approx(
        math.expm1(sum(row["portfolio_log"] for row in rows[1:]))
    )


@pytest.mark.parametrize("count", [None, 64])
def test_sample_metrics_use_same_period_and_log_returns(count):
    data = load_portfolio()
    rows = data["returns"] if count is None else data["returns"][-count:]
    result = performance(rows, data["periods_per_year"], data["annual_risk_free_rate"])
    logs = [r["portfolio_log"] for r in rows[1:]]
    expected_sharpe = (
        (mean(logs) - math.log1p(0.02) / 252) / stdev(logs) * math.sqrt(252)
    )
    assert result["cumulative"] == pytest.approx(math.expm1(sum(logs)))
    assert result["path"][-1]["portfolio"] == result["cumulative"]
    assert result["path"][0]["portfolio"] == 0
    assert result["sharpe"] == pytest.approx(expected_sharpe)
    assert result["observations"] == len(rows) - 1


def test_mdd_includes_initial_capital_and_differs_from_current_drawdown():
    # Wealth: 100 -> 80 -> 120 -> 108. Period MDD -20%, current -10%.
    rows = [
        dict(date=str(i), portfolio_log=math.log(r), benchmark_log=0)
        for i, r in enumerate([1, 0.8, 1.5, 0.9])
    ]
    result = performance(rows)
    assert result["mdd"] == pytest.approx(-0.2)
    assert result["current_drawdown"] == pytest.approx(-0.1)
    assert result["cumulative"] == pytest.approx(0.08)


def test_flat_returns_have_no_infinite_sharpe():
    rows = [dict(date=str(i), portfolio_log=0, benchmark_log=0) for i in range(4)]
    result = performance(rows)
    assert result["sharpe"] is None
    assert result["mdd"] == 0
    with pytest.raises(SampleDataError):
        performance(rows[:1])
