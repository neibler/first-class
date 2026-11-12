from types import SimpleNamespace

import pandas as pd
import pytest

from backend.investment_ai.model_runs import explicit_windows, plan_runs
from tests.test_contract_v3 import config


def definition():
    return dict(
        train_start="2022-01-03",
        train_end="2022-12-30",
        test_start="2023-01-02",
        test_end="2023-12-29",
    )


def test_exact_boundary_and_partial_final_year():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    spec = definition()
    spec["test_end"] = "2023-09-29"
    w = explicit_windows(dates, [spec])[0]
    assert dates[w["split"] - 1] == pd.Timestamp(spec["train_end"])
    assert dates[w["split"]] == pd.Timestamp(spec["test_start"])
    assert dates[w["stop"] - 1] == pd.Timestamp(spec["test_end"])


@pytest.mark.parametrize(
    "change",
    [
        {"train_start": "2021-01-01"},
        {"test_end": "2024-01-01"},
        {"test_start": "2022-12-30"},
        {"test_start": "2023-01-03"},
        {"test_start": "invalid"},
    ],
)
def test_reject_invalid_periods(change):
    with pytest.raises(ValueError):
        explicit_windows(
            pd.bdate_range("2022-01-03", "2023-12-29"), [{**definition(), **change}]
        )


def test_reject_overlapping_tests():
    with pytest.raises(ValueError, match="non-overlapping"):
        explicit_windows(
            pd.bdate_range("2022-01-03", "2023-12-29"), [definition(), definition()]
        )


def test_config_plan_uses_dates_instead_of_fraction():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    c = {**config(), "training_windows": [definition()], "test_fraction": 0.2}
    windows = plan_runs(c, SimpleNamespace(dates=dates), [17])
    assert dates[windows[0]["split"]] == pd.Timestamp("2023-01-02")


def test_default_fraction_without_explicit_windows_is_unchanged():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    c = config()
    w = plan_runs(c, SimpleNamespace(dates=dates), [17])[0]
    assert w["split"] == int(len(dates) * (1 - c["test_fraction"]))
