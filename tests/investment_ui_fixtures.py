"""UI-only fixtures: no model files, training or network are used by these tests."""

import pytest
from frontend import investment_client
from frontend.sample_data import load_portfolio, performance


@pytest.fixture(autouse=True)
def investment_ui_fixture(request, monkeypatch):
    if request.node.module.__name__.split(".")[-1] not in {
        "test_dashboard",
        "test_ui_pages",
    }:
        return

    def payload(observations=None):
        data = load_portfolio()
        if observations is not None:
            data["returns"] = data["returns"][-(observations + 1) :]
        data["display_metrics"] = performance(
            data["returns"], data["periods_per_year"], data["annual_risk_free_rate"]
        )
        data.update(data_source="synthetic", stopped=False, cash_weight=0.0)
        return data

    monkeypatch.setattr(investment_client, "load_portfolio", payload)
