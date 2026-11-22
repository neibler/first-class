import pytest
from fastapi.testclient import TestClient
from backend.investment_ai.api import create_app, Metrics
from backend.investment_ai.backtest import run_backtest, performance
from backend.investment_ai import baseline_execution as b
from tests.test_contract_v3 import service, data, config


@pytest.mark.parametrize("policy", [None, b.POLICY, b.CASH_POLICY])
def test_computed_backtest_metrics_pass_real_http_contract(tmp_path, policy):
    obj = service(tmp_path)
    c = config()
    if policy is not None:
        c["execution_policy"] = policy
    obj.config = c
    history = run_backtest(data(), c, 0, 4, strategy="spy_buy_hold")
    obj.histories = {"ppo": history, "spy_buy_hold": history.copy()}
    app = create_app()
    app.state.service = obj
    with TestClient(app, raise_server_exceptions=False) as client:
        # Lifespan may load the installed model; use the controlled history here.
        app.state.service = obj
        for query in ("", "?observations=2"):
            response = client.get("/backtest" + query)
            assert response.status_code == 200, response.text
            body = response.json()
            selected = history if not query else history.tail(2)
            expected = performance(selected)["total_trade_notional_ratio"]
            for key in ("metrics", "benchmark_metrics"):
                assert body[key]["total_trade_notional_ratio"] == pytest.approx(
                    expected
                )


def test_older_metrics_without_new_field_still_validate():
    history = run_backtest(data(), config(), 0, 4, strategy="spy_buy_hold")
    metrics = performance(history)
    metrics.pop("total_trade_notional_ratio")
    assert Metrics.model_validate(metrics).total_trade_notional_ratio is None
