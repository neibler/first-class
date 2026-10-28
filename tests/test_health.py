import pytest
from fastapi.testclient import TestClient
from backend.investment_ai.api import create_app


@pytest.fixture
def client(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("No test model configured")

    monkeypatch.setattr("backend.investment_ai.api.InvestmentService", missing)
    with TestClient(create_app()) as value:
        yield value


def test_health_distinguishes_liveness_and_readiness(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert client.get("/ready").status_code == 503


def test_openapi_documents_investment_contract(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/ready", "/optimize", "/backtest"} <= set(schema["paths"])
    assert client.get("/docs").status_code == 200


def test_inference_without_model_is_unavailable(client):
    assert client.post("/optimize", json={}).status_code == 503
    assert client.get("/backtest").status_code == 503
    assert client.post("/research").status_code == 404
    assert client.post("/explain").status_code == 404
