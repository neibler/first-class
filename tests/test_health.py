import pytest
from fastapi.testclient import TestClient

from backend.main import app


def test_health_reports_live_server_without_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_openapi_describes_health_and_swagger_is_available():
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        assert client.get("/docs").status_code == 200
    assert set(schema["paths"]) == {"/health"}
    model = schema["components"]["schemas"]["HealthResponse"]
    assert set(model["required"]) == {"status", "model_loaded"}
    assert model["properties"]["model_loaded"]["type"] == "boolean"


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/optimize"),
        ("POST", "/explain"),
        ("POST", "/research"),
        ("GET", "/backtest"),
    ],
)
def test_unimplemented_endpoints_do_not_return_fabricated_results(method, path):
    with TestClient(app) as client:
        assert client.request(method, path).status_code == 404
