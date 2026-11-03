"""Merged app wiring; no private model, database, or external request is used."""

from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from backend.investment_ai import api
from backend.news_research import settings, vector_store
from backend import research_api


def test_factory_preserves_investment_lifecycle_and_one_research_route(monkeypatch):
    service = SimpleNamespace(run_id="test-model")
    factory = Mock(return_value=service)
    monkeypatch.setattr(api, "InvestmentService", factory)
    app = api.create_app("test-runtime")

    # FastAPI 0.142 keeps included routers as branches, not flattened routes.
    def leaves(routes):
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                yield from leaves(included.routes)
            else:
                yield route

    routes = [r for r in leaves(app.routes) if r.path == "/research"]
    assert len(routes) == 1 and routes[0].methods == {"POST"}
    with TestClient(app) as client:
        factory.assert_called_once_with("test-runtime")
        assert app.state.service is service
        assert client.get("/ready").json()["model_version"] == "test-model"
        assert client.get("/research/status").status_code == 404
        assert {"/optimize", "/backtest", "/research"} <= set(
            client.get("/openapi.json").json()["paths"]
        )
        # Research errors are sanitized; investment validation stays FastAPI's.
        response = client.post("/research", json={"question": "", "as_of": "bad"})
        assert response.json()["detail"]["code"] == "invalid_research_request"
        response = client.get("/backtest?observations=1")
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)
    assert app.state.service is None


def test_factory_uses_canonical_contract_without_ticker(monkeypatch):
    monkeypatch.setattr(api, "InvestmentService", Mock(return_value=None))
    run = Mock(
        return_value={
            "status": "insufficient_evidence",
            "question": "QQQ?",
            "as_of": "2025-04-10",
            "answer": "판단할 근거가 부족합니다",
            "sources": [],
            "risk_tags": [],
            "plan": [],
            "log": [],
            "disclaimer": "교육용",
        }
    )
    monkeypatch.setattr(research_api, "run_research", run)
    payload = {"question": " QQQ? ", "as_of": "2025-04-10"}
    with TestClient(api.create_app()) as client:
        response = client.post("/research", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_evidence"
        assert run.call_args.args[0].question == "QQQ?"
        assert (
            client.post("/research", json={**payload, "ticker": None}).status_code
            == 422
        )
        assert run.call_count == 1


@pytest.mark.parametrize("configured", ["", "custom/chroma", "absolute"])
def test_news_path_is_shared_and_independent_of_working_directory(
    monkeypatch, tmp_path, configured
):
    root = tmp_path / "project"
    root.mkdir()
    absolute = tmp_path / "absolute-store"
    value = str(absolute) if configured == "absolute" else configured
    (root / ".env").write_text(f"NEWS_CHROMA_PATH={value}\n", encoding="utf-8")
    monkeypatch.setattr(settings, "PROJECT", root)
    monkeypatch.delenv("NEWS_CHROMA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = (
        absolute
        if configured == "absolute"
        else root / (configured or "data/processed/news/chroma")
    )
    assert settings.news_db_path() == expected
    assert not expected.exists()  # Resolving configuration never creates a DB.
    monkeypatch.setenv("NEWS_CHROMA_PATH", "override")
    assert settings.news_db_path() == root / "override"


def test_vector_cli_search_only_opens_existing_store(monkeypatch, tmp_path):
    opener = Mock(return_value=Mock())
    monkeypatch.setattr(vector_store, "open_collection", opener)
    monkeypatch.setattr(vector_store, "search", Mock(return_value=[]))
    monkeypatch.setattr(vector_store.sys.stdout, "reconfigure", Mock(), raising=False)
    assert vector_store.main(["--db", str(tmp_path), "search", "chips"]) == 0
    assert opener.call_args.args == (tmp_path,)
    assert opener.call_args.kwargs["create"] is False
    assert isinstance(
        opener.call_args.kwargs["embedding_function"], vector_store.LocalEmbedding
    )
