"""Offline HTTP + real agent adapter tests; never open the user's DB or call an LLM."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend import research_service as service
from backend.news_research import research_agent as agent
from backend.news_research import settings as news_settings
from backend.news_research import vector_store as store

BODY = {"question": "  최근 QQQ 위험?  ", "as_of": "2025-04-10"}
SECRET = "secret-value-and-private-path"


class ScriptedClient:
    def __init__(self):
        self.options, self.calls = [], []
        self.closed = False
        self.error = None
        self.answers = [
            {
                "goal": "private planner reasoning",
                "focus_etfs": ["QQQ"],
                "queries": [
                    {
                        "query": "chip export restrictions",
                        "etf": "QQQ",
                        "lookback_days": 30,
                    }
                ],
            },
            {"scores": [{"i": 0, "relevance": 0.9}], "comment": "private grading"},
            {
                "answer": "수출 규제 위험이 있습니다 [1].",
                "risk_tags": [
                    {
                        "ticker": "QQQ",
                        "risk_type": "regulation",
                        "risk_level": 0.6,
                        "confidence": 0.8,
                        "direction": "increase",
                        "evidence": [1],
                    }
                ],
            },
        ]
        self.messages = NS(create=self.create)
        self.beta = NS(messages=self.messages)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return NS(
            stop_reason="end_turn",
            usage=NS(input_tokens=10, output_tokens=5),
            content=[NS(type="text", text=json.dumps(self.answers.pop(0)))],
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.fixture
def offline(monkeypatch, tmp_path):
    # Guard all tests, including validation failures, against real credentials/DB.
    db_path = tmp_path / "chroma"
    db_path.mkdir()
    (db_path / "chroma.sqlite3").touch()
    env = {"ANTHROPIC_API_KEY": SECRET, "RESEARCH_DB_PATH": str(db_path)}
    monkeypatch.setattr(news_settings, "env_value", lambda key, *a: env.get(key, ""))
    client = ScriptedClient()
    factory = Mock(return_value=client)
    monkeypatch.setattr(anthropic, "Anthropic", factory)
    hit = {
        "id": "original-chunk-id",
        "text": "Chip export restrictions.",
        "metadata": {
            "title": "Chip export rules",
            "source": "nytimes",
            "date": "2025-04-03",
            "url": "https://example.com/a%2Fb?q=1",
        },
        "distance": 0.2,
    }
    search = Mock(return_value=[hit])
    opener = Mock(return_value=search)
    monkeypatch.setattr(agent, "default_search", opener)
    with TestClient(app) as http:
        yield NS(
            http=http,
            client=client,
            factory=factory,
            search=search,
            opener=opener,
            env=env,
            db_path=db_path,
        )


def test_real_agent_adapter_preserves_results_and_inputs(offline):
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "complete"
    assert data["question"] == BODY["question"].strip()
    assert data["as_of"] == BODY["as_of"]
    assert data["answer"] == "수출 규제 위험이 있습니다 [1]."
    assert data["sources"] == [
        {
            "n": 1,
            "title": "Chip export rules",
            "source": "nytimes",
            "date": "2025-04-03",
            "url": "https://example.com/a%2Fb?q=1",
        }
    ]
    assert data["risk_tags"] == [
        {
            "ticker": "QQQ",
            "risk_type": "regulation",
            "risk_level": 0.6,
            "confidence": 0.8,
            "risk_score": 0.48,
            "direction": "increase",
            "date": "2025-04-10",
            "evidence": [1],
        }
    ]
    assert data["plan"] == [
        {"query": "chip export restrictions", "etf": "QQQ", "lookback_days": 30}
    ]
    assert data["disclaimer"].startswith("본 분석은 교육 목적")
    for stage in ("Planner", "Researcher", "Verifier", "Analyst"):
        assert any(f"[{stage}]" in line for line in data["log"])
    assert "private" not in "\n".join(data["log"])
    offline.opener.assert_called_once_with(offline.db_path, local_embeddings_only=True)
    offline.search.assert_called_once_with(
        "chip export restrictions", 8, etf="QQQ", since="2025-03-11", until="2025-04-10"
    )
    assert offline.client.calls[0]["messages"][0]["content"] == BODY["question"].strip()
    assert "2025-04-10" in offline.client.calls[0]["system"]
    offline.factory.assert_called_once_with(api_key=SECRET, timeout=20.0, max_retries=0)
    assert all(
        o["max_retries"] == 0 and 0 < o["timeout"] <= 20 for o in offline.client.options
    )
    assert offline.client.closed


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "question": ""},
        {**BODY, "question": " \n\t "},
        {**BODY, "question": "x" * 2001},
        {**BODY, "question": 42},
        {**BODY, "as_of": "2025-02-30"},
        {**BODY, "as_of": "20250410"},
        {**BODY, "as_of": "2025-04-10T00:00:00"},
        {**BODY, "as_of": 1744243200},
        {"question": "hello"},
        {"as_of": "2025-04-10"},
        {},
        {**BODY, "api_key": SECRET},
        {**BODY, "db_path": SECRET},
    ],
)
def test_invalid_inputs_never_start_research(offline, body):
    response = offline.http.post("/research", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_research_request"
    assert SECRET not in response.text
    offline.factory.assert_not_called()
    offline.opener.assert_not_called()


def test_malformed_json_does_not_echo_input(offline):
    response = offline.http.post(
        "/research",
        content='{"api_key": "' + SECRET,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422 and SECRET not in response.text


def test_max_length_and_dates_outside_demo_range_are_valid(offline):
    response = offline.http.post(
        "/research", json={"question": " " + "a" * 2000 + " ", "as_of": "2026-10-29"}
    )
    assert response.status_code == 200
    assert len(response.json()["question"]) == 2000


@pytest.mark.parametrize(
    "missing,code",
    [
        ("key", "research_key_missing"),
        ("db", "research_db_missing"),
    ],
)
def test_missing_setup_is_safe_and_health_unchanged(offline, missing, code):
    if missing == "key":
        offline.env["ANTHROPIC_API_KEY"] = ""
    else:
        offline.env["RESEARCH_DB_PATH"] = str(offline.db_path / "absent")
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == code
    assert str(offline.db_path) not in response.text and SECRET not in response.text
    offline.factory.assert_not_called()
    assert offline.http.get("/health").json() == {"status": "ok", "model_loaded": False}


@pytest.mark.parametrize(
    "kind,status,code",
    [
        ("timeout", 504, "research_timeout"),
        ("upstream", 502, "research_upstream_error"),
        ("invalid", 502, "research_invalid_result"),
        ("unexpected", 500, "research_failed"),
    ],
)
def test_llm_failures_do_not_become_answers(offline, kind, status, code):
    request = httpx.Request("POST", "https://example.com")
    errors = {
        "timeout": anthropic.APITimeoutError(request=request),
        "upstream": anthropic.APIConnectionError(message=SECRET, request=request),
        "invalid": agent.AgentError(SECRET),
        "unexpected": RuntimeError(SECRET),
    }
    offline.client.error = errors[kind]
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert SECRET not in response.text
    assert offline.client.closed


@pytest.mark.parametrize(
    "kind,code",
    [
        ("open", "research_db_unavailable"),
        ("query", "research_db_unavailable"),
        ("embedding", "research_embedding_unavailable"),
    ],
)
def test_db_and_embedding_failures(offline, kind, code):
    if kind == "query":
        offline.search.side_effect = RuntimeError(SECRET)
    else:
        offline.opener.side_effect = (
            store.StoreUnavailable(SECRET)
            if kind == "open"
            else store.EmbeddingUnavailable(SECRET)
        )
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == code
    assert SECRET not in response.text


def test_no_evidence_keeps_agent_message_and_skips_analyst(offline):
    offline.client.answers = [offline.client.answers[0]] + [
        {"query": "rewritten query", "reason": "private rewrite reasoning"}
    ] * 2
    offline.search.return_value = []
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["answer"] == "기준일까지의 뉴스에서 질문에 맞는 근거를 찾지 못했습니다."
    assert data["sources"] == data["risk_tags"] == []
    assert len(offline.client.calls) == 3  # planner + two rewrites, no analyst
    assert "private" not in "\n".join(data["log"])


def test_no_citations_is_an_error_not_a_success(offline):
    offline.client.answers[-1] = {"answer": "unsupported claim", "risk_tags": []}
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "research_invalid_result"


def test_adapter_validates_return_structure(offline, monkeypatch):
    monkeypatch.setattr(agent, "research", lambda *a, **kw: {"answer": SECRET})
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 502 and SECRET not in response.text


def test_budget_stops_before_next_paid_call(offline, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(service, "time", NS(monotonic=lambda: now[0]))

    def expired_search(*args, **kwargs):
        now[0] = service.RESEARCH_BUDGET + 1
        return []

    offline.search.side_effect = expired_search
    response = offline.http.post("/research", json=BODY)
    assert response.status_code == 504
    assert len(offline.client.calls) == 1  # No grader/rewriter after deadline.
    assert offline.client.closed and not service._research_lock.locked()


def test_busy_request_does_not_block_health_or_start_more_work(offline):
    entered, release = threading.Event(), threading.Event()

    def paused_search(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return offline.search.return_value

    offline.search.side_effect = paused_search
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(offline.http.post, "/research", json=BODY)
        try:
            assert entered.wait(5)
            response = offline.http.post("/research", json=BODY)
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "research_busy"
            assert offline.http.get("/health").status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200
    assert not service._research_lock.locked()


def test_openapi_exposes_nested_contracts(offline):
    schema = offline.http.get("/openapi.json").json()
    operation = schema["paths"]["/research"]["post"]
    assert operation["requestBody"]["required"]
    assert set(operation["responses"]) == {"200", "422", "500", "502", "503", "504"}
    models = schema["components"]["schemas"]
    assert set(models["ResearchRequest"]["required"]) == {"question", "as_of"}
    assert models["ResearchRequest"]["properties"]["question"]["maxLength"] == 2000
    assert models["ResearchResponse"]["properties"]["sources"]["items"][
        "$ref"
    ].endswith("/ResearchSource")
    assert models["ResearchResponse"]["properties"]["risk_tags"]["items"][
        "$ref"
    ].endswith("/ResearchRiskTag")
    assert offline.http.get("/docs").status_code == 200


def test_app_import_never_imports_optional_runtime():
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'anthropic', 'chromadb', 'langgraph', 'yaml'}:
            raise AssertionError('Heavy startup import: ' + fullname)
sys.meta_path.insert(0, Block())
from backend.main import app
assert '/research' in app.openapi()['paths']
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
