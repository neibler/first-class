from datetime import date
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from streamlit.testing.v1 import AppTest

from backend.news_research import api
from frontend import research_client


def result():
    return {
        "question": "QQQ risk?",
        "as_of": "2025-04-10",
        "answer": "근거 기반 답변 [1]",
        "risk_tags": [],
        "sources": [
            {
                "n": 1,
                "title": "News",
                "source": "test",
                "date": "2025-04-09",
                "url": "https://example.com/news",
            }
        ],
        "log": ["[Researcher] 검색 1건"],
        "plan": [],
        "disclaimer": "교육용",
    }


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    monkeypatch.setattr(api, "readiness", lambda: {"configured": True, "messages": []})
    with TestClient(app) as client:
        yield client


def test_research_http_success_and_input_contract(client, monkeypatch):
    seen = []

    def run(request):
        seen.append(request)
        return result()

    monkeypatch.setattr(api, "run_research", run)
    payload = {"question": " QQQ risk? ", "as_of": "2025-04-10", "ticker": "qqq"}
    response = client.post("/research", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["sources"][0]["url"] == "https://example.com/news"
    assert seen[0].ticker == "QQQ" and seen[0].as_of == date(2025, 4, 10)
    for update in [{"question": " "}, {"as_of": "2025-02-30"}, {"ticker": "UNKNOWN"}]:
        assert client.post("/research", json={**payload, **update}).status_code == 422
    assert len(seen) == 1


def test_research_missing_configuration_and_sanitized_failure(client, monkeypatch):
    def fail(request):
        raise RuntimeError("PRIVATE_KEY_DO_NOT_DISPLAY")

    monkeypatch.setattr(api, "run_research", fail)
    payload = {"question": "QQQ risk?", "as_of": "2025-04-10"}
    response = client.post("/research", json=payload)
    assert response.status_code == 502
    assert "PRIVATE_KEY" not in response.text
    monkeypatch.setattr(
        api, "readiness", lambda: {"configured": False, "messages": ["설정 필요"]}
    )
    assert client.get("/research/status").json()["configured"] is False
    assert client.post("/research", json=payload).status_code == 503


def test_busy_request_is_not_run(client, monkeypatch):
    monkeypatch.setattr(api, "run_research", lambda req: pytest.fail("must not call"))
    api.request_lock.acquire()
    try:
        assert (
            client.post(
                "/research", json={"question": "q", "as_of": "2025-04-10"}
            ).status_code
            == 409
        )
    finally:
        api.request_lock.release()


def test_explicit_ticker_survives_agent_fallback(monkeypatch):
    from backend.news_research import research_agent

    calls = []
    monkeypatch.setattr(
        research_agent,
        "default_search",
        lambda path: lambda query, k, **f: calls.append(f) or [],
    )

    def research(question, as_of, **kwargs):
        kwargs["search"]("test", 5, etf=None, until=as_of)
        return result()

    monkeypatch.setattr(research_agent, "research", research)
    api.run_research(
        api.ResearchRequest(question="q", as_of="2025-04-10", ticker="QQQ")
    )
    assert calls == [{"etf": "QQQ", "until": "2025-04-10"}]


def test_research_page_submit_and_failure_clears_old_answer(monkeypatch):
    monkeypatch.setattr(
        research_client, "status", lambda: {"configured": True, "messages": []}
    )
    seen = []

    def request(*args):
        seen.append(args)
        return result()

    monkeypatch.setattr(research_client, "research", request)
    page = AppTest.from_file("frontend/app_pages/research.py", default_timeout=15).run()
    page.text_area[0].set_value("QQQ risk?")
    page.selectbox[0].set_value("QQQ")
    page.button[0].click().run()
    assert not page.exception
    assert seen == [("QQQ risk?", "2025-04-10", "QQQ")]
    assert any("근거 기반 답변" in item.value for item in page.markdown)

    def failed(*args):
        raise research_client.ResearchAPIError("실패 안내")

    monkeypatch.setattr(research_client, "research", failed)
    page.button[0].click().run()
    assert not page.exception and page.error[0].value == "실패 안내"
    assert not any("근거 기반 답변" in item.value for item in page.markdown)


def test_research_page_disables_submit_when_unconfigured(monkeypatch):
    monkeypatch.setattr(
        research_client,
        "status",
        lambda: {"configured": False, "messages": ["설정 필요"]},
    )
    page = AppTest.from_file("frontend/app_pages/research.py", default_timeout=15).run()
    assert not page.exception
    assert page.button[0].disabled
    assert page.info[0].value == "설정 필요"


def test_research_client_uses_real_contract_and_timeout(monkeypatch):
    seen = []

    def request(method, url, **kwargs):
        seen.append((method, url, kwargs))
        return httpx.Response(200, json=result(), request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", request)
    assert research_client.research("q", "2025-04-10", "QQQ")["answer"]
    assert seen[0][2]["json"]["ticker"] == "QQQ"
    assert seen[0][2]["timeout"] == 300.0
