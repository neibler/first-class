import json
import re
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from backend.news_research import research_agent as ra

NEWS_MAP = {
    "etfs": {
        "QQQ": {"role": "나스닥 100 대형 기술주"},
        "XLE": {"role": "미국 에너지"},
    }
}


def _hit(n, title, date="2025-04-01", source="nytimes"):
    return {
        "id": f"c{n}",
        "text": f"{title}\n\nBody of article {n}.",
        "metadata": {
            "url": f"https://example.com/{n}",
            "title": title,
            "source": source,
            "date": date,
        },
        "distance": 0.3,
    }


class FakeLLM:
    """Scripted answers per role; the grader scores titles containing 'good'."""

    def __init__(self, planner, rewriter=None, analyst=None):
        self.answers = {"planner": planner, "rewriter": rewriter, "analyst": analyst}
        self.calls, self.systems = [], {}

    def json(self, role, system, user, schema):
        self.calls.append((role, user))
        self.systems[role] = system
        if role == "grader":
            items = re.findall(r"^\[(\d+)\] (.*)$", user, flags=re.M)
            return {
                "scores": [
                    {"i": int(i), "relevance": 0.9 if "good" in title else 0.2}
                    for i, title in items
                ],
                "comment": "평가",
            }
        answer = self.answers[role]
        return answer(user) if callable(answer) else answer


class FakeSearch:
    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, query, k, **filters):
        self.calls.append({"query": query, **filters})
        return self.results(query, filters) if callable(self.results) else []


@pytest.fixture
def settings():
    return ra.load_settings()


def test_low_relevance_triggers_a_rewrite_and_only_good_hits_become_evidence(settings):
    planner = {
        "goal": "QQQ와 에너지 위험 점검",
        "focus_etfs": ["QQQ"],
        "queries": [
            {"query": "tech export ban", "etf": "QQQ", "lookback_days": 30},
            {"query": "oil supply", "etf": "", "lookback_days": 0},
        ],
    }
    by_query = {
        "tech export ban": [_hit(1, "weak match"), _hit(2, "another weak one")],
        "Nvidia China chip export ban": [_hit(3, "good: Nvidia faces China ban")],
        "oil supply": [_hit(4, "good: OPEC cuts output", source="marketwatch")],
    }
    analyst = {
        "answer": "엔비디아는 중국 수출 규제를 받는다 [1]. 원유 공급이 줄었다 [2]. 없는 근거 [9].",
        "risk_tags": [
            {
                "ticker": "QQQ",
                "risk_type": "regulation",
                "risk_level": 0.6,
                "direction": "increase",
                "confidence": 0.8,
                "evidence": [1],
            },
            {
                "ticker": "XLE",
                "risk_type": "commodity_risk",
                "risk_level": 0.5,
                "direction": "increase",
                "confidence": 0.7,
                "evidence": [9],  # not a listed evidence number: dropped
            },
        ],
    }
    llm = FakeLLM(
        planner,
        rewriter={"query": "Nvidia China chip export ban", "reason": "기업명을 명시"},
        analyst=analyst,
    )
    search = FakeSearch(lambda q, f: by_query.get(q, []))
    result = ra.research(
        "최근 QQQ 위험?",
        "2025-04-10",
        llm=llm,
        search=search,
        settings=settings,
        news_map=NEWS_MAP,
    )
    # every search stops at the as-of date; the first plan step keeps its ETF filter
    assert all(c["until"] == "2025-04-10" for c in search.calls)
    assert search.calls[0] == {
        "query": "tech export ban",
        "etf": "QQQ",
        "since": "2025-03-11",
        "until": "2025-04-10",
    }
    assert search.calls[2]["since"] == "2024-10-12"  # default 180-day lookback
    assert [c["query"] for c in search.calls] == [
        "tech export ban",
        "Nvidia China chip export ban",
        "oil supply",
    ]
    log = "\n".join(result["log"])
    for step in ("[Planner]", "[Researcher]", "[Verifier] 검색어 수정", "[Analyst]"):
        assert step in log
    analyst_input = [u for role, u in llm.calls if role == "analyst"][0]
    assert (
        "Nvidia faces China ban" in analyst_input and "weak match" not in analyst_input
    )
    assert [s["n"] for s in result["sources"]] == [1, 2]
    assert result["sources"][0]["url"] == "https://example.com/3"
    # [9] has no source behind it: it leaves the answer and the log says so
    assert "[9]" not in result["answer"] and "[2]" in result["answer"]
    assert "근거 목록에 없는 인용 [9] 삭제" in log
    assert result["risk_tags"] == [
        {
            "ticker": "QQQ",
            "risk_type": "regulation",
            "risk_score": 0.48,
            "risk_level": 0.6,
            "direction": "increase",
            "confidence": 0.8,
            "date": "2025-04-10",
            "evidence": [1],
        }
    ]
    assert result["disclaimer"].startswith("본 분석은 교육 목적")


def test_planner_and_rewriter_are_told_the_as_of_date(settings):
    planner = {
        "goal": "g",
        "focus_etfs": [],
        "queries": [{"query": "bond selloff", "etf": "", "lookback_days": 30}],
    }
    llm = FakeLLM(planner, rewriter={"query": "Treasury yields jump", "reason": "r"})
    search = FakeSearch(lambda q, f: [_hit(1, "unrelated")])
    ra.research(
        "?", "2022-10-15", llm=llm, search=search, settings=settings, news_map=NEWS_MAP
    )
    for role in ("planner", "rewriter"):
        assert "Today is 2022-10-15" in llm.systems[role]
        assert "ETF tickers" in llm.systems[role]


def test_rewrites_stop_at_the_limit_and_no_evidence_skips_the_analyst(settings):
    planner = {
        "goal": "g",
        "focus_etfs": [],
        "queries": [{"query": "vague", "etf": "", "lookback_days": 60}],
    }
    attempts = iter(range(10))
    llm = FakeLLM(
        planner, rewriter=lambda user: {"query": f"try {next(attempts)}", "reason": "r"}
    )
    search = FakeSearch(lambda q, f: [_hit(1, "unrelated")])
    result = ra.research(
        "?", "2025-01-31", llm=llm, search=search, settings=settings, news_map=NEWS_MAP
    )
    rewrites = [role for role, _ in llm.calls if role == "rewriter"]
    assert len(rewrites) == settings["retrieval"]["max_rewrites"]
    assert len(search.calls) == settings["retrieval"]["max_rewrites"] + 1
    assert "analyst" not in [role for role, _ in llm.calls]
    assert result["sources"] == [] and result["risk_tags"] == []
    assert "근거를 찾지 못했습니다" in result["answer"]


def test_missing_etf_flags_fall_back_to_an_unfiltered_search(settings):
    planner = {
        "goal": "g",
        "focus_etfs": ["XLE"],
        "queries": [{"query": "refinery outage", "etf": "XLE", "lookback_days": 30}],
    }
    analyst = {"answer": "정제소 가동 중단 [1].", "risk_tags": []}
    llm = FakeLLM(planner, analyst=analyst)
    search = FakeSearch(
        lambda q, f: [] if f.get("etf") else [_hit(1, "good refinery fire")]
    )
    result = ra.research(
        "?", "2025-06-01", llm=llm, search=search, settings=settings, news_map=NEWS_MAP
    )
    assert search.calls[1].get("etf") is None
    assert "ETF 조건 없이 재검색" in "\n".join(result["log"])
    assert [s["n"] for s in result["sources"]] == [1]


class FakeClient:
    def __init__(self, stop_reason="end_turn", text='{"ok": true}'):
        self.calls = []

        def create(**params):
            self.calls.append(params)
            return NS(
                stop_reason=stop_reason,
                content=[NS(type="thinking"), NS(type="text", text=text)],
                usage=NS(input_tokens=100, output_tokens=20),
            )

        self.messages = NS(create=create)
        self.beta = NS(messages=NS(create=create))


def test_claude_calls_use_structured_output_and_fallback_only_where_offered():
    client = FakeClient()
    llm = ra.ClaudeLLM(
        client,
        {"planner": "claude-opus-5-5", "grader": "claude-haiku-5-5"},
        {"planner": "low"},
    )
    assert llm.json("planner", "SYS", "Q", {"type": "object"}) == {"ok": True}
    opus = client.calls[0]
    assert opus["betas"] == [ra.FALLBACK_BETA] and opus["fallbacks"] == "default"
    assert opus["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert "temperature" not in opus and opus["messages"][-1]["role"] == "user"
    llm.json("grader", "SYS", "Q", {"type": "object"})
    assert "fallbacks" not in client.calls[1]  # Haiku 5.5 has no server-side fallback
    assert llm.usage["claude-opus-5-5:input"] == 100


def test_a_refusal_is_reported_as_an_agent_error():
    llm = ra.ClaudeLLM(FakeClient("refusal"), {"planner": "claude-opus-5-5"}, {})
    with pytest.raises(ra.AgentError, match="거절"):
        llm.json("planner", "S", "Q", {})


def test_an_unreadable_answer_is_reported_as_an_agent_error():
    llm = ra.ClaudeLLM(FakeClient(text=""), {"planner": "claude-opus-5-5"}, {})
    with pytest.raises(ra.AgentError, match="JSON"):
        llm.json("planner", "S", "Q", {})


def test_a_malformed_as_of_fails_before_any_model_call(settings):
    llm = FakeLLM(planner={"goal": "g", "focus_etfs": [], "queries": []})
    with pytest.raises(ra.AgentError, match="기준일"):
        ra.research(
            "?",
            "2025/04/10",
            llm=llm,
            search=FakeSearch(None),
            settings=settings,
            news_map=NEWS_MAP,
        )
    assert llm.calls == []


def test_schemas_restrict_tickers_and_risk_types():
    shapes = ra.schemas(["QQQ", "XLE"])
    tag = shapes["analyst"]["properties"]["risk_tags"]["items"]
    assert tag["properties"]["ticker"]["enum"] == ["QQQ", "XLE"]
    assert "none" not in tag["properties"]["direction"]["enum"]
    assert json.dumps(shapes)  # serialisable for the API


def test_default_search_opens_existing_collection_and_forwards_filters(
    monkeypatch, tmp_path
):
    from backend.news_research import vector_store

    embedding, collection = Mock(), Mock()
    monkeypatch.setattr(vector_store, "LocalEmbedding", Mock(return_value=embedding))
    opener = Mock(return_value=collection)
    query = Mock(return_value=[_hit(1, "good")])
    monkeypatch.setattr(vector_store, "open_collection", opener)
    monkeypatch.setattr(vector_store, "search", query)
    search = ra.default_search(tmp_path, local_embeddings_only=True)
    opener.assert_called_once_with(tmp_path, embedding_function=embedding, create=False)
    embedding._download_model_if_not_exists.assert_called_once_with()
    result = search("chips", 8, until="2025-04-10", etf="QQQ")
    query.assert_called_once_with(collection, "chips", 8, until="2025-04-10", etf="QQQ")
    assert result == [_hit(1, "good")]


def test_default_llm_has_explicit_timeout_and_no_retries(monkeypatch, settings):
    import anthropic
    from backend.news_research import settings as secrets

    monkeypatch.setattr(secrets, "env_value", lambda name: "test-only")
    factory = Mock()
    monkeypatch.setattr(anthropic, "Anthropic", factory)
    ra.default_llm(settings)
    factory.assert_called_once_with(api_key="test-only", timeout=20.0, max_retries=0)
