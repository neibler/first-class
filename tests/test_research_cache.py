"""Exact and similar-question reuse of completed research answers."""

import sqlite3

import numpy as np
import pytest

from backend import research_cache, research_service
from backend.research_cache import ResearchCache, context_key, normalize
from backend.research_schemas import ResearchRequest

PAYLOAD = {
    "status": "complete",
    "question": "금(GLD) 위험은?",
    "as_of": "2025-04-10",
    "answer": "금리 기대 변화가 컸습니다 [1].",
    "sources": [
        {"n": 1, "title": "t", "source": "s", "date": "2025-04-01", "url": "https://x"}
    ],
    "risk_tags": [],
    "plan": [],
    "log": [],
    "disclaimer": "d",
}


def words(text):
    """Toy embedding: bag of a few words, enough to make similar/unrelated."""
    vocab = ["금", "gld", "위험", "반도체", "smh", "뉴스"]
    lowered = text.casefold()
    return np.array([lowered.count(w) for w in vocab], dtype=float) + 1e-9


def test_normalized_question_matches_exactly(tmp_path):
    cache = ResearchCache(tmp_path, embed=None)
    cache.save("금(GLD) 위험은?", "2025-04-10", "ctx", PAYLOAD)
    assert normalize("  금(GLD)   위험은? ") == normalize("금(gld) 위험은?")
    payload, kind, stored = cache.lookup("  금(gld)  위험은? ", "2025-04-10", "ctx")
    assert (kind, stored) == ("exact", "금(GLD) 위험은?")
    # Another date or another DB/settings context is a different question.
    assert cache.lookup("금(GLD) 위험은?", "2025-04-11", "ctx") is None
    assert cache.lookup("금(GLD) 위험은?", "2025-04-10", "other") is None


def test_similar_question_needs_the_threshold(tmp_path):
    cache = ResearchCache(tmp_path, embed=words, threshold=0.95)
    cache.save("금 GLD 위험", "2025-04-10", "ctx", PAYLOAD)
    hit = cache.lookup("GLD 금 위험은 뭐야", "2025-04-10", "ctx")
    assert hit is not None and hit[1] == "similar"
    assert cache.lookup("반도체 SMH 뉴스", "2025-04-10", "ctx") is None


def test_missing_embedding_model_keeps_the_exact_step(tmp_path):
    def broken(text):
        raise RuntimeError("no model")

    cache = ResearchCache(tmp_path, embed=broken)
    cache.save("q", "2025-04-10", "ctx", PAYLOAD)
    assert cache.lookup("q", "2025-04-10", "ctx")[1] == "exact"
    assert cache.lookup("other q", "2025-04-10", "ctx") is None


def test_context_changes_with_the_news_db(tmp_path):
    db = tmp_path / "chroma"
    db.mkdir()
    with sqlite3.connect(db / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE t (x)")
    before = context_key(db)
    with sqlite3.connect(db / "chroma.sqlite3") as conn:
        conn.execute("INSERT INTO t VALUES (1)")
    assert before and context_key(db) != before
    assert context_key(tmp_path / "absent") is None


def test_context_changes_with_the_agent_prompts(tmp_path, monkeypatch):
    # Prompts live in research_agent.py; editing them must invalidate answers.
    root = tmp_path / "repo"
    for name in research_cache.SETTINGS_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("v1", encoding="utf-8")
    db = tmp_path / "chroma"
    db.mkdir()
    (db / "chroma.sqlite3").write_bytes(b"db")
    monkeypatch.setattr(research_cache, "ROOT", root)
    before = context_key(db)
    (root / "backend/news_research/research_agent.py").write_text("v2", "utf-8")
    assert context_key(db) != before


@pytest.fixture
def service(tmp_path, monkeypatch):
    """run_research with a fake paid step that counts calls."""
    db = tmp_path / "chroma"
    db.mkdir()
    with sqlite3.connect(db / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE embedding_metadata (key TEXT, int_value INTEGER)")
        conn.execute("INSERT INTO embedding_metadata VALUES ('date_int', 20170106)")
        conn.execute("INSERT INTO embedding_metadata VALUES ('date_int', 20261124)")
    monkeypatch.setenv("NEWS_CHROMA_PATH", str(db))
    calls = []

    def paid(request):
        calls.append(request)
        return research_service.ResearchResponse.model_validate(
            {**PAYLOAD, "question": request.question}
        )

    monkeypatch.setattr(research_service, "_locked_research", paid)
    return calls


def request(question="금(GLD) 위험은?", refresh=False):
    return ResearchRequest(question=question, as_of="2025-04-10", refresh=refresh)


def test_second_identical_request_skips_the_paid_run(service):
    first = research_service.run_research(request())
    second = research_service.run_research(request(" 금(gld) 위험은? "))
    assert len(service) == 1
    assert first.cached is None and second.cached == "exact"
    # The response still answers the question as asked.
    assert second.question == " 금(gld) 위험은? ".strip()


def test_refresh_runs_again_and_replaces_the_answer(service):
    research_service.run_research(request())
    fresh = research_service.run_research(request(refresh=True))
    assert len(service) == 2 and fresh.cached is None


def test_cache_failure_never_blocks_research(service, monkeypatch):
    monkeypatch.setattr(research_cache, "get_cache", lambda: None)
    research_service.run_research(request())
    research_service.run_research(request())
    assert len(service) == 2


def test_only_completed_answers_are_stored(service, monkeypatch):
    def no_evidence(req):
        service.append(req)
        return research_service.ResearchResponse.model_validate(
            {
                **PAYLOAD,
                "status": "insufficient_evidence",
                "sources": [],
                "answer": "근거 없음",
            }
        )

    monkeypatch.setattr(research_service, "_locked_research", no_evidence)
    research_service.run_research(request())
    research_service.run_research(request())
    assert len(service) == 2
