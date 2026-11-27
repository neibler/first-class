"""News DB date range and the paid-call-free refusal of dates outside it."""

import sqlite3
import sys
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import research_service
from backend.research_api import router
from backend.research_schemas import ResearchRequest


def make_db(folder, dates):
    """Minimal chroma.sqlite3 with Chroma's metadata table and date_int rows."""
    folder.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(folder / "chroma.sqlite3") as db:
        db.execute(
            "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, "
            "string_value TEXT, int_value INTEGER)"
        )
        db.executemany(
            "INSERT INTO embedding_metadata VALUES (?, 'date_int', NULL, ?)",
            enumerate(dates),
        )
        db.execute("INSERT INTO embedding_metadata VALUES (99, 'year', NULL, 1999)")
    return folder


@pytest.fixture(autouse=True)
def fresh_cache():
    research_service._coverage_cache.clear()
    yield
    research_service._coverage_cache.clear()


@pytest.fixture
def news_db(tmp_path, monkeypatch):
    folder = make_db(tmp_path / "chroma", [20250410, 20170106, 20261124])
    monkeypatch.setenv("NEWS_CHROMA_PATH", str(folder))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return folder


def test_range_is_min_and_max_news_date(news_db):
    assert research_service.news_coverage(news_db) == ("2017-01-06", "2026-11-24")


def test_range_is_reread_only_when_the_db_changes(news_db):
    research_service.news_coverage(news_db)
    with sqlite3.connect(news_db / "chroma.sqlite3") as db:
        db.execute(
            "INSERT INTO embedding_metadata VALUES (7, 'date_int', NULL, 20270102)"
        )
    assert research_service.news_coverage(news_db)[1] == "2027-01-02"


@pytest.mark.parametrize("dates", [[], None])
def test_unreadable_range_is_none(tmp_path, dates):
    folder = tmp_path / "db"
    if dates is not None:
        make_db(folder, dates)
    assert research_service.news_coverage(folder) is None


@pytest.mark.parametrize("as_of", ["2017-01-05", "2026-11-25"])
def test_outside_dates_are_refused_before_any_paid_call(news_db, monkeypatch, as_of):
    # Importing the SDK would fail, so reaching Claude would be a 503, not 422.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    request = ResearchRequest(question="TLT 위험?", as_of=as_of)
    with pytest.raises(research_service.ResearchFailure) as error:
        research_service.run_research(request)
    assert (error.value.status, error.value.code) == (422, "research_out_of_range")
    assert "2017-01-06 ~ 2026-11-24" in error.value.message


def test_edge_dates_pass_the_range_check(news_db, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    request = ResearchRequest(
        question="TLT 위험?", as_of=date(2026, 11, 24).isoformat()
    )
    with pytest.raises(research_service.ResearchFailure) as error:
        research_service.run_research(request)
    assert error.value.code == "research_unavailable"  # got past the range check


def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_coverage_endpoint(news_db):
    response = client().get("/research/coverage")
    assert response.status_code == 200
    assert response.json() == {"start": "2017-01-06", "end": "2026-11-24"}


def test_coverage_endpoint_without_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWS_CHROMA_PATH", str(tmp_path / "absent"))
    response = client().get("/research/coverage")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "research_db_missing"
