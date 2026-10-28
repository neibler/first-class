import sqlite3
from datetime import date

import pytest

from backend.investment_ai.news_search import search_news


@pytest.fixture
def news_db(tmp_path):
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE embedding_metadata "
            "(id INTEGER, key TEXT, string_value TEXT, bool_value INTEGER)"
        )
        for i, published in enumerate(
            ["2023-12-31", "2024-01-05T23:59:00", "2024-01-06", "2024-03-01"]
        ):
            conn.executemany(
                "INSERT INTO embedding_metadata VALUES (?, ?, ?, ?)",
                [
                    (i, "title", "market news", None),
                    (i, "date", published, None),
                    (i, "etf_QQQ", None, 1),
                    (i, "url", f"https://example.com/{i}", None),
                ],
            )
    return path


@pytest.mark.parametrize("as_of", ["2024-01-05", "20240105", "2024-W01-5"])
def test_search_excludes_future_articles_for_every_supported_date_format(
    news_db, as_of
):
    try:
        date.fromisoformat(as_of)
    except ValueError:
        pytest.skip("This Python version rejects alternate ISO dates")
    rows = search_news("market", as_of, "QQQ", db_path=news_db)
    assert [r["id"] for r in rows] == [1, 0]
    assert rows[0]["url"] == "https://example.com/1"


@pytest.mark.parametrize("as_of", ["2024-02-30", "not-a-date"])
def test_search_rejects_invalid_cutoff(news_db, as_of):
    with pytest.raises(ValueError):
        search_news("market", as_of, "QQQ", db_path=news_db)
