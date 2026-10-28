"""Read-only keyword retrieval from the supplied Chroma SQLite snapshot.

No pickle loading, embedding download, LLM call or database mutation.
This adapter is keyword retrieval, not semantic search or generated explanations.
"""

import argparse
from datetime import date
import json
from pathlib import Path
import sqlite3

DEFAULT_DB = (
    Path(__file__).resolve().parents[2] / "data/processed/news/chroma/chroma.sqlite3"
)


def search_news(query, as_of, ticker, limit=5, db_path=DEFAULT_DB):
    as_of = date.fromisoformat(as_of).isoformat()
    if not 1 <= limit <= 50 or not query.strip():
        raise ValueError("Provide a query and limit between 1 and 50")
    from .settings import read_config

    assets = read_config(
        Path(__file__).resolve().parents[2] / "config/investment_ai.json"
    )["assets"]
    if ticker not in assets:
        raise ValueError("Unknown ETF")
    connection = sqlite3.connect(
        Path(db_path).resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        rows = connection.execute(
            """
            SELECT title.id, title.string_value, d.string_value
            FROM embedding_metadata title
            JOIN embedding_metadata d ON d.id=title.id AND d.key='date'
            JOIN embedding_metadata etf ON etf.id=title.id AND etf.key=?
            WHERE title.key='title' AND instr(lower(title.string_value), lower(?)) > 0
              AND substr(d.string_value,1,10) <= ? AND etf.bool_value=1
            ORDER BY d.string_value DESC, title.id LIMIT ?
        """,
            ("etf_" + ticker, query, as_of, limit),
        ).fetchall()
        result = []
        for item_id, title, published in rows:
            metadata = dict(
                connection.execute(
                    """SELECT key, string_value
                FROM embedding_metadata WHERE id=?
                AND key IN ('url','source','chroma:document')""",
                    (item_id,),
                )
            )
            result.append(
                {"id": item_id, "title": title, "date": published, **metadata}
            )
        return result
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            search_news(args.query, args.as_of, args.ticker),
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
