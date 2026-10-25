import contextlib
import hashlib
import json
import math

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.errors import NotFoundError

from backend.news_research import vector_store as vs


class WordHashEmbedding(EmbeddingFunction):
    """Bag-of-words hashed into 64 dims: shared words mean nearby vectors."""

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            vector = [0.0] * 64
            for word in text.lower().split():
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    @staticmethod
    def name():
        return "word-hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return WordHashEmbedding()


def _chunk(i, text, date, etfs, source="nytimes"):
    return {
        "id": f"{source}:{i}:0",
        "text": text,
        "metadata": {
            "url": f"https://example.com/{i}",
            "title": text,
            "source": source,
            "date": date,
            "date_int": int(date.replace("-", "")),
            **{f"etf_{t}": True for t in etfs},
        },
    }


def _collection(tmp_path):
    chunks = [
        _chunk(1, "oil prices jump after OPEC cut", "2025-04-01", ["XLE"]),
        _chunk(
            2,
            "Nvidia faces new China export ban on chips",
            "2025-04-10",
            ["SMH", "QQQ"],
        ),
        _chunk(
            3, "oil prices fall as supply rises", "2025-05-01", ["XLE"], "marketwatch"
        ),
        _chunk(4, "Fed holds rates steady", "2025-04-02", ["TLT"]),
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    client = chromadb.EphemeralClient()
    # in-memory clients share one store per process: start every test empty
    with contextlib.suppress(NotFoundError):
        client.delete_collection(vs.COLLECTION)
    collection = vs.open_collection(
        embedding_function=WordHashEmbedding(), client=client
    )
    return collection, path


def test_build_is_resumable(tmp_path):
    collection, path = _collection(tmp_path)
    assert vs.build(collection, path, batch_size=3) == (4, 0)
    assert vs.build(collection, path, batch_size=3) == (0, 4)
    assert collection.count() == 4


def test_search_ranks_by_meaning_and_filters_by_etf_date_and_source(tmp_path):
    collection, path = _collection(tmp_path)
    vs.build(collection, path)
    hits = vs.search(collection, "oil prices", k=2)
    assert {h["metadata"]["url"][-1] for h in hits} == {"1", "3"}
    assert hits[0]["distance"] <= hits[1]["distance"]
    # a backtest on 2025-04-15 must not see the May article
    dated = vs.search(collection, "oil prices", k=5, until="2025-04-15")
    assert all(h["metadata"]["date_int"] <= 20250415 for h in dated)
    assert "3" not in {h["metadata"]["url"][-1] for h in dated}
    smh = vs.search(collection, "chips", k=5, etf="SMH")
    assert [h["metadata"]["url"][-1] for h in smh] == ["2"]
    mw = vs.search(collection, "oil", k=5, sources=["marketwatch"])
    assert [h["metadata"]["source"] for h in mw] == ["marketwatch"]


def test_where_filter_shapes():
    assert vs.where_filter() is None
    assert vs.where_filter(etf="SPY") == {"etf_SPY": True}
    both = vs.where_filter(since="2025-01-01", until="2025-12-31")
    assert both == {
        "$and": [
            {"date_int": {"$gte": 20250101}},
            {"date_int": {"$lte": 20251231}},
        ]
    }
