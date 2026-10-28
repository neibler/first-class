"""Store RAG chunks in ChromaDB and search them for the research agent.

    python -m backend.news_research.vector_store build
    python -m backend.news_research.vector_store search "Nvidia China export ban" \
        --etf SMH --until 2025-04-30

Embeddings come from ChromaDB's built-in all-MiniLM-L6-v2 (ONNX, CPU only,
downloaded once to ~/.cache/chroma). The database lives in
data/processed/news/chroma and is git-ignored. Building skips chunk ids that
are already stored, so an interrupted build resumes. Search can filter by ETF,
source and date range, so a backtest on day D only sees news dated up to D.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = Path(__file__).resolve().parents[2] / "data/processed/news/chroma"
CHUNKS_PATH = Path("data/processed/news/rag/chunks.jsonl")
COLLECTION = "financial_news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


def open_collection(
    db_path: Path = DB_PATH, embedding_function=None, client=None, create=True
) -> chromadb.Collection:
    client = client or chromadb.PersistentClient(path=str(db_path))
    if not create:
        return client.get_collection(
            COLLECTION,
            embedding_function=embedding_function or DefaultEmbeddingFunction(),
        )
    return client.get_or_create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL},
        embedding_function=embedding_function or DefaultEmbeddingFunction(),
    )


def _batches(path: Path, size: int):
    batch = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                batch.append(json.loads(line))
                if len(batch) == size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def build(
    collection: chromadb.Collection, chunks_path: Path, batch_size: int = 256
) -> tuple[int, int]:
    """Embed and store every chunk not stored yet; returns (added, skipped)."""
    added = skipped = 0
    started = time.monotonic()
    for n, batch in enumerate(_batches(chunks_path, batch_size), 1):
        stored = set(collection.get(ids=[c["id"] for c in batch], include=[])["ids"])
        new = [c for c in batch if c["id"] not in stored]
        skipped += len(batch) - len(new)
        if new:
            collection.add(
                ids=[c["id"] for c in new],
                documents=[c["text"] for c in new],
                metadatas=[c["metadata"] for c in new],
            )
            added += len(new)
        if n % 40 == 0:
            rate = added / (time.monotonic() - started)
            logger.info(
                "조각 %d개 저장, %d개 건너뜀 (초당 %.0f개)", added, skipped, rate
            )
    return added, skipped


def where_filter(
    etf: str | None = None,
    since: str | None = None,
    until: str | None = None,
    sources: list[str] | None = None,
) -> dict | None:
    """ChromaDB metadata filter; dates are inclusive YYYY-MM-DD."""
    conditions: list[dict] = []
    if etf:
        conditions.append({f"etf_{etf}": True})
    if since:
        conditions.append({"date_int": {"$gte": int(since.replace("-", ""))}})
    if until:
        conditions.append({"date_int": {"$lte": int(until.replace("-", ""))}})
    if sources:
        conditions.append({"source": {"$in": list(sources)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(
    collection: chromadb.Collection, query: str, k: int = 5, **filters
) -> list[dict]:
    """Top-k chunks for a query; distance is cosine (0 = same meaning)."""
    result = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter(**filters),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": i, "text": text, "metadata": meta, "distance": distance}
        for i, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChromaDB 뉴스 벡터 저장소")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="조각을 임베딩해 저장 (이어서 가능)")
    b.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    b.add_argument("--batch-size", type=int, default=256)
    s = sub.add_parser("search", help="질문과 의미가 가까운 뉴스 찾기")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--etf")
    s.add_argument("--since", help="YYYY-MM-DD (포함)")
    s.add_argument("--until", help="YYYY-MM-DD (포함)")
    s.add_argument("--sources", nargs="*")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    collection = open_collection(args.db)
    if args.command == "build":
        added, skipped = build(collection, args.chunks, args.batch_size)
        logger.info(
            "완료: 새로 저장 %d개, 이미 있음 %d개, 전체 %d개",
            added,
            skipped,
            collection.count(),
        )
        return 0
    hits = search(
        collection,
        args.query,
        args.k,
        etf=args.etf,
        since=args.since,
        until=args.until,
        sources=args.sources,
    )
    sys.stdout.reconfigure(encoding="utf-8")
    for hit in hits:
        meta = hit["metadata"]
        print(
            f"[{hit['distance']:.3f}] {meta['date']} {meta['source']} | {meta['title']}"
        )
        print(f"        {meta['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
