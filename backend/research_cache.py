"""Two-step cache for /research answers: exact request, then a very similar question.

1. Exact: the normalized question + as_of + news DB stamp + agent settings hash
   is the key. Same request -> the stored answer at once, no Claude call.
2. Similar: within the same as_of, DB and settings, a question whose embedding
   (the news DB's own local all-MiniLM-L6-v2) has cosine similarity >= the
   threshold reuses that answer, marked so the page can offer a fresh run.

Stored with diskcache (SQLite on disk): survives restarts, needs no extra
server, expires after a week. Only completed research is stored. Any cache
problem is logged and ignored; research then runs as before.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILES = ("config/agent.yaml", "config/etf_news_map.yaml")
TTL_SECONDS = 7 * 24 * 3600
SIZE_LIMIT = 100 * 1024 * 1024
MAX_INDEX = 200  # similar-question candidates kept per (as_of, DB, settings)
SIMILARITY = float(os.environ.get("RESEARCH_SIMILARITY", "0.95"))


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def context_key(db_path: Path) -> str | None:
    """Changes whenever the news DB file or the agent settings change."""
    try:
        stat = (Path(db_path) / "chroma.sqlite3").stat()
        settings = b"".join((ROOT / name).read_bytes() for name in SETTINGS_FILES)
    except OSError:
        return None
    return _digest(
        f"{stat.st_size}-{stat.st_mtime_ns}", hashlib.sha256(settings).hexdigest()
    )


class ResearchCache:
    def __init__(self, directory: Path, embed=None, threshold: float = SIMILARITY):
        import diskcache

        self.store = diskcache.Cache(str(directory), size_limit=SIZE_LIMIT)
        self.embed = embed  # text -> vector, or None to skip the similar step
        self.threshold = threshold

    def _vector(self, question: str):
        if self.embed is None:
            return None
        try:
            vector = np.asarray(self.embed(question), dtype=float)
        except Exception:  # embedding model missing or failing: exact step only
            logger.warning("research_cache_embedding_unavailable")
            return None
        norm = np.linalg.norm(vector)
        return vector / norm if norm else None

    def lookup(self, question: str, as_of: str, context: str):
        """(payload, "exact" | "similar", stored question) or None."""
        bucket = _digest(as_of, context)
        exact = self.store.get("answer:" + _digest(normalize(question), bucket))
        if exact is not None:
            return exact["payload"], "exact", exact["question"]
        vector = self._vector(question)
        if vector is None:
            return None
        best, score = None, -1.0
        for entry in self.store.get("index:" + bucket, []):
            similarity = float(np.dot(vector, np.asarray(entry["vector"])))
            if similarity > score:
                best, score = entry, similarity
        if best is None or score < self.threshold:
            return None
        found = self.store.get("answer:" + best["key"])
        if found is None:
            return None
        return found["payload"], "similar", found["question"]

    def save(self, question: str, as_of: str, context: str, payload: dict) -> None:
        bucket = _digest(as_of, context)
        key = _digest(normalize(question), bucket)
        self.store.set(
            "answer:" + key,
            {"question": question, "payload": payload},
            expire=TTL_SECONDS,
        )
        vector = self._vector(question)
        if vector is None:
            return
        with self.store.transact():
            index = [
                e for e in self.store.get("index:" + bucket, []) if e["key"] != key
            ]
            index.append({"key": key, "question": question, "vector": vector.tolist()})
            self.store.set("index:" + bucket, index[-MAX_INDEX:], expire=TTL_SECONDS)


_cache = None


def local_embedding():
    """The news DB's own embedding model; never downloads (raises if missing)."""
    from backend.news_research.vector_store import LocalEmbedding

    model = LocalEmbedding()
    return lambda text: model([text])[0]


def get_cache() -> ResearchCache | None:
    """Process-wide cache, or None when diskcache or the folder is unusable."""
    global _cache
    if _cache is None:
        directory = Path(
            os.environ.get("RESEARCH_CACHE_DIR") or ROOT / "results" / "research_cache"
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            try:
                embed = local_embedding()
            except Exception:
                embed = None
            _cache = ResearchCache(directory, embed)
        except Exception:
            logger.warning("research_cache_unavailable")
            return None
    return _cache
