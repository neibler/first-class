"""Lazy adapter: synchronous work runs in FastAPI's worker thread pool."""

import logging
import threading
import time
from pathlib import Path

from pydantic import ValidationError

from backend.research_schemas import ResearchRequest, ResearchResponse

ROOT = Path(__file__).resolve().parents[1]
CALL_TIMEOUT = 20.0
RESEARCH_BUDGET = 90.0
# Chroma may reuse process-local internals even with new client instances.
# Serialize the entire workflow; never queue more paid work behind it.
_research_lock = threading.Lock()
logger = logging.getLogger(__name__)


class ResearchFailure(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(code)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ResearchFailure(504, "research_timeout", "리서치 시간이 초과됐습니다.")
    return remaining


class _BoundedLLM:
    def __init__(self, client, agent, settings, deadline):
        self.client, self.agent = client, agent
        self.settings, self.deadline = settings, deadline

    def json(self, role, system, user, schema):
        timeout = min(CALL_TIMEOUT, _remaining(self.deadline))
        client = self.client.with_options(timeout=timeout, max_retries=0)
        llm = self.agent.ClaudeLLM(
            client, self.settings["models"], self.settings["effort"]
        )
        result = llm.json(role, system, user, schema)
        _remaining(self.deadline)
        return result


_coverage_cache: dict = {}


def news_coverage(db_path: Path) -> tuple[str, str] | None:
    """(first, last) news date in the DB, from Chroma's indexed date_int metadata.

    A read-only SQLite min/max (well under a second on the 200k-chunk DB),
    cached until chroma.sqlite3 changes. None when it cannot be read, so a
    coverage problem never blocks research on its own.
    """
    import sqlite3

    sqlite = Path(db_path) / "chroma.sqlite3"
    try:
        stat = sqlite.stat()
    except OSError:
        return None
    key = (str(sqlite), stat.st_mtime_ns, stat.st_size)
    # One read, no membership-then-index: another thread may clear the dict.
    cached = _coverage_cache.get(key)
    if cached is not None:
        return cached
    try:
        with sqlite3.connect(f"{sqlite.as_uri()}?mode=ro", uri=True) as connection:
            low, high = connection.execute(
                "SELECT min(int_value), max(int_value) FROM embedding_metadata "
                "WHERE key = 'date_int' AND int_value IS NOT NULL"
            ).fetchone()
    except sqlite3.Error:
        return None
    if low is None or high is None:
        return None

    def iso(value: int) -> str:
        text = str(value)
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

    found = (iso(low), iso(high))
    _coverage_cache.clear()
    _coverage_cache[key] = found
    return found


def coverage() -> dict:
    """Date range the AI assistant can answer for; raises ResearchFailure."""
    try:
        from backend.news_research.settings import news_db_path
    except ImportError:
        raise ResearchFailure(
            503, "research_unavailable", "리서치 실행 환경이 준비되지 않았습니다."
        ) from None
    db_path = news_db_path()
    if not (db_path / "chroma.sqlite3").is_file():
        raise ResearchFailure(
            503, "research_db_missing", "리서치 DB가 준비되지 않았습니다."
        )
    found = news_coverage(db_path)
    if found is None:
        raise ResearchFailure(
            503, "research_db_unavailable", "리서치 DB를 사용할 수 없습니다."
        )
    return {"start": found[0], "end": found[1]}


def _run(request: ResearchRequest) -> ResearchResponse:
    # Kept inside the request so the base API image works without RAG packages.
    try:
        from backend.news_research.settings import env_value, news_db_path
    except ImportError:
        raise ResearchFailure(
            503, "research_unavailable", "리서치 실행 환경이 준비되지 않았습니다."
        ) from None
    api_key = env_value("ANTHROPIC_API_KEY", ROOT / ".env")
    if not api_key:
        raise ResearchFailure(
            503, "research_key_missing", "리서치 API 키가 설정되지 않았습니다."
        )
    db_path = news_db_path()
    if not (db_path / "chroma.sqlite3").is_file():
        raise ResearchFailure(
            503, "research_db_missing", "리서치 DB가 준비되지 않았습니다."
        )
    # Before any paid call: a date outside the DB can only end in "no evidence".
    found = news_coverage(db_path)
    if found and not found[0] <= request.as_of.isoformat() <= found[1]:
        raise ResearchFailure(
            422,
            "research_out_of_range",
            f"기준일이 뉴스 DB 범위({found[0]} ~ {found[1]}) 밖입니다.",
        )
    try:
        import anthropic

        from backend.news_research import research_agent as agent
        from backend.news_research import vector_store
    except ImportError:
        raise ResearchFailure(
            503, "research_unavailable", "리서치 실행 환경이 준비되지 않았습니다."
        ) from None

    deadline = time.monotonic() + RESEARCH_BUDGET
    try:
        settings = agent.load_settings(ROOT / "config/agent.yaml")
        news_map = agent.load_settings(ROOT / "config/etf_news_map.yaml")
        search = agent.default_search(db_path, local_embeddings_only=True)

        def bounded_search(query, k, **filters):
            _remaining(deadline)
            try:
                result = search(query, k, **filters)
            except vector_store.EmbeddingUnavailable:
                raise
            except Exception:
                raise ResearchFailure(
                    503, "research_db_unavailable", "리서치 DB 검색에 실패했습니다."
                ) from None
            _remaining(deadline)
            return result

        with anthropic.Anthropic(
            api_key=api_key, timeout=CALL_TIMEOUT, max_retries=0
        ) as client:
            result = agent.research(
                request.question,
                request.as_of.isoformat(),
                llm=_BoundedLLM(client, agent, settings, deadline),
                search=bounded_search,
                settings=settings,
                news_map=news_map,
            )
        _remaining(deadline)
        response = ResearchResponse.model_validate(result)
        if response.question != request.question or response.as_of != request.as_of:
            raise ValueError("Research input mismatch.")
        return response
    except ResearchFailure:
        raise
    except vector_store.EmbeddingUnavailable:
        raise ResearchFailure(
            503,
            "research_embedding_unavailable",
            "로컬 임베딩 모델이 준비되지 않았습니다.",
        ) from None
    except vector_store.StoreUnavailable:
        raise ResearchFailure(
            503, "research_db_unavailable", "리서치 DB를 사용할 수 없습니다."
        ) from None
    except (anthropic.APITimeoutError, TimeoutError):
        raise ResearchFailure(
            504, "research_timeout", "리서치 시간이 초과됐습니다."
        ) from None
    except anthropic.APIError:
        raise ResearchFailure(
            502, "research_upstream_error", "외부 리서치 서비스 요청에 실패했습니다."
        ) from None
    except (agent.AgentError, ValidationError, ValueError, KeyError, TypeError):
        raise ResearchFailure(
            502, "research_invalid_result", "리서치 결과를 검증하지 못했습니다."
        ) from None


def _cache_context():
    """(cache, context key) or (None, None); research works without either."""
    try:
        from backend.news_research.settings import news_db_path
        from backend.research_cache import context_key, get_cache

        cache = get_cache()
        context = context_key(news_db_path()) if cache else None
    except Exception:
        logger.warning("research_cache_unavailable")
        return None, None
    return (cache, context) if context else (None, None)


def _cached_answer(request: ResearchRequest, cache, context):
    try:
        found = cache.lookup(request.question, request.as_of.isoformat(), context)
    except Exception:
        logger.warning("research_cache_read_failed")
        return None
    if found is None:
        return None
    payload, kind, stored_question = found
    try:
        # A similar question's answer is shown under the asked question, with the
        # stored question kept so the page can say which one it answers.
        return ResearchResponse.model_validate(
            {
                **payload,
                "question": request.question,
                "cached": kind,
                "cached_question": stored_question,
            }
        )
    except ValidationError:
        return None


def run_research(request: ResearchRequest) -> ResearchResponse:
    cache, context = _cache_context()
    if cache and not request.refresh:
        hit = _cached_answer(request, cache, context)
        if hit is not None:
            return hit
    response = _locked_research(request)
    if cache and response.status == "complete":
        try:
            payload = response.model_dump(
                mode="json", exclude={"cached", "cached_question"}
            )
            cache.save(request.question, request.as_of.isoformat(), context, payload)
        except Exception:
            logger.warning("research_cache_write_failed")
    return response


def _locked_research(request: ResearchRequest) -> ResearchResponse:
    if not _research_lock.acquire(blocking=False):
        raise ResearchFailure(
            503, "research_busy", "리서치가 진행 중입니다. 완료 후 다시 요청하세요."
        )
    try:
        return _run(request)
    except ResearchFailure:
        raise
    except Exception:
        # Never log raw exceptions, questions, credentials, or filesystem paths.
        logger.error("research_failed")
        raise ResearchFailure(
            500, "research_failed", "리서치 처리 중 오류가 발생했습니다."
        ) from None
    finally:
        _research_lock.release()
