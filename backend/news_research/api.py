"""Optional news research endpoints, independent of the investment model."""

from datetime import date
from importlib.util import find_spec
import logging
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from backend.news_research.settings import env_value

PROJECT = Path(__file__).resolve().parents[2]
router = APIRouter(prefix="/research", tags=["research"])
request_lock = Lock()
logger = logging.getLogger(__name__)


def database_path():
    configured = env_value("NEWS_CHROMA_PATH", PROJECT / ".env")
    path = Path(configured) if configured else PROJECT / "data/processed/news/chroma"
    return path if path.is_absolute() else PROJECT / path


def readiness():
    dependencies = all(
        find_spec(name) is not None for name in ("anthropic", "chromadb", "langgraph")
    )
    key = bool(env_value("ANTHROPIC_API_KEY", PROJECT / ".env"))
    database = (database_path() / "chroma.sqlite3").is_file()
    missing = []
    if not dependencies:
        missing.append(
            "리서치 패키지를 설치하세요: pip install -r backend/news_research/requirements.txt"
        )
    if not database:
        missing.append(
            "공유 뉴스 DB를 data/processed/news/chroma에 넣거나 NEWS_CHROMA_PATH를 설정하세요."
        )
    if not key:
        missing.append("서버의 .env에 ANTHROPIC_API_KEY를 설정하세요.")
    return {
        "configured": dependencies and database and key,
        "dependencies_available": dependencies,
        "database_available": database,
        "api_key_configured": key,
        "messages": missing,
        "note": "설정 확인이며 API 키 유효성·모델 접근권한·DB 검색 성공을 보장하지 않습니다.",
    }


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000)
    as_of: date
    ticker: str | None = None

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value):
        if value is not None:
            import yaml

            assets = yaml.safe_load(
                (PROJECT / "config/etf_news_map.yaml").read_text(encoding="utf-8")
            )["etfs"]
            value = value.upper()
            if value not in assets:
                raise ValueError("Unknown ETF")
        return value


class ResearchSource(BaseModel):
    n: int
    title: str
    source: str
    date: str
    url: HttpUrl


class ResearchRiskTag(BaseModel):
    ticker: str
    risk_type: str
    risk_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_level: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    direction: str
    date: str
    evidence: list[int]


class ResearchResponse(BaseModel):
    question: str
    as_of: str
    answer: str
    risk_tags: list[ResearchRiskTag]
    sources: list[ResearchSource]
    plan: list[dict]
    log: list[str]
    disclaimer: str


def run_research(request):
    from backend.news_research.research_agent import default_search, research

    search = default_search(database_path())

    def filtered_search(query, k, **filters):
        # A selected ticker remains in force even for the agent's fallback search.
        if request.ticker:
            filters["etf"] = request.ticker
        return search(query, k, **filters)

    return research(request.question, request.as_of.isoformat(), search=filtered_search)


@router.get("/status")
def research_status():
    return readiness()


@router.post("", response_model=ResearchResponse)
def research_endpoint(request: ResearchRequest):
    status = readiness()
    if not status["configured"]:
        raise HTTPException(503, " ".join(status["messages"]))
    if not request_lock.acquire(blocking=False):
        raise HTTPException(
            409, "리서치 요청이 진행 중입니다. 완료 후 다시 시도하세요."
        )
    try:
        # Validate before returning; never expose provider errors to the client.
        return ResearchResponse.model_validate(run_research(request))
    except Exception as exc:
        logger.warning("RESEARCH_FAILED: %s", type(exc).__name__)
        raise HTTPException(
            502,
            "리서치를 완료하지 못했습니다. 서버의 API 키·모델 설정·뉴스 DB를 확인하세요.",
        ) from exc
    finally:
        request_lock.release()
