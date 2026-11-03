"""HTTP-only access to the API; no backend or model imports."""

import os
import re
from datetime import date
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field, StrictBool, ValidationError, model_validator


class HealthStatus(BaseModel):
    status: Literal["ok"]
    model_loaded: StrictBool


class APIError(Exception):
    """A user-displayable API connection or response error."""


# Backend: 90-second cooperative budget, 20 seconds per LLM call.
# This is a network read timeout, not a server cancellation mechanism.
RESEARCH_TIMEOUT = httpx.Timeout(120.0, connect=5.0, write=10.0, pool=5.0)
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Reference = Annotated[int, Field(gt=0, strict=True)]


class ResearchSource(BaseModel):
    n: Reference
    title: str
    source: str
    date: str
    url: str = Field(min_length=1)


class ResearchRiskTag(BaseModel):
    ticker: str
    risk_type: str
    risk_score: Score
    risk_level: Score
    direction: Literal["increase", "decrease", "neutral"]
    confidence: Score
    date: date
    evidence: list[Reference] = Field(min_length=1)


class ResearchPlan(BaseModel):
    query: str
    etf: str
    lookback_days: int


class ResearchResult(BaseModel):
    """HTTP response contract only; the UI never imports backend modules."""

    status: Literal["complete", "insufficient_evidence"]
    question: str
    as_of: date
    answer: str = Field(min_length=1)
    sources: list[ResearchSource]
    risk_tags: list[ResearchRiskTag]
    plan: list[ResearchPlan]
    log: list[str]
    disclaimer: str

    @model_validator(mode="after")
    def check_evidence(self):
        numbers = {source.n for source in self.sources}
        references = {int(n) for n in re.findall(r"\[(\d+)\]", self.answer)}
        for tag in self.risk_tags:
            references.update(tag.evidence)
        if len(numbers) != len(self.sources) or references - numbers:
            raise ValueError("Invalid citation references")
        if self.status == "complete" and not numbers:
            raise ValueError("Missing sources")
        if self.status == "insufficient_evidence" and (numbers or self.risk_tags):
            raise ValueError("Inconsistent evidence status")
        return self


RESEARCH_ERRORS = {
    "invalid_research_request": "질문은 앞뒤 공백 정리 후 1~2,000자로 입력하고 기준일을 선택해 주세요.",
    "research_key_missing": "서버에 리서치 API 키가 설정되지 않았습니다. 서버 담당자에게 설정을 요청해 주세요.",
    "research_db_missing": "서버에 뉴스 DB가 준비되지 않았습니다. 서버 담당자에게 DB 경로를 확인해 주세요.",
    "research_db_unavailable": "서버에서 뉴스 DB를 사용할 수 없습니다. 서버 담당자에게 확인해 주세요.",
    "research_embedding_unavailable": "서버의 검색용 임베딩 모델이 준비되지 않았습니다. 서버 담당자에게 확인해 주세요.",
    "research_unavailable": "서버의 리서치 실행 환경이 준비되지 않았습니다. 서버 담당자에게 확인해 주세요.",
    "research_busy": "다른 리서치가 실행 중입니다. 잠시 후 분석 버튼을 다시 눌러 주세요.",
    "research_upstream_error": "외부 분석 서비스 요청에 실패했습니다. 잠시 후 다시 실행하거나 서버 담당자에게 확인해 주세요.",
    "research_invalid_result": "분석 결과를 검증하지 못했습니다. 질문을 조정하거나 서버 담당자에게 확인해 주세요.",
    "research_timeout": "서버의 리서치 처리 시간이 초과되었습니다. 잠시 후 다시 실행해 주세요.",
    "research_failed": "리서치 처리 중 오류가 발생했습니다. 서버 담당자에게 확인해 주세요.",
    "connection_failed": "API에 연결할 수 없습니다. 서버 실행 상태와 API 주소를 확인해 주세요.",
    "client_timeout": "응답 대기 시간이 초과되었습니다. 서버 분석은 계속될 수 있으므로 잠시 기다린 뒤 다시 실행해 주세요.",
    "invalid_response": "서버의 리서치 응답 형식이 올바르지 않습니다. 서버 담당자에게 확인해 주세요.",
    "invalid_api_url": "API 주소 설정이 올바르지 않습니다. 서버 담당자에게 확인해 주세요.",
    "http_error": "리서치 요청을 처리하지 못했습니다. 서버 담당자에게 확인해 주세요.",
}


class ResearchAPIError(APIError):
    def __init__(self, code: str):
        self.code = code if code in RESEARCH_ERRORS else "http_error"
        super().__init__(RESEARCH_ERRORS[self.code])


def safe_source_url(value: str) -> str | None:
    """Only explicit HTTP(S) links; reject credentials and ambiguous URL syntax."""
    if any(char.isspace() or ord(char) < 32 or char in "\\<>\x7f" for char in value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.lower() in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
        ):
            parsed.port  # Validate malformed ports without changing the original URL.
            return value
    except ValueError:
        pass
    return None


def get_api_base_url() -> str:
    """Use the same endpoint identity for HTTP calls and session reuse."""
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def post_research(question: str, as_of: str | None) -> ResearchResult:
    """One explicit POST, no retry or fallback. Input validation belongs to the API."""
    question = question.strip()
    base_url = get_api_base_url()
    try:
        response = httpx.post(
            f"{base_url}/research",
            json={"question": question, "as_of": as_of},
            timeout=RESEARCH_TIMEOUT,
            follow_redirects=False,
        )
        if response.status_code != 200:
            code = (
                "invalid_research_request"
                if response.status_code == 422
                else "http_error"
            )
            try:
                body = response.json()
                detail = body.get("detail") if isinstance(body, dict) else None
                candidate = detail.get("code") if isinstance(detail, dict) else None
                if isinstance(candidate, str) and candidate in RESEARCH_ERRORS:
                    code = candidate
            except ValueError:
                pass
            raise ResearchAPIError(code)
        result = ResearchResult.model_validate(response.json())
        if result.question != question or result.as_of.isoformat() != as_of:
            raise ValueError("Research response does not match the request")
        return result
    except httpx.TimeoutException:
        raise ResearchAPIError("client_timeout") from None
    except httpx.InvalidURL:
        raise ResearchAPIError("invalid_api_url") from None
    except httpx.HTTPError:
        raise ResearchAPIError("connection_failed") from None
    except (ValueError, ValidationError):
        raise ResearchAPIError("invalid_response") from None


def get_health() -> HealthStatus:
    base_url = get_api_base_url()
    try:
        response = httpx.get(f"{base_url}/health", timeout=3.0)
        response.raise_for_status()
        return HealthStatus.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise APIError("API 응답 시간이 초과되었습니다.") from exc
    except httpx.HTTPError as exc:
        raise APIError("API에 연결할 수 없습니다. 서버 상태를 확인해 주세요.") from exc
    except (ValueError, ValidationError, httpx.InvalidURL) as exc:
        raise APIError("API 주소 또는 상태 응답 형식이 올바르지 않습니다.") from exc
