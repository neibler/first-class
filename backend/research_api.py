"""Research HTTP route; no model initialization at import time."""

from fastapi import APIRouter, HTTPException, Request

from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.research_schemas import (
    ResearchErrorResponse,
    ResearchRequest,
    ResearchResponse,
)
from backend.research_service import ResearchFailure, coverage, run_research

router = APIRouter()


@router.get(
    "/research/coverage",
    responses={code: {"model": ResearchErrorResponse} for code in (503,)},
)
def research_coverage() -> dict:
    """First and last news date in the DB; requests outside it are refused."""
    try:
        return coverage()
    except ResearchFailure as exc:
        raise HTTPException(
            status_code=exc.status,
            detail={"code": exc.code, "message": exc.message},
        ) from None


@router.post(
    "/research",
    response_model=ResearchResponse,
    responses={
        code: {"model": ResearchErrorResponse} for code in (422, 500, 502, 503, 504)
    },
)
def research(request: ResearchRequest) -> ResearchResponse:
    """Return research and execution summaries, including no-evidence results."""
    try:
        return run_research(request)
    except ResearchFailure as exc:
        raise HTTPException(
            status_code=exc.status,
            detail={"code": exc.code, "message": exc.message},
        ) from None


async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == "/research":
        # Pydantic's default errors echo input, including forbidden secret fields.
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_research_request",
                    "message": "질문과 기준일(YYYY-MM-DD)을 확인하세요.",
                }
            },
        )
    return await request_validation_exception_handler(request, exc)
