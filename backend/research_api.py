"""Research HTTP route; no model initialization at import time."""

from fastapi import APIRouter, HTTPException

from backend.research_schemas import (
    ResearchErrorResponse,
    ResearchRequest,
    ResearchResponse,
)
from backend.research_service import ResearchFailure, run_research

router = APIRouter()


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
