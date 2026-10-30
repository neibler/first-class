"""Minimal API serving; data collection and model training are not startup tasks."""

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.research_api import router as research_router

app = FastAPI(title="First Class Robo-Advisor")
app.include_router(research_router)


@app.exception_handler(RequestValidationError)
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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report server liveness; model loading is not implemented yet."""
    return HealthResponse(status="ok", model_loaded=False)
