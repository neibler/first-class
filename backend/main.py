"""Minimal API serving; data collection and model training are not startup tasks."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="First Class Robo-Advisor")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report server liveness; model loading is not implemented yet."""
    return HealthResponse(status="ok", model_loaded=False)
