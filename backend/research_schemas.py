"""HTTP contracts for the existing news research workflow."""

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Reference = Annotated[int, Field(gt=0, strict=True)]


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    as_of: date

    @field_validator("question", mode="before")
    @classmethod
    def trim_question(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("as_of", mode="before")
    @classmethod
    def require_iso_date(cls, value):
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Use YYYY-MM-DD.")
        return value


class ResearchSource(BaseModel):
    n: Reference
    title: str
    source: str
    date: str
    # Keep original citation URLs verbatim (including their escaping).
    url: str = Field(min_length=1)


class ResearchRiskTag(BaseModel):
    ticker: str
    risk_type: Literal[
        "recession",
        "inflation",
        "interest_rate",
        "regulation",
        "geopolitical",
        "earnings",
        "market_risk",
        "credit_risk",
        "commodity_risk",
        "currency_risk",
    ]
    risk_score: Score
    risk_level: Score
    direction: Literal["increase", "decrease", "neutral"]
    confidence: Score
    date: date
    evidence: list[Reference] = Field(min_length=1)


class ResearchPlanStep(BaseModel):
    query: str
    etf: str
    lookback_days: int


class ResearchResponse(BaseModel):
    status: Literal["complete", "insufficient_evidence"]
    question: str
    as_of: date
    answer: str = Field(min_length=1)
    sources: list[ResearchSource]
    risk_tags: list[ResearchRiskTag]
    plan: list[ResearchPlanStep]
    log: list[str]
    disclaimer: str

    @model_validator(mode="after")
    def check_citations(self):
        numbers = {source.n for source in self.sources}
        if len(numbers) != len(self.sources):
            raise ValueError("Duplicate source numbers.")
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", self.answer)}
        for tag in self.risk_tags:
            cited.update(tag.evidence)
        if cited - numbers:
            raise ValueError("Citation without a source.")
        if self.status == "complete" and not numbers:
            raise ValueError("Completed research needs sources.")
        if self.status == "insufficient_evidence" and (numbers or self.risk_tags):
            raise ValueError("No-evidence result must have empty evidence.")
        return self


class ResearchErrorDetail(BaseModel):
    code: str
    message: str


class ResearchErrorResponse(BaseModel):
    detail: ResearchErrorDetail
