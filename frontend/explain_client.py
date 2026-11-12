"""HTTP contract for saved SHAP explanations; no model or SHAP imports."""

import math
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from frontend.api_client import APIError, get_api_base_url

Number = Annotated[float, Field(allow_inf_nan=False)]
Unit = Annotated[float, Field(allow_inf_nan=False, ge=0, le=1)]
Day = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
TIMEOUT = httpx.Timeout(10.0, connect=3.0)
# Server sums float64 SHAP values; allow rounding, not a different decision.
ADDITIVITY_TOLERANCE = 1e-6


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class Importance(Contract):
    feature: str
    mean_abs_shap: Annotated[Number, Field(ge=0)]


class Overview(Contract):
    features: list[str] = Field(min_length=1)
    feature_importance: list[Importance] | None = None
    assets: list[str]
    note: str

    @model_validator(mode="after")
    def check_names(self):
        if len(set(self.features)) != len(self.features) or len(
            set(self.assets)
        ) != len(self.assets):
            raise ValueError("Duplicate features or assets")
        return self


class Decision(Contract):
    date: Day
    base_value: Number
    final_value: Number
    feature_values: list[Unit]
    shap_values: list[Number]
    feature_values_raw: list[Number] | None = None


class AssetExplanation(Contract):
    ticker: str
    name: str
    features: list[str] = Field(min_length=1)
    output_unit: Literal["target_weight"]
    note: str
    decisions: list[Decision] = Field(min_length=1)

    @model_validator(mode="after")
    def check_decisions(self):
        count = len(self.features)
        dates = [d.date for d in self.decisions]
        if len(set(dates)) != len(dates):
            raise ValueError("Duplicate decision dates")
        for decision in self.decisions:
            raw = decision.feature_values_raw
            if (
                len(decision.feature_values) != count
                or len(decision.shap_values) != count
                or (raw is not None and len(raw) != count)
            ):
                raise ValueError("Feature and SHAP lengths differ")
            if not math.isclose(
                decision.base_value + sum(decision.shap_values),
                decision.final_value,
                abs_tol=ADDITIVITY_TOLERANCE,
            ):
                raise ValueError("base + SHAP sum differs from final value")
        self.decisions.sort(key=lambda d: d.date)
        return self


class ExplainAPIError(APIError):
    """Only safe, user-facing messages; never include response bodies."""

    def __init__(self, message, missing=False):
        super().__init__(message)
        self.missing = missing


def _get(params=None):
    try:
        response = httpx.get(
            f"{get_api_base_url()}/performance/explain",
            params=params,
            timeout=TIMEOUT,
            follow_redirects=False,
        )
    except httpx.TimeoutException as exc:
        raise ExplainAPIError(
            "SHAP API 응답 시간이 초과되었습니다. 다시 시도해 주세요."
        ) from exc
    except httpx.InvalidURL as exc:
        raise ExplainAPIError(
            "API 주소가 올바르지 않습니다. 관리자에게 확인을 요청해 주세요."
        ) from exc
    except httpx.RequestError as exc:
        raise ExplainAPIError(
            "SHAP API에 연결할 수 없습니다. 서버 연결 상태를 확인해 주세요."
        ) from exc
    if response.status_code == 503:
        raise ExplainAPIError(
            "SHAP 해석 결과가 아직 준비되지 않았습니다. "
            "서버의 분석 결과 연결 상태를 확인해 주세요."
        )
    if response.status_code == 404 and params:
        raise ExplainAPIError(
            "선택한 자산의 SHAP 결과가 없습니다. 결과를 새로고침해 주세요.",
            missing=True,
        )
    if response.status_code != 200:
        raise ExplainAPIError(
            "SHAP 결과를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
        )
    try:
        return response.json()
    except ValueError as exc:
        raise _invalid() from exc


def _invalid():
    return ExplainAPIError(
        "SHAP 응답 형식이 올바르지 않습니다. 관리자에게 확인을 요청해 주세요."
    )


def parse_overview(data) -> Overview:
    return Overview.model_validate(data)


def parse_asset(data, ticker: str, features: list[str]) -> AssetExplanation:
    result = AssetExplanation.model_validate(data)
    if result.ticker != ticker or result.features != features:
        raise ValueError("Response does not match the requested asset")
    return result


def get_overview() -> Overview:
    """Feature names, optional importance and asset list: one GET, no cache."""
    data = _get()
    try:
        return parse_overview(data)
    except (ValidationError, ValueError) as exc:
        raise _invalid() from exc


def get_asset(ticker: str, features: list[str]) -> AssetExplanation:
    """All saved decisions for one asset; dates are chosen on the page."""
    data = _get({"ticker": ticker})
    try:
        return parse_asset(data, ticker, features)
    except (ValidationError, ValueError) as exc:
        raise _invalid() from exc
