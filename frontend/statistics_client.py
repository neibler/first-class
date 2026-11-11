"""HTTP contract for saved ANOVA results; no analysis or model imports."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from frontend.api_client import APIError, get_api_base_url

Number = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[Number, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]
TIMEOUT = httpx.Timeout(10.0, connect=3.0)


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class TestValues(Contract):
    f_statistic: Number | None
    p_value: Probability | None
    eta_squared: Probability | None
    significant: bool
    effect_size_label: str | None = None
    partial_eta_squared: Probability | None = None


class TukeyPair(Contract):
    group1: str
    group2: str
    mean_difference: Number
    p_adjusted: Probability
    lower: Number
    upper: Number
    significant: bool


class Tukey(Contract):
    test: Literal["tukey_hsd"]
    alpha: Probability
    value: Literal["period_return"]
    pairs: list[TukeyPair]


class Assumptions(Contract):
    shapiro_p_value: dict[str, Probability | None] = Field(default_factory=dict)
    levene_p_value: Probability | None = None
    note: str | None = None


class Nonparametric(Contract):
    test: Literal["kruskal_wallis"]
    h_statistic: Number | None
    p_value: Probability | None


class Independence(Contract):
    available: bool
    note: str | None = None
    rows_per_group: dict[str, Count] = Field(default_factory=dict)
    distinct_dates_per_group: dict[str, Count] = Field(default_factory=dict)
    duplicated_group_date_rows: Count | None = None
    dates_shared_across_groups: Count | None = None


class OneWay(TestValues):
    test: Literal["one_way_anova"]
    label: str
    groups_label: str
    groups: list[str] = Field(min_length=2)
    n_per_group: dict[str, Count]
    group_mean: dict[str, Number | None]
    group_std: dict[str, Number | None]
    posthoc: Tukey | None = None
    interpretation: str | None = None
    sample_source: str | None = None
    assumptions: Assumptions | None = None
    nonparametric: Nonparametric | None = None
    independence: Independence | None = None

    @model_validator(mode="after")
    def check_groups(self):
        groups = set(self.groups)
        if len(groups) != len(self.groups) or any(
            set(values) != groups
            for values in (self.n_per_group, self.group_mean, self.group_std)
        ):
            raise ValueError("Inconsistent ANOVA groups")
        return self


class TwoWay(Contract):
    test: Literal["two_way_anova"]
    label: str
    value: Literal["period_return"]
    factors: list[str] = Field(min_length=2, max_length=2)
    terms: dict[str, TestValues] = Field(min_length=1)
    interaction_included: bool
    n: Count
    cell_counts: dict[str, Count]
    posthoc: dict[str, Tukey] | None = None
    posthoc_note: str | None = None
    interpretation: str | None = None

    @model_validator(mode="after")
    def check_factors(self):
        if len(set(self.factors)) != 2 or not set(self.factors) <= self.terms.keys():
            raise ValueError("Missing ANOVA factor results")
        return self


class Envelope(Contract):
    results: list[dict[str, Any]]


@dataclass
class StatisticsResult:
    results: list[OneWay | TwoWay]
    unsupported_count: int = 0


class StatisticsAPIError(APIError):
    """Only safe, user-facing messages; never include response bodies."""


def parse_statistics(data: Any) -> StatisticsResult:
    payload = Envelope.model_validate(data)
    results = []
    unsupported = 0
    for item in payload.results:
        kind = item.get("test")
        if not isinstance(kind, str):
            raise ValueError("Missing test type")
        if kind == "one_way_anova":
            results.append(OneWay.model_validate(item))
        elif kind == "two_way_anova":
            results.append(TwoWay.model_validate(item))
        else:
            unsupported += 1
    return StatisticsResult(results, unsupported)


def get_statistics() -> StatisticsResult:
    """One GET per page execution, with no cache, retries or fallback data."""
    try:
        response = httpx.get(
            f"{get_api_base_url()}/performance/statistics",
            timeout=TIMEOUT,
            follow_redirects=False,
        )
        if response.status_code == 503:
            raise StatisticsAPIError(
                "통계 분석 결과가 아직 준비되지 않았습니다. "
                "서버의 분석 결과 연결 상태를 확인해 주세요."
            )
        if response.status_code != 200:
            raise StatisticsAPIError(
                "통계 결과를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
            )
        return parse_statistics(response.json())
    except httpx.TimeoutException as exc:
        raise StatisticsAPIError(
            "통계 API 응답 시간이 초과되었습니다. 다시 시도해 주세요."
        ) from exc
    except httpx.RequestError as exc:
        raise StatisticsAPIError(
            "통계 API에 연결할 수 없습니다. 서버 연결 상태를 확인해 주세요."
        ) from exc
    except (ValidationError, ValueError, httpx.InvalidURL) as exc:
        raise StatisticsAPIError(
            "통계 응답 형식 또는 API 주소가 올바르지 않습니다. "
            "관리자에게 확인을 요청해 주세요."
        ) from exc
