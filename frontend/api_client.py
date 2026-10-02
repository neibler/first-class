"""HTTP-only access to the API; no backend or model imports."""

import os
from typing import Literal

import httpx
from pydantic import BaseModel, StrictBool, ValidationError


class HealthStatus(BaseModel):
    status: Literal["ok"]
    model_loaded: StrictBool


class APIError(Exception):
    """A user-displayable API connection or response error."""


def get_health() -> HealthStatus:
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
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
