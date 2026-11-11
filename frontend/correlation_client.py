"""HTTP-only market relationship client. No synthetic fallback."""

import math
import httpx
from frontend.api_client import APIError, get_api_base_url


def load_correlations(ticker, window=60):
    try:
        response = httpx.get(
            get_api_base_url() + f"/performance/correlation/{ticker}",
            params={"window": window, "top_k": 5},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if (
            data["ticker"] != ticker
            or data["window"] != window
            or data["status"] not in {"ok", "insufficient_data"}
        ):
            raise ValueError("Invalid correlation response")
        for row in data["all_relationships"]:
            value = row["correlation"]
            if (
                not math.isfinite(value)
                or not -1 <= value <= 1
                or row["name"] == ticker
            ):
                raise ValueError("Invalid correlation")
        return data
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise APIError(
            "자산 관계를 불러오지 못했습니다. API 서버와 실제 시장 데이터 설정을 확인하세요."
        ) from None
