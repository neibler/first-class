"""HTTP-only market relationship client. No synthetic fallback."""

import math
import httpx
from frontend.api_client import APIError, get_api_base_url


def load_correlation_matrix(window=60, weights=None):
    """ETF x ETF correlations and, with weights, the holdings' average one."""
    params = {"window": window}
    if weights:
        params["weights"] = ",".join(f"{t}:{w:.6f}" for t, w in weights.items())
    try:
        response = httpx.get(
            get_api_base_url() + "/performance/correlation-matrix",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        size = len(data["tickers"])
        if data["window"] != window or len(data["matrix"]) != size:
            raise ValueError("Invalid correlation matrix")
        for row in data["matrix"]:
            if len(row) != size:
                raise ValueError("Invalid correlation matrix")
            for value in row:
                if value is not None and not (
                    math.isfinite(value) and -1 <= value <= 1
                ):
                    raise ValueError("Invalid correlation")
        mix = data["portfolio"]
        if mix is not None and not -1 <= mix["average_correlation"] <= 1:
            raise ValueError("Invalid portfolio correlation")
        return data
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError, TypeError):
        raise APIError(
            "ETF 관계 지도를 불러오지 못했습니다. API 서버와 실제 시장 데이터 설정을 확인하세요."
        ) from None


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
