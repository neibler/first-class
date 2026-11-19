"""HTTP-only client for risk, rebalancing and per-asset history. No local math."""

import math
from datetime import date

import httpx

from frontend.api_client import APIError, get_api_base_url


def _get(path, params, message):
    try:
        response = httpx.get(get_api_base_url() + path, params=params, timeout=20.0)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        raise APIError(message) from None


def _date(value):
    """A YYYY-MM-DD string, the format every insights API date uses."""
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _finite(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def load_risk(observations=None):
    """Server VaR/CVaR. Loss sizes arrive positive, as the server defines them."""
    message = "위험 지표(VaR·CVaR)를 불러오지 못했습니다. 서버와 모델 준비 상태를 확인해 주세요."
    params = {} if observations is None else {"observations": observations}
    data = _get("/risk", params, message)
    try:
        for side in (data["portfolio"], data["benchmark_risk"]):
            for key in ("var", "cvar"):
                value = side[key]
                if value is not None and not _finite(value):
                    raise ValueError("Invalid risk value")
        if not 0 < data["level"] < 1:
            raise ValueError("Invalid level")
    except (KeyError, TypeError, ValueError):
        raise APIError(message) from None
    return data


def load_rebalancing(observations=None, limit=30):
    message = (
        "비중 변화 기록을 불러오지 못했습니다. 서버와 모델 준비 상태를 확인해 주세요."
    )
    params = {"limit": limit}
    if observations is not None:
        params["observations"] = observations
    data = _get("/rebalancing", params, message)
    try:
        for event in data["events"]:
            for item in event["changes"]:
                if not _finite(item["before"], item["after"], item["change"]):
                    raise ValueError("Invalid weight change")
                if not (0 <= item["before"] <= 1 and 0 <= item["after"] <= 1):
                    raise ValueError("Weight out of range")
        if not _finite(data["average_turnover"]):
            raise ValueError("Invalid turnover")
    except (KeyError, TypeError, ValueError):
        raise APIError(message) from None
    return data


def load_asset_history(ticker, observations=252):
    message = "자산 가격·지표 흐름을 불러오지 못했습니다. 서버와 모델 준비 상태를 확인해 주세요."
    data = _get(f"/assets/{ticker}/history", {"observations": observations}, message)
    try:
        if data["ticker"] != ticker or not data["rows"]:
            raise ValueError("Unexpected asset history")
        if not _date(data["as_of"]) or not isinstance(data["observations"], int):
            raise ValueError("Invalid asset history metadata")
        for row in data["rows"]:
            if not _date(row["date"]):
                raise ValueError("Missing asset history date")
            if not _finite(row["cumulative_return"], row["rsi"], row["macd"]):
                raise ValueError("Invalid asset history value")
            close = row["close"]
            if close is not None and not (_finite(close) and close > 0):
                raise ValueError("Invalid close price")
    except (KeyError, TypeError, ValueError):
        raise APIError(message) from None
    return data
