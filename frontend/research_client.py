"""HTTP adapter for the research service; never handles provider credentials."""

import os
import httpx


class ResearchAPIError(ValueError):
    pass


def _request(method, path, **kwargs):
    base = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.request(
            method, base + path, timeout=300.0 if method == "POST" else 5.0, **kwargs
        )
        if response.status_code in (409, 503, 502):
            raise ResearchAPIError(
                response.json().get("detail", "리서치를 사용할 수 없습니다.")
            )
        response.raise_for_status()
        return response.json()
    except ResearchAPIError:
        raise
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ResearchAPIError(
            "리서치 서버 응답을 받지 못했습니다. 서버 상태를 확인하세요."
        ) from exc


def status():
    return _request("GET", "/research/status")


def research(question, as_of, ticker=None):
    return _request(
        "POST",
        "/research",
        json={"question": question, "as_of": as_of, "ticker": ticker},
    )
