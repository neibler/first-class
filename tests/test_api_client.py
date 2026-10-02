import httpx
import pytest

from frontend import api_client


def test_health_uses_configured_http_url_and_timeout(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://api:8000/")

    def get(url, *, timeout):
        assert url == "http://api:8000/health"
        assert timeout == 3.0
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"status": "ok", "model_loaded": False},
        )

    monkeypatch.setattr(httpx, "get", get)
    assert api_client.get_health().model_loaded is False


def test_health_defaults_to_local_api(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)

    def get(url, *, timeout):
        assert url == "http://127.0.0.1:8000/health"
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"status": "ok", "model_loaded": False},
        )

    monkeypatch.setattr(httpx, "get", get)
    assert api_client.get_health().status == "ok"


@pytest.mark.parametrize(
    "error,message",
    [
        (httpx.ConnectError("connection refused"), "연결할 수 없습니다"),
        (httpx.ReadTimeout("timeout"), "시간이 초과"),
        (httpx.InvalidURL("bad URL"), "올바르지 않습니다"),
    ],
)
def test_connection_errors_are_displayable(monkeypatch, error, message):
    def get(*args, **kwargs):
        raise error

    monkeypatch.setattr(httpx, "get", get)
    with pytest.raises(api_client.APIError, match=message):
        api_client.get_health()


def test_http_failure_is_not_reported_as_healthy(monkeypatch):
    response = httpx.Response(
        503,
        request=httpx.Request("GET", "http://api:8000/health"),
        json={"status": "ok", "model_loaded": False},
    )
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: response)
    with pytest.raises(api_client.APIError, match="연결할 수 없습니다"):
        api_client.get_health()


@pytest.mark.parametrize(
    "body",
    [
        b"not JSON",
        b"{}",
        b'{"status":"ok","model_loaded":"false"}',
        b'{"status":"failed","model_loaded":false}',
    ],
)
def test_invalid_health_response_is_not_reported_as_healthy(monkeypatch, body):
    response = httpx.Response(
        200, request=httpx.Request("GET", "http://api:8000/health"), content=body
    )
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: response)
    with pytest.raises(api_client.APIError, match="올바르지 않습니다"):
        api_client.get_health()
