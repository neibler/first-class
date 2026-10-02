from streamlit.testing.v1 import AppTest

from frontend import api_client


def test_page_shows_api_success_and_missing_model(monkeypatch):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    page = AppTest.from_file("frontend/app.py").run()
    assert not page.exception
    assert not page.error
    assert page.success[0].value == "API 연결 정상"
    assert page.warning[0].value == "모델이 로드되지 않았습니다."


def test_refresh_recovers_from_api_failure(monkeypatch):
    def unavailable():
        raise api_client.APIError("API에 연결할 수 없습니다.")

    monkeypatch.setattr(api_client, "get_health", unavailable)
    page = AppTest.from_file("frontend/app.py").run()
    assert not page.exception
    assert not page.success
    assert not page.warning
    assert page.error[0].value == "API에 연결할 수 없습니다."

    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    page.button[0].click().run()
    assert not page.exception
    assert not page.error
    assert page.success[0].value == "API 연결 정상"
    assert page.warning[0].value == "모델이 로드되지 않았습니다."
