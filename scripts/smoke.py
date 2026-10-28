"""Check the model-free Compose deployment over real HTTP, without fake results."""

import os
from urllib.error import HTTPError
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

from frontend.api_client import get_health

assert os.environ["API_BASE_URL"] == "http://api:8000"
health = get_health()
assert health.status == "ok"
assert health.model_loaded is False

with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=3) as response:
    assert response.status == 200

# CI does not ship private datasets or trained models. Liveness remains healthy,
# while readiness and portfolio data must explicitly report unavailable.
for endpoint in ("/ready", "/backtest"):
    try:
        with urlopen(f"http://api:8000{endpoint}", timeout=3) as response:
            raise AssertionError(f"{endpoint}: expected 503, got {response.status}")
    except HTTPError as exc:
        assert exc.code == 503, f"{endpoint}: unexpected HTTP {exc.code}"


def assert_model_unavailable_page(page):
    assert not page.exception, page.exception
    assert [item.value for item in page.error] == [
        "투자 분석 결과를 불러오지 못했습니다. 서버와 모델 준비 상태를 확인해 주세요."
    ]
    assert [item.value for item in page.success] == ["API 연결 정상"]
    assert [item.value for item in page.warning] == ["모델이 로드되지 않았습니다."]
    assert not page.metric, "Do not display fabricated portfolio metrics"


# Execute the page and its HTTP client, not just the Streamlit health endpoint.
page = AppTest.from_file("frontend/app.py", default_timeout=15).run()
assert_model_unavailable_page(page)
page.button(key="refresh_health").click().run()
assert_model_unavailable_page(page)
print("PASS: real HTTP health, expected 503, unavailable UI, and health refresh")
