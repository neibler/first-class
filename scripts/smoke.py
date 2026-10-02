"""Run inside the dashboard container against the real Compose API."""

import os
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

from frontend.api_client import get_health

assert os.environ["API_BASE_URL"] == "http://api:8000"
health = get_health()
assert health.status == "ok"
assert health.model_loaded is False

with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=3) as response:
    assert response.status == 200

# A server health check alone does not execute the Streamlit page.
# AppTest runs the actual page and its HTTP client without mocked responses.
page = AppTest.from_file("frontend/app.py", default_timeout=15).run()
assert not page.exception, page.exception
assert not page.error, page.error
assert page.success[0].value == "API 연결 정상"
assert page.warning[0].value == "모델이 로드되지 않았습니다."
page.button[0].click().run()
assert not page.exception
assert not page.error
assert page.success[0].value == "API 연결 정상"
print("PASS: dashboard -> http://api:8000/health; model_loaded=false; page refresh")
