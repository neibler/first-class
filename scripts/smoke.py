"""Check the Compose deployment with the shared model over real HTTP."""

import os
import json
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

from frontend.api_client import get_health
from frontend.charts import allocation
from frontend.weight_format import format_weight

assert format_weight(0.105) == "11%"
assert allocation([{"ticker": "SPY", "weight": 1.0}]).to_dict()
print("PASS: allocation and weight_format imports inside dashboard image")

assert os.environ["API_BASE_URL"] == "http://api:8000"
health = get_health()
assert health.status == "ok"
# The API image installs the Git-tracked shared model selected by serving.json.
assert health.model_loaded is True, "Shared model was not loaded in the API image"

with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=3) as response:
    assert response.status == 200

with urlopen("http://api:8000/ready", timeout=3) as response:
    assert response.status == 200
with urlopen("http://api:8000/backtest", timeout=60) as response:
    backtest = json.load(response)
    assert response.status == 200 and backtest["path"], "Empty backtest path"
with urlopen("http://api:8000/risk", timeout=30) as response:
    # The sign depends on the data (gain-only windows give a negative VaR);
    # the contract is a numeric value.
    var = json.load(response)["portfolio"]["var"]
    assert isinstance(var, (int, float)) and not isinstance(var, bool)
with urlopen("http://api:8000/rebalancing?limit=1", timeout=30) as response:
    assert len(json.load(response)["events"]) == 1
with urlopen("http://api:8000/assets/SPY/history?observations=20", timeout=30) as r:
    assert len(json.load(r)["rows"]) == 20
print("PASS: risk, rebalancing and asset history APIs in the API image")


def assert_model_ready_page(page):
    assert not page.exception, page.exception
    assert not page.error, [item.value for item in page.error]
    assert [item.value for item in page.success] == ["API 연결 정상"]
    assert "모델이 로드되지 않았습니다." not in [item.value for item in page.warning]
    assert page.metric, "Portfolio metrics from the real backtest are missing"


# Execute the page and its HTTP client, not just the Streamlit health endpoint.
page = AppTest.from_file("frontend/app.py", default_timeout=60).run()
assert_model_ready_page(page)
page.button(key="refresh_health").click().run()
assert_model_ready_page(page)
print("PASS: shared model loaded, real backtest, portfolio UI, and health refresh")

page.switch_page("app_pages/research.py").run()
assert not page.exception, page.exception
assert page.text_area(key="research_question").value == ""
assert str(page.date_input(key="research_date").value) == "2025-04-10"
assert len(page.get("page_link")) == 4
assert not page.button(key="research_submit").disabled
print("PASS: research form, demo date, four menu links; no research submitted")

page.switch_page("app_pages/anova.py").run()
assert not page.exception, page.exception
assert any("통계 분석 결과가 아직 준비되지" in w.value for w in page.warning)
assert not page.dataframe
assert len(page.get("page_link")) == 5
page.button(key="refresh_statistics").click().run()
assert not page.exception, page.exception
assert any("통계 분석 결과가 아직 준비되지" in w.value for w in page.warning)
print("PASS: statistics HTTP 503 guidance, navigation and refresh")


with urlopen(
    "http://api:8000/performance/correlation/SPY?window=60", timeout=30
) as response:
    correlation = json.load(response)
    assert response.status == 200 and correlation["status"] == "ok"
    assert correlation["ticker"] == "SPY" and correlation["relationships"]
print("PASS: correlation reads installed training inputs")
