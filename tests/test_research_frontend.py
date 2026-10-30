"""HTTP client and real multipage UI with test-only research responses."""

from datetime import date
from unittest.mock import Mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from backend.research_schemas import ResearchResponse
from frontend import api_client


@pytest.fixture
def body():
    data = {
        "status": "complete",
        "question": "최근 QQQ 위험?",
        "as_of": "2025-04-10",
        "answer": "테스트용 분석 [1].",
        "sources": [
            {
                "n": 1,
                "title": "테스트 기사",
                "source": "example",
                "date": "2025-04-03",
                "url": "https://example.com/a%2Fb?q=1",
            }
        ],
        "risk_tags": [
            {
                "ticker": "QQQ",
                "risk_type": "regulation",
                "risk_score": 0.48,
                "risk_level": 0.6,
                "confidence": 0.8,
                "direction": "increase",
                "date": "2025-04-10",
                "evidence": [1],
            }
        ],
        "plan": [{"query": "export restrictions", "etf": "QQQ", "lookback_days": 30}],
        "log": ["[Planner] 검색 계획 완료", "[Analyst] 위험 태그 1개"],
        "disclaimer": "테스트용 교육 목적 면책 안내",
    }
    # Keep the frontend fixture checked against the real API contract.
    ResearchResponse.model_validate(data)
    return data


def response(status, body):
    return httpx.Response(
        status, json=body, request=httpx.Request("POST", "http://api:8000/research")
    )


@pytest.fixture
def post(monkeypatch, body):
    mock = Mock(return_value=response(200, body))
    monkeypatch.setattr(httpx, "post", mock)
    return mock


@pytest.fixture
def page(monkeypatch, post):
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    app = AppTest.from_file("frontend/app.py", default_timeout=10).run()
    app.switch_page("app_pages/research.py").run()
    assert not app.exception
    return app


@pytest.mark.parametrize(
    "base,url",
    [
        (None, "http://127.0.0.1:8000/research"),
        ("http://api:8000/", "http://api:8000/research"),
    ],
)
def test_http_contract_and_separate_timeout(monkeypatch, post, body, base, url):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    if base:
        monkeypatch.setenv("API_BASE_URL", base)
    result = api_client.post_research("  최근 QQQ 위험?  ", "2025-04-10")
    post.assert_called_once_with(
        url,
        json={"question": "최근 QQQ 위험?", "as_of": "2025-04-10"},
        timeout=api_client.RESEARCH_TIMEOUT,
        follow_redirects=False,
    )
    assert api_client.RESEARCH_TIMEOUT.read == 120
    assert api_client.RESEARCH_TIMEOUT.connect == 5
    assert result.model_dump(mode="json") == body


@pytest.mark.parametrize(
    "status,code",
    [
        (422, "invalid_research_request"),
        (503, "research_key_missing"),
        (503, "research_db_missing"),
        (503, "research_db_unavailable"),
        (503, "research_embedding_unavailable"),
        (503, "research_unavailable"),
        (503, "research_busy"),
        (502, "research_upstream_error"),
        (502, "research_invalid_result"),
        (504, "research_timeout"),
        (500, "research_failed"),
    ],
)
def test_http_errors_are_safe_and_never_retried(post, status, code):
    post.return_value = response(
        status, {"detail": {"code": code, "message": "PRIVATE /server/path"}}
    )
    with pytest.raises(api_client.ResearchAPIError) as caught:
        api_client.post_research("최근 QQQ 위험?", "2025-04-10")
    assert caught.value.code == code
    assert "PRIVATE" not in str(caught.value)
    assert post.call_count == 1


@pytest.mark.parametrize(
    "error,code",
    [
        (httpx.ConnectError("PRIVATE"), "connection_failed"),
        (httpx.ReadTimeout("PRIVATE"), "client_timeout"),
        (httpx.InvalidURL("PRIVATE"), "invalid_api_url"),
    ],
)
def test_transport_errors_are_safe(post, error, code):
    post.side_effect = error
    with pytest.raises(api_client.ResearchAPIError) as caught:
        api_client.post_research("최근 QQQ 위험?", "2025-04-10")
    assert caught.value.code == code
    assert "PRIVATE" not in str(caught.value)
    assert post.call_count == 1


@pytest.mark.parametrize("kind", ["json", "shape", "question", "date", "citation"])
def test_bad_responses_are_not_displayed_as_success(post, body, kind):
    if kind == "json":
        post.return_value = httpx.Response(200, text="PRIVATE")
    else:
        if kind == "shape":
            body.pop("sources")
        elif kind == "question":
            body["question"] = "다른 질문"
        elif kind == "date":
            body["as_of"] = "2025-04-11"
        else:
            body["risk_tags"][0]["evidence"] = [99]
        post.return_value = response(200, body)
    with pytest.raises(api_client.ResearchAPIError) as caught:
        api_client.post_research("최근 QQQ 위험?", "2025-04-10")
    assert caught.value.code == "invalid_response"


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,test",
        "file:///secret",
        "//example.com",
        "https://user:pass@example.com",
        "https://example.com\\@evil.com",
        "https://example.com\n",
        "https://example.com:bad",
        "https://",
        "<script>",
    ],
)
def test_unsafe_urls_never_become_links(url):
    assert api_client.safe_source_url(url) is None


@pytest.mark.parametrize(
    "url", ["http://example.com/news", "https://example.com/a%2Fb?q=1"]
)
def test_safe_urls_are_preserved(url):
    assert api_client.safe_source_url(url) == url


def submit(page, question="최근 QQQ 위험?", as_of=date(2025, 4, 10)):
    page.text_area(key="research_question").set_value(question)
    page.date_input(key="research_date").set_value(as_of)
    page.button(key="research_submit").click().run()
    assert not page.exception


def test_no_post_before_submit_or_on_rerun_and_navigation(page, post):
    assert page.date_input(key="research_date").value == date(2025, 4, 10)
    assert "질문과 기준일" in page.info[0].value
    assert "2023~2025" in " ".join(item.value for item in page.caption)
    page.text_area(key="research_question").set_value("아직 제출하지 않은 질문").run()
    page.run()
    page.switch_page("app_pages/shap.py").run()
    page.switch_page("app_pages/research.py").run()
    assert not page.exception
    assert page.text_area(key="research_question").value == "아직 제출하지 않은 질문"
    assert len(page.get("page_link")) == 6
    post.assert_not_called()


def test_submit_displays_response_and_preserves_it_without_duplicate_post(
    page, post, body
):
    submit(page, "  최근 QQQ 위험?  ")
    assert post.call_count == 1
    assert post.call_args.kwargs["json"] == {
        "question": body["question"],
        "as_of": body["as_of"],
    }
    text = "\n".join(item.value for item in page.text)
    for expected in (
        body["answer"],
        body["disclaimer"],
        body["log"][0],
        "테스트 기사",
        "2025-04-10",
    ):
        assert expected in text
    assert page.dataframe[0].value.iloc[0]["뉴스 위험 지표"] == 0.48
    assert page.get("link_button")[0].proto.url == body["sources"][0]["url"]
    assert page.expander[0].label == "완료 후 실행 요약 로그"
    page.text_area(key="research_question").set_value("새 질문").run()
    page.date_input(key="research_date").set_value(date(2024, 1, 1)).run()
    page.switch_page("app_pages/anova.py").run()
    page.switch_page("app_pages/research.py").run()
    assert not page.exception and post.call_count == 1
    assert "최근 QQQ 위험?" in "\n".join(item.value for item in page.text)
    assert "요청 기준일: 2025-04-10" in "\n".join(item.value for item in page.text)
    assert page.text_area(key="research_question").value == "새 질문"
    assert page.date_input(key="research_date").value == date(2024, 1, 1)


def test_no_evidence_has_no_fabricated_sources_or_tags(page, post, body):
    body.update(
        status="insufficient_evidence",
        answer="근거를 찾지 못했습니다.",
        sources=[],
        risk_tags=[],
    )
    post.return_value = response(200, body)
    submit(page)
    assert "근거가 부족" in page.warning[0].value
    assert not page.dataframe and not page.get("link_button")
    assert body["answer"] in [item.value for item in page.text]


@pytest.mark.parametrize(
    "code",
    [
        "invalid_research_request",
        "research_key_missing",
        "research_unavailable",
        "research_busy",
        "research_upstream_error",
        "research_timeout",
        "connection_failed",
        "client_timeout",
    ],
)
def test_page_error_states_and_explicit_retry(page, post, code):
    if code in {"connection_failed", "client_timeout"}:
        post.side_effect = (
            httpx.ConnectError("PRIVATE")
            if code == "connection_failed"
            else httpx.ReadTimeout("PRIVATE")
        )
    else:
        post.return_value = response(
            422 if code == "invalid_research_request" else 503,
            {"detail": {"code": code, "message": "PRIVATE"}},
        )
    submit(page, "   " if code == "invalid_research_request" else "최근 QQQ 위험?")
    assert page.error[0].value == api_client.RESEARCH_ERRORS[code]
    assert not page.dataframe
    page.run()
    assert post.call_count == 1
    page.button(key="research_submit").click().run()
    assert not page.exception and post.call_count == 2


def test_untrusted_response_text_is_plain_text_and_links_are_filtered(page, post, body):
    malicious = '<img src=x onerror="alert(1)"> [bad](javascript:alert(1))'
    body.update(answer=malicious + " [1]", log=[malicious], disclaimer=malicious)
    body["sources"][0].update(title=malicious, url="javascript:alert(1)")
    post.return_value = response(200, body)
    submit(page)
    assert not page.get("link_button")
    assert malicious in "\n".join(item.value for item in page.text)
    assert all(malicious not in item.value for item in page.markdown)


def test_request_consumed_before_http_without_frontend_lock(page, post, body):
    import streamlit as st

    def during_request(*args, **kwargs):
        assert "research_running" not in st.session_state
        assert "research_pending" not in st.session_state
        return response(200, body)

    post.side_effect = during_request
    submit(page)
    assert not page.button(key="research_submit").disabled
    assert not page.text_area(key="research_question").disabled
    assert not page.date_input(key="research_date").disabled
    assert post.call_count == 1


@pytest.mark.parametrize("question", ["최근 QQQ 위험?", " \t최근 QQQ 위험?\n "])
def test_same_success_is_reused_with_full_response(page, post, body, question):
    submit(page)
    submit(page, question)
    assert post.call_count == 1
    assert "동일한 요청의 이전 분석 결과입니다." in [item.value for item in page.info]
    saved = page.session_state["research_last_success"]
    assert saved["result"].model_dump(mode="json") == body
    assert page.session_state["research_outcome"]["result"] == saved["result"]
    text = "\n".join(item.value for item in page.text)
    for expected in (body["answer"], body["disclaimer"], *body["log"]):
        assert expected in text
    assert page.get("link_button")[0].proto.url == body["sources"][0]["url"]
    assert page.dataframe[0].value.iloc[0]["뉴스 위험 지표"] == 0.48


@pytest.mark.parametrize(
    "question,as_of",
    [
        ("최근 SPY 위험?", date(2025, 4, 10)),
        ("최근 QQQ 위험!", date(2025, 4, 10)),
        ("최근 qqq 위험?", date(2025, 4, 10)),
        ("최근  QQQ 위험?", date(2025, 4, 10)),
        ("최근 QQQ 위험?", date(2025, 4, 11)),
    ],
)
def test_changed_request_replaces_only_success_slot(page, post, body, question, as_of):
    submit(page)
    changed = dict(
        body, question=question, as_of=as_of.isoformat(), answer="B 결과 [1]"
    )
    post.return_value = response(200, changed)
    submit(page, question, as_of)
    assert post.call_count == 2
    assert page.session_state["research_last_success"]["result"].answer == "B 결과 [1]"
    post.return_value = response(200, body)
    submit(page)
    assert post.call_count == 3  # A was replaced, not kept in a history list.
    assert (
        page.session_state["research_last_success"]["result"].model_dump(mode="json")
        == body
    )
    assert not page.session_state["research_outcome"]["reused"]


@pytest.mark.parametrize("kind", ["error", "insufficient_evidence", "invalid_response"])
def test_unsuccessful_b_is_displayed_but_keeps_success_a(page, post, body, kind):
    submit(page)
    if kind == "error":
        post.return_value = response(503, {"detail": {"code": "research_busy"}})
    elif kind == "invalid_response":
        post.return_value = response(200, {})
    else:
        post.return_value = response(
            200,
            dict(
                body,
                question="B 질문",
                status="insufficient_evidence",
                answer="B 근거 부족",
                sources=[],
                risk_tags=[],
            ),
        )
    submit(page, "B 질문")
    assert post.call_count == 2
    assert "B 질문" in [item.value for item in page.text]
    assert body["answer"] not in [item.value for item in page.text]
    assert not page.dataframe and not page.get("link_button")
    if kind == "insufficient_evidence":
        assert "판단할 근거가 부족합니다" in page.warning[0].value
    else:
        assert page.error
    assert (
        page.session_state["research_last_success"]["result"].model_dump(mode="json")
        == body
    )
    submit(page)
    assert post.call_count == 2
    assert page.session_state["research_outcome"]["reused"]
    assert body["answer"] in [item.value for item in page.text]
    assert not page.error


def test_no_evidence_without_success_is_requested_again(page, post, body):
    post.return_value = response(
        200,
        dict(body, status="insufficient_evidence", sources=[], risk_tags=[])
        | {"answer": "근거 없음"},
    )
    submit(page)
    submit(page)
    assert post.call_count == 2
    assert page.session_state["research_last_success"] is None


def test_new_session_has_no_other_sessions_success(page, post):
    submit(page)
    other = AppTest.from_file("frontend/app.py", default_timeout=10).run()
    other.switch_page("app_pages/research.py").run()
    assert not other.exception
    assert other.session_state["research_last_success"] is None
    assert other.session_state["research_outcome"] is None
    assert post.call_count == 1
    submit(other)
    assert post.call_count == 2
    submit(page)
    assert post.call_count == 2


def test_endpoint_change_invalidates_success_without_automatic_post(
    page, post, monkeypatch
):
    monkeypatch.setenv("API_BASE_URL", "http://first-api:8000/")
    submit(page)
    monkeypatch.setenv("API_BASE_URL", "http://second-api:8000")
    page.run()
    assert page.session_state["research_last_success"] is None
    assert post.call_count == 1
    # Even switching back must not resurrect an older endpoint's result.
    monkeypatch.setenv("API_BASE_URL", "http://first-api:8000")
    submit(page)
    assert post.call_count == 2
    submit(page)
    assert post.call_count == 2


@pytest.mark.parametrize("as_of", [date(2022, 12, 31), date(2026, 1, 1)])
def test_outside_demo_range_warns_without_blocking(page, post, body, as_of):
    post.return_value = response(200, dict(body, as_of=as_of.isoformat()))
    submit(page, as_of=as_of)
    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["as_of"] == as_of.isoformat()
    assert "자료 범위(2023~2025년) 밖" in page.warning[0].value
    assert page.success


def test_invalid_input_after_success_still_reaches_api_validation(page, post):
    submit(page)
    post.return_value = response(422, {"detail": {"code": "invalid_research_request"}})
    submit(page, "  ")
    assert post.call_count == 2
    assert page.error[0].value == api_client.RESEARCH_ERRORS["invalid_research_request"]
    assert page.session_state["research_last_success"] is not None
    assert page.session_state["research_outcome"]["result"] is None
