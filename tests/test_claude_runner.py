import json
from collections import Counter
from types import SimpleNamespace as NS

import pytest

from backend.news_research import claude_runner as cr
from backend.news_research import risk_tagger as rt


def _run_dir(tmp_path, n_articles=5):
    articles = [
        {
            "url": f"https://www.nytimes.com/2024/09/0{i}/x.html",
            "title": f"Headline {i}",
            "summary": "Summary.",
            "published_at": f"2024-09-0{i}T14:00:00+00:00",
        }
        for i in range(1, n_articles + 1)
    ]
    requests, keys = rt.build_requests(articles, per_request=2)
    rt.write_jsonl(tmp_path / "requests.jsonl", requests)
    rt.write_jsonl(tmp_path / "keys.jsonl", keys)
    (tmp_path / "prompt.md").write_text("RUBRIC", encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps(rt.label_schema()))
    return requests


def _message(stop_reason="end_turn", text='{"labels": []}'):
    content = [NS(type="thinking", thinking=""), NS(type="text", text=text)]
    usage = NS(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=0,
    )
    return NS(stop_reason=stop_reason, content=content, usage=usage)


class FakeBatches:
    """Message Batches stand-in: each batch ends after one poll."""

    def __init__(self, outcome):
        self.outcome, self.created, self.polls, self.batches = outcome, [], 0, {}

    def create(self, requests):
        batch_id = f"b{len(self.created) + 1}"
        self.created.append(requests)
        self.batches[batch_id] = [r["custom_id"] for r in requests]
        return NS(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        self.polls += 1
        status = "ended" if self.polls % 2 == 0 else "in_progress"
        counts = NS(succeeded=0, errored=0, processing=1)
        return NS(processing_status=status, request_counts=counts)

    def results(self, batch_id):
        for custom_id in self.batches[batch_id]:
            yield NS(custom_id=custom_id, result=self.outcome(custom_id))


def _client(outcome):
    return NS(messages=NS(batches=FakeBatches(outcome)))


def test_params_cache_the_rubric_force_json_and_send_no_urls(tmp_path):
    requests = _run_dir(tmp_path)
    params = cr.batch_params(requests[0], "RUBRIC", rt.label_schema(), cr.MODEL)
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"] == rt.label_schema()
    assert params["messages"][-1]["role"] == "user"
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in params
    assert "nytimes.com" not in json.dumps(params) and "2024-09" not in json.dumps(
        params
    )


def test_failed_answers_are_resent_and_finished_batches_not(tmp_path):
    _run_dir(tmp_path)  # 3 requests

    def first_try(custom_id):
        if custom_id == "r000002":
            return NS(type="succeeded", message=_message(stop_reason="refusal"))
        if custom_id == "r000003":
            return NS(type="errored", error=NS(type="api_error"))
        return NS(type="succeeded", message=_message())

    client = _client(first_try)
    written, failed, usage = cr.run(tmp_path, client, poll=0, sleep=lambda s: None)
    assert (written, failed) == (1, 2)
    assert usage["cache_read_input_tokens"] == 2000  # refusals are billed too
    assert rt.answered(tmp_path / "outputs_claude-haiku-5-5.jsonl") == {"r000001"}
    state = json.loads((tmp_path / "claude_batches_claude-haiku-5-5.json").read_text())
    assert state["batches"] == [{"id": "b1", "requests": 3, "collected": True}]

    client = _client(lambda custom_id: NS(type="succeeded", message=_message()))
    assert cr.run(tmp_path, client, poll=0, sleep=lambda s: None)[:2] == (2, 0)
    resent = [r["custom_id"] for r in client.messages.batches.created[0]]
    assert resent == ["r000002", "r000003"]


def test_an_interrupted_run_waits_for_its_batch_instead_of_resubmitting(tmp_path):
    _run_dir(tmp_path, n_articles=2)  # 1 request
    state = {"batches": [{"id": "b1", "requests": 1, "collected": False}]}
    (tmp_path / "claude_batches_claude-haiku-5-5.json").write_text(json.dumps(state))
    client = _client(lambda custom_id: NS(type="succeeded", message=_message()))
    client.messages.batches.batches["b1"] = ["r000001"]
    assert cr.run(tmp_path, client, poll=0, sleep=lambda s: None)[:2] == (1, 0)
    assert client.messages.batches.created == []


def test_cost_uses_batch_prices():
    usage = Counter(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cr.cost(usage) == pytest.approx(0.30)
    assert cr.cost(usage, "claude-sonnet-5-5") == pytest.approx(6.0)
    assert cr.cost(usage, "unknown-model") is None


def test_effort_and_tag_keep_settings_apart(tmp_path):
    requests = _run_dir(tmp_path, n_articles=2)
    params = cr.batch_params(requests[0], "R", rt.label_schema(), cr.MODEL, "medium")
    assert params["output_config"]["effort"] == "medium"
    client = _client(lambda custom_id: NS(type="succeeded", message=_message()))
    cr.run(tmp_path, client, poll=0, sleep=lambda s: None, effort="medium", tag="m2")
    sent = client.messages.batches.created[0][0]["params"]["output_config"]
    assert sent["effort"] == "medium"
    assert rt.answered(tmp_path / "outputs_m2.jsonl") == {"r000001"}
    assert not (tmp_path / "outputs_claude-haiku-5-5.jsonl").exists()
