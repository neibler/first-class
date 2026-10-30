import json
from argparse import Namespace

import httpx

from backend.news_research import ollama_runner as runner
from backend.news_research import risk_tagger as rt


def _run_dir(tmp_path, n_articles=5):
    articles = [
        {
            "url": f"https://www.nytimes.com/2024/09/0{i}/x.html",
            "title": f"Headline {i}",
            "summary": None,
            "published_at": f"2024-09-0{i}T14:00:00+00:00",
        }
        for i in range(1, n_articles + 1)
    ]
    requests, keys = rt.build_requests(articles, per_request=2)
    rt.write_jsonl(tmp_path / "requests.jsonl", requests)
    rt.write_jsonl(tmp_path / "keys.jsonl", keys)
    (tmp_path / "prompt.md").write_text("RUBRIC", encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps(rt.label_schema()))
    (tmp_path / "manifest.json").write_text('{"prompt_version": "v2"}')
    return requests


def _reply(url, labels, status=200):
    content = json.dumps({"labels": labels})
    body = {"message": {"content": content}, "eval_count": 40, "prompt_eval_count": 900}
    return httpx.Response(status, request=httpx.Request("POST", url), json=body)


def test_body_carries_rubric_schema_and_deterministic_settings(tmp_path):
    requests = _run_dir(tmp_path)
    body = runner.chat_body(requests[0], "RUBRIC", rt.label_schema(), "qwen2.5:7b")
    assert body["messages"][0] == {"role": "system", "content": "RUBRIC"}
    assert "Headline 1" in body["messages"][1]["content"]
    assert "nytimes.com" not in json.dumps(body)
    assert body["format"] == rt.label_schema() and body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": runner.NUM_CTX}


def test_answers_are_saved_and_a_rerun_only_retries_failures(tmp_path):
    _run_dir(tmp_path)  # 5 articles -> 3 requests of up to 2

    def flaky(url, **kwargs):
        if "Headline 3" in kwargs["json"]["messages"][1]["content"]:
            return _reply(url, [], status=500)
        return _reply(url, [])

    assert runner.run(tmp_path, "m", post=flaky) == (2, 1)
    with (tmp_path / "outputs.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert {r["request_id"] for r in rows} == {"r000001", "r000003"}
    assert rows[0]["output_tokens"] == 40 and rows[0]["model"] == "m"

    asked = []

    def healthy(url, **kwargs):
        asked.append(kwargs["json"]["messages"][1]["content"])
        return _reply(url, [])

    assert runner.run(tmp_path, "m", post=healthy) == (1, 0)
    assert len(asked) == 1 and "Headline 3" in asked[0]
    assert runner.answered(tmp_path / "outputs.jsonl") == {
        "r000001",
        "r000002",
        "r000003",
    }


def test_outputs_merge_into_labels(tmp_path):
    _run_dir(tmp_path, n_articles=1)
    label = {
        "id": "a01",
        "market_relevant": True,
        "risk_types": ["earnings"],
        "risk_level": 0.3,
        "direction": "increase",
        "confidence": 0.6,
        "evidence": "Headline 1",
    }
    runner.run(tmp_path, "m", post=lambda url, **kwargs: _reply(url, [label]))
    labels = tmp_path / "labels.jsonl"
    rt.merge(Namespace(run=tmp_path, labels=labels, tagger="m"))
    row = json.loads(labels.read_text(encoding="utf-8"))
    assert row["url"].endswith("/01/x.html") and row["risk_types"] == ["earnings"]
    assert row["tagger"] == "m" and row["evidence_found"] is True
