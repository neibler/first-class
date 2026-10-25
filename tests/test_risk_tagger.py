import json
from argparse import Namespace

import yaml

from backend.news_research import risk_tagger as rt


def _article(
    i, title="Fed Signals Rate Cut", summary="Officials see cooling inflation."
):
    return {
        "url": f"https://www.nytimes.com/2024/09/{i:02d}/business/fed-{i}.html",
        "title": title,
        "summary": summary,
        "published_at": f"2024-09-{i:02d}T14:00:00+00:00",
        "source": "nytimes",
    }


def test_requests_hide_urls_and_dates_and_keep_the_map_locally():
    articles = [_article(i) for i in range(1, 6)]
    requests, keys = rt.build_requests(articles, per_request=2)
    assert [len(r["items"]) for r in requests] == [2, 2, 1]
    sent = json.dumps(requests) + "".join(rt.user_prompt(r) for r in requests)
    for article in articles:
        assert article["url"] not in sent and article["published_at"] not in sent
    assert "2024" not in sent
    assert keys[0]["urls"] == {"a01": articles[0]["url"], "a02": articles[1]["url"]}
    assert (
        keys[2]["texts"]["a01"]
        == "Fed Signals Rate Cut\nOfficials see cooling inflation."
    )


def test_summary_is_sent_only_when_the_article_has_one():
    title_only = {**_article(1), "summary": None, "title": "  Oil   Jumps "}
    assert rt.article_input(title_only) == {"title": "Oil Jumps"}
    assert "summary" in rt.article_input(_article(2))


def test_prompt_drops_maintainer_notes_and_schema_matches_the_team_types():
    prompt = rt.load_prompt()
    assert "<!--" not in prompt and "판단 예시" not in prompt
    assert prompt.startswith("# Risk tagging prompt v2")
    assert rt.prompt_version(rt.PROMPT_PATH) == "v2"
    with open("config/etf_news_map.yaml", encoding="utf-8") as f:
        assert list(rt.RISK_TYPES) == yaml.safe_load(f)["risk_types"]
    item = rt.label_schema()["properties"]["labels"]["items"]
    assert item["properties"]["risk_types"]["items"]["enum"] == list(rt.RISK_TYPES)


KEY = {
    "urls": {"a01": "u1", "a02": "u2", "a03": "u3"},
    "texts": {
        "a01": "Fed Signals Rate Cut\nOfficials see cooling inflation.",
        "a02": "Pumped-Up Prices",
        "a03": "Oil Jumps",
    },
}


def test_outputs_are_coerced_into_the_rubric_and_gaps_reported():
    output = """```json
{"labels": [
 {"id": "a01", "market_relevant": true, "risk_types": ["interest_rate", "made_up",
  "interest_rate", "inflation"], "risk_level": 1.4, "direction": "decrease",
  "confidence": "0.8", "evidence": "officials see  cooling inflation"},
 {"id": "a02", "market_relevant": false, "risk_types": ["inflation"],
  "risk_level": 0.5, "direction": "increase", "confidence": 0.3, "evidence": ""},
 {"id": "zz", "market_relevant": true, "risk_types": ["recession"],
  "risk_level": 0.5, "direction": "increase", "confidence": 0.9, "evidence": "x"}
]}
```"""
    labels, missing = rt.parse_output(output, KEY)
    by_url = {label["url"]: label for label in labels}
    first = by_url["u1"]
    assert first["risk_types"] == ["interest_rate", "inflation"]
    assert first["risk_level"] == 1.0 and first["confidence"] == 0.8
    assert first["evidence_found"] is True  # case and spacing do not matter
    irrelevant = by_url["u2"]
    assert irrelevant["risk_types"] == [] and irrelevant["risk_level"] == 0.0
    assert irrelevant["direction"] == "none"
    assert missing == ["a03"]


def test_unusable_output_leaves_every_article_for_a_retry():
    assert rt.parse_output("not json", KEY) == ([], ["a01", "a02", "a03"])
    bad_number = '{"labels": [{"id": "a03", "market_relevant": true, "risk_types":'
    bad_number += ' ["recession"], "risk_level": "high", "direction": "increase",'
    bad_number += ' "confidence": 0.5, "evidence": ""}]}'
    assert rt.parse_output(bad_number, KEY)[1] == ["a01", "a02", "a03"]


def test_quote_not_in_the_text_is_flagged():
    raw = {
        "market_relevant": True,
        "risk_types": ["commodity_risk"],
        "risk_level": 0.4,
        "direction": "increase",
        "confidence": 0.7,
        "evidence": "OPEC cuts output",
    }
    assert rt.clean_label(raw, "Oil Jumps")["evidence_found"] is False


def test_selection_skips_labeled_restricts_and_samples_reproducibly():
    articles = [_article(i) for i in range(1, 11)] + [_article(1)]
    labeled = {articles[0]["url"]}
    chosen = rt.select_articles(articles, labeled)
    assert len(chosen) == 9 and chosen[0]["url"] == articles[1]["url"]
    only = {articles[2]["url"], articles[3]["url"], articles[0]["url"]}
    assert len(rt.select_articles(articles, labeled, only)) == 2
    dated = rt.select_articles(articles, set(), since="2024-09-03", until="2024-09-05")
    assert [x["published_at"][:10] for x in dated] == [
        "2024-09-03",
        "2024-09-04",
        "2024-09-05",
    ]
    a = rt.select_articles(articles, set(), sample=4, seed=7)
    b = rt.select_articles(articles, set(), sample=4, seed=7)
    assert a == b and len(a) == 4


def test_prepare_then_merge_appends_new_labels_once(tmp_path):
    articles_path = tmp_path / "articles.jsonl"
    rt.write_jsonl(articles_path, [_article(i) for i in range(1, 4)])
    run, labels = tmp_path / "run", tmp_path / "labels.jsonl"
    common = {"labels": labels, "prompt": rt.PROMPT_PATH, "per_request": 2}
    rt.prepare(
        Namespace(
            articles=[str(articles_path)],
            out=run,
            urls=None,
            sample=None,
            seed=0,
            since=None,
            until=None,
            **common,
        )
    )
    requests = [json.loads(line) for line in (run / "requests.jsonl").open()]
    answer = {
        "labels": [
            {
                "id": "a01",
                "market_relevant": True,
                "risk_types": ["interest_rate"],
                "risk_level": 0.4,
                "direction": "decrease",
                "confidence": 0.8,
                "evidence": "Fed Signals Rate Cut",
            }
        ]
    }
    outputs = [{"request_id": requests[0]["request_id"], "output": json.dumps(answer)}]
    rt.write_jsonl(run / "outputs.jsonl", outputs)
    merge_args = Namespace(run=run, labels=labels, tagger="test-model")
    rt.merge(merge_args)
    rt.merge(merge_args)  # merging again adds nothing
    rows = [json.loads(line) for line in labels.open(encoding="utf-8")]
    assert len(rows) == 1
    assert rows[0]["prompt_version"] == "v2" and rows[0]["tagger"] == "test-model"
    # the two articles the model skipped are prepared again, the labeled one is not
    rt.prepare(
        Namespace(
            articles=[str(articles_path)],
            out=tmp_path / "run2",
            urls=None,
            sample=None,
            seed=0,
            since=None,
            until=None,
            **common,
        )
    )
    manifest = json.loads((tmp_path / "run2" / "manifest.json").read_text())
    assert manifest["articles"] == 2
