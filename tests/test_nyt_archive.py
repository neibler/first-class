import gzip
import json

import httpx
import pytest

from backend.news_research import nyt_archive as nyt
from backend.news_research.term_matcher import (
    build_matcher,
    risk_only_terms,
    term_etfs,
)

NEWS_MAP = {
    "companies": {
        "AAPL": {"aliases": ["Apple"]},
        "MRK": {"aliases": ["Merck"], "exclude": ["Merck KGaA"]},
    },
    "etfs": {
        "QQQ": {"holdings": {"AAPL": 0.07}, "keywords": []},
        "XLE": {"holdings": {}, "keywords": ["OPEC", "OPEC+"]},
        "GLD": {"holdings": {}, "keywords": [], "nyt_tags": ["Gold"]},
        "TIP": {"holdings": {}, "keywords": ["TIPS"]},
    },
    "risk_type_keywords": {"inflation": ["inflation"]},
}


def _doc(**overrides):
    doc = {
        "web_url": "https://www.nytimes.com/2025/01/02/business/apple.html",
        "headline": {"main": "Apple Shares Slide on China Tariff Fears"},
        "abstract": "The company warned that new tariffs could hurt iPhone sales.",
        "lead_paragraph": "",
        "pub_date": "2025-01-02T15:30:00+0000",
        "section_name": "Business Day",
        "type_of_material": "News",
    }
    return doc | overrides


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple shares fell", ["Apple"]),
        ("an apple a day", []),
        ("Merck KGaA raised its outlook", []),
        ("Merck beat estimates", ["Merck"]),
        ("Inflation cooled in May", ["inflation"]),
        ("OPEC+ agreed to cut output", ["OPEC", "OPEC+"]),
        ("Investors piled into TIPS", ["TIPS"]),
        ("Last-Minute Tax Tips", []),
    ],
)
def test_matcher_case_rules_and_excludes(text, expected):
    assert build_matcher(NEWS_MAP)(text) == expected


def test_api_key_comes_from_env_or_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("API_PORT=8000\nNYT_API_KEY='from-dotenv'\n", encoding="utf-8")
    monkeypatch.delenv("NYT_API_KEY", raising=False)
    assert nyt.load_api_key(env) == "from-dotenv"

    monkeypatch.setenv("NYT_API_KEY", "from-env")
    assert nyt.load_api_key(env) == "from-env"

    monkeypatch.delenv("NYT_API_KEY")
    with pytest.raises(nyt.NytError, match="NYT_API_KEY"):
        nyt.load_api_key(tmp_path / "missing.env")


def test_month_range_crosses_years():
    assert list(nyt.iter_months("2024-11", "2025-02")) == [
        (2024, 11),
        (2024, 12),
        (2025, 1),
        (2025, 2),
    ]


def test_article_keeps_relevant_news_with_short_summary():
    match = build_matcher(NEWS_MAP)
    long_doc = _doc(abstract="Apple " + "x" * 400, pub_date="2025-01-02T15:30:00.886Z")
    article = nyt.to_article(long_doc, match)

    assert article["published_at"] == "2025-01-02T15:30:00+00:00"
    assert article["matched_terms"] == ["Apple"]
    assert len(article["summary"]) == nyt.SUMMARY_MAX_CHARS
    assert nyt.to_article(_doc(section_name="Arts"), match) is None
    assert nyt.to_article(_doc(type_of_material="Op-Ed"), match) is None
    no_terms = _doc(headline={"main": "A Quiet Day"}, abstract="Nothing happened.")
    assert nyt.to_article(no_terms, match) is None


def test_nyt_tags_count_as_matches_and_are_kept():
    tagged = _doc(
        headline={"main": "Prices Keep Climbing"},
        abstract="Shoppers paid more again.",
        keywords=[
            {"name": "Subject", "value": "Inflation (Economics)", "rank": 1},
            {"name": "Subject", "value": "internal-storyline-no", "rank": 2},
        ],
    )
    article = nyt.to_article(tagged, build_matcher(NEWS_MAP))
    assert article["matched_terms"] == ["inflation"]
    assert article["nyt_keywords"] == ["Inflation (Economics)"]


def test_world_news_with_only_risk_words_needs_an_economic_tag():
    match, risk_only = build_matcher(NEWS_MAP), risk_only_terms(NEWS_MAP)
    politics = _doc(
        section_name="World",
        headline={"main": "Inflation of Egos at the Summit"},
        abstract="Leaders traded barbs.",
    )
    assert risk_only == frozenset({"inflation"})
    assert nyt.to_article(politics, match, risk_only) is None

    economic = politics | {"keywords": [{"value": "Inflation (Economics)"}]}
    assert nyt.to_article(economic, match, risk_only) is not None
    oil = politics | {"headline": {"main": "OPEC Meets as Leaders Trade Barbs"}}
    assert nyt.to_article(oil, match, risk_only)["matched_terms"] == ["OPEC"]


def test_exact_nyt_tags_link_articles_to_etfs():
    match, risk_only = build_matcher(NEWS_MAP), risk_only_terms(NEWS_MAP)
    rally = _doc(
        headline={"main": "Central Banks Keep Buying"},
        abstract="Reserves grew again.",
        keywords=[{"value": "Gold"}],
    )
    article = nyt.to_article(rally, match, risk_only)
    assert article["matched_terms"] == ["nyt:Gold"]
    globes = rally | {"keywords": [{"value": "Golden Globes (Awards)"}]}
    assert nyt.to_article(globes, match, risk_only) is None
    heist = rally | {"section_name": "World"}  # tags alone don't count in politics
    assert nyt.to_article(heist, match, risk_only) is None


def test_lead_paragraph_is_ignored_so_all_years_match_alike():
    # The archive dropped lead paragraphs in 2025; older years must not use them
    lead_only = _doc(
        headline={"main": "A Quiet Day"},
        abstract="Nothing happened.",
        lead_paragraph="Apple shares fell sharply.",
    )
    assert nyt.to_article(lead_only, build_matcher(NEWS_MAP)) is None

    etfs = term_etfs(NEWS_MAP)
    assert etfs["nyt:Gold"] == {"GLD"} and etfs["Apple"] == {"QQQ"}
    assert "inflation" not in etfs  # risk words reach ETFs via risk_exposure


def test_rate_limit_is_retried_and_bad_key_stops(monkeypatch):
    statuses = [429, 200]
    sleeps = []

    def get(url, *, params, timeout):
        assert url.endswith("/2025/1.json") and params == {"api-key": "k"}
        status = statuses.pop(0)
        body = {"response": {"docs": [_doc()]}} if status == 200 else {}
        return httpx.Response(status, request=httpx.Request("GET", url), json=body)

    monkeypatch.setattr(httpx, "get", get)
    assert len(nyt.fetch_month(2025, 1, "k", sleep=sleeps.append)) == 1
    assert sleeps == [60.0]

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(401, request=httpx.Request("GET", url)),
    )
    with pytest.raises(nyt.NytError, match="키가 올바르지 않"):
        nyt.fetch_month(2025, 1, "bad", sleep=sleeps.append)


def test_cached_month_is_processed_without_key_or_network(monkeypatch, tmp_path):
    def get(*args, **kwargs):
        raise AssertionError("cached month must not call the API")

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.delenv("NYT_API_KEY", raising=False)
    (tmp_path / "raw").mkdir()
    docs = [_doc(), _doc(section_name="Sports")]
    (tmp_path / "raw" / "2025-01.json.gz").write_bytes(
        gzip.compress(json.dumps(docs).encode("utf-8"))
    )
    news_map = tmp_path / "map.yaml"
    news_map.write_text(json.dumps(NEWS_MAP), encoding="utf-8")  # JSON is YAML

    exit_code = nyt.main(
        ["--start=2025-01", "--end=2025-01", f"--map={news_map}", f"--out={tmp_path}"]
    )
    lines = (tmp_path / "articles" / "2025-01.jsonl").read_text(encoding="utf-8")
    assert exit_code == 0
    assert [json.loads(x)["section"] for x in lines.splitlines()] == ["Business Day"]
