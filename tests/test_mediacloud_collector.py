import gzip
import json

import httpx
import pytest

from backend.news_research import mediacloud_collector as mc
from backend.news_research.term_matcher import build_matcher, risk_only_terms

NEWS_MAP = {
    "companies": {"AAPL": {"aliases": ["Apple"]}},
    "etfs": {"QQQ": {"holdings": {"AAPL": 0.07}, "keywords": []}},
    "risk_type_keywords": {"geopolitical": ["war"]},
}


def _story(i, title="Apple shares fall", language="en", day="2025-06-02"):
    return {
        "id": f"s{i}",
        "url": f"https://www.reuters.com/a/{i}",
        "title": title,
        "language": language,
        "publish_date": day,
        "media_name": "reuters.com",
    }


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _client(fake=None, **kwargs):
    fake = fake or FakeTime()
    return mc.MediaCloudClient("k", sleep=fake.sleep, clock=fake.clock, **kwargs)


@pytest.mark.parametrize(
    "raw,fixed",
    [
        ("Irelandâ\x80\x99s peat bogs", "Ireland’s peat bogs"),
        ("Château owners sue", "Château owners sue"),
        ("Plain title", "Plain title"),
    ],
)
def test_mojibake_titles_are_repaired_without_touching_clean_text(raw, fixed):
    assert mc.fix_mojibake(raw) == fixed


def test_only_english_titles_with_config_terms_become_articles():
    match = build_matcher(NEWS_MAP)
    article = mc.to_article(_story(1), match, "reuters", "reuters.com")

    assert article["published_at"] == "2025-06-02"
    assert article["time_precision"] == "day"
    assert article["matched_terms"] == ["Apple"]
    pt = _story(2, title="Apple cai", language="pt")
    assert mc.to_article(pt, match, "reuters", "reuters.com") is None
    quiet = _story(3, title="Rain expected Monday")
    assert mc.to_article(quiet, match, "reuters", "reuters.com") is None


@pytest.mark.parametrize(
    "url,title",
    [
        (
            "https://www.marketwatch.com/~r/marketwatch/realtimeheadlines"
            "/~3/x/quotes.asp",
            "Apple Inc.",
        ),
        ("https://www.marketwatch.com/investing/stock/aapl", "Apple shares fall"),
        ("https://www.marketwatch.com/guides/solar/apple-review/", "Apple review"),
        (
            "https://www.marketwatch.com/bulletins/redirect/go",
            "AAPL Stock Price | Apple Inc. Stock Quote (U.S.: Nasdaq)",
        ),
    ],
)
def test_quote_and_guide_pages_are_not_articles(url, title):
    story = {**_story(1, title), "url": url}
    assert mc.to_article(story, build_matcher(NEWS_MAP), "mw", "mw.com") is None


def test_repeated_headlines_keep_the_earliest_direct_link():
    match = build_matcher(NEWS_MAP)
    stories = [
        {**_story(1, "Apple shares fall", day="2025-06-03")},
        {
            **_story(2, "Apple Shares Fall", day="2025-06-02"),
            "url": "https://www.marketwatch.com/bulletins/redirect/go?x=1",
        },
        {
            **_story(3, "Apple shares fall", day="2025-06-02"),
            "url": "https://www.marketwatch.com/story/apple-shares-fall",
        },
        _story(4, "Apple warns", day="2025-06-03"),
    ]
    kept = mc.drop_repeats([mc.to_article(s, match, "mw", "mw.com") for s in stories])
    assert [(a["published_at"], a["url"]) for a in kept] == [
        ("2025-06-02", "https://www.marketwatch.com/story/apple-shares-fall"),
        ("2025-06-03", "https://www.reuters.com/a/4"),
    ]


def test_daily_rank_puts_company_news_before_risk_words():
    match = build_matcher(NEWS_MAP)
    articles = [
        mc.to_article(_story(1, "War fears grow"), match, "reuters", "reuters.com"),
        mc.to_article(_story(2, "Apple warns"), match, "reuters", "reuters.com"),
        mc.to_article(
            _story(3, "War and Apple", day="2025-06-03"), match, "reuters", "x"
        ),
    ]
    mc.add_daily_rank(articles, risk_only_terms(NEWS_MAP))
    ranks = {a["url"][-1]: a["daily_rank"] for a in articles}
    assert ranks == {"2": 1, "1": 2, "3": 1}


def test_pages_are_spaced_and_rate_limits_retried(monkeypatch):
    fake = FakeTime()
    statuses = [429, 200, 200]

    def get(url, *, headers, params, timeout):
        assert headers["Authorization"] == "Token k"
        assert params["ss"] == "1089" and params["page_size"] == 1000
        status = statuses.pop(0)
        body = {"stories": [_story(1)], "pagination_token": None}
        return httpx.Response(status, request=httpx.Request("GET", url), json=body)

    monkeypatch.setattr(httpx, "get", get)
    client = _client(fake)
    day = mc.date(2025, 6, 2)
    client.fetch_page(1089, "language:en", day, day)
    client.fetch_page(1089, "language:en", day, day)
    assert fake.sleeps == [60.0, 31.0]  # backoff, then the 2/minute spacing


def test_bad_key_is_reported(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: httpx.Response(401, request=httpx.Request("GET", url)),
    )
    with pytest.raises(mc.MediaCloudError, match="MEDIACLOUD_API_KEY"):
        _client().fetch_page(1089, "*", mc.date(2025, 6, 1), mc.date(2025, 6, 30))


class PagedClient:
    """Two pages; optionally fails after the first to simulate an interruption."""

    def __init__(self, fail_after_first=False, reject_token=False):
        self.calls, self.fail, self.reject = [], fail_after_first, reject_token

    def fetch_page(self, source_id, query, start, end, token=None):
        self.calls.append(token)
        if token and self.reject:
            raise mc.MediaCloudRejected("HTTP 400")
        if token is None:
            if self.fail and len(self.calls) > 1:
                raise mc.MediaCloudError("HTTP 429")
            return [_story(1), _story(2)], "t1"
        if self.fail:
            raise mc.MediaCloudError("HTTP 429")
        return [_story(2), _story(3)], None


def test_interrupted_month_resumes_from_saved_token(tmp_path):
    with pytest.raises(mc.MediaCloudError):
        mc.fetch_month(PagedClient(fail_after_first=True), 2025, 6, tmp_path, 1, "*")
    assert json.loads((tmp_path / "2025-06.state.json").read_text())["token"] == "t1"

    resumed = PagedClient()
    stories = mc.fetch_month(resumed, 2025, 6, tmp_path, 1, "*")
    assert resumed.calls == ["t1"]
    assert sorted(s["id"] for s in stories) == ["s1", "s2", "s3"]  # s2 deduped
    assert (tmp_path / "2025-06.jsonl.gz").exists()
    assert not (tmp_path / "2025-06.state.json").exists()
    assert mc.fetch_month(PagedClient(fail_after_first=True), 2025, 6, tmp_path, 1, "*")


def test_rejected_resume_token_restarts_the_month(tmp_path):
    (tmp_path / "2025-06.part.jsonl").write_text(json.dumps(_story(9)) + "\n")
    (tmp_path / "2025-06.state.json").write_text(json.dumps({"token": "expired"}))
    client = PagedClient(reject_token=True)
    client.reject = True

    with pytest.raises(mc.MediaCloudRejected):  # the fresh "t1" page is refused too
        mc.fetch_month(client, 2025, 6, tmp_path, 1, "*")
    assert client.calls == ["expired", None, "t1"]
    saved = (tmp_path / "2025-06.part.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in saved] == ["s1", "s2"]  # s9 discarded


def test_finished_month_file_is_gzip_jsonl(tmp_path):
    mc.fetch_month(PagedClient(), 2025, 6, tmp_path, 1, "*")
    lines = gzip.decompress((tmp_path / "2025-06.jsonl.gz").read_bytes()).splitlines()
    assert len(lines) == 4  # raw pages kept as-is; dedupe happens on read
