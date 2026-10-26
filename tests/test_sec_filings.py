import json
from datetime import date

import httpx
import pytest

from backend.news_research import sec_filings as sec

SUBMISSION = """<SEC-DOCUMENT>0000320193-24-000067.txt : 20240502
<SEC-HEADER>0000320193-24-000067.hdr.sgml : 20240502
<ACCEPTANCE-DATETIME>20240502163034
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>aapl-20240502.htm
<TEXT>
<html><head><title>aapl-20240502</title></head><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey
0000320193</ix:hidden></ix:header></div>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 8-K</p>
<p><b>Item&#160;2.02 Results of Operations and Financial Condition.</b></p>
<p>On May&nbsp;2, 2024, Apple Inc. issued a press release regarding
Apple&#8217;s financial results for its second fiscal quarter.</p>
<p><b>Item 9.01 Financial Statements and Exhibits.</b></p>
<table><tr><td>99.1</td><td>Press release</td></tr></table>
<p>SIGNATURE</p>
<p>Pursuant to the requirements of the Securities Exchange Act of 1934</p>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>a8-kex991.htm
<TEXT>
<html><body><p>Apple reports second quarter results</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-101.SCH
<SEQUENCE>3
<FILENAME>aapl-20240502.xsd
<TEXT>
<XBRL>schema</XBRL>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def _columns(*rows):
    keys = rows[0].keys()
    return {key: [row[key] for row in rows] for key in keys}


# EDGAR's clock for an 8-K accepted 2024-05-02 16:30:34 New York time
AFTER_CLOSE = "2024-05-03T00:30:34.000Z"


def _row(accession, form="8-K", day="2024-05-02", items="2.02,9.01", at=AFTER_CLOSE):
    return {
        "accessionNumber": accession,
        "form": form,
        "filingDate": day,
        "acceptanceDateTime": at,
        "items": items,
        "primaryDocument": "aapl-20240502.htm",
    }


class FakeClient:
    """Serves canned EDGAR responses and records every URL asked for."""

    def __init__(self, responses):
        self.responses, self.urls = responses, []

    def get(self, url):
        self.urls.append(url)
        if url not in self.responses:
            raise sec.SecNotFound(url)
        body = self.responses[url]
        return body.encode() if isinstance(body, str) else json.dumps(body).encode()


SUBMISSIONS = "https://data.sec.gov/submissions/"
APPLE = SUBMISSIONS + "CIK0000320193.json"
OLDER = SUBMISSIONS + "CIK0000320193-submissions-001.json"
OLDEST = SUBMISSIONS + "CIK0000320193-submissions-002.json"


def _apple_responses():
    return {
        APPLE: {
            "filings": {
                "recent": _columns(
                    _row("0000320193-24-000067"),
                    _row("0000320193-24-000066", form="4", items=""),
                    _row("0000320193-24-000065", items="5.07", day="2024-03-01"),
                ),
                "files": [
                    {
                        "name": "CIK0000320193-submissions-001.json",
                        "filingFrom": "2018-01-01",
                        "filingTo": "2019-12-31",
                    },
                    {
                        "name": "CIK0000320193-submissions-002.json",
                        "filingFrom": "2010-01-01",
                        "filingTo": "2017-12-31",
                    },
                ],
            }
        },
        OLDER: _columns(
            _row("0000320193-19-000010", day="2019-01-29"),
            _row("0000320193-18-000100", day="2018-11-01"),
        ),
    }


def test_config_cik_overrides_a_ticker_that_moved_and_foreign_is_skipped():
    ciks = {"XOM": 2115436, "AAPL": 320193, "TSM": 1046179}
    xom = {"name": "ExxonMobil", "sec_cik": 34088}
    assert sec.company_cik("XOM", xom, ciks) == 34088
    assert sec.company_cik("AAPL", {"name": "Apple Inc"}, ciks) == 320193
    assert sec.company_cik("TSM", {"foreign": True}, ciks) is None
    assert sec.company_cik("ZZZZ", {}, ciks) is None


def test_filings_are_8k_in_range_and_old_pages_read_only_when_needed(tmp_path):
    client = FakeClient(_apple_responses())
    rows = sec.list_filings(
        client, 320193, date(2019, 1, 1), date(2025, 12, 31), tmp_path
    )
    assert sorted(r["accessionNumber"] for r in rows) == [
        "0000320193-19-000010",
        "0000320193-24-000065",
        "0000320193-24-000067",
    ]
    assert OLDEST not in client.urls
    again = FakeClient({})
    sec.list_filings(again, 320193, date(2019, 1, 1), date(2025, 12, 31), tmp_path)
    assert again.urls == []  # served from the gzip cache


def test_markup_the_html_parser_rejects_still_yields_text():
    body = (
        "<html><body><p>Revenue rose &amp; margins held</p>"
        "<![GCALEC x]><style>p {}</style><p>Costs fell</p></body></html>"
    )
    assert sec.html_to_text(body) == "Revenue rose & margins held\n\nCosts fell"


def test_filing_text_keeps_items_and_press_release_without_boilerplate():
    text, summary = sec.filing_text(SUBMISSION, "8-K", ["2.02", "9.01"])
    assert text.startswith("Item 2.02 Results of Operations")
    assert "May 2, 2024" in text and "Apple’s financial results" in text
    assert "99.1 Press release" in text
    assert "[EX-99.1]\nApple reports second quarter results" in text
    for hidden in ("UNITED STATES", "EntityCentralIndexKey", "Pursuant", "schema"):
        assert hidden not in text
    assert summary.startswith("Item 2.02") and len(summary) <= 300


@pytest.mark.parametrize(
    "clock,accepted",
    [
        # XOM, index page 06:31:49: this clock is true UTC
        ("2025-10-31T10:31:49.000Z", "2025-10-31T06:31:49-04:00"),
        # AAPL, index page 16:30:25: the later candidate, still after the close
        ("2025-08-01T00:30:25.000Z", "2025-07-31T20:30:25-04:00"),
        # index page 20:18:19: the later candidate (00:18) is outside EDGAR hours
        ("2020-07-31T04:18:19.000Z", "2020-07-30T20:18:19-04:00"),
    ],
)
def test_edgar_clock_never_places_a_filing_early(clock, accepted):
    assert sec.accepted_at(clock).isoformat() == accepted


def test_record_title_url_and_new_york_time():
    after_close = sec.to_record(
        _row("0000320193-24-000067"), "AAPL", "Apple Inc", 320193, ["QQQ"]
    )
    assert after_close["title"] == (
        "Apple Inc 8-K: Results of Operations and Financial Condition"
    )
    assert after_close["url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000067/aapl-20240502.htm"
    )
    assert after_close["published_at"] == "2024-05-02T20:30:34-04:00"
    morning = sec.to_record(
        _row("x", at="2024-05-02T17:05:00.000Z"), "AAPL", "Apple Inc", 1, []
    )
    assert morning["published_at"] == "2024-05-02T13:05:00-04:00"
    undated = sec.to_record(_row("y", items="9.01", at=""), "AAPL", "Apple Inc", 1, [])
    assert undated["title"].endswith("Financial Statements and Exhibits")
    assert undated["time_precision"] == "day"
    assert undated["published_at"] == "2024-05-02"


def test_company_run_fetches_text_once_and_resumes(tmp_path):
    responses = _apple_responses()
    txt = (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000067/0000320193-24-000067.txt"
    )
    responses[txt] = SUBMISSION
    client = FakeClient(responses)
    out = tmp_path / "AAPL.jsonl"
    args = ("AAPL", "Apple Inc", 320193, ["SPY", "QQQ"])
    period = (date(2019, 1, 1), date(2025, 12, 31), tmp_path / "cache", out)
    assert sec.collect_company(client, *args, *period) == (3, 1)
    records = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {r["accession"]: r for r in records}
    assert by_id["0000320193-24-000065"]["text"] is None  # vote results: routine
    assert by_id["0000320193-24-000067"]["summary"].startswith("Item 2.02")
    # fetched files carry the exact time in their header
    assert by_id["0000320193-24-000067"]["published_at"] == (
        "2024-05-02T16:30:34-04:00"
    )
    assert by_id["0000320193-24-000067"]["etfs"] == ["SPY", "QQQ"]
    # 2019 8-K has no submission file in the fake: kept as a listing only
    assert by_id["0000320193-19-000010"]["text"] is None
    again = FakeClient({})
    assert sec.collect_company(again, *args, *period) == (0, 0)
    assert again.urls == []


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_requests_are_spaced_identified_and_rate_limits_retried(monkeypatch):
    fake = FakeTime()
    statuses = [429, 200, 200]

    def get(url, *, headers, timeout, follow_redirects):
        assert headers["User-Agent"] == "FirstClass a@b.c"
        return httpx.Response(
            statuses.pop(0), request=httpx.Request("GET", url), content=b"{}"
        )

    monkeypatch.setattr(sec.httpx, "get", get)
    client = sec.SecClient(
        "FirstClass a@b.c", sleep=fake.sleep, clock=fake.clock, backoff=30.0
    )
    assert client.get("https://www.sec.gov/a") == b"{}"
    assert client.get("https://www.sec.gov/b") == b"{}"
    assert fake.sleeps == [30.0, 1.1]


def test_blocked_request_points_at_the_user_agent(monkeypatch):
    def get(url, **kwargs):
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr(sec.httpx, "get", get)
    with pytest.raises(sec.SecError, match="SEC_USER_AGENT"):
        sec.SecClient("x a@b.c", sleep=lambda s: None).get("https://www.sec.gov/a")


def test_missing_user_agent_stops_before_any_request(monkeypatch, caplog):
    monkeypatch.setattr(sec, "env_value", lambda name: "")
    monkeypatch.setattr(sec.httpx, "get", lambda *a, **k: pytest.fail("no request"))
    assert sec.main([]) == 2
    assert "SEC_USER_AGENT" in caplog.text
