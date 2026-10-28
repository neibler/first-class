from datetime import date

import pytest

from backend.news_research import risk_scoring as rs

# Thu 2022-06-16, Fri 06-17, (weekend), Mon 06-20 closed (Juneteenth), Tue 06-21
DAYS = [date(2022, 6, 16), date(2022, 6, 17), date(2022, 6, 21)]
NEWS_MAP = {
    "risk_types": ["interest_rate", "inflation", "earnings"],
    "companies": {"XOM": {"aliases": ["Exxon"]}, "NVDA": {"aliases": ["Nvidia"]}},
    "etfs": {
        "TLT": {
            "holdings": {},
            "keywords": [],
            "risk_exposure": {"interest_rate": 1.0, "inflation": 0.9},
            "hedge_for": ["recession"],
        },
        "TIP": {
            "holdings": {},
            "keywords": [],
            "risk_exposure": {"interest_rate": 0.6},
            "hedge_for": ["inflation"],
        },
        "XLE": {
            "holdings": {"XOM": 0.24},
            "keywords": ["OPEC"],
            "risk_exposure": {"interest_rate": 0.2, "earnings": 0.5},
            "hedge_for": [],
        },
        "SPY": {
            "holdings": {"NVDA": 0.0836},
            "keywords": [],
            "risk_exposure": {"earnings": 0.7},
            "hedge_for": [],
        },
    },
}


def _tag(url, when, types, level=0.75, confidence=0.9, terms=(), relevant=True):
    article = {"url": url, "published_at": when, "source": "nytimes"}
    article["matched_terms"] = list(terms)
    label = {
        "market_relevant": relevant,
        "risk_types": types,
        "risk_level": level,
        "confidence": confidence,
        "evidence": url,
    }
    return article, label


def _rows(tagged, **kw):
    rows = rs.score_days(tagged, NEWS_MAP, DAYS, **kw)
    return {(r["date"], r["ticker"]): r for r in rows}


@pytest.mark.parametrize(
    "published,precision,expected",
    [
        ("2022-06-16T19:59:00+00:00", "second", date(2022, 6, 16)),  # 15:59 ET
        ("2022-06-16T20:01:00+00:00", "second", date(2022, 6, 17)),  # after close
        ("2022-06-16T13:05:00-04:00", "second", date(2022, 6, 16)),  # SEC, morning
        ("2022-06-16T16:30:34-04:00", "second", date(2022, 6, 17)),  # SEC, 16:30
        ("2022-06-18T14:00:00+00:00", "second", date(2022, 6, 21)),  # Sat -> Tue
        ("2022-06-16", "day", date(2022, 6, 17)),  # date only: next trading day
        ("2022-06-21T21:00:00+00:00", "second", None),  # beyond the calendar
    ],
)
def test_articles_land_on_the_first_trading_day_they_were_known(
    published, precision, expected
):
    assert rs.trading_day(published, precision, DAYS) == expected


def test_holdings_link_by_weight_and_keywords_fully():
    links = rs.direct_link_index(NEWS_MAP, rs.ScoringParams())
    assert links["Exxon"] == {"XLE": 1.0}  # 24% x 5 -> capped at 1
    assert links["Nvidia"]["SPY"] == pytest.approx(0.418)
    assert links["OPEC"] == {"XLE": 1.0}


def test_rate_hike_spreads_by_exposure_and_skips_hedges():
    fed = _tag("fed", "2022-06-16T18:00:00+00:00", ["interest_rate", "inflation"])
    day = _rows([fed])
    score = 0.75 * 0.9
    assert day[("2022-06-16", "TLT")]["risk_score"] == pytest.approx(score, abs=1e-4)
    assert day[("2022-06-16", "TLT")]["risk_type"] == "interest_rate"
    assert day[("2022-06-16", "XLE")]["risk_score"] == pytest.approx(score * 0.2, 1e-3)
    # TIP hedges inflation, so only the rate path counts
    assert day[("2022-06-16", "TIP")]["risk_score"] == pytest.approx(score * 0.6, 1e-3)
    assert day[("2022-06-16", "SPY")]["risk_score"] == 0.0  # no rate exposure


def test_each_risk_type_carries_forward_on_its_own():
    fed = _tag("fed", "2022-06-16T18:00:00+00:00", ["interest_rate"])
    cpi = _tag("cpi", "2022-06-17T13:00:00+00:00", ["inflation"], level=0.2)
    rows = _rows([fed, cpi])
    friday, tuesday = rows[("2022-06-17", "TLT")], rows[("2022-06-21", "TLT")]
    # a small inflation story does not erase Thursday's rate risk
    assert friday["risk_score"] == pytest.approx(0.675, abs=1e-4)
    assert friday["risk_type"] == "interest_rate"
    assert (friday["has_news"], friday["days_since_news"]) == (1, 0)
    assert (tuesday["has_news"], tuesday["days_since_news"]) == (0, 1)
    assert tuesday["risk_score"] == friday["risk_score"]


def test_secondary_mentions_can_raise_but_not_lower_a_risk():
    cpi = _tag("cpi", "2022-06-16T13:00:00+00:00", ["inflation"], level=0.7)
    fed = _tag("fed", "2022-06-17T18:00:00+00:00", ["interest_rate", "inflation"])
    friday = _rows([cpi, fed])[("2022-06-17", "TLT")]
    # inflation stays at Thursday's 0.7*0.9*0.9; the hike's half-weight
    # inflation mention (0.675*0.9*0.5) does not replace it
    assert friday["risk_score"] == pytest.approx(0.675, abs=1e-4)
    assert friday["risk_type"] == "interest_rate;inflation"

    only_secondary = _rows([fed])[("2022-06-17", "TLT")]
    assert only_secondary["risk_type"] == "interest_rate"


def test_optional_ttl_expires_stale_risk_types():
    fed = _tag("fed", "2022-06-16T18:00:00+00:00", ["interest_rate"])
    params = rs.ScoringParams(type_ttl=1)
    rows = {
        (r["date"], r["ticker"]): r
        for r in rs.score_days([fed], NEWS_MAP, DAYS, params=params)
    }
    assert rows[("2022-06-17", "TLT")]["risk_score"] == pytest.approx(0.675, 1e-3)
    tuesday = rows[("2022-06-21", "TLT")]  # two trading days later: expired
    assert (tuesday["risk_score"], tuesday["risk_type"]) == (0.0, "")
    assert tuesday["days_since_news"] == 2
    # without expiry the value would carry on
    forever = rs.score_days(
        [fed], NEWS_MAP, DAYS, params=rs.ScoringParams(type_ttl=None)
    )
    assert forever[-4]["ticker"] == "TLT" and forever[-4]["risk_score"] > 0
    assert rs.ScoringParams().type_ttl == 20  # default agreed for the PPO feed


def test_many_similar_articles_do_not_inflate_the_score():
    one = [_tag("a0", "2022-06-16T13:00:00+00:00", ["earnings"], terms=["Exxon"])]
    many = [
        _tag(f"a{i}", "2022-06-16T13:00:00+00:00", ["earnings"], terms=["Exxon"])
        for i in range(10)
    ]
    key = ("2022-06-16", "XLE")
    assert _rows(one)[key]["risk_score"] == _rows(many)[key]["risk_score"]


def test_irrelevant_news_and_days_before_any_news_score_zero():
    noise = _tag("x", "2022-06-16T13:00:00+00:00", [], relevant=False)
    first = _rows([noise])[("2022-06-16", "TLT")]
    assert first["risk_score"] == 0.0 and first["has_news"] == 0
    assert first["days_since_news"] == rs.MAX_STALE_DAYS


@pytest.mark.parametrize(
    "ttl,expected,url", [(1, 0.09, "new"), (2, 0.8, "old"), (None, 0.8, "old")]
)
def test_secondary_news_cannot_revive_expired_risk(ttl, expected, url):
    old = _tag(
        "old", "2022-06-16T13:00:00+00:00", ["interest_rate"], level=0.8, confidence=1.0
    )
    # TLT has no earnings exposure: only the secondary rate risk affects it.
    new = _tag(
        "new", "2022-06-21T13:00:00+00:00", ["earnings", "interest_rate"], level=0.2
    )
    row = _rows([old, new], params=rs.ScoringParams(type_ttl=ttl))[
        ("2022-06-21", "TLT")
    ]
    assert row["risk_score"] == pytest.approx(expected)
    assert row["evidence_url"] == url
    assert (row["has_news"], row["days_since_news"]) == (1, 0)


def test_default_ttl_expires_and_secondary_news_starts_a_new_score():
    from datetime import timedelta

    days = [DAYS[0] + timedelta(days=i) for i in range(50)]
    days = [
        day for day in days if day.weekday() < 5
    ]  # Explicit synthetic trading calendar.
    old = _tag(
        "old", f"{days[0]}T13:00:00+00:00", ["interest_rate"], level=0.8, confidence=1.0
    )
    new = _tag(
        "new", f"{days[30]}T13:00:00+00:00", ["earnings", "interest_rate"], level=0.2
    )
    rows = [
        r for r in rs.score_days([old, new], NEWS_MAP, days) if r["ticker"] == "TLT"
    ]
    assert rows[20]["risk_score"] == 0.8
    assert rows[21]["risk_score"] == rows[29]["risk_score"] == 0.0
    assert rows[30]["risk_score"] == rows[31]["risk_score"] == 0.09
    assert rows[30]["evidence_url"] == "new"


def test_future_and_after_close_articles_do_not_change_earlier_scores():
    morning = _tag("morning", "2022-06-16T13:00:00+00:00", ["interest_rate"], level=0.2)
    close = _tag("close", "2022-06-16T20:00:00+00:00", ["interest_rate"], level=1)
    future = _tag("future", "2022-06-22T13:00:00+00:00", ["interest_rate"], level=1)
    baseline = _rows([morning])
    assert _rows([morning, future]) == baseline
    changed = _rows([morning, close])
    assert changed[("2022-06-16", "TLT")] == baseline[("2022-06-16", "TLT")]
    assert changed[("2022-06-17", "TLT")]["risk_score"] == 0.9
