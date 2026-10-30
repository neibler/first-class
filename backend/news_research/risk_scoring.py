"""Turn LLM risk tags on articles into daily per-ETF risk scores for the PPO.

Batch command only; never run at API startup:

    python -m backend.news_research.risk_scoring --labels labels.jsonl \\
        --articles "data/raw/news/nyt/articles/*.jsonl" --calendar trading_days.csv

Steps (all deterministic, no LLM):
  1. trading day: NYT after 16:00 ET or on a closed day -> next trading day;
     date-only sources (Reuters via Media Cloud) -> next trading day
  2. article score = risk_level x confidence x source weight
  3. link to each ETF = max(direct link, risk exposure); secondary risk types
     count half; a risk the ETF hedges (hedge_for) counts zero
  4. per (ETF, risk type, day): mean of the top-k article values; on days
     without such news the previous value carries forward (team rule, per type)
     for up to 20 trading days, then that risk type expires
  5. ETF risk_score = largest risk-type value; risk_type = that type(s)
Output columns follow the team rule: date, ticker, risk_score, risk_type,
has_news (+ days_since_news and the evidence behind the score).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from backend.news_research.term_matcher import NYT_TAG_PREFIX

NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
MAX_STALE_DAYS = 20  # days_since_news is capped here (team rule 6 proposal)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringParams:
    holding_scale: float = 5.0  # weight x 5, capped at 1: a 20%+ holding links fully
    secondary_factor: float = 0.5  # later risk types count half
    top_k: int = 3  # mean of the strongest k articles: insensitive to volume
    type_share: float = 0.8  # report every type within 80% of the top score
    # A risk type expires this many trading days after its last news, so rare
    # types (credit, market crash) do not linger for months. None = never.
    type_ttl: int | None = 20


def trading_day(
    published_at: str, time_precision: str, trading_days: list[date]
) -> date | None:
    """First trading day on which the article could have been known."""
    if time_precision == "day":
        index = bisect.bisect_right(trading_days, date.fromisoformat(published_at[:10]))
    else:
        local = datetime.fromisoformat(published_at).astimezone(NEW_YORK)
        index = bisect.bisect_left(trading_days, local.date())
        if (
            index < len(trading_days)
            and trading_days[index] == local.date()
            and local.time() >= MARKET_CLOSE
        ):
            index += 1
    return trading_days[index] if index < len(trading_days) else None


def direct_link_index(news_map: dict, params: ScoringParams) -> dict[str, dict]:
    """Matcher label -> {ETF: strength}; holdings scale with their weight."""
    index: dict[str, dict[str, float]] = defaultdict(dict)

    def put(label: str, ticker: str, strength: float) -> None:
        index[label][ticker] = max(index[label].get(ticker, 0.0), strength)

    for ticker, etf in news_map["etfs"].items():
        for symbol, weight in etf["holdings"].items():
            strength = min(1.0, weight * params.holding_scale)
            for alias in news_map["companies"][symbol]["aliases"]:
                put(alias, ticker, strength)
        for keyword in etf["keywords"]:
            put(keyword, ticker, 1.0)
        for tag in etf.get("nyt_tags", []):
            put(NYT_TAG_PREFIX + tag, ticker, 1.0)
    return dict(index)


def article_values(
    article: dict,
    label: dict,
    news_map: dict,
    links: dict[str, dict],
    source_weights: dict[str, float],
    params: ScoringParams,
) -> dict[tuple[str, str], tuple[float, bool]]:
    """(ETF, risk type) -> (value, is primary type); empty if not market relevant."""
    if not label.get("market_relevant") or not label.get("risk_types"):
        return {}
    score = (
        float(label["risk_level"])
        * float(label["confidence"])
        * source_weights.get(article.get("source", ""), 1.0)
    )
    direct: dict[str, float] = defaultdict(float)
    for term in article.get("matched_terms", []):
        for ticker, strength in links.get(term, {}).items():
            direct[ticker] = max(direct[ticker], strength)

    values = {}
    for ticker, etf in news_map["etfs"].items():
        for position, risk_type in enumerate(label["risk_types"]):
            if risk_type in etf.get("hedge_for", []):
                continue
            link = max(direct[ticker], etf["risk_exposure"].get(risk_type, 0.0))
            factor = 1.0 if position == 0 else params.secondary_factor
            value = score * link * factor
            key = (ticker, risk_type)
            if value > 0 and value > values.get(key, (0.0, False))[0]:
                values[key] = (value, position == 0)
    return values


def score_days(
    tagged: list[tuple[dict, dict]],
    news_map: dict,
    trading_days: list[date],
    source_weights: dict[str, float] | None = None,
    params: ScoringParams = ScoringParams(),
) -> list[dict]:
    """Daily rows for every trading day x ETF (team ETF order)."""
    links = direct_link_index(news_map, params)
    source_weights = source_weights or {}
    # (day, ETF, type) -> [(value, is_primary, article, label)]
    found: dict[tuple[date, str, str], list] = defaultdict(list)
    for article, label in tagged:
        day = trading_day(
            article["published_at"],
            article.get("time_precision", "second"),
            trading_days,
        )
        if day is None:
            continue
        values = article_values(article, label, news_map, links, source_weights, params)
        for (ticker, risk_type), (value, primary) in values.items():
            found[(day, ticker, risk_type)].append((value, primary, article, label))

    # (ETF, type) -> (value, article, label, trading-day index of the news
    # behind the value); expiry counts from that index
    state: dict[tuple[str, str], tuple[float, dict, dict, int]] = {}
    last_news: dict[str, int] = {}
    ttl = params.type_ttl
    rows = []
    for day_index, day in enumerate(trading_days):
        for ticker in news_map["etfs"]:
            updated = False
            for risk_type in news_map["risk_types"]:
                hits = found.get((day, ticker, risk_type))
                if not hits:
                    continue  # no news of this type today: keep yesterday's value
                carried = state.get((ticker, risk_type))
                if carried and ttl is not None and day_index - carried[3] > ttl:
                    carried = None  # an expired risk is not revived by new mentions
                state[(ticker, risk_type)] = _update(hits, carried, day_index, params)
                updated = True
            if updated:
                last_news[ticker] = day_index
            rows.append(
                _row(day, ticker, news_map, state, last_news, day_index, params)
            )
    return rows


def _update(
    hits: list, carried: tuple | None, day_index: int, params: ScoringParams
) -> tuple:
    """New (value, article, label, set_on) for one ETF and risk type on a news day.

    risk_level describes an article's first (primary) risk type, so only
    primary mentions may set the value - up or down. A secondary mention is a
    half-weight proxy: it can raise the value but never lower it. set_on is the
    day of the news behind the value: keeping a larger older value does not
    restart its expiry clock. carried is None when there is none or it expired.
    """
    primary = sorted((h for h in hits if h[1]), key=lambda h: h[0], reverse=True)
    secondary = max((h for h in hits if not h[1]), key=lambda h: h[0], default=None)
    if primary:
        top = primary[: params.top_k]
        best = (sum(h[0] for h in top) / len(top), top[0][2], top[0][3], day_index)
    else:
        best = carried or (0.0, None, None, day_index)
    if secondary and secondary[0] > best[0]:
        best = (secondary[0], secondary[2], secondary[3], day_index)
    return best


def _row(day, ticker, news_map, state, last_news, day_index, params) -> dict:
    ttl = params.type_ttl
    current = [
        (state[(ticker, t)][:3], t)
        for t in news_map["risk_types"]
        if (ticker, t) in state
        and (ttl is None or day_index - state[(ticker, t)][3] <= ttl)
    ]
    since = (
        min(day_index - last_news[ticker], MAX_STALE_DAYS)
        if ticker in last_news
        else MAX_STALE_DAYS
    )
    if not current:  # before the first news: team rule 6
        return {
            "date": day.isoformat(),
            "ticker": ticker,
            "risk_score": 0.0,
            "risk_type": "",
            "has_news": 0,
            "days_since_news": since,
            "evidence_url": "",
            "evidence": "",
        }
    current.sort(key=lambda c: c[0][0], reverse=True)
    (best, article, label), _ = current[0]
    types = [t for (s, _, _), t in current if s >= best * params.type_share]
    return {
        "date": day.isoformat(),
        "ticker": ticker,
        "risk_score": round(best, 4),
        "risk_type": ";".join(types),
        "has_news": int(last_news.get(ticker) == day_index),
        "days_since_news": since,
        "evidence_url": article["url"],
        "evidence": label.get("evidence", ""),
    }


def load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="뉴스 위험 태그 → ETF별 일별 risk_score"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True, help="url이 있는 태그 jsonl"
    )
    parser.add_argument(
        "--articles", nargs="+", required=True, help="기사 jsonl (glob)"
    )
    parser.add_argument(
        "--calendar", type=Path, required=True, help="거래일 CSV (date)"
    )
    parser.add_argument("--map", type=Path, default=Path("config/etf_news_map.yaml"))
    parser.add_argument(
        "--sources", type=Path, default=Path("config/news_sources.yaml")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/news_risk.csv")
    )
    parser.add_argument(
        "--type-ttl",
        type=int,
        default=ScoringParams.type_ttl,
        help="위험 유형별 만료 거래일 수 (기본 20, 0이면 만료 없음)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    news_map = yaml.safe_load(args.map.read_text(encoding="utf-8"))
    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["sources"]
    source_weights = {key: s["weight"] for key, s in sources.items()}
    with args.calendar.open(encoding="utf-8") as f:
        trading_days = sorted(date.fromisoformat(r["date"]) for r in csv.DictReader(f))

    labels = {row["url"]: row for row in load_jsonl(args.labels)}
    articles = {a["url"]: a for a in load_jsonl(args.articles)}
    tagged = [
        (articles[url], label) for url, label in labels.items() if url in articles
    ]
    missing = len(labels) - len(tagged)
    params = ScoringParams(type_ttl=args.type_ttl or None)
    rows = score_days(tagged, news_map, trading_days, source_weights, params)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "태그 %d건(기사 없음 %d건) → %d행 저장: %s",
        len(tagged),
        missing,
        len(rows),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
