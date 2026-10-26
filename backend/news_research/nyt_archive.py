"""Collect New York Times articles month by month via the NYT Archive API.

Batch command only; never run at API startup:

    python -m backend.news_research.nyt_archive --start 2025-01 --end 2025-12

Needs NYT_API_KEY (environment variable or .env). One request returns a whole
month; it is cached gzip-compressed under data/raw/news/nyt/raw so re-runs do
not call the API again. Only business/world news mentioning a term from
config/etf_news_map.yaml is kept. Output is git-ignored and not redistributed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from backend.news_research.settings import env_value, iter_months
from backend.news_research.term_matcher import build_matcher, risk_only_terms

API_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
SUMMARY_MAX_CHARS = 300  # team rule: summary within 300 characters
SECTIONS = {
    "Business Day",
    "Business",
    "Economy",
    "Your Money",
    "Technology",
    "World",
    "U.S.",
    "Climate",
    "Real Estate",
}
# Drops op-eds, letters, reviews; live blogs carry big events (e.g. 2025 tariffs)
MATERIALS = {"news", "news analysis", "brief", "live blog post"}
# In these sections a risk word alone ("war", "Justice Department") is mostly
# politics: keep such articles only when NYT tagged an economic subject.
GATED_SECTIONS = {"World", "U.S.", "Climate"}
ECONOMIC_TAGS = {
    "International Trade and World Market",
    "Customs (Tariff)",
    "Protectionism (Trade)",
    "Embargoes and Sanctions",
    "Oil (Petroleum) and Gasoline",
    "Natural Gas",
    "Economic Conditions and Trends",
    "United States Economy",
    "Recession and Depression",
    "Inflation (Economics)",
    "Prices (Fares, Fees and Rates)",
    "Interest Rates",
    "Federal Reserve System",
    "Stocks and Bonds",
    "Banking and Financial Institutions",
    "Currency",
    "Labor and Jobs",
    "Supply Chain",
    "Factories and Manufacturing",
    "Agriculture and Farming",
    "Company Reports",
}

logger = logging.getLogger(__name__)


class NytError(Exception):
    """A user-displayable NYT API or configuration error."""


def load_api_key(env_path: Path = Path(".env")) -> str:
    key = env_value("NYT_API_KEY", env_path)
    if not key:
        raise NytError(
            "NYT_API_KEY가 없습니다. .env에 NYT_API_KEY=발급받은키 를 추가하세요."
        )
    return key


def fetch_month(
    year: int,
    month: int,
    api_key: str,
    max_retries: int = 4,
    backoff: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    # Error messages never include the URL: its query string holds the API key
    url = API_URL.format(year=year, month=month)
    reason = ""
    for attempt in range(max_retries + 1):
        try:
            response = httpx.get(url, params={"api-key": api_key}, timeout=120.0)
        except httpx.HTTPError as exc:
            reason = f"네트워크 오류({type(exc).__name__})"
        else:
            if response.status_code == 200:
                try:
                    return response.json()["response"]["docs"]
                except (ValueError, KeyError, TypeError) as exc:
                    raise NytError("NYT 응답 형식이 예상과 다릅니다.") from exc
            if response.status_code in (401, 403):
                raise NytError(
                    "NYT API 키가 올바르지 않거나 Archive API가 활성화되지 않았습니다."
                )
            if response.status_code != 429 and response.status_code < 500:
                raise NytError(f"NYT API 오류(HTTP {response.status_code})")
            reason = f"HTTP {response.status_code}"
        if attempt < max_retries:
            delay = backoff * 2**attempt
            logger.warning(
                "NYT %s → %.0f초 후 재시도 (%d/%d)",
                reason,
                delay,
                attempt + 1,
                max_retries,
            )
            sleep(delay)
    raise NytError(f"NYT API {reason}가 계속돼 중단했습니다.")


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_article(
    doc: dict,
    match: Callable[..., list[str]],
    risk_only: frozenset[str] = frozenset(),
) -> dict | None:
    if doc.get("section_name") not in SECTIONS:
        return None
    if (doc.get("type_of_material") or "").lower() not in MATERIALS:
        return None
    title = (doc.get("headline") or {}).get("main") or ""
    summary = doc.get("abstract") or doc.get("snippet") or ""
    # NYT's own tags, e.g. "Interest Rates", "Customs (Tariff)", "Meta Platforms Inc"
    tags = [
        k["value"]
        for k in doc.get("keywords") or []
        if k.get("value") and not k["value"].startswith("internal-")
    ]
    # No lead_paragraph: the 2025+ archive omits it, so using it for 2019~2024
    # would make news features shift between PPO train and test years
    text = "\n".join([title, summary, *tags])
    # Tags alone link ETFs only outside politics sections ("Gold" on a museum heist)
    terms = match(text, () if doc["section_name"] in GATED_SECTIONS else tags)
    if not terms:
        return None
    if (
        doc["section_name"] in GATED_SECTIONS
        and set(terms) <= risk_only
        and not ECONOMIC_TAGS.intersection(tags)
    ):
        return None
    published = datetime.fromisoformat(doc["pub_date"]).astimezone(timezone.utc)
    return {
        "url": doc["web_url"],
        "title": title,
        "published_at": published.isoformat(timespec="seconds"),  # same as GDELT
        "source": "nytimes",
        "domain": "nytimes.com",
        "summary": _truncate(summary, SUMMARY_MAX_CHARS),
        "section": doc["section_name"],
        "matched_terms": terms,
        "nyt_keywords": tags,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NYT Archive API 월별 기사 수집기")
    parser.add_argument("--start", required=True, help="시작 월 YYYY-MM")
    parser.add_argument("--end", required=True, help="종료 월 YYYY-MM (포함)")
    parser.add_argument("--map", type=Path, default=Path("config/etf_news_map.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/news/nyt"))
    parser.add_argument("--min-interval", type=float, default=12.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its URLs contain the key
    with args.map.open(encoding="utf-8") as f:
        news_map = yaml.safe_load(f)
    match, risk_only = build_matcher(news_map), risk_only_terms(news_map)
    raw_dir, article_dir = args.out / "raw", args.out / "articles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    article_dir.mkdir(parents=True, exist_ok=True)

    api_key = None
    try:
        for year, month in iter_months(args.start, args.end):
            label = f"{year}-{month:02d}"
            cache = raw_dir / f"{label}.json.gz"
            if cache.exists():
                docs = json.loads(gzip.decompress(cache.read_bytes()))
            else:
                if api_key is not None:
                    time.sleep(args.min_interval)
                api_key = api_key or load_api_key()
                docs = fetch_month(year, month, api_key)
                cache.write_bytes(gzip.compress(json.dumps(docs).encode("utf-8")))
            articles = [a for d in docs if (a := to_article(d, match, risk_only))]
            with (article_dir / f"{label}.jsonl").open("w", encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
            logger.info(
                "%s: 전체 %d건 → 관련 기사 %d건 저장", label, len(docs), len(articles)
            )
    except NytError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("완료: %s", article_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
