"""Collect news headlines month by month via the Media Cloud search API.

Batch command only; never run at API startup:

    python -m backend.news_research.mediacloud_collector --start 2019-01 --end 2025-12

Needs MEDIACLOUD_API_KEY (environment variable or .env). Media Cloud returns
titles, URLs and publish *dates* only (no time of day, no body), so scoring must
apply these articles from the next trading day to avoid look-ahead. Search
allows 2 requests/minute; each month is paged 1,000 stories at a time and
checkpointed under data/raw/news/mediacloud/<source>/raw so interrupted runs
resume. Output is git-ignored and not redistributed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import time
from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

from backend.news_research.settings import env_value, iter_months
from backend.news_research.term_matcher import build_matcher, risk_only_terms

API_URL = "https://search.mediacloud.org/api/search/story-list"
PLATFORM = "onlinenews-mediacloud"
PAGE_SIZE = 1000
REUTERS_SOURCE_ID = 1089  # 73819 is a near-empty duplicate "reuters.com" entry
# Quote, tool and affiliate-guide pages that MarketWatch feeds mix in with news
NON_ARTICLE_PATH = re.compile(r"quotes\.asp$|^/(?:investing|guides|tools)/")
NON_ARTICLE_TITLE = re.compile(
    r"Stock (?:Price|Quote) [|(]"
    r"|^(?:Dow Jones Industrial Average|Multiple Quotes Tool)$"
)
REDIRECT_PATH = re.compile(r"^/~r/|/redirect/")

logger = logging.getLogger(__name__)


class MediaCloudError(Exception):
    """A user-displayable Media Cloud API or configuration error."""


class MediaCloudRejected(MediaCloudError):
    """The server refused the request itself (4xx), e.g. an expired page token."""


class MediaCloudClient:
    """Story-list pages with the server's 2/minute pace and backoff on 429/5xx."""

    def __init__(
        self,
        api_key: str,
        min_interval: float = 31.0,
        max_retries: int = 4,
        backoff: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Accept": "application/json",
        }
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.clock = clock
        self._last_request: float | None = None

    def fetch_page(
        self,
        source_id: int,
        query: str,
        start: date,
        end: date,
        token: str | None = None,
    ) -> tuple[list[dict], str | None]:
        params = {
            "q": query,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "platform": PLATFORM,
            "ss": str(source_id),
            "page_size": PAGE_SIZE,
        }
        if token:
            params["pagination_token"] = token
        reason = ""
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = httpx.get(
                    API_URL, headers=self.headers, params=params, timeout=120.0
                )
            except httpx.HTTPError as exc:
                reason = f"네트워크 오류({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return data["stories"], data.get("pagination_token")
                    except (ValueError, KeyError, TypeError) as exc:
                        raise MediaCloudError(
                            "Media Cloud 응답 형식이 다릅니다."
                        ) from exc
                if response.status_code in (401, 403):
                    raise MediaCloudError("MEDIACLOUD_API_KEY가 올바르지 않습니다.")
                if response.status_code != 429 and response.status_code < 500:
                    raise MediaCloudRejected(
                        f"Media Cloud 요청 거부(HTTP {response.status_code})"
                    )
                reason = f"HTTP {response.status_code}"
            finally:
                self._last_request = self.clock()
            if attempt < self.max_retries:
                delay = self.backoff * 2**attempt
                logger.warning(
                    "Media Cloud %s → %.0f초 후 재시도 (%d/%d)",
                    reason,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self.sleep(delay)
        raise MediaCloudError(f"Media Cloud {reason}가 계속돼 중단했습니다.")

    def _wait_turn(self) -> None:
        if self._last_request is None:
            return
        remaining = self.min_interval - (self.clock() - self._last_request)
        if remaining > 0:
            self.sleep(remaining)


def fetch_month(
    client: MediaCloudClient,
    year: int,
    month: int,
    raw_dir: Path,
    source_id: int,
    query: str,
) -> list[dict]:
    """Return a month of raw stories, resuming from the last saved page."""
    label = f"{year}-{month:02d}"
    final = raw_dir / f"{label}.jsonl.gz"
    if final.exists():
        lines = gzip.decompress(final.read_bytes()).decode("utf-8").splitlines()
        # A crash between saving a page and its token re-fetches that page once
        unique = {}
        for line in filter(None, lines):
            story = json.loads(line)
            unique.setdefault(story.get("id") or story["url"], story)
        return list(unique.values())

    part, state = raw_dir / f"{label}.part.jsonl", raw_dir / f"{label}.state.json"
    token = None
    if part.exists() and state.exists():
        token = json.loads(state.read_text(encoding="utf-8")).get("token")
    if not token:  # no usable checkpoint: start the month over
        part.unlink(missing_ok=True)
    start, end = date(year, month, 1), date(year, month, monthrange(year, month)[1])
    pages = 0
    while True:
        try:
            stories, token = client.fetch_page(source_id, query, start, end, token)
        except MediaCloudRejected:
            if not (token and pages == 0):
                raise
            logger.warning(
                "%s: 저장된 페이지 토큰이 거부돼 처음부터 다시 받습니다.", label
            )
            part.unlink(missing_ok=True)
            state.unlink(missing_ok=True)
            token = None
            continue
        with part.open("a", encoding="utf-8") as f:
            for story in stories:
                f.write(json.dumps(story, ensure_ascii=False) + "\n")
        pages += 1
        logger.info("%s: %d페이지째 %d건", label, pages, len(stories))
        if not token:
            break
        state.write_text(json.dumps({"token": token}), encoding="utf-8")
    final.write_bytes(gzip.compress(part.read_bytes()))
    part.unlink()
    state.unlink(missing_ok=True)
    return fetch_month(client, year, month, raw_dir, source_id, query)


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 read as Latin-1/CP1252: "Irelandâ\\x80\\x99s" -> "Ireland’s"."""
    if "â" not in text and "Ã" not in text:
        return text
    for encoding in ("latin-1", "cp1252"):
        try:
            return text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def to_article(
    story: dict, match: Callable[..., list[str]], source: str, domain: str
) -> dict | None:
    if story.get("language") != "en":
        return None
    if NON_ARTICLE_PATH.search(urlparse(story["url"]).path):
        return None
    title = " ".join(fix_mojibake(story.get("title") or "").split())
    if NON_ARTICLE_TITLE.search(title):
        return None
    terms = match(title)
    if not terms:
        return None
    return {
        "url": story["url"],
        "title": title,
        "published_at": str(story["publish_date"])[:10],
        "time_precision": "day",  # no time of day: apply from the next trading day
        "source": source,
        "domain": domain,
        "summary": None,
        "matched_terms": terms,
    }


def drop_repeats(articles: list[dict]) -> list[dict]:
    """Keep one article per headline, from its earliest day.

    MarketWatch publishes the same story under several feed and redirect URLs
    (about 40% of a 2022 week); repeats would crowd the daily top-N. A direct
    URL is preferred over a redirect so evidence links stay usable.
    """
    kept: dict[str, dict] = {}
    for article in sorted(
        articles,
        key=lambda a: (
            a["published_at"],
            bool(REDIRECT_PATH.search(urlparse(a["url"]).path)),
        ),
    ):
        kept.setdefault(article["title"].casefold(), article)
    return list(kept.values())


def add_daily_rank(articles: list[dict], risk_only: frozenset[str]) -> None:
    """Rank each day's articles: company/ETF terms first, then more matches.

    Scoring can keep the top N per day so that Media Cloud's very uneven yearly
    volume (Reuters 2019 ~40x 2023) does not inflate risk in high-volume years.
    """
    by_day: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        by_day[article["published_at"]].append(article)
    for day_articles in by_day.values():
        day_articles.sort(
            key=lambda a: (
                -sum(t not in risk_only for t in a["matched_terms"]),
                -len(a["matched_terms"]),
                a["url"],
            )
        )
        for rank, article in enumerate(day_articles, 1):
            article["daily_rank"] = rank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Media Cloud 월별 헤드라인 수집기")
    parser.add_argument("--start", required=True, help="시작 월 YYYY-MM")
    parser.add_argument("--end", required=True, help="종료 월 YYYY-MM (포함)")
    parser.add_argument("--source-id", type=int, default=REUTERS_SOURCE_ID)
    parser.add_argument("--source", default="reuters")
    parser.add_argument("--domain", default="reuters.com")
    parser.add_argument("--query", default="language:en")
    parser.add_argument("--map", type=Path, default=Path("config/etf_news_map.yaml"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-interval", type=float, default=31.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = args.out or Path("data/raw/news/mediacloud") / args.source
    raw_dir, article_dir = out / "raw", out / "articles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    article_dir.mkdir(parents=True, exist_ok=True)
    with args.map.open(encoding="utf-8") as f:
        news_map = yaml.safe_load(f)
    match, risk_only = build_matcher(news_map), risk_only_terms(news_map)

    api_key = env_value("MEDIACLOUD_API_KEY")
    if not api_key:
        logger.error("MEDIACLOUD_API_KEY가 없습니다. .env에 추가하세요.")
        return 2
    client = MediaCloudClient(api_key, min_interval=args.min_interval)
    try:
        for year, month in iter_months(args.start, args.end):
            label = f"{year}-{month:02d}"
            stories = fetch_month(
                client, year, month, raw_dir, args.source_id, args.query
            )
            articles = drop_repeats(
                [
                    a
                    for s in stories
                    if (a := to_article(s, match, args.source, args.domain))
                ]
            )
            add_daily_rank(articles, risk_only)
            with (article_dir / f"{label}.jsonl").open("w", encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
            logger.info(
                "%s: 전체 %d건 → 관련 기사 %d건 저장",
                label,
                len(stories),
                len(articles),
            )
    except MediaCloudError as exc:
        logger.error("%s 같은 명령을 다시 실행하면 이어서 수집합니다.", exc)
        return 2
    logger.info("완료: %s", article_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
