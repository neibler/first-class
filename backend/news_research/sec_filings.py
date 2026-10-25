"""Collect SEC EDGAR 8-K filings of the ETF holdings for the research agent.

Batch command only; never run at API startup:

    python -m backend.news_research.sec_filings --start 2019-01-01 --end 2025-12-31

The research agent has to cross-check news against company filings, so every
8-K of the US companies in config/etf_news_map.yaml is saved with its items,
acceptance time, document URL and text (the 8-K item section plus EX-99
exhibits such as earnings press releases). SEC asks every client to identify
itself: set SEC_USER_AGENT (environment variable or .env) to a name and contact
e-mail, e.g. "FirstClass Capstone you@example.com". Requests are spaced 1.1 s
apart (project rule: at most one per second). Foreign issuers file 6-K instead,
mostly daily share-buyback notices, and are skipped. Each filing is appended as
soon as it is fetched, so an interrupted run resumes where it stopped. Output
lives under data/raw/filings/sec, is git-ignored and is not redistributed.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import yaml

from backend.news_research.settings import env_value

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"
FORMS = frozenset({"8-K"})
NEW_YORK = ZoneInfo("America/New_York")
SUMMARY_MAX_CHARS = 300  # project rule: summary within 300 characters
TEXT_MAX_CHARS = 100_000
ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to the Code of Ethics",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
# Vote results, bylaw edits and exhibit-only filings: listed, text not fetched
ROUTINE_ITEMS = frozenset(
    {"3.02", "3.03", "5.03", "5.04", "5.05", "5.07", "5.08", "9.01"}
)

_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S)
_TEXT = re.compile(r"<TEXT>(.*?)</TEXT>", re.S)
_HTML = re.compile(r"<(?:html|body|p|div|table|font)\b", re.I)
_ITEM = re.compile(r"\bItem\s*(\d)\s*\.\s*(\d{2})", re.I)
_SIGNATURE = re.compile(r"\n(?:SIGNATURES?|Signatures?)\n")
_ACCEPTANCE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
EDGAR_OPEN, EDGAR_CLOSE = dt_time(6, 0), dt_time(22, 30)  # 22:00 plus late stamps

logger = logging.getLogger(__name__)


class SecError(Exception):
    """A user-displayable EDGAR or configuration error."""


class SecNotFound(SecError):
    """EDGAR has no document at that address."""


class SecClient:
    """EDGAR GETs spaced min_interval apart, with backoff on 429/5xx."""

    def __init__(
        self,
        user_agent: str,
        min_interval: float = 1.1,
        max_retries: int = 4,
        backoff: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.headers = {"User-Agent": user_agent}
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.clock = clock
        self._last_request: float | None = None

    def get(self, url: str) -> bytes:
        reason = ""
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = httpx.get(
                    url, headers=self.headers, timeout=60.0, follow_redirects=True
                )
            except httpx.HTTPError as exc:
                reason = f"네트워크 오류({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code == 404:
                    raise SecNotFound(f"EDGAR에 문서가 없습니다: {url}")
                if response.status_code == 403:
                    raise SecError(
                        "SEC가 요청을 거부했습니다(HTTP 403). SEC_USER_AGENT에"
                        " 이름과 연락처 이메일이 있는지 확인하세요."
                    )
                if response.status_code != 429 and response.status_code < 500:
                    raise SecError(f"EDGAR 요청 거부(HTTP {response.status_code})")
                reason = f"HTTP {response.status_code}"
            finally:
                self._last_request = self.clock()
            if attempt < self.max_retries:
                delay = self.backoff * 2**attempt
                logger.warning(
                    "EDGAR %s → %.0f초 후 재시도 (%d/%d)",
                    reason,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self.sleep(delay)
        raise SecError(f"EDGAR {reason}가 계속돼 중단했습니다.")

    def _wait_turn(self) -> None:
        if self._last_request is None:
            return
        remaining = self.min_interval - (self.clock() - self._last_request)
        if remaining > 0:
            self.sleep(remaining)


def cached_json(client: SecClient, url: str, cache: Path) -> dict:
    """GET a JSON document once and keep it gzip-compressed for re-runs."""
    if cache.exists():
        return json.loads(gzip.decompress(cache.read_bytes()))
    raw = client.get(url)
    data = json.loads(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(raw))
    return data


def load_ciks(client: SecClient, cache_dir: Path) -> dict[str, int]:
    """Ticker -> SEC CIK for every company EDGAR lists."""
    rows = cached_json(client, TICKERS_URL, cache_dir / "company_tickers.json.gz")
    return {row["ticker"].upper(): int(row["cik_str"]) for row in rows.values()}


def company_cik(ticker: str, company: dict, ciks: dict[str, int]) -> int | None:
    """CIK of a US 8-K filer; config sec_cik wins when the ticker changed hands."""
    if company.get("foreign"):
        return None
    return company.get("sec_cik") or ciks.get(ticker.upper())


def filing_rows(block: dict) -> list[dict]:
    """EDGAR's column arrays ({"form": [...], ...}) as one dict per filing."""
    columns = {key: value for key, value in block.items() if isinstance(value, list)}
    count = len(block.get("accessionNumber", []))
    return [{key: value[i] for key, value in columns.items()} for i in range(count)]


def list_filings(
    client: SecClient, cik: int, start: date, end: date, cache_dir: Path
) -> list[dict]:
    """8-K rows filed in [start, end]; older pages are read only if they overlap."""
    name = f"CIK{cik:010d}.json"
    filings = cached_json(
        client, SUBMISSIONS_URL.format(name=name), cache_dir / f"{name}.gz"
    )["filings"]
    rows = filing_rows(filings["recent"])
    for page in filings.get("files", []):
        if page["filingTo"] < start.isoformat() or page["filingFrom"] > end.isoformat():
            continue
        rows += filing_rows(
            cached_json(
                client,
                SUBMISSIONS_URL.format(name=page["name"]),
                cache_dir / f"{page['name']}.gz",
            )
        )
    unique = {row["accessionNumber"]: row for row in rows}
    return [
        row
        for row in unique.values()
        if row.get("form") in FORMS
        and start.isoformat() <= row["filingDate"] <= end.isoformat()
    ]


def accepted_at(value: str) -> datetime:
    """Latest New York acceptance time the submissions JSON clock allows.

    That clock is true UTC for some filings and UTC plus New York's offset again
    for others (filing index pages: XOM 06:31:49 reads "10:31:49Z", AAPL
    16:30:25 reads "00:30:25Z" the next day), with no way to tell them apart.
    Reading it as UTC gives the later candidate, so a filing is never placed
    before it was public; the earlier one is used only when the later one falls
    outside EDGAR's 6:00-22:00 hours. Prefer header_accepted when the file is
    at hand.
    """
    later = (
        datetime.fromisoformat(value[:19])
        .replace(tzinfo=timezone.utc)
        .astimezone(NEW_YORK)
    )
    earlier = (later.replace(tzinfo=None) + later.utcoffset()).replace(tzinfo=NEW_YORK)
    for candidate in (later, earlier):
        if EDGAR_OPEN <= candidate.time() <= EDGAR_CLOSE:
            return candidate
    return later


def header_accepted(submission: str) -> datetime | None:
    """Exact acceptance time from the SGML header of a full submission file."""
    match = _ACCEPTANCE.search(submission[:2000])
    if not match:
        return None
    return datetime.strptime(match[1], "%Y%m%d%H%M%S").replace(tzinfo=NEW_YORK)


def split_documents(submission: str) -> list[dict]:
    """Documents in an EDGAR full submission file (.txt): type, file, body."""
    documents = []
    for block in _DOCUMENT.findall(submission):
        body = _TEXT.search(block)
        documents.append(
            {
                "type": _header(block, "TYPE"),
                "filename": _header(block, "FILENAME"),
                "body": body.group(1) if body else "",
            }
        )
    return documents


def _header(block: str, name: str) -> str:
    match = re.search(rf"<{name}>([^\n<]*)", block)
    return match.group(1).strip() if match else ""


class _TextExtractor(HTMLParser):
    BLOCKS = frozenset(
        {"p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
    )
    # ix:header holds the hidden inline-XBRL facts of the cover page
    HIDDEN = frozenset({"script", "style", "title", "ix:header"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.HIDDEN:
            self.hidden += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.HIDDEN:
            self.hidden = max(0, self.hidden - 1)
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def html_to_text(body: str) -> str:
    """Readable text of an EDGAR document; plain-text exhibits pass through."""
    if _HTML.search(body):
        parser = _TextExtractor()
        try:
            parser.feed(body)
            parser.close()
            body = "".join(parser.parts)
        except AssertionError:  # markup html.parser rejects, e.g. "<![GCALEC"
            body = _strip_tags(body)
    lines = (" ".join(line.split()) for line in body.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _strip_tags(body: str) -> str:
    """Cruder fallback: drop hidden parts, break lines at blocks, remove tags."""
    body = re.sub(r"(?is)<(script|style|title|ix:header)\b.*?</\1\s*>", " ", body)
    body = re.sub(r"(?i)</?(?:p|div|br|tr|li|table|h[1-6])\b[^>]*>", "\n", body)
    return html.unescape(re.sub(r"<[^>]*>", " ", body))


def item_section(text: str, items: list[str]) -> str:
    """The 8-K body from its first reported item up to the signature block."""
    start = next(
        (m.start() for m in _ITEM.finditer(text) if f"{m[1]}.{m[2]}" in items), 0
    )
    signature = _SIGNATURE.search(text, start)
    stop = signature.start() if signature else len(text)
    return text[start:stop].strip()


def filing_text(submission: str, form: str, items: list[str]) -> tuple[str, str]:
    """(Text for RAG, summary): the item section first, then EX-99 exhibits."""
    parts = []
    for document in split_documents(submission):
        if document["type"] == form:
            parts.insert(0, item_section(html_to_text(document["body"]), items))
        elif document["type"].startswith("EX-99"):
            parts.append(f"[{document['type']}]\n{html_to_text(document['body'])}")
    text = "\n\n".join(part for part in parts if part)
    return text[:TEXT_MAX_CHARS], _truncate(text, SUMMARY_MAX_CHARS)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_record(row: dict, ticker: str, name: str, cik: int, etfs: list[str]) -> dict:
    items = [item for item in (row.get("items") or "").split(",") if item]
    titles = [ITEM_TITLES.get(i, f"Item {i}") for i in items if i != "9.01"]
    record = {
        "url": ARCHIVE_URL.format(
            cik=cik,
            folder=row["accessionNumber"].replace("-", ""),
            name=row["primaryDocument"],
        ),
        "title": f"{name} {row['form']}: " + "; ".join(titles or [ITEM_TITLES["9.01"]]),
        "published_at": row["filingDate"],
        "source": "sec",
        "domain": "sec.gov",
        "summary": None,
        "company": ticker,
        "cik": cik,
        "form": row["form"],
        "items": items,
        "accession": row["accessionNumber"],
        "filed_on": row["filingDate"],
        "etfs": etfs,
        "text": None,
    }
    if row.get("acceptanceDateTime"):
        accepted = accepted_at(row["acceptanceDateTime"])
        record["published_at"] = accepted.isoformat(timespec="seconds")
    else:  # no acceptance time: apply from the next trading day
        record["time_precision"] = "day"
    return record


def holders(news_map: dict) -> dict[str, list[str]]:
    """Company ticker -> ETFs that hold it, in the config's ETF order."""
    etfs_of: dict[str, list[str]] = defaultdict(list)
    for etf, spec in news_map["etfs"].items():
        for company in spec.get("holdings") or {}:
            etfs_of[company].append(etf)
    return etfs_of


def collect_company(
    client: SecClient,
    ticker: str,
    name: str,
    cik: int,
    etfs: list[str],
    start: date,
    end: date,
    cache_dir: Path,
    out_path: Path,
) -> tuple[int, int]:
    """Append the company's new 8-K records; returns (saved, with text)."""
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(line)["accession"] for line in f if line.strip()}
    rows = list_filings(client, cik, start, end, cache_dir / "submissions")
    saved = with_text = 0
    for row in sorted(rows, key=lambda r: r["filingDate"]):
        accession = row["accessionNumber"]
        if accession in done:
            continue
        record = to_record(row, ticker, name, cik, etfs)
        if not set(record["items"]) <= ROUTINE_ITEMS:
            url = ARCHIVE_URL.format(
                cik=cik, folder=accession.replace("-", ""), name=f"{accession}.txt"
            )
            try:
                submission = client.get(url).decode("utf-8", errors="replace")
            except SecNotFound:
                logger.warning(
                    "%s: %s 본문이 없어 목록만 저장합니다.", ticker, accession
                )
            else:
                record["text"], record["summary"] = filing_text(
                    submission, row["form"], record["items"]
                )
                exact = header_accepted(submission)
                if exact:
                    record["published_at"] = exact.isoformat(timespec="seconds")
                    record.pop("time_precision", None)
                with_text += 1
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        saved += 1
    return saved, with_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SEC EDGAR 8-K 공시 수집기")
    parser.add_argument("--start", default="2019-01-01", help="접수일 시작 YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="접수일 끝 (포함)")
    parser.add_argument("--tickers", help="일부 기업만: 쉼표로 구분 (기본: 전체)")
    parser.add_argument("--map", type=Path, default=Path("config/etf_news_map.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/filings/sec"))
    parser.add_argument("--min-interval", type=float, default=1.1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    user_agent = env_value("SEC_USER_AGENT")
    if "@" not in user_agent:
        logger.error(
            "SEC_USER_AGENT가 없습니다. .env에 "
            'SEC_USER_AGENT="팀이름 연락처@이메일" 한 줄을 추가하세요.'
        )
        return 2
    with args.map.open(encoding="utf-8") as f:
        news_map = yaml.safe_load(f)
    companies = news_map["companies"]
    wanted = args.tickers.split(",") if args.tickers else list(companies)
    unknown = [ticker for ticker in wanted if ticker not in companies]
    if unknown:
        logger.error("매핑 설정에 없는 기업입니다: %s", ", ".join(unknown))
        return 2
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    out_dir, cache_dir = args.out / "filings", args.out / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    etfs_of = holders(news_map)

    client = SecClient(user_agent, min_interval=args.min_interval)
    total = total_text = 0
    try:
        ciks = load_ciks(client, cache_dir)
        for ticker in wanted:
            company = companies[ticker]
            cik = company_cik(ticker, company, ciks)
            if cik is None:
                logger.info("%s: 미국 8-K 제출 기업이 아니어서 건너뜁니다.", ticker)
                continue
            saved, with_text = collect_company(
                client,
                ticker,
                company.get("name", ticker),
                cik,
                etfs_of[ticker],
                start,
                end,
                cache_dir,
                out_dir / f"{ticker}.jsonl",
            )
            total, total_text = total + saved, total_text + with_text
            logger.info("%s: 8-K %d건 저장 (본문 %d건)", ticker, saved, with_text)
    except SecError as exc:
        logger.error("%s 같은 명령을 다시 실행하면 이어서 수집합니다.", exc)
        return 2
    logger.info("완료: 새로 저장 %d건 (본문 %d건) → %s", total, total_text, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
