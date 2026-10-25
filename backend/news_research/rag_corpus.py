"""Build the RAG corpus: clean, de-duplicate, chunk and tag news and filings.

    python -m backend.news_research.rag_corpus --out RAG_DIR/chunks.jsonl

Sources (only articles already matched to the ETFs, see etf_news_map.yaml):
- NYT: title, summary and, when the archive has it, the lead paragraph
- MarketWatch: headline only (Media Cloud gives no body)
- Yahoo Finance full text (2017-2023 and 2025-03..09), titles matched to the map
- SEC 8-K filings: item section and EX-99 press releases

Each line of the output is one chunk: {"id", "text", "metadata"}. Metadata keeps
the original url for citations, the title, source, a New York calendar date
(date, date_int, year) for point-in-time filtering, one etf_<TICKER> flag per
linked ETF, and the risk label when the article was tagged. Long bodies are cut
to MAX_CHARS per source (the opening carries the news; tables and boilerplate
follow) and split into ~CHUNK_CHARS pieces on paragraph boundaries.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from backend.news_research.mediacloud_collector import fix_mojibake
from backend.news_research.sec_filings import holders
from backend.news_research.term_matcher import build_matcher, term_etfs

NEW_YORK = ZoneInfo("America/New_York")
CHUNK_CHARS = 1000  # ~250 tokens: inside the embedding model's 256-token window
OVERLAP = 150
MAX_CHARS = {"yahoo": 4000, "sec": 6000}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

logger = logging.getLogger(__name__)


def clean(text: str | None) -> str:
    """Repair mojibake and collapse runs of spaces, keeping paragraph breaks."""
    text = fix_mojibake(text or "")
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n|\n", text))
    return "\n\n".join(p for p in paragraphs if p)


def ny_date(published_at: str) -> str:
    """Calendar date in New York; a bare YYYY-MM-DD is taken as given."""
    if len(published_at) <= 10 or "T" not in published_at:
        return published_at[:10]
    moment = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        return moment.date().isoformat()
    return moment.astimezone(NEW_YORK).date().isoformat()


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Pack paragraphs into chunks of at most `size` characters.

    A paragraph longer than `size` is split at sentence ends, and hard-cut as a
    last resort; consecutive pieces of one long paragraph share `overlap`
    characters so a sentence cut at the border is still found.
    """
    pieces: list[str] = []
    for paragraph in filter(None, text.split("\n\n")):
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        current = ""
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > size:  # a "sentence" with no breaks at all
                pieces.append(sentence[:size])
                sentence = sentence[size - overlap :]
            if current and len(current) + 1 + len(sentence) > size:
                pieces.append(current)
                current = current[-overlap:].lstrip() + " " + sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > size:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def doc_chunks(doc: dict, max_chars: int | None = None) -> list[dict]:
    """Chunks of one document; every chunk text starts with the title."""
    body = doc["body"][:max_chars] if max_chars else doc["body"]
    parts = chunk_text(body) if body else [""]
    digest = hashlib.sha1(doc["url"].encode("utf-8")).hexdigest()[:16]
    date = doc["date"]
    base = {
        "url": doc["url"],
        "title": doc["title"],
        "source": doc["source"],
        "date": date,
        "date_int": int(date.replace("-", "")),
        "year": int(date[:4]),
        **{f"etf_{ticker}": True for ticker in doc["etfs"]},
        **doc.get("extra", {}),
    }
    return [
        {
            "id": f"{doc['source']}:{digest}:{n}",
            "text": f"{doc['title']}\n\n{part}".strip(),
            "metadata": {**base, "chunk": n, "chunks": len(parts)},
        }
        for n, part in enumerate(parts)
    ]


def _jsonl(pattern: str) -> Iterator[dict]:
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            yield from (json.loads(line) for line in f if line.strip())


def _etfs_for(terms: list[str], label_etfs: dict[str, set[str]]) -> list[str]:
    found: set[str] = set()
    for term in terms:
        found |= label_etfs.get(term, set())
    return sorted(found)


def label_extra(label: dict | None) -> dict:
    """Risk-label metadata for a tagged article (empty when untagged)."""
    if not label:
        return {}
    extra = {"risk_relevant": bool(label["market_relevant"])}
    if label["market_relevant"]:
        risk_types = label.get("risk_types") or []
        extra |= {
            "risk_types": ";".join(risk_types),
            "risk_level": float(label["risk_level"]),
            "risk_direction": label["direction"],
        }
        if risk_types:
            extra["risk_type"] = risk_types[0]
    return extra


def nyt_docs(
    pattern: str, raw_dir: Path, label_etfs: dict, labels: dict
) -> Iterator[dict]:
    articles = list(_jsonl(pattern))
    wanted = {article["url"] for article in articles}
    leads: dict[str, str] = {}  # only for the matched articles, not the archive
    for path in sorted(raw_dir.glob("*.json.gz")):
        for raw in json.loads(gzip.decompress(path.read_bytes())):
            if raw.get("lead_paragraph") and raw.get("web_url") in wanted:
                leads[raw["web_url"]] = raw["lead_paragraph"]
    for article in articles:
        summary = clean(article.get("summary"))
        lead = clean(leads.get(article["url"]))
        if lead and summary and (lead.startswith(summary[:60]) or summary in lead):
            summary = ""  # the lead paragraph already says it
        yield {
            "url": article["url"],
            "title": clean(article["title"]),
            "body": "\n\n".join(p for p in (summary, lead) if p),
            "source": "nytimes",
            "date": ny_date(article["published_at"]),
            "etfs": _etfs_for(article["matched_terms"], label_etfs),
            "extra": label_extra(labels.get(article["url"])),
        }


def mediacloud_docs(pattern: str, source: str, label_etfs: dict) -> Iterator[dict]:
    for article in _jsonl(pattern):
        yield {
            "url": article["url"],
            "title": clean(article["title"]),
            "body": "",
            "source": source,
            "date": ny_date(article["published_at"]),
            "etfs": _etfs_for(article["matched_terms"], label_etfs),
        }


def yahoo_docs(
    path: Path, match, label_etfs: dict, etfs_of: dict[str, list[str]]
) -> Iterator[dict]:
    import pyarrow.parquet as pq

    # ~200k rows with bodies: stream in batches instead of loading the file
    rows = (
        row
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000)
        for row in batch.to_pylist()
    )
    for row in rows:
        extra = row["extra_fields"]
        extra = json.loads(extra) if isinstance(extra, str) else extra or {}
        title, _, rest = clean(row["text"]).partition("\n\n")
        terms = match(title)
        url = extra.get("url") or extra.get("link")
        if not terms or not url or len(rest) < 300:
            continue  # unrelated to the ETFs, uncitable, or no real body
        etfs = set(_etfs_for(terms, label_etfs))
        stocks = extra.get("stocks") or []
        if isinstance(stocks, str):  # some rows store "['AAPL', 'XOM']"
            stocks = re.findall(r"[A-Z][A-Z.]*", stocks)
        for ticker in stocks:
            etfs.update(etfs_of.get(ticker, []))
        published = row["date"]
        published = (
            published.isoformat() if hasattr(published, "isoformat") else str(published)
        )
        yield {
            "url": url,
            "title": title,
            "body": rest,
            "source": "yahoo",
            "date": ny_date(published),
            "etfs": sorted(etfs),
            "extra": {
                "publisher": extra.get("publisher") or extra.get("news_outlet") or ""
            },
        }


def sec_docs(pattern: str) -> Iterator[dict]:
    for filing in _jsonl(pattern):
        body = filing["text"] or filing["summary"] or ""
        yield {
            "url": filing["url"],
            "title": filing["title"],
            "body": clean(body),
            "source": "sec",
            "date": ny_date(filing["published_at"]),
            "etfs": filing["etfs"],
            "extra": {"company": filing["company"], "form": filing["form"]},
        }


def build(args: argparse.Namespace) -> dict[str, int]:
    with args.map.open(encoding="utf-8") as f:
        news_map = yaml.safe_load(f)
    match = build_matcher(news_map)
    label_etfs = term_etfs(news_map)
    etfs_of = holders(news_map)
    labels = {row["url"]: row for path in args.labels for row in _jsonl(path)}

    sources = [
        ("nytimes", nyt_docs(args.nyt, args.nyt_raw, label_etfs, labels)),
        ("marketwatch", mediacloud_docs(args.marketwatch, "marketwatch", label_etfs)),
        *[("yahoo", yahoo_docs(p, match, label_etfs, etfs_of)) for p in args.yahoo],
        ("sec", sec_docs(args.sec)),
    ]
    seen_urls: set[str] = set()
    counts: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for name, docs in sources:
            for doc in docs:
                if doc["url"] in seen_urls:
                    counts[f"{name}_duplicate"] += 1
                    continue
                seen_urls.add(doc["url"])
                counts[f"{name}_docs"] += 1
                for chunk in doc_chunks(doc, MAX_CHARS.get(doc["source"])):
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    counts[f"{name}_chunks"] += 1
            logger.info(
                "%s: 문서 %d건, 조각 %d개",
                name,
                counts[f"{name}_docs"],
                counts[f"{name}_chunks"],
            )
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG 말뭉치 만들기 (정리·중복 제거·조각·태깅)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/news/rag/chunks.jsonl")
    )
    parser.add_argument("--map", type=Path, default=Path("config/etf_news_map.yaml"))
    parser.add_argument("--nyt", default="data/raw/news/nyt/articles/*.jsonl")
    parser.add_argument("--nyt-raw", type=Path, default=Path("data/raw/news/nyt/raw"))
    parser.add_argument(
        "--marketwatch", default="data/raw/news/mediacloud/marketwatch/articles/*.jsonl"
    )
    parser.add_argument(
        "--yahoo",
        nargs="*",
        type=Path,
        default=sorted(Path("data/raw/news/yahoo").glob("*.parquet")),
    )
    parser.add_argument("--sec", default="data/raw/filings/sec/filings/*.jsonl")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["data/processed/news/labels/labels_v4_sonnet.jsonl"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    counts = build(args)
    logger.info("완료: %s → %s", counts, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
