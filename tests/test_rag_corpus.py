import gzip
import json
from argparse import Namespace

import pyarrow as pa
import pyarrow.parquet as pq

from backend.news_research import rag_corpus as rc
from backend.news_research.term_matcher import build_matcher, term_etfs

NEWS_MAP = {
    "companies": {"AAPL": {"aliases": ["Apple"]}, "XOM": {"aliases": ["Exxon"]}},
    "etfs": {
        "QQQ": {"holdings": {"AAPL": 0.07}, "keywords": ["Nasdaq"]},
        "XLE": {"holdings": {"XOM": 0.2}, "keywords": ["oil prices"]},
    },
    "risk_type_keywords": {"geopolitical": ["war"]},
}


def test_clean_repairs_text_and_keeps_paragraphs():
    raw = "Irelandâ\x80\x99s   peat\n\n\n  bogs  \nare wet"
    assert rc.clean(raw) == "Ireland’s peat\n\nbogs\n\nare wet"
    assert rc.clean(None) == ""


def test_dates_are_new_york_calendar_days():
    assert rc.ny_date("2021-03-01T03:00:00+00:00") == "2021-02-28"
    assert rc.ny_date("2024-05-02T16:30:34-04:00") == "2024-05-02"
    assert rc.ny_date("2025-06-02") == "2025-06-02"


def test_chunks_respect_size_keep_order_and_overlap_long_paragraphs():
    assert rc.chunk_text("Short note.") == ["Short note."]
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    chunks = rc.chunk_text("\n\n".join(paragraphs), size=400, overlap=50)
    assert all(len(c) <= 400 for c in chunks) and len(chunks) > 1
    assert chunks[0].startswith("Paragraph 0") and "Paragraph 9" in chunks[-1]
    long_paragraph = " ".join(f"Sentence number {i} ends here." for i in range(60))
    pieces = rc.chunk_text(long_paragraph, size=300, overlap=60)
    assert all(len(p) <= 300 for p in pieces)
    for i in range(60):
        assert any(f"Sentence number {i} ends" in p for p in pieces)
    # neighbours share text so a sentence cut at the border is still found
    assert pieces[1][:20] in pieces[0]


def test_doc_chunks_carry_citation_date_and_etf_metadata():
    doc = {
        "url": "https://example.com/a",
        "title": "Apple warns",
        "body": "First paragraph.\n\n" + "More text. " * 200,
        "source": "yahoo",
        "date": "2025-04-03",
        "etfs": ["QQQ", "SPY"],
        "extra": {"publisher": "Reuters"},
    }
    chunks = rc.doc_chunks(doc, max_chars=1500)
    assert len(chunks) > 1 and len({c["id"] for c in chunks}) == len(chunks)
    # the body was cut to 1500 characters before chunking
    prefix = len("Apple warns") + 2
    assert sum(len(c["text"]) - prefix for c in chunks) < 1500 + 300  # + overlaps
    assert all(c["text"].startswith("Apple warns\n\n") for c in chunks)
    meta = chunks[1]["metadata"]
    assert meta["url"] == doc["url"] and meta["date_int"] == 20250403
    assert meta["year"] == 2025 and meta["etf_QQQ"] and meta["etf_SPY"]
    assert "etf_XLE" not in meta and meta["chunk"] == 1
    assert meta["chunks"] == len(chunks)
    assert meta["publisher"] == "Reuters"
    assert rc.doc_chunks(doc, 1500)[0]["id"] == chunks[0]["id"]  # stable ids


def test_risk_labels_become_metadata():
    label = {
        "market_relevant": True,
        "risk_types": ["regulation", "inflation"],
        "risk_level": 0.6,
        "direction": "increase",
    }
    assert rc.label_extra(label) == {
        "risk_relevant": True,
        "risk_type": "regulation",
        "risk_types": "regulation;inflation",
        "risk_level": 0.6,
        "risk_direction": "increase",
    }
    assert rc.label_extra({"market_relevant": False}) == {"risk_relevant": False}
    assert rc.label_extra(None) == {}


def test_nyt_docs_add_the_lead_paragraph_without_repeating_the_summary(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw = [
        {"web_url": "u1", "lead_paragraph": "Apple said sales fell. More detail."},
        {"web_url": "u2", "lead_paragraph": "Oil prices rose sharply on Monday."},
    ]
    (raw_dir / "2025-04.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    articles = tmp_path / "2025-04.jsonl"
    rows = [
        {
            "url": "u1",
            "title": "Apple Warns",
            "summary": "Apple said sales fell.",
            "published_at": "2025-04-04T01:00:00+00:00",
            "matched_terms": ["Apple"],
        },
        {
            "url": "u2",
            "title": "Energy Shock",
            "summary": "A different summary.",
            "published_at": "2025-04-04T15:00:00+00:00",
            "matched_terms": ["oil prices", "war"],
        },
    ]
    articles.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    labels = {
        "u2": {
            "market_relevant": True,
            "risk_types": ["commodity_risk"],
            "risk_level": 0.5,
            "direction": "increase",
        }
    }
    docs = list(rc.nyt_docs(str(articles), raw_dir, term_etfs(NEWS_MAP), labels))
    assert docs[0]["body"] == "Apple said sales fell. More detail."
    assert docs[0]["date"] == "2025-04-03" and docs[0]["etfs"] == ["QQQ"]
    assert docs[1]["body"].startswith("A different summary.\n\nOil prices rose")
    assert docs[1]["etfs"] == ["XLE"]  # "war" is risk-only: no direct ETF
    assert docs[1]["extra"]["risk_type"] == "commodity_risk"


def test_yahoo_rows_need_a_matched_title_a_url_and_a_real_body(tmp_path):
    body = "Body sentence about the company. " * 15
    rows = {
        "date": ["2025-04-03T13:30:00Z"] * 4,
        "text": [
            f"Apple cuts iPhone orders\n\n{body}",
            f"Garden tips for spring\n\n{body}",
            "Exxon beats estimates\n\nToo short.",
            f"Exxon raises its dividend\n\n{body}",
        ],
        "extra_fields": [
            json.dumps(
                {
                    "link": "https://y/1",
                    "publisher": "Zacks",
                    "stocks": "['AAPL', 'XOM']",
                }
            ),
            json.dumps({"link": "https://y/2"}),
            json.dumps({"url": "https://y/3"}),
            # other rows store the stocks as a real list
            json.dumps({"link": "https://y/4", "stocks": ["XOM"]}),
        ],
    }
    path = tmp_path / "yahoo.parquet"
    pq.write_table(pa.table(rows), path)
    etfs_of = {"AAPL": ["QQQ"], "XOM": ["XLE"]}
    docs = list(
        rc.yahoo_docs(path, build_matcher(NEWS_MAP), term_etfs(NEWS_MAP), etfs_of)
    )
    assert [d["url"] for d in docs] == ["https://y/1", "https://y/4"]
    assert docs[1]["etfs"] == ["XLE"]
    assert docs[0]["etfs"] == ["QQQ", "XLE"]  # title match plus listed stocks
    assert docs[0]["date"] == "2025-04-03" and docs[0]["extra"]["publisher"] == "Zacks"


def test_build_writes_each_url_once(tmp_path):
    def article(url):
        return {
            "url": url,
            "title": "Nasdaq falls",
            "summary": None,
            "published_at": "2025-01-02",
            "matched_terms": ["Nasdaq"],
        }

    (tmp_path / "raw").mkdir()
    nyt = tmp_path / "nyt.jsonl"
    nyt.write_text(json.dumps(article("dup")) + "\n", encoding="utf-8")
    mw = tmp_path / "mw.jsonl"
    mw.write_text(
        json.dumps(article("dup")) + "\n" + json.dumps(article("mw1")), encoding="utf-8"
    )
    news_map = tmp_path / "map.yaml"
    news_map.write_text(json.dumps(NEWS_MAP), encoding="utf-8")
    out = tmp_path / "chunks.jsonl"
    counts = rc.build(
        Namespace(
            map=news_map,
            nyt=str(nyt),
            nyt_raw=tmp_path / "raw",
            marketwatch=str(mw),
            yahoo=[],
            sec=str(tmp_path / "none-*.jsonl"),
            labels=[],
            out=out,
        )
    )
    assert counts["nytimes_docs"] == 1 and counts["marketwatch_docs"] == 1
    assert counts["marketwatch_duplicate"] == 1
    chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [c["metadata"]["source"] for c in chunks] == ["nytimes", "marketwatch"]
    assert chunks[0]["text"] == "Nasdaq falls" and chunks[0]["metadata"]["etf_QQQ"]
