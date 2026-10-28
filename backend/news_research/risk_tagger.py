"""Engine-neutral parts of LLM risk tagging: build model inputs, check outputs.

The model runs elsewhere (Ollama on this PC, an open model on a Colab GPU, or a
hosted API). Every engine reads requests.jsonl with the system prompt in
prompt.md, and writes one line per request to outputs.jsonl:
{"request_id": "...", "output": "<the model's JSON text>"}.

    python -m backend.news_research.risk_tagger prepare --out RUN_DIR \
        --articles "data/raw/news/nyt/articles/*.jsonl"
    (engine: RUN_DIR/requests.jsonl -> RUN_DIR/outputs.jsonl)
    python -m backend.news_research.risk_tagger merge --run RUN_DIR \
        --labels data/processed/news/labels/labels_v2.jsonl --tagger MODEL

Requests carry only a short id, the headline and, when there is one, the
summary - never the URL (NYT URLs embed the date) or the publish time, so the
model cannot anchor on when the story ran. The id -> URL map stays in
keys.jsonl. Articles already in the labels file are skipped by prepare, so
anything an engine missed is simply prepared again.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROMPT_PATH = Path("config/prompts/risk_tagging_v2.md")
RISK_TYPES = (
    "recession",
    "inflation",
    "interest_rate",
    "regulation",
    "geopolitical",
    "earnings",
    "market_risk",
    "credit_risk",
    "commodity_risk",
    "currency_risk",
)
DIRECTIONS = ("increase", "decrease", "neutral", "none")
PER_REQUEST = 20
INSTRUCTION = (
    "Label every article below. Return one entry per id in `labels`, in the "
    "same order. Each article is a JSON object with an id, a title and, when "
    "available, a summary."
)

logger = logging.getLogger(__name__)


def load_prompt(path: Path = PROMPT_PATH) -> str:
    """The system prompt, without the Korean maintainer notes in <!-- -->."""
    text = path.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.S).strip()


def prompt_version(path: Path) -> str:
    match = re.search(r"_(v\d+)$", path.stem)
    return match[1] if match else path.stem


def label_schema() -> dict:
    """JSON schema of a model answer, usable as a structured-output format."""
    label = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "market_relevant": {"type": "boolean"},
            "risk_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(RISK_TYPES)},
            },
            "risk_level": {"type": "number"},
            "direction": {"type": "string", "enum": list(DIRECTIONS)},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
        },
        "required": [
            "id",
            "market_relevant",
            "risk_types",
            "risk_level",
            "direction",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": label}},
        "required": ["labels"],
        "additionalProperties": False,
    }


def article_input(article: dict) -> dict:
    """What the model sees of an article: title and, if present, summary."""
    item = {"title": " ".join(article["title"].split())}
    summary = " ".join((article.get("summary") or "").split())
    if summary:
        item["summary"] = summary
    return item


def build_requests(
    articles: list[dict], per_request: int = PER_REQUEST
) -> tuple[list[dict], list[dict]]:
    """Group articles into model requests; the URL map goes to keys only."""
    requests, keys = [], []
    for start in range(0, len(articles), per_request):
        request_id = f"r{start // per_request + 1:06d}"
        items, urls, texts = [], {}, {}
        for n, article in enumerate(articles[start : start + per_request], 1):
            item_id = f"a{n:02d}"
            shown = article_input(article)
            items.append({"id": item_id, **shown})
            urls[item_id] = article["url"]
            texts[item_id] = "\n".join(shown.values())
        requests.append({"request_id": request_id, "items": items})
        keys.append({"request_id": request_id, "urls": urls, "texts": texts})
    return requests, keys


def user_prompt(request: dict) -> str:
    lines = (json.dumps(item, ensure_ascii=False) for item in request["items"])
    return INSTRUCTION + "\n\n" + "\n".join(lines)


def _fraction(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, 0.0), 1.0)


def _normalize(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"')
    return " ".join(text.replace("”", '"').split()).casefold()


def clean_label(raw: dict, source_text: str) -> dict | None:
    """Coerce one model label into the rubric's ranges; None if unusable."""
    level, confidence = _fraction(raw.get("risk_level")), _fraction(
        raw.get("confidence")
    )
    if level is None or confidence is None:
        return None
    types = []
    for risk_type in raw.get("risk_types") or []:
        if risk_type in RISK_TYPES and risk_type not in types:
            types.append(risk_type)
    relevant = raw.get("market_relevant") is True and bool(types)
    direction = raw.get("direction")
    if not relevant:
        types, level, direction = [], 0.0, "none"
    elif direction not in DIRECTIONS or direction == "none":
        direction = "neutral"
    evidence = " ".join(str(raw.get("evidence") or "").split())
    return {
        "market_relevant": relevant,
        "risk_types": types,
        "risk_level": level,
        "direction": direction,
        "confidence": confidence,
        "evidence": evidence,
        "evidence_found": not evidence
        or _normalize(evidence) in _normalize(source_text),
    }


def parse_output(text: str, key: dict) -> tuple[list[dict], list[str]]:
    """Labels (with URLs) for one request, and the ids the model missed."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return [], list(key["urls"])
    entries = data.get("labels", []) if isinstance(data, dict) else data
    labels, seen = [], set()
    for entry in entries if isinstance(entries, list) else []:
        item_id = entry.get("id") if isinstance(entry, dict) else None
        if item_id not in key["urls"] or item_id in seen:
            continue
        label = clean_label(entry, key["texts"][item_id])
        if label is None:
            continue
        seen.add(item_id)
        labels.append({"url": key["urls"][item_id], **label})
    return labels, [item_id for item_id in key["urls"] if item_id not in seen]


def load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not Path(path).exists():
                continue
            with open(path, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def answered(outputs: Path) -> set[str]:
    """Request ids an engine already answered in an outputs jsonl file."""
    if not outputs.exists():
        return set()
    with outputs.open(encoding="utf-8") as f:
        return {
            row["request_id"]
            for row in map(json.loads, filter(str.strip, f))
            if row.get("output")
        }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_articles(
    articles: list[dict],
    labeled: set[str],
    only_urls: set[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
    since: str | None = None,
    until: str | None = None,
) -> list[dict]:
    """Unlabeled articles (optionally restricted / dated / sampled), oldest first.

    since/until are inclusive YYYY-MM-DD bounds on the published_at date.
    """
    unique = {a["url"]: a for a in articles}
    chosen = [
        a
        for url, a in unique.items()
        if url not in labeled
        and (only_urls is None or url in only_urls)
        and (since is None or a["published_at"][:10] >= since)
        and (until is None or a["published_at"][:10] <= until)
    ]
    if sample is not None and sample < len(chosen):
        chosen = random.Random(seed).sample(chosen, sample)
    return sorted(chosen, key=lambda a: (a["published_at"], a["url"]))


def prepare(args: argparse.Namespace) -> int:
    articles = load_jsonl(args.articles)
    labeled = {row["url"] for row in load_jsonl([str(args.labels)])}
    only = {row["url"] for row in load_jsonl(args.urls)} if args.urls else None
    chosen = select_articles(
        articles, labeled, only, args.sample, args.seed, args.since, args.until
    )
    if only is not None and len(only - labeled) > len(chosen):
        logger.warning(
            "지정한 URL %d건 중 %d건이 기사 파일에 없습니다.",
            len(only - labeled),
            len(only - labeled) - len(chosen),
        )
    requests, keys = build_requests(chosen, args.per_request)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "requests.jsonl", requests)
    write_jsonl(args.out / "keys.jsonl", keys)
    (args.out / "prompt.md").write_text(load_prompt(args.prompt), encoding="utf-8")
    (args.out / "schema.json").write_text(
        json.dumps(label_schema(), indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": prompt_version(args.prompt),
        "per_request": args.per_request,
        "articles": len(chosen),
        "requests": len(requests),
        "sources": sorted({a.get("source", "") for a in chosen}),
        "period": [args.since, args.until],
        "input_chars": sum(len(user_prompt(r)) for r in requests),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "기사 %d건 → 요청 %d개 (요청당 %d건) 저장: %s",
        len(chosen),
        len(requests),
        args.per_request,
        args.out,
    )
    return 0


def merge(args: argparse.Namespace) -> int:
    keys = {
        row["request_id"]: row for row in load_jsonl([str(args.run / "keys.jsonl")])
    }
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    labeled = {row["url"] for row in load_jsonl([str(args.labels)])}
    new, missing, unanswered = [], 0, set(keys)
    for row in load_jsonl([str(args.run / getattr(args, "outputs", "outputs.jsonl"))]):
        key = keys.get(row.get("request_id"))
        if key is None:
            continue
        unanswered.discard(row["request_id"])
        labels, missed = parse_output(row.get("output") or "", key)
        missing += len(missed)
        for label in labels:
            if label["url"] not in labeled:
                labeled.add(label["url"])
                new.append(
                    {
                        **label,
                        "prompt_version": manifest["prompt_version"],
                        "tagger": args.tagger,
                        "request_id": row["request_id"],
                    }
                )
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    with args.labels.open("a", encoding="utf-8") as f:
        for label in new:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    missing += sum(len(keys[r]["urls"]) for r in unanswered)
    unfound = sum(not label["evidence_found"] for label in new)
    logger.info(
        "새 라벨 %d건 저장 (근거 문구 불일치 %d건), 빠진 기사 %d건"
        " - prepare를 다시 실행하면 빠진 기사만 다시 요청합니다.",
        len(new),
        unfound,
        missing,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 위험 태깅 입력 준비·결과 병합")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="태깅할 기사를 모델 요청 파일로 만들기")
    p.add_argument("--articles", nargs="+", required=True, help="기사 jsonl (glob)")
    p.add_argument("--out", type=Path, required=True, help="실행 폴더")
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/news/labels/labels_v2.jsonl"),
        help="이미 태깅한 라벨 (여기 있는 기사는 건너뜀)",
    )
    p.add_argument("--urls", nargs="+", help="이 파일의 url만 태깅 (검증 표본용)")
    p.add_argument("--since", help="발행일 시작 YYYY-MM-DD (포함)")
    p.add_argument("--until", help="발행일 끝 YYYY-MM-DD (포함)")
    p.add_argument("--sample", type=int, help="무작위 표본 크기")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--per-request", type=int, default=PER_REQUEST)
    p.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    m = sub.add_parser("merge", help="모델 출력을 검사해 라벨 파일에 추가")
    m.add_argument("--run", type=Path, required=True, help="prepare 실행 폴더")
    m.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/news/labels/labels_v2.jsonl"),
    )
    m.add_argument("--tagger", required=True, help="모델 이름 (라벨에 기록)")
    m.add_argument("--outputs", default="outputs.jsonl", help="엔진 결과 파일 이름")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return prepare(args) if args.command == "prepare" else merge(args)


if __name__ == "__main__":
    sys.exit(main())
