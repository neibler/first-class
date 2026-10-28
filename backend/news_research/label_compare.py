"""Compare a tagger's labels with reference labels on the same articles.

Used to accept or reject a model before it tags every article:

    python -m backend.news_research.label_compare \
        --reference data/processed/news/validation/reference_200_v2.jsonl \
        --candidate data/processed/news/validation/labels_<model>.jsonl

Labels are joined on URL. Results are reported overall and, when a reference
file mixes rubric versions, per prompt_version. The 200 references were
labeled by Claude in a chat session, so they measure agreement with the rubric
as applied there, not accuracy against human judgment; the same labeler
re-labeling 100 of them under v2 agreed at κ 0.88, which bounds what any
tagger can be expected to reach.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Acceptance bar for a tagger (agreed in the validation plan)
THRESHOLDS = {"kappa": 0.80, "first_type": 0.85, "level_spearman": 0.70}


def ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            result[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def cohen_kappa(xs: list[bool], ys: list[bool]) -> float | None:
    if not xs:
        return None
    observed = sum(a == b for a, b in zip(xs, ys)) / len(xs)
    px, py = sum(xs) / len(xs), sum(ys) / len(ys)
    expected = px * py + (1 - px) * (1 - py)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def compare(reference: list[dict], candidate: list[dict]) -> dict:
    """Agreement metrics on articles both files labeled (joined on url)."""
    cand = {row["url"]: row for row in candidate}
    pairs = [(ref, cand[ref["url"]]) for ref in reference if ref["url"] in cand]
    rel = [(bool(r["market_relevant"]), bool(c["market_relevant"])) for r, c in pairs]
    both = [(r, c) for r, c in pairs if r["market_relevant"] and c["market_relevant"]]
    tp = fp = fn = 0
    for r, c in both:
        ref_types, cand_types = set(r["risk_types"]), set(c["risk_types"])
        tp += len(ref_types & cand_types)
        fp += len(cand_types - ref_types)
        fn += len(ref_types - cand_types)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    levels = [(float(r["risk_level"]), float(c["risk_level"])) for r, c in both]
    scores = [
        (
            float(r["risk_level"]) * float(r["confidence"]),
            float(c["risk_level"]) * float(c["confidence"]),
        )
        for r, c in pairs
    ]
    return {
        "articles": len(reference),
        "compared": len(pairs),
        "relevant_agreement": (sum(a == b for a, b in rel) / len(rel) if rel else None),
        "kappa": cohen_kappa([a for a, _ in rel], [b for _, b in rel]),
        "both_relevant": len(both),
        "first_type": (
            sum(r["risk_types"][:1] == c["risk_types"][:1] for r, c in both) / len(both)
            if both
            else None
        ),
        "type_precision": precision,
        "type_recall": recall,
        "type_f1": (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        ),
        "level_spearman": spearman([a for a, _ in levels], [b for _, b in levels]),
        "level_mae": (
            sum(abs(a - b) for a, b in levels) / len(levels) if levels else None
        ),
        "direction": (
            sum(r["direction"] == c["direction"] for r, c in both) / len(both)
            if both
            else None
        ),
        "score_spearman": spearman([a for a, _ in scores], [b for _, b in scores]),
    }


def passes(metrics: dict) -> dict[str, bool | None]:
    """Each threshold met (True), missed (False) or not measurable (None)."""
    return {
        name: None if metrics.get(name) is None else metrics[name] >= bar
        for name, bar in THRESHOLDS.items()
    }


def disagreements(reference: list[dict], candidate: list[dict]) -> list[dict]:
    """Relevance flips, different first types and level gaps of 0.3 or more."""
    cand = {row["url"]: row for row in candidate}
    found = []
    for r in reference:
        c = cand.get(r["url"])
        if c is None:
            continue
        if r["market_relevant"] != c["market_relevant"]:
            kind = "관련 여부"
        elif r["market_relevant"] and r["risk_types"][:1] != c["risk_types"][:1]:
            kind = "첫 유형"
        elif abs(float(r["risk_level"]) - float(c["risk_level"])) >= 0.3:
            kind = "위험 수준"
        else:
            continue
        found.append({"kind": kind, "url": r["url"], "reference": r, "candidate": c})
    return found


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


ROWS = [
    ("비교한 기사", "compared"),
    ("시장 관련 여부 일치", "relevant_agreement"),
    ("관련 여부 κ", "kappa"),
    ("둘 다 관련", "both_relevant"),
    ("첫 번째 위험 유형 일치", "first_type"),
    ("위험 유형 F1", "type_f1"),
    ("위험 수준 Spearman (둘 다 관련)", "level_spearman"),
    ("위험 수준 평균 절대 오차", "level_mae"),
    ("방향 일치", "direction"),
    ("점수(수준×확신도) Spearman", "score_spearman"),
]


def report(reference: list[dict], candidate: list[dict], titles: dict) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in reference:
        groups[f"기준 {row.get('prompt_version', '?')}"].append(row)
    columns = {"전체": compare(reference, candidate)}
    if len(groups) > 1:
        columns.update(
            {name: compare(rows, candidate) for name, rows in sorted(groups.items())}
        )
    lines = ["| 지표 | " + " | ".join(columns) + " |"]
    lines.append("|---|" + "---|" * len(columns))
    for label, key in ROWS:
        digits = 3 if key == "level_mae" else 2
        values = " | ".join(_fmt(m[key], digits) for m in columns.values())
        lines.append(f"| {label} | {values} |")
    lines.append("")
    lines.append("통과 기준: " + ", ".join(f"{k} ≥ {v}" for k, v in THRESHOLDS.items()))
    for name, metrics in columns.items():
        verdict = passes(metrics)
        failed = [k for k, ok in verdict.items() if ok is False]
        unknown = [k for k, ok in verdict.items() if ok is None]
        if failed:
            status = f"미달 ({', '.join(failed)})"
        elif unknown:
            status = f"판단 불가 - 표본 부족 ({', '.join(unknown)})"
        else:
            status = "통과"
        lines.append(f"- {name}: {status}")
    lines.append("\n## 불일치 사례")
    for d in disagreements(reference, candidate)[:30]:
        r, c = d["reference"], d["candidate"]
        title = titles.get(d["url"], d["url"])
        lines.append(
            f"- [{d['kind']}] {title[:90]}\n"
            f"  기준 {r['risk_types']} {r['risk_level']} {r['direction']}"
            f" / 비교 {c['risk_types']} {c['risk_level']} {c['direction']}"
        )
    return "\n".join(lines)


def titles_from_keys(keys: list[dict]) -> dict[str, str]:
    """URL -> title from risk_tagger keys.jsonl rows (title is the first line)."""
    return {
        url: key["texts"][item_id].splitlines()[0]
        for key in keys
        for item_id, url in key["urls"].items()
    }


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="태깅 결과를 기준 라벨과 비교")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--articles", nargs="*", default=[], help="제목 표시용 기사 jsonl"
    )
    parser.add_argument("--keys", nargs="*", default=[], help="제목 표시용 keys.jsonl")
    parser.add_argument("--out", type=Path, help="보고서 저장 경로 (.md)")
    args = parser.parse_args(argv)

    titles = {}
    for path in args.articles:
        for pattern_path in sorted(Path().glob(path)) if "*" in path else [Path(path)]:
            titles.update({a["url"]: a["title"] for a in _load(pattern_path)})
    for path in args.keys:
        titles.update(titles_from_keys(_load(Path(path))))
    text = report(_load(args.reference), _load(args.candidate), titles)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
