"""LangGraph research agent: plan -> retrieve -> verify -> (rewrite) -> analyze.

    python -m backend.news_research.research_agent "최근 QQQ에 위험한 뉴스가 있어?" \
        --as-of 2025-04-10

Planner turns the question into up to N English search queries; Researcher
searches ChromaDB for each; Verifier has the LLM score every hit and, when the
best hits score below the bar, Rewriter rewrites the query and searches again
(Self-Correction, at most max_rewrites times); Analyst answers in Korean from
the accepted evidence only, citing it as [n], and turns the risk events it
finds into risk tags (ticker, risk_type, risk_level x confidence = risk_score).

Searches only see news dated on or before `as_of`, so the agent can be replayed
at past dates without look-ahead. Steps append execution summaries to `log`,
returned after completion (not private reasoning or live events). `research()`
is the function
the FastAPI POST /research endpoint calls. LLM calls go through the `LLM`
protocol; ClaudeLLM uses the Anthropic SDK and another provider can be added
behind the same two-argument interface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol, TypedDict
from zoneinfo import ZoneInfo

import yaml
from langgraph.graph import END, START, StateGraph

from backend.news_research.risk_tagger import DIRECTIONS, RISK_TYPES

AGENT_CONFIG = Path("config/agent.yaml")
NEWS_MAP = Path("config/etf_news_map.yaml")
# Models that take the server-side refusal fallback ("default" form)
FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NEW_YORK = ZoneInfo("America/New_York")

Search = Callable[..., list[dict]]


class AgentError(Exception):
    """A user-displayable failure of the research agent."""


class LLM(Protocol):
    def json(self, role: str, system: str, user: str, schema: dict) -> dict: ...


class ClaudeLLM:
    """Structured-output calls to Claude, one model and effort per agent role."""

    def __init__(self, client, models: dict, effort: dict, max_tokens: int = 8000):
        self.client, self.models, self.effort = client, models, effort
        self.max_tokens = max_tokens
        self.usage: Counter = Counter()

    def json(self, role: str, system: str, user: str, schema: dict) -> dict:
        model = self.models[role]
        params = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.effort.get(role, "low"),
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if model in FALLBACK_MODELS:
            response = self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**params)
        self.usage[f"{model}:input"] += response.usage.input_tokens
        self.usage[f"{model}:output"] += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise AgentError("요청이 모델의 안전 정책으로 거절됐습니다.")
        if response.stop_reason == "max_tokens":
            raise AgentError(f"{role} 응답이 길이 제한에 걸렸습니다.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentError(f"{role} 응답을 JSON으로 읽을 수 없습니다.") from exc


class AgentState(TypedDict, total=False):
    status: str
    question: str
    as_of: str
    goal: str
    plan: list[dict]
    current: int
    query: str
    rewrites: int
    candidates: list[dict]
    search_score: float
    evidence: list[dict]
    answer: str
    risk_tags: list[dict]
    sources: list[dict]
    log: list[str]


def _schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def schemas(etfs: list[str]) -> dict[str, dict]:
    query = _schema(
        {
            "query": {"type": "string"},
            "etf": {"type": "string", "enum": [*etfs, ""]},
            "lookback_days": {"type": "integer"},
        }
    )
    tag = _schema(
        {
            "ticker": {"type": "string", "enum": etfs},
            "risk_type": {"type": "string", "enum": list(RISK_TYPES)},
            "risk_level": {"type": "number"},
            "direction": {
                "type": "string",
                "enum": [d for d in DIRECTIONS if d != "none"],
            },
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "integer"}},
        }
    )
    return {
        "planner": _schema(
            {
                "goal": {"type": "string"},
                "focus_etfs": {
                    "type": "array",
                    "items": {"type": "string", "enum": etfs},
                },
                "queries": {"type": "array", "items": query},
            }
        ),
        "grader": _schema(
            {
                "scores": {
                    "type": "array",
                    "items": _schema(
                        {"i": {"type": "integer"}, "relevance": {"type": "number"}}
                    ),
                },
                "comment": {"type": "string"},
            }
        ),
        "rewriter": _schema(
            {"query": {"type": "string"}, "reason": {"type": "string"}}
        ),
        "analyst": _schema(
            {"answer": {"type": "string"}, "risk_tags": {"type": "array", "items": tag}}
        ),
    }


PLANNER = """You plan news research for a robo-advisor holding these 15 US-listed ETFs:
{etfs}

Today is {as_of}. You know nothing that happened after this date: do not name
later events, levels or outcomes.

Turn the user's question (often in Korean) into at most {max_queries} focused
English search queries for a news database (New York Times, MarketWatch, Yahoo
Finance, SEC 8-K filings). Each query covers one angle (results, regulation,
rates, conflict, commodity supply...). {query_style}

Set `etf` to a ticker when the query is about that ETF's holdings or theme,
otherwise "". Match lookback_days to the time frame of the question: a specific
day or week 7-14, a month 30-45, "recent" 30-90, a broad trend 180-365
(default {lookback}). Write `goal` in Korean, one sentence."""

# Short, headline-like queries embed far better than keyword lists
QUERY_STYLE = """Write each query like a news headline: 4-12 words, naming
companies, policies, indicators or events. Do not put ETF tickers or fund names
in the query - the news rarely mentions them."""

GRADER = """Rate how useful each news item is for answering the search query,
from 0 to 1: 1 = directly about the query's subject and risk, 0.5 = related
background, 0 = unrelated. Judge only from the given text. Score every item by
its number `i`. Put a one-sentence Korean comment in `comment`."""

REWRITER = """Today is {as_of}. You know nothing that happened after this date:
do not name later events, levels or outcomes.

The news search results were not relevant enough. Write one better English
search query for the same need: name the companies, events, policies or
indicators more specifically, or use common synonyms. Do not repeat a query
that was already tried. {query_style} Give the reason in Korean in one
sentence."""

ANALYST = """You are an investment research analyst for a robo-advisor holding
these 15 ETFs:
{etfs}

Answer the user's question in Korean using ONLY the numbered evidence. Put the
citation [n] right after each claim; never cite a number that is not listed. If
the evidence is not enough, say what is missing. Keep it to 3-8 sentences or
bullets.

Then list risk events the evidence shows as `risk_tags`, at most one per
(ticker, risk_type), only for ETFs the evidence supports:
- risk_type: recession, inflation, interest_rate, regulation (tariffs, trade
  disputes, export controls, antitrust, fines), geopolitical (war, conflict,
  sanctions tied to them), earnings, market_risk (market-wide moves only),
  credit_risk, commodity_risk, currency_risk
- risk_level: current level 0-1 (0.2 small single-company news, 0.5 major
  policy or large-cap profit decline, 0.7 broad selloff or conflict escalation,
  0.85+ crisis language)
- direction relative to before the news; confidence 0-1; evidence = the [n]
  numbers it rests on."""


def _etf_lines(news_map: dict) -> str:
    return "\n".join(
        f"- {ticker}: {spec.get('role', '')}"
        for ticker, spec in news_map["etfs"].items()
    )


def _snippet(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_graph(llm: LLM, search: Search, settings: dict, news_map: dict):
    """Compile the LangGraph workflow for one configuration."""
    retrieval = settings["retrieval"]
    etfs = list(news_map["etfs"])
    shapes = schemas(etfs)
    etf_lines = _etf_lines(news_map)

    def log(state: AgentState, *lines: str) -> list[str]:
        return [*state.get("log", []), *lines]

    def plan(state: AgentState) -> AgentState:
        system = PLANNER.format(
            etfs=etf_lines,
            as_of=state["as_of"],
            query_style=QUERY_STYLE,
            max_queries=retrieval["max_subqueries"],
            lookback=retrieval["lookback_days"],
        )
        result = llm.json("planner", system, state["question"], shapes["planner"])
        queries = [q for q in result["queries"] if q["query"].strip()]
        queries = queries[: retrieval["max_subqueries"]]
        steps = " / ".join(f"{n}) {q['query']}" for n, q in enumerate(queries, 1))
        return {
            "goal": result["goal"],
            "plan": queries,
            "current": 0,
            "query": queries[0]["query"] if queries else "",
            "rewrites": 0,
            "evidence": [],
            "log": log(
                state,
                f"[Planner] 하위 검색 {len(queries)}개 계획 완료",
                f"[Planner] 검색 계획: {steps or '없음'}",
            ),
        }

    def retrieve(state: AgentState) -> AgentState:
        step = state["plan"][state["current"]]
        until = date.fromisoformat(state["as_of"])
        lookback = step.get("lookback_days") or retrieval["lookback_days"]
        days = min(max(int(lookback), 7), 365)
        since = (until - timedelta(days=days)).isoformat()
        etf = step.get("etf") or None
        hits = search(
            state["query"],
            retrieval["k"],
            etf=etf,
            since=since,
            until=until.isoformat(),
        )
        note = f", ETF {etf}" if etf else ""
        if not hits and etf:  # the ETF flag may be missing on relevant news
            hits = search(
                state["query"], retrieval["k"], since=since, until=until.isoformat()
            )
            note += " → ETF 조건 없이 재검색"
        return {
            "candidates": hits,
            "log": log(
                state,
                f"[Researcher] 검색어 '{state['query']}'"
                f" ({since}~{until}{note}): {len(hits)}건",
            ),
        }

    def verify(state: AgentState) -> AgentState:
        hits = state["candidates"]
        if not hits:
            return {
                "search_score": 0.0,
                "log": log(state, "[Verifier] 검색 결과가 없습니다."),
            }
        items = "\n\n".join(
            f"[{i}] {h['metadata']['title']} ({h['metadata']['date']}, "
            f"{h['metadata']['source']})\n{_snippet(h['text'], 500)}"
            for i, h in enumerate(hits)
        )
        result = llm.json(
            "grader",
            GRADER,
            f"Search query: {state['query']}\n\n{items}",
            shapes["grader"],
        )
        scores = {
            s["i"]: min(max(float(s["relevance"]), 0.0), 1.0) for s in result["scores"]
        }
        ranked = sorted(scores.values(), reverse=True)[:3]
        score = sum(ranked) / len(ranked) if ranked else 0.0
        evidence = {e["url"]: e for e in state.get("evidence", [])}
        kept = 0
        for i, hit in enumerate(hits):
            relevance = scores.get(i, 0.0)
            if relevance < retrieval["min_hit_relevance"]:
                continue
            meta = hit["metadata"]
            old = evidence.get(meta["url"])
            if old is None or relevance > old["relevance"]:
                kept += old is None
                evidence[meta["url"]] = {
                    "url": meta["url"],
                    "title": meta["title"],
                    "source": meta["source"],
                    "date": meta["date"],
                    "text": hit["text"],
                    "relevance": relevance,
                    "query": state["query"],
                }
        bar = retrieval["min_search_relevance"]
        verdict = "충분" if score >= bar else "부족"
        return {
            "search_score": score,
            "evidence": list(evidence.values()),
            "log": log(
                state,
                f"[Verifier] 관련성 {score:.2f} (기준 {bar}) → {verdict}, 근거 {kept}건 채택",
            ),
        }

    def after_verify(state: AgentState) -> str:
        weak = state["search_score"] < retrieval["min_search_relevance"]
        if weak and state["rewrites"] < retrieval["max_rewrites"]:
            return "rewriter"
        return "next_query"

    def rewrite(state: AgentState) -> AgentState:
        tried = [state["plan"][state["current"]]["query"], state["query"]]
        user = (
            f"User question: {state['question']}\n"
            f"Queries already tried: {json.dumps(sorted(set(tried)))}\n"
            f"Best relevance so far: {state['search_score']:.2f}"
        )
        system = REWRITER.format(as_of=state["as_of"], query_style=QUERY_STYLE)
        result = llm.json("rewriter", system, user, shapes["rewriter"])
        return {
            "query": result["query"],
            "rewrites": state["rewrites"] + 1,
            "log": log(
                state,
                f"[Verifier] 검색어 수정: '{state['query']}' → '{result['query']}'",
            ),
        }

    def advance(state: AgentState) -> AgentState:
        current = state["current"] + 1
        if current >= len(state["plan"]):
            return {"current": current}
        return {
            "current": current,
            "query": state["plan"][current]["query"],
            "rewrites": 0,
        }

    def after_advance(state: AgentState) -> str:
        return "researcher" if state["current"] < len(state["plan"]) else "analyst"

    def analyze(state: AgentState) -> AgentState:
        evidence = sorted(
            state.get("evidence", []), key=lambda e: (-e["relevance"], e["date"])
        )[: retrieval["max_evidence"]]
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "answer": "기준일까지의 뉴스에서 질문에 맞는 근거를 찾지 못했습니다.",
                "risk_tags": [],
                "sources": [],
                "log": log(state, "[Analyst] 근거가 없어 분석을 생략합니다."),
            }
        numbered = "\n\n".join(
            f"[{n}] {e['title']} ({e['date']}, {e['source']})\n"
            + _snippet(e["text"], 900)
            for n, e in enumerate(evidence, 1)
        )
        user = (
            f"Question: {state['question']}\nAs of: {state['as_of']}\n\n"
            f"Evidence:\n{numbered}"
        )
        system = ANALYST.format(etfs=etf_lines)
        result = llm.json("analyst", system, user, shapes["analyst"])
        valid = set(range(1, len(evidence) + 1))
        answer = result["answer"]
        answer_refs = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        notes = []
        if invalid := sorted(answer_refs - valid):
            # a citation with no source behind it is dropped from the answer text
            answer = re.sub(
                r"\s?\[(\d+)\]",
                lambda m: m.group(0) if int(m.group(1)) in valid else "",
                answer,
            )
            notes.append(f"[Analyst] 근거 목록에 없는 인용 {invalid} 삭제")
        cited = answer_refs & valid
        tags = []
        for tag in result["risk_tags"]:
            refs = sorted(set(tag["evidence"]) & valid)
            if not refs:
                continue  # a tag must rest on listed evidence
            level = min(max(float(tag["risk_level"]), 0.0), 1.0)
            confidence = min(max(float(tag["confidence"]), 0.0), 1.0)
            tags.append(
                {
                    "ticker": tag["ticker"],
                    "risk_type": tag["risk_type"],
                    "risk_score": round(level * confidence, 3),
                    "risk_level": level,
                    "direction": tag["direction"],
                    "confidence": confidence,
                    "date": state["as_of"],
                    "evidence": refs,
                }
            )
            cited.update(refs)
        sources = [
            {
                "n": n,
                "title": evidence[n - 1]["title"],
                "source": evidence[n - 1]["source"],
                "date": evidence[n - 1]["date"],
                "url": evidence[n - 1]["url"],
            }
            for n in sorted(cited)
        ]
        if not answer.strip() or not sources:
            raise AgentError("분석 결과에 답변 또는 유효한 출처가 없습니다.")
        return {
            "status": "complete",
            "answer": answer,
            "risk_tags": tags,
            "sources": sources,
            "log": log(
                state,
                *notes,
                f"[Analyst] 근거 {len(evidence)}건 중 {len(sources)}건 인용,"
                f" 위험 태그 {len(tags)}개",
            ),
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan)
    graph.add_node("researcher", retrieve)
    graph.add_node("verifier", verify)
    graph.add_node("rewriter", rewrite)
    graph.add_node("next_query", advance)
    graph.add_node("analyst", analyze)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner", lambda s: "researcher" if s["plan"] else "analyst"
    )
    graph.add_edge("researcher", "verifier")
    graph.add_conditional_edges("verifier", after_verify)
    graph.add_edge("rewriter", "researcher")
    graph.add_conditional_edges("next_query", after_advance)
    graph.add_edge("analyst", END)
    return graph.compile()


def load_settings(path: Path = AGENT_CONFIG) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_llm(settings: dict) -> ClaudeLLM:
    import anthropic

    from backend.news_research.settings import env_value

    api_key = env_value("ANTHROPIC_API_KEY")
    if not api_key:
        raise AgentError("ANTHROPIC_API_KEY가 없습니다. .env에 추가하세요.")
    return ClaudeLLM(
        anthropic.Anthropic(api_key=api_key, timeout=20.0, max_retries=0),
        settings["models"],
        settings["effort"],
    )


def default_search(
    db_path: Path | None = None, *, local_embeddings_only: bool = False
) -> Search:
    from backend.news_research import vector_store

    embedding = None
    if local_embeddings_only:
        embedding = vector_store.LocalEmbedding()
        embedding._download_model_if_not_exists()
    collection = vector_store.open_collection(
        db_path or vector_store.DB_PATH, embedding_function=embedding, create=False
    )

    def search(query: str, k: int, **filters) -> list[dict]:
        return vector_store.search(collection, query, k, **filters)

    return search


def research(
    question: str,
    as_of: str | None = None,
    *,
    llm: LLM | None = None,
    search: Search | None = None,
    settings: dict | None = None,
    news_map: dict | None = None,
) -> dict:
    """Answer an investment question from the news store, with citations.

    Returns answer, risk_tags, sources, log, plan, as_of and disclaimer - the
    body of the POST /research response.
    """
    settings = settings or load_settings()
    if news_map is None:
        with NEWS_MAP.open(encoding="utf-8") as f:
            news_map = yaml.safe_load(f)
    as_of = as_of or datetime.now(NEW_YORK).date().isoformat()
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise AgentError(
            f"기준일 형식이 올바르지 않습니다: {as_of} (YYYY-MM-DD)"
        ) from exc
    graph = build_graph(
        llm or default_llm(settings), search or default_search(), settings, news_map
    )
    state = graph.invoke(
        {"question": question, "as_of": as_of, "log": []},
        config={"recursion_limit": 80},
    )
    return {
        "status": state["status"],
        "question": question,
        "as_of": as_of,
        "answer": state["answer"],
        "risk_tags": state.get("risk_tags", []),
        "sources": state.get("sources", []),
        "plan": state.get("plan", []),
        "log": state.get("log", []),
        "disclaimer": settings["disclaimer"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="뉴스 리서치 에이전트")
    parser.add_argument("question")
    parser.add_argument("--as-of", help="기준일 YYYY-MM-DD (이날까지의 뉴스만 사용)")
    parser.add_argument("--db", type=Path, help="ChromaDB 경로 (기본: 프로젝트 DB)")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = research(args.question, args.as_of, search=default_search(args.db))
    except AgentError as exc:
        print(f"오류: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("\n".join(result["log"]), end="\n\n")
    print(result["answer"], end="\n\n")
    for tag in result["risk_tags"]:
        print(
            f"- {tag['ticker']} {tag['risk_type']} risk_score {tag['risk_score']}"
            f" ({tag['direction']}, 근거 {tag['evidence']})"
        )
    print()
    for src in result["sources"]:
        print(
            f"[{src['n']}] {src['title']} ({src['date']}, {src['source']}) {src['url']}"
        )
    print("\n" + result["disclaimer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
