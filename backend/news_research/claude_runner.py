"""Run prepared risk-tagging requests through the Claude Message Batches API.

    python -m backend.news_research.claude_runner --run RUN_DIR

Reads RUN_DIR/requests.jsonl, prompt.md and schema.json (made by risk_tagger
prepare), sends them as one Message Batch to claude-haiku-5-5 (half the
per-token price of single calls), waits for the batch to end and appends one
line per request to RUN_DIR/outputs_<model>.jsonl. Submitted batch ids are
kept in RUN_DIR/claude_batches_<model>.json, so an interrupted run waits for
the same batch instead of paying for it twice; requests that failed are sent
again on the next run. Needs ANTHROPIC_API_KEY (environment or .env). Only ids,
titles and summaries leave this machine - never URLs or dates.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from backend.news_research.risk_tagger import answered, user_prompt
from backend.news_research.settings import env_value

if TYPE_CHECKING:  # the SDK is slow to import; main() loads it when it runs
    import anthropic

MODEL = "claude-haiku-5-5"
EFFORT = "low"  # classification; thinking stays adaptive but short
MAX_TOKENS = 8000  # 20 labels (~1.5k tokens) plus thinking, with room to spare
USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
# Batch prices per million tokens (pricing page, 2026-10): half the standard
# rates, with the cache discounts stacked on top. Haiku: prompts up to 100K.
PRICES = {
    "claude-haiku-5-5": {
        "input_tokens": 0.05,
        "output_tokens": 0.25,
        "cache_creation_input_tokens": 0.0625,
        "cache_read_input_tokens": 0.005,
    },
    "claude-sonnet-5-5": {
        "input_tokens": 1.0,
        "output_tokens": 5.0,
        "cache_creation_input_tokens": 1.25,
        "cache_read_input_tokens": 0.05,
    },
}

logger = logging.getLogger(__name__)


def batch_params(
    request: dict, system: str, schema: dict, model: str, effort: str = EFFORT
) -> dict:
    """Messages API parameters for one prepared request."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # The rubric is identical in every request, so cache it
        "system": [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": user_prompt(request)}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
    }


def cost(usage: Counter, model: str = MODEL) -> float | None:
    """Batch cost in dollars, or None for a model without a price entry."""
    prices = PRICES.get(model)
    if prices is None:
        return None
    return sum(usage[key] * price for key, price in prices.items()) / 1_000_000


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def wait(
    client: anthropic.Anthropic,
    batch_id: str,
    poll: float,
    sleep: Callable[[float], None],
) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return
        counts = batch.request_counts
        logger.info(
            "배치 %s 처리 중: 완료 %d, 진행 %d",
            batch_id,
            counts.succeeded + counts.errored,
            counts.processing,
        )
        sleep(poll)


def collect(
    client: anthropic.Anthropic, batch_id: str, outputs: Path, model: str
) -> tuple[int, int, Counter]:
    """Append succeeded answers; returns (written, failed, token usage)."""
    written = failed = 0
    usage: Counter = Counter()
    with outputs.open("a", encoding="utf-8") as out:
        for item in client.messages.batches.results(batch_id):
            result = item.result
            if result.type != "succeeded":
                failed += 1
                logger.warning(
                    "%s: %s - 다음 실행에서 재요청", item.custom_id, result.type
                )
                continue
            message = result.message
            for key in USAGE_KEYS:
                usage[key] += getattr(message.usage, key, 0) or 0
            text = next((b.text for b in message.content if b.type == "text"), "")
            if message.stop_reason != "end_turn" or not text:
                # refusal, or cut off at max_tokens before finishing the JSON
                failed += 1
                logger.warning(
                    "%s: stop_reason=%s - 다음 실행에서 재요청",
                    item.custom_id,
                    message.stop_reason,
                )
                continue
            row = {
                "request_id": item.custom_id,
                "output": text,
                "model": model,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written, failed, usage


def run(
    run_dir: Path,
    client: anthropic.Anthropic,
    model: str = MODEL,
    poll: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    effort: str = EFFORT,
    tag: str | None = None,
) -> tuple[int, int, Counter]:
    """Finish pending batches, then submit and finish unanswered requests.

    tag names the outputs/state files (default: the model), so one prepared run
    can be tagged under several settings without mixing results.
    """
    tag = tag or model
    outputs = run_dir / f"outputs_{tag}.jsonl"
    state_path = run_dir / f"claude_batches_{tag}.json"
    state = _load_state(state_path)
    written = failed = 0
    usage: Counter = Counter()

    def finish(entry: dict) -> None:
        nonlocal written, failed
        wait(client, entry["id"], poll, sleep)
        w, f, u = collect(client, entry["id"], outputs, model)
        written, failed = written + w, failed + f
        usage.update(u)
        entry["collected"] = True
        _save_state(state_path, state)

    for entry in state["batches"]:
        if not entry["collected"]:
            logger.info("이전에 보낸 배치 %s 결과를 기다립니다.", entry["id"])
            finish(entry)

    system = (run_dir / "prompt.md").read_text(encoding="utf-8")
    schema = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    done = answered(outputs)
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as f:
        todo = [
            r
            for r in map(json.loads, filter(str.strip, f))
            if r["request_id"] not in done
        ]
    if todo:
        batch = client.messages.batches.create(
            requests=[
                {
                    "custom_id": r["request_id"],
                    "params": batch_params(r, system, schema, model, effort),
                }
                for r in todo
            ]
        )
        entry = {
            "id": batch.id,
            "requests": len(todo),
            "collected": False,
        }
        state["batches"].append(entry)
        _save_state(state_path, state)
        logger.info("배치 %s 제출: 요청 %d개", batch.id, len(todo))
        finish(entry)
    return written, failed, usage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claude Batch API로 위험 태깅 실행")
    parser.add_argument("--run", type=Path, required=True, help="prepare 실행 폴더")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--effort", default=EFFORT, help="low / medium / high")
    parser.add_argument(
        "--tag", help="결과·상태 파일 이름 (기본: 모델 이름). 설정별로 따로 저장할 때"
    )
    parser.add_argument("--poll", type=float, default=30.0, help="상태 확인 간격(초)")
    args = parser.parse_args(argv)
    import anthropic

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    api_key = env_value("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY가 없습니다. .env에 추가하세요.")
        return 2
    client = anthropic.Anthropic(api_key=api_key)
    try:
        written, failed, usage = run(
            args.run, client, args.model, args.poll, effort=args.effort, tag=args.tag
        )
    except anthropic.AuthenticationError:
        logger.error("API 키가 올바르지 않습니다 (401).")
        return 2
    except anthropic.APIStatusError as exc:
        logger.error("Claude API 오류 %s (%s): %s", exc.status_code, exc.type, exc)
        return 2
    except anthropic.APIConnectionError:
        logger.error("Claude API에 연결하지 못했습니다. 다시 실행하면 이어서 합니다.")
        return 2
    logger.info(
        "응답 %d개 저장, 실패 %d개 | 토큰 입력 %d, 출력 %d, 캐시 읽기 %d, 캐시 쓰기 %d"
        " | 예상 비용 $%s",
        written,
        failed,
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
        f"{spent:.3f}" if (spent := cost(usage, args.model)) is not None else "?",
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
