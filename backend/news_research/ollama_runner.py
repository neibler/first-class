"""Run prepared risk-tagging requests through an Ollama server.

Works the same on a Colab GPU or this laptop once Ollama is serving a model:

    python -m backend.news_research.ollama_runner --run RUN_DIR --model qwen2.5:7b

Reads RUN_DIR/requests.jsonl, prompt.md and schema.json (made by
risk_tagger prepare), asks Ollama for JSON matching the schema at temperature
0, and appends one line per request to RUN_DIR/outputs.jsonl as soon as it
returns. Re-running skips requests already answered, so a dropped Colab
session resumes where it stopped. The model only ever sees ids, titles and
summaries; URLs stay in keys.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from backend.news_research.risk_tagger import answered, user_prompt

OLLAMA_URL = "http://localhost:11434/api/chat"
# System prompt (~1.6k tokens) + 20 articles in and out fit well inside 8k;
# Ollama's default context is smaller and would silently cut the prompt.
NUM_CTX = 8192

logger = logging.getLogger(__name__)


def chat_body(
    request: dict, system: str, schema: dict, model: str, num_ctx: int = NUM_CTX
) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt(request)},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }


def run(
    run_dir: Path,
    model: str,
    workers: int = 1,
    url: str = OLLAMA_URL,
    num_ctx: int = NUM_CTX,
    outputs_name: str = "outputs.jsonl",
    post: Callable[..., httpx.Response] = httpx.post,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[int, int]:
    """Answer every unanswered request; returns (answered now, failed)."""
    system = (run_dir / "prompt.md").read_text(encoding="utf-8")
    schema = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    outputs = run_dir / outputs_name
    done = answered(outputs)
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as f:
        todo = [
            r
            for r in map(json.loads, filter(str.strip, f))
            if r["request_id"] not in done
        ]
    logger.info("요청 %d개 남음 (이미 답한 요청 %d개)", len(todo), len(done))

    lock, started = threading.Lock(), clock()
    finished = failed = generated = 0

    def ask(request: dict) -> dict:
        body = chat_body(request, system, schema, model, num_ctx)
        response = post(url, json=body, timeout=600.0)
        response.raise_for_status()
        data = response.json()
        return {
            "request_id": request["request_id"],
            "output": data["message"]["content"],
            "model": model,
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        }

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        outputs.open("a", encoding="utf-8") as out,
    ):
        futures = {pool.submit(ask, request): request for request in todo}
        for future in as_completed(futures):
            request_id = futures[future]["request_id"]
            try:
                row = future.result()
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                failed += 1
                logger.warning("%s 실패(%s) - 다시 실행하면 재시도", request_id, exc)
                continue
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                finished += 1
                generated += row["output_tokens"] or 0
                if finished % 10 == 0 or finished == len(todo):
                    elapsed = max(clock() - started, 1e-9)
                    left = (len(todo) - finished) * elapsed / finished
                    logger.info(
                        "%d/%d 완료, 출력 %.0f토큰/초, 남은 시간 약 %.0f분",
                        finished,
                        len(todo),
                        generated / elapsed,
                        left / 60,
                    )
    return finished, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ollama로 위험 태깅 요청 실행")
    parser.add_argument("--run", type=Path, required=True, help="prepare 실행 폴더")
    parser.add_argument("--model", required=True, help="예: qwen2.5:7b")
    parser.add_argument("--workers", type=int, default=1, help="동시 요청 수")
    parser.add_argument("--url", default=OLLAMA_URL)
    parser.add_argument("--num-ctx", type=int, default=NUM_CTX)
    parser.add_argument(
        "--outputs", default="outputs.jsonl", help="결과 파일 이름 (모델별로 따로)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    finished, failed = run(
        args.run, args.model, args.workers, args.url, args.num_ctx, args.outputs
    )
    logger.info("이번 실행: 응답 %d개, 실패 %d개", finished, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
