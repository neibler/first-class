"""FastAPI router serving saved analysis results (과제 25번).

웹 담당 사용법 (backend/main.py)
    from backend.performance.api import router as performance_router
    app.include_router(performance_router)

투자 AI 의 기존 GET /backtest 는 모델을 직접 호출하는 라이브 엔드포인트라
이름이 겹치지 않게 /performance 아래에 둔다.

    GET /performance/backtest                 전략 비교 + 일별 경로 + 12개 지표
    GET /performance/explain?ticker=&date=    SHAP 요약/개별 결정 해석
    GET /performance/statistics               ANOVA·Tukey 결과
    GET /performance/summary                  한 장 요약
    GET /performance/health                   결과 파일 존재 여부

결과 파일은 run_analysis.py 가 results/api/*.json 에 만들어 둔 것을 그대로 읽는다.
RESULTS_DIR 환경변수로 경로를 바꿀 수 있다.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/performance", tags=["performance"])


def results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "results")).expanduser()


@lru_cache(maxsize=16)
def _read(path_text: str, mtime: float) -> dict:
    """mtime 을 키에 포함해 파일이 바뀌면 캐시가 자동으로 무효화되게 한다."""
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def load(name: str, subfolder: str = "api") -> dict:
    path = results_dir() / subfolder / name
    if not path.exists():
        raise HTTPException(
            503,
            f"분석 결과가 아직 없습니다: {path}. "
            "python -m backend.performance.run_analysis 를 먼저 실행하세요.",
        )
    return _read(str(path), path.stat().st_mtime)


@router.get("/health")
def health() -> dict:
    root = results_dir()
    files = {
        "backtest": root / "api" / "backtest.json",
        "explain": root / "api" / "explain.json",
        "statistics": root / "statistics" / "anova.json",
        "summary": root / "summary.json",
    }
    available = {name: path.exists() for name, path in files.items()}
    return {
        "status": "ok",
        "results_dir": str(root.resolve()),
        "available": available,
        "ready": all(available.values()),
    }


@router.get("/backtest")
def backtest(
    strategy: str | None = Query(default=None, description="단일 전략만 필터"),
    include_path: bool = Query(default=True, description="일별 경로 포함 여부"),
) -> dict:
    payload = dict(load("backtest.json"))
    if strategy:
        selected = [s for s in payload["strategies"] if s["strategy"] == strategy]
        if not selected:
            raise HTTPException(404, f"알 수 없는 전략: {strategy}")
        payload["strategies"] = selected
        # 일별 경로도 같은 전략만 남긴다. 요약만 거르면 프론트에서 요약과
        # 그래프의 전략 목록이 어긋난다.
        payload["path"] = [
            {
                key: value
                for key, value in row.items()
                if key in ("date", "regime", strategy)
            }
            for row in payload.get("path", [])
        ]
    if not include_path:
        payload.pop("path", None)
    return payload


@router.get("/explain")
def explain(
    ticker: str | None = Query(default=None),
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict:
    payload = load("explain.json")
    if ticker is None:
        return {
            "features": payload["features"],
            "feature_importance": payload.get("feature_importance"),
            "assets": list(payload["assets"]),
            "note": payload["note"],
        }
    asset = payload["assets"].get(ticker)
    if asset is None:
        raise HTTPException(404, f"SHAP 결과에 없는 자산: {ticker}")
    decisions = asset["decisions"]
    if date is not None:
        decisions = [d for d in decisions if d["date"] == date]
        if not decisions:
            raise HTTPException(404, f"{ticker} 의 {date} 결정 해석이 없습니다")
    return {
        "ticker": ticker,
        "name": asset["name"],
        "features": payload["features"],
        "output_unit": payload["output_unit"],
        "note": payload["note"],
        "decisions": decisions,
    }


@router.get("/statistics")
def statistics() -> dict:
    return {"results": load("anova.json", subfolder="statistics")}


@router.get("/summary")
def summary() -> dict:
    path = results_dir() / "summary.json"
    if not path.exists():
        raise HTTPException(503, f"요약 파일이 없습니다: {path}")
    return _read(str(path), path.stat().st_mtime)


SERVING_CONFIG = Path(__file__).resolve().parents[2] / "config/serving.json"


@router.get("/model-shap")
def model_shap(
    model: str | None = Query(default=None, pattern="^(ppo|recurrent_ppo|sac|td3)$")
):
    from backend.investment_ai.settings import ROOT
    import hashlib

    config = (
        json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
        if SERVING_CONFIG.exists()
        else {}
    )
    selected = os.environ.get("INVESTMENT_AI_RUN") or config.get("model_path")
    if selected:
        # Same precedence and path semantics as InvestmentService.
        configured = Path(selected)
        if not os.environ.get("INVESTMENT_AI_RUN") and not configured.is_absolute():
            configured = ROOT / configured
        if configured.name == "model.zip":
            configured = configured.parent
        bundle = configured.resolve()
        try:
            kind = json.loads((bundle / "config.json").read_text(encoding="utf-8")).get(
                "model_type", "ppo"
            )
        except (OSError, ValueError):
            raise HTTPException(503, "활성 모델 폴더를 확인하세요.") from None
        if model is not None and model != kind:
            raise HTTPException(404, "현재 활성 투자 모델의 SHAP만 제공합니다.")
        model = kind
    else:
        # Backward-compatible shared registry only when no active model is set.
        model = model or "ppo"
        registry = (
            Path(__file__).resolve().parents[2] / "shared/trained_models_manifest.json"
        )
        if not registry.exists():
            raise HTTPException(503, "활성 모델 또는 공유 모델 목록이 없습니다.")
        item = json.loads(registry.read_text(encoding="utf-8"))["models"].get(model)
        if not item:
            raise HTTPException(404, "모델이 없습니다.")
        bundle = (ROOT / item["path"]).resolve()
        if not bundle.is_relative_to(ROOT.resolve()):
            raise HTTPException(503, "모델 경로 설정 오류")
    path = bundle / "shap.json"
    if not path.exists() or not (bundle / "model.zip").exists():
        raise HTTPException(
            503,
            "이 모델의 실제 SHAP 결과가 없습니다. 해석 계산 또는 공유본 설치가 필요합니다.",
        )
    payload = _read(str(path), path.stat().st_mtime_ns)
    digest = hashlib.sha256((bundle / "model.zip").read_bytes()).hexdigest()
    if payload["model_sha256"] != digest or payload["model_type"] != model:
        raise HTTPException(409, "모델과 SHAP 결과가 다릅니다. 다시 계산하세요.")
    from .shap_context import attach_market_context

    from .shap_decisions import attach_decisions

    return attach_decisions(attach_market_context(payload, bundle), bundle)


@router.get("/correlation/{ticker}")
def correlation(
    ticker: str,
    window: int = Query(60),
    top_k: int = Query(5, ge=1, le=30),
    as_of: str | None = None,
):
    from datetime import date
    from .correlation import CONFIG, load_correlation_data, analyze_correlations

    if window not in CONFIG.windows:
        raise HTTPException(422, "window는 20, 60, 120, 252 중 하나여야 합니다.")
    if as_of is not None:
        try:
            if date.fromisoformat(as_of).isoformat() != as_of:
                raise ValueError("Invalid date")
        except ValueError:
            raise HTTPException(422, "as_of는 YYYY-MM-DD 형식이어야 합니다.") from None
    try:
        data = load_correlation_data()
    except (OSError, ValueError, KeyError):
        raise HTTPException(
            503, "실제 시장 입력을 읽지 못했습니다. 입력 데이터와 설정을 확인하세요."
        ) from None
    if ticker.upper() not in data.assets:
        raise HTTPException(404, "분석 대상 ETF가 아닙니다.")
    return analyze_correlations(data, ticker.upper(), window, top_k, as_of)
