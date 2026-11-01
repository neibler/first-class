"""News risk-tag effect analysis (과제 17번).

두 가지 방식을 모두 제공한다. 보고서에는 어느 쪽인지 반드시 밝혀야 한다.

1) ablation (기본, 빠름)
   같은 PPO 모델에 risk_score 입력만 0 으로 바꿔 다시 백테스트한다.
   "학습된 정책이 뉴스 입력에 실제로 반응하는가"를 본다. 재학습이 없다.

2) retrain (정확, 느림)
   risk_score 를 0 으로 만든 데이터로 PPO 를 처음부터 다시 학습해 비교한다.
   "뉴스 AI 를 붙이면 성능이 좋아지는가"에 대한 제대로 된 답이다.

SHAP 쪽에서도 risk_score 피처의 평균 |SHAP| 순위를 함께 확인한다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_ORDER, performance_metrics
from .strategies import run_strategy


def without_risk(data):
    """risk_score 를 전부 0 으로 만든 사본 (뉴스 AI 미적용 상태)."""
    return replace(data, risk=np.zeros_like(data.risk))


def ablation(
    data,
    config: dict,
    start: int,
    end: int,
    models,
    scaler=None,
    benchmark_returns=None,
) -> dict:
    """같은 모델, risk 입력만 제거해 성능 차이를 본다."""
    with_tag = run_strategy(data, config, start, end, "ppo", models, scaler)
    without_tag = run_strategy(
        without_risk(data),
        {**config, "window_id": f"{config.get('window_id', 'standalone')}_no_risk"},
        start,
        end,
        "ppo",
        models,
        scaler,
    )
    annual_rf = config.get("risk_free_annual", 0.0)
    rows = []
    for label, history in (
        ("with_risk_tag", with_tag),
        ("without_risk_tag", without_tag),
    ):
        values = performance_metrics(
            history.portfolio_return.to_numpy(dtype=float), benchmark_returns, annual_rf
        )
        rows.append(
            {"variant": label, **{k: values[k] for k in ["days", *METRIC_ORDER]}}
        )
    table = pd.DataFrame(rows)
    return {
        "method": "ablation",
        "table": table,
        "histories": {"with_risk_tag": with_tag, "without_risk_tag": without_tag},
        "difference": difference(table),
        "weight_distance": weight_distance(with_tag, without_tag, data.assets),
    }


def retrain(
    data,
    config: dict,
    train_end: int,
    start: int,
    end: int,
    seeds,
    output: Path,
    train_fn=None,
    benchmark_returns=None,
) -> dict:
    """risk 없는 데이터로 PPO 를 재학습해 비교한다 (느림)."""
    from .walk_forward import default_train_fn

    train_fn = train_fn or default_train_fn
    output = Path(output)
    annual_rf = config.get("risk_free_annual", 0.0)
    rows, histories = [], {}
    for label, dataset in (
        ("with_risk_tag", data),
        ("without_risk_tag", without_risk(data)),
    ):
        models, scaler = [], None
        for seed in seeds:
            directory = output / label / f"seed_{seed}"
            model, scaler = train_fn(
                dataset,
                {**config, "window_id": f"risk_tag_{label}"},
                train_end,
                seed,
                directory,
            )
            models.append(model)
        history = run_strategy(dataset, config, start, end, "ppo", models, scaler)
        histories[label] = history
        values = performance_metrics(
            history.portfolio_return.to_numpy(dtype=float), benchmark_returns, annual_rf
        )
        rows.append(
            {"variant": label, **{k: values[k] for k in ["days", *METRIC_ORDER]}}
        )
    table = pd.DataFrame(rows)
    return {
        "method": "retrain",
        "table": table,
        "histories": histories,
        "difference": difference(table),
    }


def difference(table: pd.DataFrame) -> dict:
    """with - without 차이를 주요 지표에 대해 뽑는다."""
    indexed = table.set_index("variant")
    keys = ("cumulative_return", "sharpe", "max_drawdown", "annualized_volatility")
    result = {}
    for key in keys:
        if key not in indexed.columns:
            continue
        a, b = indexed.loc["with_risk_tag", key], indexed.loc["without_risk_tag", key]
        result[key] = None if pd.isna(a) or pd.isna(b) else float(a - b)
    return result


def weight_distance(with_tag: pd.DataFrame, without_tag: pd.DataFrame, assets) -> dict:
    """두 변형의 일별 비중 차이 — 성과가 같아도 결정이 달라졌는지 본다."""
    columns = [f"current_weight_{a.lower()}" for a in assets]
    size = min(len(with_tag), len(without_tag))
    if not size:
        return {"mean_l1_distance": None, "max_l1_distance": None}
    left = with_tag[columns].to_numpy(dtype=float)[:size]
    right = without_tag[columns].to_numpy(dtype=float)[:size]
    distance = np.abs(left - right).sum(axis=1)
    return {
        "mean_l1_distance": float(distance.mean()),
        "max_l1_distance": float(distance.max()),
        "days_compared": int(size),
    }


def shap_risk_importance(importance: pd.DataFrame, feature_name="리스크 태그 점수"):
    """SHAP 평균 |SHAP| 표에서 risk_score 피처의 순위만 뽑는다."""
    subset = importance[importance.feature == feature_name]
    if subset.empty:
        return {"feature": feature_name, "mean_abs_shap": None, "mean_rank": None}
    return {
        "feature": feature_name,
        "mean_abs_shap": float(subset.mean_abs_shap.mean()),
        "mean_rank": float(subset["rank"].mean()),
        "top_assets": subset.nlargest(3, "mean_abs_shap").ticker.tolist(),
    }


def interpret(result: dict) -> str:
    """보고서용 한국어 해석 문장."""
    diff = result["difference"]
    method = "입력 차단(ablation)" if result["method"] == "ablation" else "재학습"
    pieces = []
    if diff.get("cumulative_return") is not None:
        pieces.append(f"누적수익률 {diff['cumulative_return']:+.2%}")
    if diff.get("sharpe") is not None:
        pieces.append(f"Sharpe {diff['sharpe']:+.3f}")
    if diff.get("max_drawdown") is not None:
        pieces.append(f"MDD {diff['max_drawdown']:+.2%}")
    changes = ", ".join(pieces) if pieces else "비교 가능한 지표 없음"
    distance = result.get("weight_distance", {}).get("mean_l1_distance")
    weight_note = (
        f" 일별 비중의 평균 L1 거리는 {distance:.4f} 로, 0 에 가까울수록 "
        "뉴스 입력이 결정을 거의 바꾸지 않았다는 뜻입니다."
        if distance is not None
        else ""
    )
    return (
        f"뉴스 리스크 태그 효과({method}): 태그 적용 대비 미적용의 차이는 "
        f"{changes} 입니다.{weight_note}"
    )
