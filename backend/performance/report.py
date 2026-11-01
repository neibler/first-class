"""Persist analysis outputs and build API payloads (과제 24·25번).

폴더 구조
    results/
    ├── metrics/      성과지표·전략 비교·국면별 표
    ├── backtest/     전략별 일별 히스토리, 누적수익률·drawdown 그림
    ├── shap/         shap.json, summary.png, force.png, 중요도 표
    └── statistics/   anova.csv, tukey.csv, 해석 텍스트
    results/api/      웹 담당이 /backtest, /explain 로 그대로 내보낼 JSON

JSON 은 NaN 을 쓰지 않는다 (investment_ai.settings.write_json 과 동일 규칙).
정의 불가 지표는 null 로 나간다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_LABELS, METRIC_ORDER
from .strategies import STRATEGY_LABELS, drawdown_frame, value_frame

SUBFOLDERS = ("metrics", "backtest", "shap", "statistics", "api")


def prepare(output: Path) -> dict[str, Path]:
    """결과 폴더를 만들고 하위 경로 사전을 돌려준다."""
    output = Path(output)
    paths = {"root": output}
    for name in SUBFOLDERS:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        paths[name] = folder
    return paths


def clean(value):
    """NaN/NumPy 타입을 JSON 이 받을 수 있는 값으로 바꾼다."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return str(value)


def write_json(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """과제 9번 성과 비교 테이블 — 한국어 라벨과 보기 좋은 반올림."""
    table = metrics.copy()
    table.index.name = "strategy"
    table = table.reset_index()
    table.insert(1, "전략", table.strategy.map(STRATEGY_LABELS).fillna(table.strategy))
    renamed = {key: f"{METRIC_LABELS[key]} ({key})" for key in METRIC_ORDER}
    return table.rename(columns=renamed)


def save_backtest(paths: dict, histories: dict, returns: pd.DataFrame) -> None:
    for name, history in histories.items():
        history.to_csv(paths["backtest"] / f"{name}_history.csv", index=False)
    returns.to_csv(paths["backtest"] / "daily_returns.csv")
    value_frame(returns).to_csv(paths["backtest"] / "portfolio_value.csv")
    drawdown_frame(returns).to_csv(paths["backtest"] / "drawdown.csv")


def backtest_payload(
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    benchmark: str = "benchmark",
    regimes: pd.Series | None = None,
    turnover: dict | None = None,
    meta: dict | None = None,
) -> dict:
    """GET /backtest 응답 본문 (웹 담당이 그대로 반환하면 된다)."""
    values = value_frame(returns)
    drawdowns = drawdown_frame(returns)
    path = []
    for i, date in enumerate(returns.index):
        row = {"date": pd.Timestamp(date).strftime("%Y-%m-%d")}
        for column in returns.columns:
            row[column] = {
                "return": float(returns[column].iloc[i]),
                "value": float(values[column].iloc[i]),
                "drawdown": float(drawdowns[column].iloc[i]),
            }
        if regimes is not None:
            row["regime"] = str(regimes.iloc[i])
        path.append(row)
    strategies = []
    for name, row in metrics.iterrows():
        strategies.append(
            {
                "strategy": name,
                "label": STRATEGY_LABELS.get(name, name),
                "is_benchmark": name == benchmark,
                "days": int(row.get("days", 0)),
                "metrics": {key: row.get(key) for key in METRIC_ORDER},
                "turnover": (turnover or {}).get(name),
            }
        )
    # 정의 불가 지표(NaN)를 null 로 바꿔 항상 JSON 직렬화 가능한 본문을 돌려준다.
    return clean(
        {
            "schema_version": "performance-1",
            "metric_labels": METRIC_LABELS,
            "benchmark": benchmark,
            "periods_per_year": 252,
            "start_date": pd.Timestamp(returns.index[0]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(returns.index[-1]).strftime("%Y-%m-%d"),
            "strategies": strategies,
            "path": path,
            **(meta or {}),
        }
    )


def explain_payload(shap_payload: dict, importance: pd.DataFrame | None = None) -> dict:
    """GET /explain 응답 본문. frontend/fixtures/shap.json 과 같은 스키마 + 요약."""
    payload = {
        "schema_version": "performance-explain-1",
        "output_unit": "target_weight",
        "link": "identity",
        "note": (
            "SHAP 값은 해당 자산 목표 비중에 대한 기여도입니다 (+0.02 = +2%p). "
            "피처 값은 모델이 보는 전처리 공간 기준이며, 인과관계나 투자 권고가 아닙니다."
        ),
        "features": shap_payload["features"],
        "assets": shap_payload["assets"],
    }
    if importance is not None and not importance.empty:
        payload["feature_importance"] = (
            importance.groupby("feature")
            .mean_abs_shap.mean()
            .sort_values(ascending=False)
            .reset_index()
            .to_dict("records")
        )
    return payload


#: 프로젝트 목표치. **지표마다 비교 대상이 다르다.**
#:   - Sharpe: 동일가중 대비 +0.2 (위험조정 성과가 단순 분산투자보다 나은가)
#:   - 누적수익률: S&P500 대비 +10%p (시장을 이겼는가)
#: 둘 다 시장과 비교하면, 동일가중보다 못한 Sharpe 도 "달성"으로 잘못 판정된다.
TARGETS = {
    "sharpe": {"reference": "equal_weight", "threshold": 0.2},
    "cumulative_return": {"reference": "benchmark_raw", "threshold": 0.10},
}
#: 기준 전략이 표에 없을 때의 대체 순서.
REFERENCE_FALLBACKS = {
    "equal_weight": ("equal_weight", "mvo", "benchmark_raw", "benchmark"),
    "benchmark_raw": ("benchmark_raw", "market_index", "benchmark"),
    "market_index": ("market_index", "benchmark_raw", "benchmark"),
    "benchmark": ("benchmark", "benchmark_raw", "market_index"),
}


def resolve_reference(metrics: pd.DataFrame, reference: str) -> str | None:
    for name in REFERENCE_FALLBACKS.get(reference, (reference,)):
        if name in metrics.index:
            return name
    return None


def target_check(
    metrics: pd.DataFrame,
    strategy: str = "ppo",
    targets: dict | None = None,
    alternatives: tuple = ("market_index", "benchmark_raw", "equal_weight", "mvo"),
) -> dict:
    """목표 달성 여부 판정 (지표마다 지정된 비교 대상 사용).

    Args:
        targets: {지표: {"reference": 전략, "threshold": 값}}. 생략 시 TARGETS.
        alternatives: 참고용으로 함께 기록할 다른 비교 대상들.

    미달이어도 그대로 보고한다. 수치 자체가 결론이기 때문이다.
    """
    targets = targets or TARGETS
    if strategy not in metrics.index:
        raise ValueError(f"지표 표에 '{strategy}' 행이 없습니다")

    def difference(key, other):
        value, base = metrics.loc[strategy, key], metrics.loc[other, key]
        if pd.isna(value) or pd.isna(base):
            return None
        return float(value) - float(base)

    rows = {}
    for key, spec in targets.items():
        threshold = float(spec["threshold"])
        reference = resolve_reference(metrics, spec["reference"])
        gap = None if reference is None else difference(key, reference)
        value = metrics.loc[strategy, key]
        rows[key] = {
            "reference": reference,
            "requested_reference": spec["reference"],
            "strategy_value": None if pd.isna(value) else float(value),
            "reference_value": (
                None
                if reference is None or pd.isna(metrics.loc[reference, key])
                else float(metrics.loc[reference, key])
            ),
            "difference": gap,
            "threshold": threshold,
            "met": None if gap is None else bool(gap >= threshold),
            # 보고서에서 다른 기준으로도 인용할 수 있도록 함께 기록한다.
            "difference_vs": {
                other: difference(key, other)
                for other in alternatives
                if other in metrics.index and other != reference
            },
        }
    met = [row["met"] for row in rows.values()]
    return {
        "strategy": strategy,
        "criteria": rows,
        "all_met": bool(met) and all(value is True for value in met),
        "note": (
            "지표마다 비교 대상이 다릅니다: Sharpe 는 동일가중, 누적수익률은 "
            "S&P500 기준입니다. 목표 미달도 유효한 결론이며, 미달이면 원인"
            "(데이터 길이, 학습량, 거래비용, Safe-Guard 발동)을 보고서에 적으세요."
        ),
    }


def summary_payload(sections: dict) -> dict:
    """담당자 간 공유용 한 장 요약 (무엇을 돌렸고 어디에 저장했는지)."""
    return {"schema_version": "performance-summary-1", **sections}


def save_all(
    output: Path,
    *,
    metrics: pd.DataFrame,
    histories: dict,
    returns: pd.DataFrame,
    walk_forward: pd.DataFrame | None = None,
    regime_table: pd.DataFrame | None = None,
    regimes: pd.Series | None = None,
    reward_table: pd.DataFrame | None = None,
    risk_tag_table: pd.DataFrame | None = None,
    anova_results: list | None = None,
    anova_table: pd.DataFrame | None = None,
    tukey: pd.DataFrame | None = None,
    shap_payload: dict | None = None,
    shap_importance: pd.DataFrame | None = None,
    turnover: dict | None = None,
    meta: dict | None = None,
) -> dict:
    """모든 산출물을 한 번에 저장하고 저장 경로 목록을 돌려준다."""
    paths = prepare(output)
    comparison_table(metrics).to_csv(
        paths["metrics"] / "strategy_comparison.csv", index=False
    )
    metrics.to_csv(paths["metrics"] / "metrics.csv", index_label="strategy")
    save_backtest(paths, histories, returns)
    if walk_forward is not None:
        walk_forward.to_csv(paths["metrics"] / "walk_forward.csv", index=False)
    if regime_table is not None:
        regime_table.to_csv(paths["metrics"] / "market_regime.csv", index=False)
    if regimes is not None:
        regimes.to_frame("regime").to_csv(paths["metrics"] / "regime_labels.csv")
    if reward_table is not None:
        reward_table.to_csv(paths["metrics"] / "reward_comparison.csv", index=False)
    if risk_tag_table is not None:
        risk_tag_table.to_csv(paths["metrics"] / "risk_tag_effect.csv", index=False)
    if anova_table is not None:
        anova_table.to_csv(paths["statistics"] / "anova.csv", index=False)
    if tukey is not None:
        tukey.to_csv(paths["statistics"] / "tukey.csv", index=False)
    if anova_results:
        write_json(paths["statistics"] / "anova.json", anova_results)
        (paths["statistics"] / "interpretation.txt").write_text(
            "\n\n".join(
                result.get("interpretation", "")
                for result in anova_results
                if result.get("interpretation")
            ),
            encoding="utf-8",
        )
    if shap_payload is not None:
        write_json(paths["shap"] / "shap.json", shap_payload)
        write_json(
            paths["api"] / "explain.json",
            explain_payload(shap_payload, shap_importance),
        )
    if shap_importance is not None:
        shap_importance.to_csv(paths["shap"] / "feature_importance.csv", index=False)
    write_json(
        paths["api"] / "backtest.json",
        backtest_payload(
            metrics, returns, regimes=regimes, turnover=turnover, meta=meta
        ),
    )
    return {name: str(path) for name, path in paths.items()}
