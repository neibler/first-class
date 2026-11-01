"""SHAP explanations for PPO portfolio decisions (과제 14~16번).

왜 관측치를 6개 피처로 묶는가
- PortfolioEnv 관측은 lookback*자산수 + 4*자산수 + 매크로 + 3 차원이다
  (15자산·lookback 20이면 363차원). 그대로 SHAP 을 돌리면 느리고, 피처 하나하나가
  "SPY 의 17일 전 로그수익률"이라 해석도 되지 않는다.
- 그래서 과제가 요구하는 해석 단위(최근 수익률 / RSI / MACD / 변동성 / 현재 비중 /
  뉴스 리스크 점수)로 묶어 자산별로 설명한다. 프론트 담당이 쓰는
  frontend/fixtures/shap.json 과 같은 스키마로 저장해 교체만 하면 되게 했다.

해석 주의 (보고서에 적을 것)
- 피처 값은 모델이 실제로 보는 전처리(scaler) 공간의 값이다. 원 단위(%, RSI 0-100)가
  아니라 학습 구간 통계로 표준화된 값이라는 뜻이다.
- 출력은 해당 자산의 목표 비중(0~1)이다. SHAP +0.02 는 비중 +2%p 기여를 뜻한다.
- 기여도는 상관이지 인과가 아니며 투자 권고도 아니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.investment_ai.portfolio_env import PortfolioEnv

FEATURE_LABELS = [
    "최근 5일 로그 수익률",
    "RSI (14일)",
    "MACD",
    "20일 변동성",
    "현재 비중",
    "리스크 태그 점수",
]
FEATURE_KEYS = [
    "recent_return_5d",
    "rsi",
    "macd",
    "volatility_20d",
    "current_weight",
    "risk_score",
]
RECENT_WINDOW = 5


@dataclass
class Layout:
    """관측 벡터 안에서 각 블록이 시작하는 위치 (training.schema_for 의 order 와 동일)."""

    lookback: int
    n_assets: int
    n_macros: int

    @property
    def returns_end(self) -> int:
        return self.lookback * self.n_assets

    def return_indices(self, asset: int) -> np.ndarray:
        return np.arange(self.lookback) * self.n_assets + asset

    @property
    def rsi0(self) -> int:
        return self.returns_end

    @property
    def macd0(self) -> int:
        return self.rsi0 + self.n_assets

    @property
    def weight0(self) -> int:
        return self.macd0 + self.n_assets

    @property
    def macro0(self) -> int:
        return self.weight0 + self.n_assets

    @property
    def risk0(self) -> int:
        return self.macro0 + self.n_macros


class PolicyExplainer:
    """관측 벡터를 6개 해석 피처로 압축/복원하고 정책 출력을 계산한다."""

    def __init__(self, data, config, models, scaler=None):
        if not models:
            raise ValueError("SHAP 분석에는 학습된 PPO 모델이 필요합니다")
        self.data, self.config, self.models = data, config, models
        self.env = PortfolioEnv(data, config, scaler=scaler)
        self.layout = Layout(
            int(config["lookback"]), len(data.assets), len(data.macros)
        )

    # -- 압축 / 복원 ------------------------------------------------------- #
    def reduce(self, observation: np.ndarray, asset: int) -> np.ndarray:
        column = observation[self.layout.return_indices(asset)]
        return np.array(
            [
                column[-RECENT_WINDOW:].mean(),
                observation[self.layout.rsi0 + asset],
                observation[self.layout.macd0 + asset],
                column.std(),
                observation[self.layout.weight0 + asset],
                observation[self.layout.risk0 + asset],
            ],
            dtype=float,
        )

    def rebuild(self, base: np.ndarray, asset: int, reduced) -> np.ndarray:
        """6개 피처 값을 강제로 적용한 관측 벡터를 만든다.

        수익률 블록은 "최근 5일 평균"과 "20일 변동성"만 바꾸고 모양은 유지한다:
        먼저 편차를 목표 변동성에 맞춰 늘리거나 줄이고, 그다음 최근 5일 평균이
        목표값이 되도록 전체를 평행이동한다.
        """
        observation = np.asarray(base, dtype=np.float32).copy()
        x = np.asarray(reduced, dtype=float)
        indices = self.layout.return_indices(asset)
        column = observation[indices].astype(float)
        current_std = column.std()
        target_std = max(float(x[3]), 0.0)
        centre = column.mean()
        if current_std > 1e-9:
            column = (column - centre) * (target_std / current_std) + centre
        column = column + (float(x[0]) - column[-RECENT_WINDOW:].mean())
        observation[indices] = column
        observation[self.layout.rsi0 + asset] = np.clip(x[1], 0.0, 1.0)
        observation[self.layout.macd0 + asset] = x[2]
        observation[self.layout.weight0 + asset] = np.clip(x[4], 0.0, 1.0)
        observation[self.layout.risk0 + asset] = np.clip(x[5], 0.0, 1.0)
        return np.nan_to_num(observation, posinf=0.0, neginf=0.0).astype(np.float32)

    # -- 정책 출력 --------------------------------------------------------- #
    def target_weights(self, observations: np.ndarray) -> np.ndarray:
        """(n, obs_dim) -> (n, n_assets) 시드 앙상블 평균 목표 비중."""
        batch = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        per_model = []
        for model in self.models:
            actions = model.predict(batch, deterministic=True)[0]
            per_model.append(
                np.vstack([self.env.action_weights(a) for a in np.atleast_2d(actions)])
            )
        return np.mean(per_model, axis=0)

    def prediction_fn(self, base: np.ndarray, asset: int):
        """KernelExplainer 가 쓰는 f: 6개 피처 -> 해당 자산 목표 비중."""

        def predict(matrix):
            rows = np.atleast_2d(np.asarray(matrix, dtype=float))
            observations = np.vstack([self.rebuild(base, asset, r) for r in rows])
            return self.target_weights(observations)[:, asset]

        return predict


def collect_observations(data, config, start: int, end: int, models, scaler=None):
    """테스트 구간을 다시 굴려 '에이전트가 실제로 본' 관측을 그대로 수집한다.

    Returns:
        (dates, observations, target_weights) — 길이가 같은 세 배열.
    """
    env = PortfolioEnv(data, config, start=start, end=end, scaler=scaler)
    observation, _ = env.reset(seed=config["seed"])
    dates, observations, weights = [], [], []
    explainer_env = PolicyExplainer(data, config, models, scaler)
    while True:
        dates.append(str(data.dates[env.t])[:10])
        observations.append(np.asarray(observation, dtype=np.float32))
        target = explainer_env.target_weights(observation)[0]
        weights.append(target)
        observation, _, terminated, truncated, _ = env.step_weights(target)
        if terminated or truncated:
            break
    return np.array(dates), np.vstack(observations), np.vstack(weights)


def explain_asset(
    explainer: PolicyExplainer,
    dates: np.ndarray,
    observations: np.ndarray,
    asset: int,
    sample: int = 60,
    background_rows: int = 1,
) -> dict:
    """한 자산에 대해 표본 날짜별 SHAP 값을 계산한다."""
    import shap

    chosen = np.arange(len(dates))[-sample:] if sample else np.arange(len(dates))
    reduced = np.vstack([explainer.reduce(observations[i], asset) for i in chosen])
    background = (
        reduced.mean(axis=0, keepdims=True)
        if background_rows <= 1
        else shap.kmeans(reduced, background_rows).data
    )
    records = []
    for row, index in zip(reduced, chosen):
        base = observations[index]
        kernel = shap.KernelExplainer(
            explainer.prediction_fn(base, asset), background, link="identity"
        )
        values = np.asarray(kernel.shap_values(row, silent=True), dtype=float).ravel()
        expected = float(np.ravel(kernel.expected_value)[0])
        records.append(
            {
                "date": str(dates[index]),
                "feature_values_raw": [float(v) for v in row],
                "shap_values": [float(v) for v in values],
                "base_value": expected,
                "final_value": float(expected + values.sum()),
            }
        )
    return {"records": records, "reduced": reduced}


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """프론트 fixture 와 같은 0-1 정규화 (색상 표현용, 해석에는 raw 사용)."""
    low, high = matrix.min(axis=0), matrix.max(axis=0)
    span = np.where(high - low < 1e-12, 1.0, high - low)
    return np.clip((matrix - low) / span, 0.0, 1.0)


def build_payload(
    data,
    config,
    models,
    scaler,
    start: int,
    end: int,
    assets=None,
    sample: int = 60,
    asset_names: dict | None = None,
) -> dict:
    """frontend/fixtures/shap.json 과 동일한 스키마의 결과를 만든다."""
    explainer = PolicyExplainer(data, config, models, scaler)
    dates, observations, _ = collect_observations(
        data, config, start, end, models, scaler
    )
    targets = list(assets or data.assets)
    payload = {"features": FEATURE_LABELS, "feature_keys": FEATURE_KEYS, "assets": {}}
    for ticker in targets:
        index = data.assets.index(ticker)
        result = explain_asset(explainer, dates, observations, index, sample=sample)
        normalized = normalize_columns(
            np.array([r["feature_values_raw"] for r in result["records"]])
        )
        decisions = []
        for record, values in zip(result["records"], normalized):
            decisions.append(
                {
                    **record,
                    "feature_values": [round(float(v), 4) for v in values],
                }
            )
        payload["assets"][ticker] = {
            "name": (asset_names or {}).get(ticker, ticker),
            "observations": [
                {
                    "date": d["date"],
                    "feature_values": d["feature_values"],
                    "shap_values": d["shap_values"],
                }
                for d in decisions
            ],
            "decisions": decisions,
        }
    return payload


def mean_absolute_importance(payload: dict) -> pd.DataFrame:
    """자산 × 피처 평균 |SHAP| — Summary Plot 과 risk tag 효과 확인에 쓴다."""
    rows = []
    for ticker, asset in payload["assets"].items():
        matrix = np.array([d["shap_values"] for d in asset["decisions"]], dtype=float)
        for name, value in zip(payload["features"], np.abs(matrix).mean(axis=0)):
            rows.append(
                {"ticker": ticker, "feature": name, "mean_abs_shap": float(value)}
            )
    frame = pd.DataFrame(rows)
    frame["rank"] = frame.groupby("ticker").mean_abs_shap.rank(ascending=False)
    return frame.sort_values(["ticker", "rank"])


def save_summary_plot(payload: dict, ticker: str, path: Path) -> Path:
    """SHAP Summary Plot (과제 15번)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    asset = payload["assets"][ticker]
    values = np.array([d["shap_values"] for d in asset["decisions"]], dtype=float)
    features = np.array([d["feature_values"] for d in asset["decisions"]], dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        values,
        features=features,
        feature_names=payload["features"],
        show=False,
        plot_size=(8, 4.5),
    )
    plt.title(f"SHAP Summary · {ticker} (출력: 목표 비중)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    return path


def save_force_plot(payload: dict, ticker: str, date: str | None, path: Path) -> Path:
    """SHAP Force Plot (과제 16번). date=None 이면 마지막 결정일."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    asset = payload["assets"][ticker]
    decision = (
        next(d for d in asset["decisions"] if d["date"] == date)
        if date
        else asset["decisions"][-1]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.force_plot(
        decision["base_value"],
        np.array(decision["shap_values"], dtype=float),
        np.array(decision["feature_values_raw"], dtype=float),
        feature_names=payload["features"],
        matplotlib=True,
        show=False,
    )
    plt.title(f"{ticker} · {decision['date']} 결정 (기준 {decision['base_value']:.3f})")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def notable_decisions(payload: dict, ticker: str, count: int = 2) -> list[str]:
    """해석할 가치가 있는 결정일을 고른다 (과제 16번).

    마지막 날을 기계적으로 쓰는 대신, 기준값에서 가장 크게 벗어난 결정을 고른다.
    "왜 이 날 비중을 크게 바꿨는가"가 설명할 거리가 있는 날이기 때문이다.
    """
    decisions = payload["assets"][ticker]["decisions"]
    ranked = sorted(
        decisions,
        key=lambda item: abs(item["final_value"] - item["base_value"]),
        reverse=True,
    )
    return [item["date"] for item in ranked[:count]]


def save_payload(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
