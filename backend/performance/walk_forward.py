"""Walk-forward validation with per-window retraining (과제 2번).

Window 를 앞으로 밀면서 매번 PPO 를 처음부터 다시 학습한다
(과제 요구: "Window 이동 시 모델을 완전히 재학습").
학습 구간이 끝난 바로 다음 구간만 테스트하므로 미래 정보가 섞이지 않는다.

기본 구간은 과제 예시와 같은 **4년 학습 / 1년 테스트**다.

    Window 1: 2019~2022 학습 -> 2023 테스트
    Window 2: 2020~2023 학습 -> 2024 테스트

거래일 기준으로 4년 = 1008일, 1년 = 252일이므로 3개 window 에는 최소
1008 + 252*3 = 1764 거래일(약 7년)이 필요하다. 데이터가 모자라면
`plan_windows(..., allow_shrink=True)` 가 실행 가능한 구간으로 줄이고
무엇을 줄였는지 `plan["warnings"]` 에 남긴다 (보고서에 반드시 적을 내용).

train_fn 을 주입할 수 있게 만든 이유
- 기본값은 investment_ai.training.train_model (PPO 실제 학습).
- 토치 없이 구조만 검증하거나(스모크 테스트), 이미 학습된 모델을 재사용할 때
  대체 함수를 넣을 수 있다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .metrics import METRIC_ORDER, performance_metrics
from .strategies import (
    INVESTABLE_BENCHMARK,
    MARKET_REFERENCE,
    market_index_returns,
    run_all,
    stitch_histories,
    turnover_summary,
)

TRADING_DAYS_PER_YEAR = 252
#: 과제 예시: 4년 학습 / 1년 테스트.
DEFAULT_TRAIN_DAYS = 4 * TRADING_DAYS_PER_YEAR
DEFAULT_TEST_DAYS = TRADING_DAYS_PER_YEAR
DEFAULT_FOLDS = 3


def required_days(train_days: int, test_days: int, folds: int) -> int:
    return train_days + test_days * folds


def expanding_windows(
    n_dates: int,
    lookback: int,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    folds: int = DEFAULT_FOLDS,
    expanding: bool = True,
) -> list[dict]:
    """학습/테스트 구간을 folds 개 만든다.

    expanding=True 면 학습 시작을 고정해 학습 구간이 길어지고(과제 예시의
    2019~2022 -> 2019~2023), False 면 길이를 유지한 채 밀어낸다(rolling).

    Returns:
        [{"window": 0, "train_start": i, "train_end": j,
          "test_start": j+1, "test_end": k}, ...]
    """
    if min(n_dates, lookback, train_days, test_days, folds) < 1:
        raise ValueError("Window parameters must be positive")
    if train_days < lookback + 1:
        raise ValueError("train_days must exceed the observation lookback")
    needed = required_days(train_days, test_days, folds)
    if needed > n_dates:
        raise ValueError(
            f"데이터가 부족합니다: 학습 {train_days}일 + 테스트 {test_days}일 x "
            f"{folds}개 window = 최소 {needed} 거래일"
            f"(약 {needed / TRADING_DAYS_PER_YEAR:.1f}년)이 필요하지만 "
            f"{n_dates}일(약 {n_dates / TRADING_DAYS_PER_YEAR:.1f}년)만 있습니다. "
            "--auto-windows 로 자동 축소하거나 --train-days/--test-days/--folds 를 "
            "직접 줄이세요."
        )
    windows = []
    for fold in range(folds):
        train_end = train_days - 1 + fold * test_days
        train_start = 0 if expanding else fold * test_days
        test_start = train_end + 1
        test_end = min(test_start + test_days - 1, n_dates - 1)
        if test_end <= test_start:
            break
        windows.append(
            {
                "window": fold,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return windows


def _largest_fitting(
    n_dates: int, requested: dict, minimum_train: int, min_test_days: int
) -> dict | None:
    """데이터에 들어가는 가장 큰 구간 조합을 찾는다.

    우선순위: window 개수 > 학습 기간 > 테스트 기간.
    walk-forward 의 핵심은 여러 구간에서 반복 검증하는 것이므로 window 수를
    가장 늦게 포기한다. 그다음은 학습 기간인데, PPO 는 학습 구간이 짧으면
    아예 학습이 되지 않아 비교 자체가 무의미해지기 때문이다. 테스트 기간은
    6개월(126거래일)까지 먼저 양보하고, 그래도 안 맞으면 더 줄인다.
    """
    preferred_test = min(requested["test_days"], 6 * 21)
    for folds in range(requested["folds"], 0, -1):
        for test_days in range(preferred_test, min_test_days - 1, -21):
            train_days = min(requested["train_days"], n_dates - test_days * folds)
            if train_days < minimum_train:
                continue
            # 학습 기간을 확보한 뒤, 남는 데이터로 테스트 기간을 다시 늘린다.
            grown = min(requested["test_days"], (n_dates - train_days) // folds)
            return {
                "train_days": int(train_days),
                "test_days": int(max(test_days, grown)),
                "folds": int(folds),
            }
    return None


def plan_windows(
    n_dates: int,
    lookback: int,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    folds: int = DEFAULT_FOLDS,
    expanding: bool = True,
    allow_shrink: bool = False,
    min_train_days: int | None = None,
    min_test_days: int = 21,
) -> tuple[list[dict], dict]:
    """요청한 구간이 데이터에 맞는지 확인하고, 필요하면 축소한다.

    축소 순서는 통계적 타당성이 덜 훼손되는 쪽부터다.
      1) window 개수를 줄인다 (최소 2개는 유지하려 한다).
      2) 그래도 안 되면 학습 기간을 줄인다 (min_train_days 까지).
      3) 마지막으로 테스트 기간을 줄인다 (min_test_days 까지).

    Returns:
        (windows, plan). plan 에는 요청값·실제값·경고 문구가 들어 있어
        보고서에 "4년 학습을 쓰지 못한 이유"를 그대로 옮길 수 있다.
    """
    minimum_train = min_train_days or max(lookback + 1, TRADING_DAYS_PER_YEAR)
    requested = {"train_days": train_days, "test_days": test_days, "folds": folds}
    warnings: list[str] = []
    used = dict(requested)

    if required_days(**used) > n_dates and allow_shrink:
        used = _largest_fitting(
            n_dates, requested, minimum_train, min_test_days
        ) or dict(requested)
        if used["folds"] != requested["folds"]:
            warnings.append(
                f"window 개수를 {requested['folds']}개에서 {used['folds']}개로 "
                "줄였습니다. window 가 1개면 walk-forward 가 아니라 단일 홀드아웃이므로 "
                "보고서에 그대로 적어야 합니다."
            )
        if used["test_days"] != requested["test_days"]:
            warnings.append(
                f"테스트 기간을 {requested['test_days']}일(1년)에서 "
                f"{used['test_days']}일"
                f"({used['test_days'] / TRADING_DAYS_PER_YEAR * 12:.0f}개월)로 "
                "줄였습니다. 구간별 표본이 작아 지표 변동이 커집니다."
            )
        if used["train_days"] != requested["train_days"]:
            warnings.append(
                f"학습 기간을 {requested['train_days']}일"
                f"({requested['train_days'] / TRADING_DAYS_PER_YEAR:.1f}년)에서 "
                f"{used['train_days']}일"
                f"({used['train_days'] / TRADING_DAYS_PER_YEAR:.1f}년)으로 "
                "줄였습니다. 과제 예시(4년 학습)보다 짧으므로 PPO 성능이 과소평가될 "
                "수 있습니다."
            )

    windows = expanding_windows(
        n_dates,
        lookback,
        train_days=used["train_days"],
        test_days=used["test_days"],
        folds=used["folds"],
        expanding=expanding,
    )
    plan = {
        "requested": requested,
        "used": used,
        "available_days": int(n_dates),
        "available_years": round(n_dates / TRADING_DAYS_PER_YEAR, 2),
        "required_days": required_days(**used),
        "scheme": "expanding" if expanding else "rolling",
        "adjusted": used != requested,
        "warnings": warnings,
    }
    return windows, plan


def label_windows(windows: list[dict], dates) -> list[dict]:
    """인덱스 구간에 실제 날짜를 붙여 보고서에 쓸 수 있게 한다."""
    return [
        {
            **window,
            "train_start_date": str(dates[window["train_start"]])[:10],
            "train_end_date": str(dates[window["train_end"]])[:10],
            "test_start_date": str(dates[window["test_start"]])[:10],
            "test_end_date": str(dates[window["test_end"]])[:10],
        }
        for window in windows
    ]


def slice_market_data(data, start: int, end: int):
    """[start, end] 구간만 남긴 MarketData 사본을 만든다.

    Rolling window 를 위해 필요하다. investment_ai.training.train_model 은
    `train_end` 만 받고 학습 시작점은 받지 않으므로(PortfolioEnv 의 start 가
    항상 lookback-1 로 고정된다), 데이터 자체를 잘라 넘기지 않으면 `--rolling`
    을 지정해도 모델은 여전히 가장 이른 날짜부터 학습하게 된다.
    """
    if not 0 <= start <= end < len(data.dates):
        raise ValueError(f"잘라낼 구간이 데이터 범위를 벗어납니다: [{start}, {end}]")
    keep = slice(start, end + 1)
    dates = data.dates[keep]
    metadata = data.news_metadata
    if metadata is not None:
        low, high = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
        metadata = metadata[
            (metadata["date"] >= low) & (metadata["date"] <= high)
        ].reset_index(drop=True)
    return replace(
        data,
        dates=dates,
        returns=data.returns[keep],
        rsi=data.rsi[keep],
        macd=data.macd[keep],
        macro=data.macro[keep],
        risk=data.risk[keep],
        news_metadata=metadata,
    )


def training_view(data, window: dict):
    """해당 window 의 학습용 데이터와 (그 데이터 기준의) 학습 종료 인덱스.

    항상 [train_start, train_end] 로 잘라서 넘긴다. rolling 이면 이전 구간을,
    expanding 이면 테스트 이후 구간을 물리적으로 제거하는 효과가 있어 학습 쪽에서
    미래 정보가 섞일 여지를 없앤다.
    """
    start, train_end = int(window["train_start"]), int(window["train_end"])
    return slice_market_data(data, start, train_end), train_end - start


def window_market_reference(data, baselines: dict):
    """구간별 Alpha/Beta 기준 수익률 — 전체 지표와 같은 정의(지수 수익률).

    Returns:
        (기준 이름, 수익률 배열). 지수를 만들 수 없으면 비용·체결지연이 포함된
        benchmark_raw, 그마저 없으면 Safe-Guard 벤치마크 순으로 내려간다.
        어떤 기준을 썼는지는 지표 표에 함께 기록된다.
    """
    if not baselines:
        return None, None
    dates = pd.to_datetime(next(iter(baselines.values())).date)
    try:
        series = market_index_returns(data, dates)
    except ValueError:
        for name in (INVESTABLE_BENCHMARK, "benchmark"):
            if name in baselines:
                return name, baselines[name].portfolio_return.to_numpy(float)
        return None, None
    return MARKET_REFERENCE, series.to_numpy(dtype=float)


def default_train_fn(data, config, train_end, seed, output):
    """투자 AI 담당의 PPO 학습을 그대로 사용한다 (torch 필요)."""
    from backend.investment_ai.training import train_model

    return train_model(data, config, train_end, seed, output)


def run_walk_forward(
    data,
    config: dict,
    output: Path,
    windows: list[dict] | None = None,
    rewards=("return", "sharpe", "return_mdd"),
    seeds=None,
    train_fn=None,
    strategies=("ppo", "mvo", "equal_weight", "benchmark", "benchmark_raw"),
    mvo_options: dict | None = None,
    plan: dict | None = None,
) -> dict:
    """모든 window 에서 reward 3종 PPO 를 재학습하고 전 전략을 백테스트한다.

    Returns:
        windows   날짜가 붙은 구간 목록
        metrics   (window x strategy x reward) 지표 표
        histories {(window, reward, strategy): DataFrame}
        models    {(window, reward): [model, ...]}  — 후속 SHAP/ablation 에 쓴다
        scalers   {(window, reward): FeatureScaler}
        stitched  {"ppo": {reward: DataFrame}, "baseline": {strategy: DataFrame}}
                  구간별 테스트 수익률을 시간순으로 이어붙인 연속 OOS 경로
        ranking   reward 별 평균 점수 (서술용. 선택 기준으로 쓰면 선택 편향)
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_fn = train_fn or default_train_fn
    seeds = list(seeds or config["seeds"])
    windows = windows or expanding_windows(len(data.dates), config["lookback"])
    windows = label_windows(windows, data.dates)
    annual_rf = config.get("risk_free_annual", 0.0)

    rows: list[dict] = []
    histories: dict = {}
    models: dict = {}
    scalers: dict = {}
    per_reward: dict = {reward: [] for reward in rewards}
    per_baseline: dict = {}

    for window in windows:
        start, end = window["test_start"] - 1, window["test_end"]
        # Rolling 이면 학습 데이터를 실제로 잘라 넘긴다 (train_start 반영).
        train_data, train_end = training_view(data, window)
        # 1) PPO: reward 종류별로 재학습 -> 시드 앙상블로 테스트 구간 백테스트
        ppo_histories = {}
        for reward in rewards:
            window_config = {
                **config,
                "reward": reward,
                "window_id": f"walkforward_{window['window']}_{reward}",
            }
            window_models, scaler = [], None
            for seed in seeds:
                directory = (
                    output
                    / "models"
                    / f"window_{window['window']}"
                    / reward
                    / f"seed_{seed}"
                )
                print(
                    f"[walk-forward] window={window['window']} reward={reward} "
                    f"seed={seed} train={window['train_start_date']}"
                    f"~{window['train_end_date']} "
                    f"({len(train_data.dates)}일) "
                    f"test={window['test_start_date']}~{window['test_end_date']}",
                    flush=True,
                )
                model, scaler = train_fn(
                    train_data, window_config, train_end, seed, directory
                )
                window_models.append(model)
            key = (window["window"], reward)
            models[key], scalers[key] = window_models, scaler
            ppo_histories[reward] = run_all(
                data,
                window_config,
                start,
                end,
                window_models,
                scaler,
                strategies=("ppo",),
            )["ppo"]

        # 2) 비학습 전략은 window 당 한 번만 (reward 와 무관)
        baseline_config = {
            **config,
            "window_id": f"walkforward_{window['window']}_baseline",
        }
        baselines = run_all(
            data,
            baseline_config,
            start,
            end,
            None,
            None,
            strategies=tuple(s for s in strategies if s != "ppo"),
            mvo_options=mvo_options,
        )
        # Alpha/Beta 기준은 전체 지표(metrics_from)와 **같은 정의**를 써야 한다.
        # 즉 거래비용·체결지연·Safe-Guard 가 없는 지수 수익률(market_index)이다.
        # 예전처럼 baselines 안에서 찾으면 market_index 는 전략이 아니라서 절대
        # 잡히지 않고 늘 안전장치 벤치마크로 떨어졌다.
        reference_name, benchmark_returns = window_market_reference(data, baselines)

        for reward, history in ppo_histories.items():
            histories[(window["window"], reward, "ppo")] = history
            per_reward[reward].append(history)
            rows.append(
                _metric_row(
                    window,
                    "ppo",
                    reward,
                    history,
                    benchmark_returns,
                    annual_rf,
                    reference_name,
                )
            )
        for name, history in baselines.items():
            histories[(window["window"], "-", name)] = history
            per_baseline.setdefault(name, []).append(history)
            reference = None if name == reference_name else benchmark_returns
            rows.append(
                _metric_row(
                    window, name, "-", history, reference, annual_rf, reference_name
                )
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "walk_forward_metrics.csv", index=False)
    pd.DataFrame(windows).to_csv(output / "walk_forward_windows.csv", index=False)
    for (window_id, reward, strategy), history in histories.items():
        directory = output / "histories" / f"window_{window_id}"
        directory.mkdir(parents=True, exist_ok=True)
        history.to_csv(directory / f"{strategy}_{reward}.csv", index=False)

    stitched = {
        "ppo": {
            reward: stitch_histories(frames)
            for reward, frames in per_reward.items()
            if frames
        },
        "baseline": {
            name: stitch_histories(frames) for name, frames in per_baseline.items()
        },
    }
    for reward, frame in stitched["ppo"].items():
        frame.to_csv(output / f"stitched_ppo_{reward}.csv", index=False)
    for name, frame in stitched["baseline"].items():
        frame.to_csv(output / f"stitched_{name}.csv", index=False)

    return {
        "windows": windows,
        "plan": plan,
        "metrics": metrics,
        "histories": histories,
        "models": models,
        "scalers": scalers,
        "stitched": stitched,
        "ranking": reward_ranking(metrics, config.get("selection_mdd_penalty", 1.0)),
    }


def reward_ranking(metrics: pd.DataFrame, mdd_penalty: float = 1.0) -> pd.DataFrame:
    """reward 별 평균 점수 (investment_ai.experiment 와 같은 기준).

    주의: 이 점수는 **테스트 구간**에서 계산된다. 이것으로 reward 를 고르면
    홀드아웃을 선택에 쓰는 것이므로 선택 편향이 생긴다. 보고서용 서술 지표로만
    쓰고, 실제 선택은 투자 AI 담당의 validation 결과(selection.json)를 따른다.
    """
    ppo = metrics[metrics.strategy == "ppo"].copy()
    if ppo.empty:
        return pd.DataFrame(columns=["reward", "score_mean", "score_std", "windows"])
    ppo["score"] = ppo.cumulative_return - mdd_penalty * ppo.max_drawdown
    ranking = (
        ppo.groupby("reward")
        .score.agg(score_mean="mean", score_std="std", windows="count")
        .reset_index()
        .sort_values("score_mean", ascending=False)
    )
    ranking["note"] = "테스트 구간 기준 — 선택에 사용하면 선택 편향"
    return ranking


def _metric_row(
    window, strategy, reward, history, benchmark_returns, annual_rf, reference=None
):
    returns = history.portfolio_return.to_numpy(dtype=float)
    values = performance_metrics(returns, benchmark_returns, annual_rf)
    return {
        "market_reference": reference,
        "window": window["window"],
        "train_start_date": window["train_start_date"],
        "train_end_date": window["train_end_date"],
        "test_start_date": window["test_start_date"],
        "test_end_date": window["test_end_date"],
        "strategy": strategy,
        "reward": reward,
        **{key: values[key] for key in ["days", *METRIC_ORDER]},
        **turnover_summary(history),
    }


def aggregate(metrics: pd.DataFrame, by=("strategy", "reward")) -> pd.DataFrame:
    """window 평균/표준편차 요약 — 보고서의 안정성 근거로 쓴다."""
    numeric = [c for c in METRIC_ORDER if c in metrics.columns]
    grouped = metrics.groupby(list(by))[numeric]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def last_window_models(result: dict, reward: str):
    """가장 최근 window 의 (models, scaler, window) — SHAP·ablation 대상.

    SHAP 과 risk-tag 분석은 "그 모델이 실제로 본 테스트 구간"에서만 돌려야 하므로
    해당 window 정보도 함께 돌려준다.
    """
    windows = result["windows"]
    if not windows:
        raise ValueError("walk-forward 결과에 window 가 없습니다")
    window = windows[-1]
    key = (window["window"], reward)
    if key not in result["models"]:
        available = sorted({r for _, r in result["models"]})
        raise ValueError(
            f"reward '{reward}' 의 학습 모델이 없습니다. 사용 가능: {available}"
        )
    return result["models"][key], result["scalers"][key], window
