"""End-to-end performance analysis pipeline (성능 분석 담당 최종 산출물 1~18).

실행 예
    # 이미 학습된 run 을 재사용해 전 분석을 수행 (가장 흔한 경우)
    python -m backend.performance.run_analysis --run runs/20261031_120000

    # Walk-forward 재학습부터 전부 수행 (오래 걸림: window x reward x seed 회 학습)
    python -m backend.performance.run_analysis --walk-forward --folds 3

    # 모델 없이 구조만 점검 (더미 데이터 + 무작위 정책)
    python -m backend.performance.run_analysis --smoke

단계
    데이터 확인 -> 전략 백테스트(PPO/MVO/동일가중/벤치마크) -> 12개 지표
    -> 그래프 -> 시장 국면 -> Reward 3종 -> SHAP -> 뉴스 Risk 효과
    -> ANOVA 3종 -> Tukey HSD -> 결과 저장 -> 웹 담당 전달용 JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.investment_ai.data_loader import load_data
from backend.investment_ai.settings import ROOT, read_config

from . import anova, market_regime, plots, report, risk_tag, shap_analysis, strategies
from .metrics import metrics_table
from .walk_forward import (
    DEFAULT_FOLDS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    aggregate,
    last_window_models,
    plan_windows,
    run_walk_forward,
)

#: Safe-Guard 가 적용된 벤치마크 (전략과 같은 조건의 대조군).
BENCHMARK = "benchmark"
#: Alpha/Beta·시장 국면 분류의 기준이 되는 순수 시장 수익률.
MARKET = strategies.MARKET_REFERENCE
#: 매매 규칙을 적용한 투자 가능한 SPY Buy & Hold.
INVESTABLE = strategies.INVESTABLE_BENCHMARK


def market_column(returns: pd.DataFrame) -> str:
    """시장 기준 열 이름. 지수 -> 투자 가능 Buy&Hold -> 안전장치 벤치마크 순."""
    for name in (MARKET, INVESTABLE, BENCHMARK):
        if name in returns.columns:
            return name
    raise ValueError("시장 기준 열(benchmark_raw/benchmark)이 없습니다")


# --------------------------------------------------------------------------- #
# 모델 로딩
# --------------------------------------------------------------------------- #
def run_config(run: Path) -> dict:
    """학습 run 에 저장된 설정을 그대로 읽는다 (schema.json 의 macro 순서 포함)."""
    run = Path(run)
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    seed_dir = run / "final" / f"seed_{selection['seeds'][0]}"
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    config["macro_columns"] = json.loads(
        (seed_dir / "schema.json").read_text(encoding="utf-8")
    )["macros"]
    return config


def load_run_models(run: Path, data):
    """runs/<id>/final/seed_* 에 저장된 최종 모델들을 불러온다."""
    from backend.investment_ai.training import load_bundle

    run = Path(run)
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    bundles = [
        load_bundle(run / "final" / f"seed_{seed}", data) for seed in selection["seeds"]
    ]
    config = bundles[0][2]
    splits = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    return (
        [bundle[0] for bundle in bundles],
        bundles[0][1],
        config,
        int(splits["test_start_index"]),
    )


class RandomPolicy:
    """스모크 테스트용 더미 정책 (torch/stable-baselines3 없이 배선만 확인)."""

    def __init__(self, n_assets: int, seed: int = 0):
        self.n_assets, self.rng = n_assets, np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):
        batch = np.atleast_2d(np.asarray(observation, dtype=float))
        # 관측에 결정적으로 의존해야 SHAP 이 의미 있는 값을 낸다.
        base = np.tanh(batch[:, : self.n_assets] * 2.0)
        actions = np.clip(base, -1, 1)
        single = np.ndim(observation) == 1
        return (actions[0] if single else actions), None


# --------------------------------------------------------------------------- #
# 단계별 수행
# --------------------------------------------------------------------------- #
def strategy_comparison(data, config, start, end, models, scaler, mvo_options):
    """과제 3~11번: 4개 전략을 같은 구간에서 백테스트 + 12개 지표."""
    histories = strategies.run_all(
        data, config, start, end, models, scaler, mvo_options=mvo_options
    )
    return (
        *(histories,),
        *metrics_from(histories, config.get("risk_free_annual", 0.0), data),
    )


def metrics_from(histories: dict, annual_rf: float, data=None):
    """히스토리 묶음 -> (수익률 표, 12지표 표, 회전율 요약).

    data 를 주면 거래비용·체결지연이 없는 **지수 수익률**(market_index) 열을
    덧붙이고, 그것을 Alpha/Beta/Information Ratio 의 기준으로 쓴다. Safe-Guard 가
    걸린 벤치마크는 낙폭 이후 현금이 되어 시장을 대표하지 못하고, benchmark_raw
    조차 첫 진입 수수료와 체결 지연의 영향을 받기 때문이다.
    """
    returns = strategies.returns_frame(histories)
    if data is not None and "SPY" in data.assets:
        returns[MARKET] = strategies.market_index_returns(data, returns.index)
    metrics = metrics_table(
        {name: returns[name] for name in returns.columns},
        benchmark_key=market_column(returns),
        annual_rf=annual_rf,
    )
    turnover = {
        name: strategies.turnover_summary(history)
        for name, history in histories.items()
    }
    return returns, metrics, turnover


def stitched_comparison(walk_forward: dict, reward: str, annual_rf: float, data=None):
    """Walk-forward 전 구간을 이어붙인 OOS 비교 (과제 1·2번의 올바른 결과).

    마지막 window 의 모델을 전 기간에 적용하면 그 모델이 학습에 쓴 구간까지
    평가에 들어간다. 그래서 각 window 는 자기 테스트 구간만 담당하고, 그
    구간들을 시간순으로 이어붙인 것이 walk-forward 백테스트 결과다.
    """
    ppo = walk_forward["stitched"]["ppo"]
    if reward not in ppo:
        raise ValueError(
            f"reward '{reward}' 의 walk-forward 결과가 없습니다. "
            f"사용 가능: {sorted(ppo)}"
        )
    histories = {"ppo": ppo[reward], **walk_forward["stitched"]["baseline"]}
    ordered = {
        name: histories[name]
        for name in ("ppo", "mvo", "equal_weight", BENCHMARK, INVESTABLE)
        if name in histories
    }
    return (ordered, *metrics_from(ordered, annual_rf, data))


def _period_rows(series: pd.Series, freq: str, reward: str, group: str):
    """일별 수익률 -> 겹치지 않는 기간 수익률 행들 (ANOVA 한 그룹의 표본)."""
    period = (1 + series.sort_index()).resample(freq).prod() - 1
    return pd.DataFrame(
        {
            "period_return": period.to_numpy(dtype=float),
            "reward": reward,
            "group": group,
            "period_end": period.index,
        }
    )


def reward_samples_from_walk_forward(walk_forward: dict, freq: str) -> pd.DataFrame:
    """Walk-forward 의 reward별 연속 OOS 경로에서 ANOVA 표본을 만든다."""
    return pd.concat(
        [
            _period_rows(
                pd.Series(
                    frame.portfolio_return.to_numpy(dtype=float),
                    index=pd.to_datetime(frame.date),
                ),
                freq,
                reward,
                "walk_forward",
            )
            for reward, frame in walk_forward["stitched"]["ppo"].items()
        ],
        ignore_index=True,
    ).dropna(subset=["period_return"])


def reward_samples_from_run(run: Path, freq: str) -> pd.DataFrame | None:
    """투자 AI 가 남긴 검증 구간 일별 기록에서 ANOVA 표본을 만든다.

    `runs/<id>/validation_models/<reward>/fold_<f>_seed_<s>/validation_history.csv`
    를 읽는다 (investment_ai.experiment 가 저장하는 경로). walk-forward 를 새로
    돌리지 않아도 Reward 3종 ANOVA 를 수행할 수 있게 하는 경로다.

    주의: 검증 fold 는 달력상 겹칠 수 있어(splits.walk_forward_splits 의 간격이
    validation_days 보다 좁을 수 있음) 표본이 완전히 독립이지는 않다. fold·seed
    단위로 따로 집계해 블록으로 다루고, 그 사실을 결과에 함께 기록한다.
    """
    root = Path(run) / "validation_models"
    if not root.is_dir():
        return None
    frames = []
    for path in sorted(root.glob("*/fold_*_seed_*/validation_history.csv")):
        history = pd.read_csv(path)
        if "portfolio_return" not in history.columns or history.empty:
            continue
        series = pd.Series(
            history.portfolio_return.to_numpy(dtype=float),
            index=pd.to_datetime(history.date),
        )
        frames.append(
            _period_rows(
                series,
                freq,
                path.parent.parent.name,
                f"{path.parent.parent.name}/{path.parent.name}",
            )
        )
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).dropna(subset=["period_return"])


def reward_samples_from_validation_csv(run: Path) -> pd.DataFrame | None:
    """`validation.csv` 의 fold×seed 성과를 Reward ANOVA 표본으로 쓴다 (폴백).

    공유 번들처럼 `validation_models/` 일별 기록이 빠져 있고 요약 CSV 만 있는
    경우를 위한 경로다. 관측 단위가 "fold×seed 의 구간 총수익률"이라 표본 수가
    작지만(reward 당 fold수×seed수), 각 행이 독립적인 학습 결과라 가정이 깨끗하다.
    """
    path = Path(run) / "validation.csv"
    if not path.exists():
        return None
    validation = pd.read_csv(path)
    needed = {"reward", "fold", "seed", "total_return"}
    if not needed.issubset(validation.columns) or validation.empty:
        return None
    return pd.DataFrame(
        {
            "period_return": validation.total_return.astype(float),
            "reward": validation.reward.astype(str),
            "group": validation.fold.astype(str)
            + "_seed"
            + validation.seed.astype(str),
        }
    ).dropna(subset=["period_return"])


def reward_tables(walk_forward: dict | None, run: Path | None, freq: str):
    """과제 13번: Reward 3종 비교표와 ANOVA 표본.

    Returns:
        (요약표, 표본 long DataFrame, 표본 출처 문구)
    """
    if walk_forward is not None:
        metrics = walk_forward["metrics"]
        table = aggregate(metrics[metrics.strategy == "ppo"], by=("reward",))
        samples = reward_samples_from_walk_forward(walk_forward, freq)
        return table, samples, "walk-forward 테스트 구간 (구간별 재학습)"
    if run is None:
        return None, None, "미수행"
    table = None
    if (Path(run) / "validation.csv").exists():
        validation = pd.read_csv(Path(run) / "validation.csv")
        columns = [
            name
            for name in ("total_return", "sharpe", "max_drawdown")
            if name in validation.columns
        ]
        if columns and "reward" in validation.columns:
            table = validation.groupby("reward")[columns].agg(["mean", "std", "count"])
            table.columns = [f"{a}_{b}" for a, b in table.columns]
            table = table.reset_index()
    samples = reward_samples_from_run(Path(run), freq)
    source = "투자 AI 검증 fold 일별 기록 (fold·seed 블록 단위, 달력상 일부 중첩 가능)"
    if samples is None:
        # 일별 기록이 없으면 validation.csv 의 fold×seed 요약으로 대체한다.
        samples = reward_samples_from_validation_csv(Path(run))
        source = (
            "validation.csv 의 fold×seed 구간 총수익률 (일별 기록이 없어 대체)"
            if samples is not None
            else "미수행 — validation_models 일별 기록도 validation.csv 도 없습니다"
        )
    return table, samples, source


def period_samples(returns: pd.DataFrame, regimes: pd.Series | None, freq: str):
    """일별 수익률 -> 겹치지 않는 기간 수익률 long 포맷 (ANOVA 표본)."""
    periods = anova.period_returns(returns, freq=freq)
    if regimes is not None:
        labels = (
            regimes.rename("regime")
            .to_frame()
            .resample(freq)
            .agg(lambda block: block.mode().iloc[0] if len(block) else np.nan)
        )
        periods = periods.join(labels)
    return anova.long_format(periods, value="period_return")


def run_statistics(
    returns: pd.DataFrame,
    regimes: pd.Series | None,
    reward_samples: pd.DataFrame | None,
    freq: str,
    reward_source: str = "",
) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """과제 18~23번: One-way 2개, Two-way 1개, Effect size, Tukey HSD."""
    results = []

    # 검증 1 — Reward 3종
    if reward_samples is not None and reward_samples.reward.nunique() >= 2:
        result = anova.run_with_posthoc(
            reward_samples,
            "period_return",
            "reward",
            label="검증 1: Reward 3종 성과 차이",
        )
        result["sample_source"] = reward_source
        result["interpretation"] += f" 표본 출처: {reward_source}."
        results.append(result)
    else:
        print(
            "[경고] Reward 3종 ANOVA 를 건너뜁니다 — "
            f"{reward_source or '표본 없음'}. "
            "--walk-forward 로 돌리거나 runs/<id>/validation_models 를 확인하세요.",
            flush=True,
        )

    # 검증 2 — 전략 3종 (벤치마크는 시장 기준이므로 제외)
    long = period_samples(returns, regimes, freq)
    strategy_frame = long[long.strategy.isin(strategies.COMPARED_STRATEGIES)].copy()
    results.append(
        anova.run_with_posthoc(
            strategy_frame,
            "period_return",
            "strategy",
            label="검증 2: PPO vs MVO vs 동일가중",
        )
    )

    # 검증 3 — 전략 × 시장 국면 (Two-way)
    if regimes is not None and strategy_frame.regime.nunique() >= 2:
        two_way = anova.two_way(
            strategy_frame.dropna(subset=["regime"]),
            "period_return",
            "strategy",
            "regime",
            label="검증 3: 전략 × 시장 국면",
        )
        two_way["interpretation"] = two_way_interpretation(two_way)
        results.append(two_way)

    return results, anova.summary_table(results), anova.tukey_table(results)


def run_paired(
    returns: pd.DataFrame, freq: str, market: str
) -> tuple[list, pd.DataFrame]:
    """같은 기간에 측정된 전략들의 대응 비교 (ANOVA 의 독립성 한계 보완).

    PPO·MVO·동일가중은 같은 날 같은 시장을 겪으므로 서로 독립이 아니다.
    차이를 직접 검정하면 그 상관을 상쇄할 수 있고, 블록 부트스트랩으로
    자기상관까지 반영한 신뢰구간을 얻는다.
    """
    periods = anova.period_returns(returns, freq=freq)
    pairs = [("ppo", "mvo"), ("ppo", "equal_weight"), ("mvo", "equal_weight")]
    if market in periods.columns:
        pairs.append(("ppo", market))
    results = [
        anova.paired_difference(periods, left, right)
        for left, right in pairs
        if left in periods.columns and right in periods.columns
    ]
    # 한 자료에서 여러 쌍을 비교하므로 Holm 보정을 적용한다.
    anova.apply_holm(results)
    for result in results:
        result["interpretation"] = anova.interpret_paired(result)
    return results, anova.paired_table(results)


def two_way_interpretation(result: dict) -> str:
    pieces = []
    for term, values in result["terms"].items():
        if values["p_value"] is None:
            pieces.append(f"{term}: 검정 불가")
            continue
        verdict = "유의" if values["significant"] else "유의하지 않음"
        pieces.append(
            f"{term}: F={values['f_statistic']:.3f}, p={values['p_value']:.4f} "
            f"({verdict}), η²={values['eta_squared']:.3f} "
            f"[{values['effect_size_label']}]"
        )
    note = (
        ""
        if result["interaction_included"]
        else " 셀당 표본이 2개 미만이라 상호작용 항은 제외했습니다."
    )
    posthoc = result.get("posthoc") or {}
    for factor, block in posthoc.items():
        significant = [
            f"{p['group1']} vs {p['group2']}"
            for p in block["pairs"]
            if p["significant"]
        ]
        note += (
            f" {factor} 사후검정(Tukey HSD) 유의한 쌍: {', '.join(significant)}."
            if significant
            else f" {factor} 는 유의했지만 Tukey HSD 에서 유의한 쌍이 없었습니다."
        )
    if posthoc:
        note += (
            " 다만 시계열 표본은 같은 시장 구간을 공유하므로 독립이 아니며,"
            " 사후검정 결과는 참고치입니다."
        )
    return f"{result['label']} — " + "; ".join(pieces) + note


def run_shap(
    data, config, models, scaler, start, end, paths, assets, sample, force_dates=None
):
    """과제 14~16번: Summary Plot, Force Plot, shap.json.

    Force Plot 은 마지막 날을 기계적으로 쓰지 않고, 기준값에서 가장 크게 벗어난
    결정일(설명할 거리가 있는 날)을 2개 고른다. --shap-force-dates 로 직접
    지정할 수도 있다.
    """
    payload = shap_analysis.build_payload(
        data, config, models, scaler, start, end, assets=assets, sample=sample
    )
    importance = shap_analysis.mean_absolute_importance(payload)
    first = next(iter(payload["assets"]))
    shap_analysis.save_summary_plot(payload, first, Path(paths["shap"]) / "summary.png")
    dates = list(force_dates or shap_analysis.notable_decisions(payload, first, 2))
    saved = []
    for order, date in enumerate(dates):
        name = "force.png" if order == 0 else f"force_{order + 1}.png"
        shap_analysis.save_force_plot(payload, first, date, Path(paths["shap"]) / name)
        saved.append({"ticker": first, "date": date, "file": name})
    payload["force_plots"] = saved
    return payload, importance


# --------------------------------------------------------------------------- #
# 엔트리 포인트
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="로보어드바이저 성능 분석 파이프라인")
    parser.add_argument("--config", type=Path, help="설정 JSON (기본: config.json)")
    parser.add_argument("--run", type=Path, help="학습 완료된 runs/<id> 경로")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--walk-forward", action="store_true", help="window 별 재학습")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--train-days",
        type=int,
        default=DEFAULT_TRAIN_DAYS,
        help=f"학습 거래일 (기본 {DEFAULT_TRAIN_DAYS} = 과제 예시의 4년)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help=f"테스트 거래일 (기본 {DEFAULT_TEST_DAYS} = 1년)",
    )
    parser.add_argument(
        "--window-scheme",
        choices=("rolling", "expanding"),
        default="rolling",
        help=(
            "walk-forward 윈도우 방식. 기본 rolling 은 과제 명세와 같이 학습 "
            "시작을 매 window 마다 이동시킨다(2019~2022 -> 2020~2023). "
            "expanding 은 학습 시작을 고정해 학습 구간이 누적된다(명세와 다름)."
        ),
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="(구버전 호환) --window-scheme rolling 과 같다",
    )
    parser.add_argument(
        "--auto-windows",
        action="store_true",
        help="데이터가 짧으면 window 수·학습 기간을 자동으로 줄이고 경고를 남긴다",
    )
    parser.add_argument(
        "--main-reward",
        help=(
            "최종 비교에 쓸 reward (기본: config/selection.json 의 선택값). "
            "테스트 구간 성과로 고르면 선택 편향이 생기므로 직접 지정하지 않는 편이 좋다"
        ),
    )
    parser.add_argument(
        "--mvo-objective", choices=["max_sharpe", "min_variance"], default="max_sharpe"
    )
    parser.add_argument(
        "--mvo-frequency",
        choices=[
            "monthly",
            "quarterly",
            "month_end",
            "quarter_end",
            "month_start",
            "quarter_start",
        ],
        default="monthly",
        help=(
            "monthly/quarterly 는 21·63 거래일 근사, "
            "month_end/quarter_end 는 달력 말일, "
            "month_start/quarter_start 는 달력 첫 거래일 기준"
        ),
    )
    parser.add_argument(
        "--mvo-strict-window",
        action="store_true",
        help=(
            "과거 252 거래일이 모이기 전에는 MVO 대신 동일가중을 유지한다 "
            "(과제가 252일 공분산을 엄격히 요구할 때)"
        ),
    )
    parser.add_argument("--regime-window", type=int, default=63)
    parser.add_argument("--regime-threshold", type=float, default=0.05)
    parser.add_argument(
        "--anova-frequency",
        default="ME",
        help="ANOVA 표본 주기 (ME=월, W=주). 표본이 적으면 W 를 쓰세요",
    )
    parser.add_argument("--shap-assets", nargs="*", default=None)
    parser.add_argument("--shap-sample", type=int, default=40)
    parser.add_argument(
        "--shap-force-dates",
        nargs="*",
        default=None,
        help="Force Plot 을 그릴 결정일 (기본: 기준값에서 가장 크게 벗어난 2일)",
    )
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--risk-tag", choices=["ablation", "retrain", "none"], default="ablation"
    )
    parser.add_argument("--smoke", action="store_true", help="더미 정책으로 배선 점검")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    config = read_config(args.config) if args.config else None
    models = scaler = None
    test_start = None

    if args.run and not args.smoke:
        config = run_config(Path(args.run))
        data, _ = load_data(config, ROOT)
        models, scaler, config, test_start = load_run_models(Path(args.run), data)
    else:
        config = config or read_config()
        data, _ = load_data(config, ROOT)
        if args.smoke:
            models = [RandomPolicy(len(data.assets), seed) for seed in config["seeds"]]

    paths = report.prepare(args.output)
    mvo_options = {
        "objective": args.mvo_objective,
        "frequency": args.mvo_frequency,
        "strict_window": args.mvo_strict_window,
    }

    # 1) Walk-forward (과제 1~2번)
    walk_forward = last_window = None
    main_reward = args.main_reward or config["reward"]
    if args.walk_forward:
        windows, plan = plan_windows(
            len(data.dates),
            config["lookback"],
            train_days=args.train_days,
            test_days=args.test_days,
            folds=args.folds,
            expanding=(args.window_scheme == "expanding" and not args.rolling),
            allow_shrink=args.auto_windows,
        )
        print(
            f"[walk-forward] 방식={plan['scheme']} "
            f"학습 {plan['used']['train_days']}일 / 테스트 "
            f"{plan['used']['test_days']}일 x {plan['used']['folds']} window",
            flush=True,
        )
        if plan["scheme_note"]:
            print(f"[경고] {plan['scheme_note']}", flush=True)
        for warning in plan["warnings"]:
            print(f"[경고] {warning}", flush=True)
        walk_forward = run_walk_forward(
            data,
            config,
            Path(args.output) / "walk_forward",
            windows=windows,
            seeds=config["seeds"],
            train_fn=(
                (lambda d, c, t, s, o: (RandomPolicy(len(d.assets), s), None))
                if args.smoke
                else None
            ),
            mvo_options=mvo_options,
            plan=plan,
        )
        # 각 window 가 학습한 모델을 그대로 이어받는다. SHAP·risk-tag 는 그 모델이
        # 실제로 본 테스트 구간(마지막 window)에서만 계산해야 한다.
        models, scaler, last_window = last_window_models(walk_forward, main_reward)
        test_start, end = last_window["test_start"], last_window["test_end"]
        start = test_start - 1
    else:
        if test_start is None:
            test_start = int(len(data.dates) * (1 - config["test_fraction"]))
        start, end = test_start - 1, len(data.dates) - 1

    if models is None:
        raise SystemExit(
            "학습된 모델이 없습니다. --run 으로 runs/<id> 를 지정하거나 "
            "--walk-forward 로 재학습하거나 --smoke 로 배선만 점검하세요."
        )

    # 2) 전략 비교와 12개 지표 (과제 3~9번)
    # Walk-forward 를 돌렸다면 구간별 테스트 수익률을 이어붙인 연속 OOS 경로가
    # 최종 비교 대상이다. 그러지 않으면 마지막 window 모델이 자기 학습 구간까지
    # 평가받게 된다.
    if walk_forward is not None:
        histories, returns, metrics, turnover = stitched_comparison(
            walk_forward, main_reward, config.get("risk_free_annual", 0.0), data
        )
    else:
        histories, returns, metrics, turnover = strategy_comparison(
            data, config, start, end, models, scaler, mvo_options
        )

    # 3) 시장 국면 (과제 12번)
    # 순수 시장 수익률로 분류한다. Safe-Guard 가 걸린 벤치마크를 쓰면 하락 후
    # 현금(수익률 0) 구간이 "횡보장"으로 잘못 분류된다.
    market = market_column(returns)
    regimes = market_regime.classify(
        returns[market].to_numpy(dtype=float),
        window=args.regime_window,
        threshold=args.regime_threshold,
        index=returns.index,
    )
    regime_table = market_regime.regime_summary(
        returns.assign(regime=regimes), config.get("risk_free_annual", 0.0), market
    )
    segments = market_regime.regime_segments(regimes)

    # 목표 판정 (Sharpe +0.2, 누적수익률 +10%p) — 저장보다 먼저 계산해야 한다.
    targets = report.target_check(metrics, "ppo")

    # 4) 그래프 (과제 10·11·12번)
    # 첫 장 그림은 전략 3종 + 시장 기준만 (Safe-Guard 벤치마크는 혼동을 준다).
    headline = plots.headline_columns(returns)
    plots.cumulative_plot(
        returns[headline], Path(paths["backtest"]) / "cumulative_return.png", segments
    )
    # 상세 위험분석용: 다섯 곡선 전부.
    plots.cumulative_plot(
        returns,
        Path(paths["backtest"]) / "cumulative_return_all.png",
        segments,
        title="누적 수익률 비교 (벤치마크 2종 포함)",
    )
    plots.drawdown_plot(returns, Path(paths["backtest"]) / "drawdown.png")
    plots.regime_bar_plot(regime_table, Path(paths["backtest"]) / "market_regime.png")

    # 5) Reward 3종 (과제 13번)
    reward_table, reward_samples, reward_source = reward_tables(
        walk_forward, args.run, args.anova_frequency
    )
    if walk_forward is not None:
        plots.reward_comparison_plot(
            walk_forward["metrics"], Path(paths["backtest"]) / "reward_comparison.png"
        )
        plots.walk_forward_plot(
            walk_forward["metrics"], Path(paths["backtest"]) / "walk_forward.png"
        )

    # SHAP·risk-tag 는 start~end (모델이 실제로 본 테스트 구간)에서만 돌린다.
    # 비교 기준이 되는 벤치마크 수익률도 같은 구간으로 잘라야 길이가 맞는다.
    model_config = {**config, "reward": main_reward}
    if last_window is not None:
        inside = (returns.index >= pd.Timestamp(last_window["test_start_date"])) & (
            returns.index <= pd.Timestamp(last_window["test_end_date"])
        )
        window_benchmark = returns.loc[inside, market].to_numpy(dtype=float)
    else:
        window_benchmark = returns[market].to_numpy(dtype=float)

    # 6) SHAP (과제 14~16번)
    shap_payload = shap_importance = None
    if not args.skip_shap:
        shap_payload, shap_importance = run_shap(
            data,
            model_config,
            models,
            scaler,
            start,
            end,
            paths,
            args.shap_assets,
            args.shap_sample,
            args.shap_force_dates,
        )

    # 7) 뉴스 Risk Tag 효과 (과제 17번)
    risk_result = None
    if args.risk_tag == "ablation":
        risk_result = risk_tag.ablation(
            data,
            model_config,
            start,
            end,
            models,
            scaler,
            window_benchmark,
        )
    elif args.risk_tag == "retrain":
        risk_result = risk_tag.retrain(
            data,
            model_config,
            start - 1,
            start,
            end,
            config["seeds"],
            Path(args.output) / "risk_tag_models",
            benchmark_returns=window_benchmark,
        )

    # 8) 통계 검증 (과제 18~23번)
    anova_results, anova_table, tukey = run_statistics(
        returns, regimes, reward_samples, args.anova_frequency, reward_source
    )
    paired_results, paired_table = run_paired(returns, args.anova_frequency, market)

    # 9) 저장 + 웹 담당 전달 (과제 24·25번)
    saved = report.save_all(
        args.output,
        metrics=metrics,
        histories=histories,
        returns=returns,
        walk_forward=walk_forward["metrics"] if walk_forward else None,
        regime_table=regime_table,
        regimes=regimes,
        reward_table=reward_table,
        risk_tag_table=risk_result["table"] if risk_result else None,
        anova_results=anova_results,
        anova_table=anova_table,
        tukey=tukey,
        shap_payload=shap_payload,
        shap_importance=shap_importance,
        turnover=turnover,
        meta={
            "run": str(args.run) if args.run else None,
            "test_start_date": str(data.dates[test_start])[:10],
            "assets": data.assets,
            "smoke": bool(args.smoke),
            "evaluation": (
                "walk_forward_stitched_out_of_sample"
                if walk_forward is not None
                else "single_holdout"
            ),
            "main_reward": main_reward,
            "market_reference": market,
            "window_plan": walk_forward["plan"] if walk_forward else None,
        },
    )
    paired_table.to_csv(Path(paths["statistics"]) / "paired.csv", index=False)
    report.write_json(Path(paths["statistics"]) / "paired.json", paired_results)
    if walk_forward is not None:
        walk_forward["ranking"].to_csv(
            Path(paths["metrics"]) / "reward_ranking.csv", index=False
        )
        report.write_json(
            Path(paths["metrics"]) / "walk_forward_plan.json", walk_forward["plan"]
        )
    if risk_result:
        report.write_json(
            Path(paths["statistics"]) / "risk_tag_effect.json",
            {
                "method": risk_result["method"],
                "difference": risk_result["difference"],
                "weight_distance": risk_result.get("weight_distance"),
                "interpretation": risk_tag.interpret(risk_result),
                "shap_risk_feature": (
                    risk_tag.shap_risk_importance(shap_importance)
                    if shap_importance is not None
                    else None
                ),
            },
        )
    report.write_json(
        Path(paths["root"]) / "summary.json",
        report.summary_payload(
            {
                "generated_for": "성능 분석 담당",
                "paths": saved,
                "strategies": list(returns.columns),
                "evaluation": (
                    "walk-forward 구간별 테스트를 이어붙인 연속 OOS 경로"
                    if walk_forward is not None
                    else "단일 홀드아웃 구간"
                ),
                "main_reward": main_reward,
                "window_plan": walk_forward["plan"] if walk_forward else None,
                "period": {
                    "start": str(returns.index[0])[:10],
                    "end": str(returns.index[-1])[:10],
                    "days": int(len(returns)),
                },
                "market_reference": market,
                "targets": targets,
                "reward_sample_source": reward_source,
                "regime_days": market_regime.regime_counts(regimes).to_dict("records"),
                "statistics": [
                    {
                        "label": result.get("label"),
                        "test": result["test"],
                        "interpretation": result.get("interpretation"),
                    }
                    for result in anova_results
                ],
                "paired_comparisons": [
                    result["interpretation"] for result in paired_results
                ],
                "risk_tag": (
                    risk_tag.interpret(risk_result) if risk_result else "미수행"
                ),
            }
        ),
    )
    report.write_json(Path(paths["metrics"]) / "targets.json", targets)
    print(f"\n완료: {Path(args.output).resolve()}")
    print(metrics.to_string())
    print("\n[목표 판정]")
    for key, row in targets["criteria"].items():
        state = "달성" if row["met"] else "미달" if row["met"] is False else "판정불가"
        difference = "-" if row["difference"] is None else f"{row['difference']:+.4f}"
        print(
            f"  {key}: vs {row['reference']} 차이 {difference} "
            f"(목표 +{row['threshold']}) -> {state}"
        )
    for result in anova_results:
        print("-", result.get("interpretation", result["test"]))
    for result in paired_results:
        print("-", result["interpretation"])
    return {"metrics": metrics, "paths": saved, "anova": anova_results}


if __name__ == "__main__":
    main()
