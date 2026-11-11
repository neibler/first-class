"""Fast tests for the performance-analysis package (torch 없이 동작)."""

import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from backend.investment_ai.data_loader import prepare_data
from backend.investment_ai.synthetic import generate
from backend.performance import (
    anova,
    plots,
    market_regime,
    metrics,
    mvo,
    report,
    shap_analysis,
    strategies,
    walk_forward,
)

ASSETS = ["SPY", "QQQ", "GLD", "TLT"]


@pytest.fixture(scope="module")
def config():
    return {
        "assets": ASSETS,
        "lookback": 20,
        "seed": 7,
        "scenario": "mixed",
        "dummy_start": "2023-01-02",
        "dummy_end": "2024-06-28",
        "dummy_macro_columns": ["vix"],
        "macro_columns": ["vix"],
        "observation_lag": 0,
        "execution_delay": 1,
        "rebalance_days": 5,
        "logit_scale": 5.0,
        "reward": "return",
        "reward_window": 20,
        "mdd_lambda": 0.1,
        "risk_free_annual": 0.0,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "max_drawdown": 0.15,
        "min_episode_days": 60,
        "random_start": False,
    }


@pytest.fixture(scope="module")
def data(config):
    frame = generate(config)
    return prepare_data(frame, ASSETS, macro_columns=["vix"], news_mode="development")


class StubPolicy:
    """관측에 결정적으로 반응하는 더미 정책 (PPO 대역)."""

    def __init__(self, n_assets):
        self.n_assets = n_assets

    def predict(self, observation, deterministic=True):
        batch = np.atleast_2d(np.asarray(observation, dtype=float))
        actions = np.clip(np.tanh(batch[:, : self.n_assets] * 2.0), -1, 1)
        return (actions[0] if np.ndim(observation) == 1 else actions), None


# --------------------------------------------------------------------------- #
# 12개 성과지표
# --------------------------------------------------------------------------- #
def test_cumulative_return_and_cagr_match_hand_calculation():
    returns = np.full(252, 0.001)
    assert metrics.cumulative_return(returns) == pytest.approx(1.001**252 - 1)
    assert metrics.cagr(returns) == pytest.approx(1.001**252 - 1)


def test_constant_returns_have_no_volatility_and_undefined_sharpe():
    returns = np.full(50, 0.001)
    assert metrics.annualized_volatility(returns) == pytest.approx(0.0)
    assert metrics.sharpe(returns) is None
    assert metrics.sortino(returns) is None


def test_var_and_cvar_are_positive_loss_magnitudes():
    returns = np.linspace(-0.05, 0.05, 200)
    var = metrics.value_at_risk(returns)
    cvar = metrics.conditional_value_at_risk(returns)
    assert var > 0 and cvar > 0
    assert cvar >= var


def test_max_drawdown_is_positive_and_matches_peak_to_trough():
    returns = np.array([0.1, -0.5, 0.2])
    assert metrics.max_drawdown(returns) == pytest.approx(0.5)


def test_beta_is_one_and_alpha_zero_against_itself():
    rng = np.random.default_rng(0)
    series = rng.normal(0.0004, 0.01, 300)
    assert metrics.beta(series, series) == pytest.approx(1.0)
    assert metrics.alpha(series, series) == pytest.approx(0.0, abs=1e-12)
    assert metrics.information_ratio(series, series) is None


def test_performance_metrics_returns_all_twelve_keys():
    rng = np.random.default_rng(1)
    strategy = rng.normal(0.0005, 0.01, 300)
    benchmark = rng.normal(0.0003, 0.01, 300)
    values = metrics.performance_metrics(strategy, benchmark)
    assert set(metrics.METRIC_ORDER) <= set(values)
    assert all(values[key] is not None for key in metrics.METRIC_ORDER)


# --------------------------------------------------------------------------- #
# MVO
# --------------------------------------------------------------------------- #
def test_mvo_respects_long_only_sum_one_and_cap():
    rng = np.random.default_rng(2)
    window = rng.normal(0.0005, 0.01, (252, len(ASSETS)))
    weights = mvo.optimize_weights(window, "max_sharpe", max_weight=0.4)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= -1e-9
    assert weights.max() <= 0.4 + 1e-6


def test_min_variance_prefers_the_calm_asset():
    rng = np.random.default_rng(3)
    window = np.column_stack(
        [
            rng.normal(0.0005, 0.03, 252),
            rng.normal(0.0005, 0.03, 252),
            rng.normal(0.0005, 0.001, 252),
            rng.normal(0.0005, 0.03, 252),
        ]
    )
    weights = mvo.optimize_weights(window, "min_variance", max_weight=0.4)
    assert weights[2] == pytest.approx(0.4, abs=1e-3)


def test_mvo_falls_back_to_equal_weight_on_tiny_sample():
    weights = mvo.optimize_weights(np.zeros((1, 4)))
    assert weights == pytest.approx(np.full(4, 0.25))


def test_mvo_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        mvo.optimize_weights(np.zeros((10, 4)), max_weight=0.2)


# --------------------------------------------------------------------------- #
# 전략 백테스트
# --------------------------------------------------------------------------- #
def test_mvo_backtest_produces_contract_columns(data, config):
    history = strategies.run_mvo(data, config, 250, 300, frequency="monthly")
    assert len(history) == 300 - 250
    for column in ("portfolio_return", "portfolio_value", "drawdown", "cash_weight"):
        assert column in history.columns
    weights = history[[f"current_weight_{a.lower()}" for a in ASSETS]].sum(axis=1)
    assert ((weights <= 1 + 1e-9) & (weights >= -1e-9)).all()


def test_strategies_share_the_same_dates(data, config):
    histories = strategies.run_all(
        data, config, 250, 300, [StubPolicy(len(ASSETS))], None
    )
    returns = strategies.returns_frame(histories)
    assert list(returns.columns) == [
        "ppo",
        "mvo",
        "equal_weight",
        "benchmark",
        "benchmark_raw",
    ]
    assert len(returns) == 50
    assert returns.index.is_monotonic_increasing


def test_stitching_rejects_overlapping_windows(data, config):
    history = strategies.run_strategy(data, config, 250, 280, "equal_weight")
    with pytest.raises(ValueError, match="겹칩니다"):
        strategies.stitch_histories([history, history])


def test_returns_frame_rejects_misaligned_strategies(data, config):
    short = strategies.run_strategy(data, config, 250, 270, "equal_weight")
    long = strategies.run_strategy(data, config, 250, 300, "benchmark")
    with pytest.raises(ValueError, match="일치하지 않습니다"):
        strategies.returns_frame({"equal_weight": short, "benchmark": long})


def test_value_and_drawdown_frames_are_consistent(data, config):
    histories = strategies.run_all(
        data, config, 250, 300, [StubPolicy(len(ASSETS))], None
    )
    returns = strategies.returns_frame(histories)
    values = strategies.value_frame(returns)
    drawdowns = strategies.drawdown_frame(returns)
    assert (drawdowns >= -1e-12).all().all()
    assert values.iloc[-1]["mvo"] == pytest.approx(
        1 + metrics.cumulative_return(returns["mvo"])
    )


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def test_default_windows_are_four_years_train_one_year_test():
    """과제 예시(2019~2022 학습 -> 2023 테스트)와 같은 기본값이어야 한다."""
    assert walk_forward.DEFAULT_TRAIN_DAYS == 1008
    assert walk_forward.DEFAULT_TEST_DAYS == 252
    windows = walk_forward.expanding_windows(1826, lookback=20)
    assert len(windows) == 3
    first = windows[0]
    assert first["train_end"] - first["train_start"] + 1 == 1008
    assert first["test_end"] - first["test_start"] + 1 == 252


def test_windows_do_not_overlap_and_move_forward():
    windows = walk_forward.expanding_windows(1826, lookback=20)
    for window in windows:
        assert window["train_end"] < window["test_start"] <= window["test_end"]
    for earlier, later in zip(windows, windows[1:]):
        assert earlier["test_end"] < later["test_start"]
        # expanding: 학습 시작은 고정, 학습 끝은 앞으로 이동
        assert earlier["train_start"] == later["train_start"]
        assert earlier["train_end"] < later["train_end"]


def test_rolling_windows_keep_the_training_length_fixed():
    windows = walk_forward.expanding_windows(1826, lookback=20, expanding=False)
    lengths = {w["train_end"] - w["train_start"] + 1 for w in windows}
    assert lengths == {1008}


def test_windows_reject_insufficient_history_with_actionable_message():
    with pytest.raises(ValueError, match="최소 1764 거래일"):
        walk_forward.expanding_windows(752, lookback=20)


def test_plan_shrinks_to_fit_and_records_what_changed():
    """팀 실제 데이터(752 거래일)에서는 4년 학습이 불가능하다."""
    windows, plan = walk_forward.plan_windows(752, lookback=20, allow_shrink=True)
    assert plan["adjusted"] is True
    assert len(windows) == 3, "walk-forward 는 window 수를 가장 늦게 포기해야 한다"
    assert plan["used"]["train_days"] < 1008
    assert plan["used"]["train_days"] >= 252, "학습 1년 미만으로는 줄이지 않는다"
    assert walk_forward.required_days(**plan["used"]) <= 752
    assert any("학습 기간" in warning for warning in plan["warnings"])


def test_plan_keeps_requested_windows_when_data_is_long_enough():
    windows, plan = walk_forward.plan_windows(1826, lookback=20, allow_shrink=True)
    assert plan["adjusted"] is False
    assert plan["warnings"] == []
    assert len(windows) == 3


def test_cli_defaults_to_the_rolling_scheme_required_by_the_spec():
    """기본 윈도우 방식은 rolling 이어야 한다.

    과제 명세와 data_process/splits/ 는 학습 시작이 매 window 마다 이동한다
    (2019~2022 -> 2020~2023). 기본값이 expanding 이면 학습 시작이 가장 이른
    날짜에 고정된 채 보고서에 실려도 아무도 알아채지 못한다.
    """
    from backend.performance import run_analysis

    args = run_analysis.parse_args(["--walk-forward"])
    assert args.window_scheme == "rolling"
    assert (args.window_scheme == "expanding" and not args.rolling) is False

    explicit = run_analysis.parse_args(
        ["--walk-forward", "--window-scheme", "expanding"]
    )
    assert (explicit.window_scheme == "expanding" and not explicit.rolling) is True

    # 구버전 호환: --rolling 은 계속 rolling 을 뜻한다.
    legacy = run_analysis.parse_args(["--walk-forward", "--rolling"])
    assert (legacy.window_scheme == "expanding" and not legacy.rolling) is False


def test_expanding_scheme_is_flagged_so_it_cannot_reach_a_report_silently():
    _, rolling = walk_forward.plan_windows(1826, lookback=20, expanding=False)
    assert rolling["scheme"] == "rolling"
    assert rolling["scheme_note"] is None

    _, expanding = walk_forward.plan_windows(1826, lookback=20, expanding=True)
    assert expanding["scheme"] == "expanding"
    assert "rolling" in expanding["scheme_note"]


def test_plan_without_shrink_still_raises():
    with pytest.raises(ValueError):
        walk_forward.plan_windows(752, lookback=20, allow_shrink=False)


def test_walk_forward_runs_with_an_injected_trainer(data, config, tmp_path):
    windows = walk_forward.expanding_windows(
        len(data.dates), lookback=20, train_days=200, test_days=40, folds=2
    )
    result = walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "wf",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=lambda d, c, t, s, o: (StubPolicy(len(d.assets)), None),
    )
    assert len(result["windows"]) == 2
    assert set(result["metrics"].strategy) == {
        "ppo",
        "mvo",
        "equal_weight",
        "benchmark",
        "benchmark_raw",
    }
    assert (tmp_path / "wf" / "walk_forward_metrics.csv").exists()
    # 학습한 모델을 버리지 않고 돌려줘야 SHAP·ablation 에 쓸 수 있다.
    assert set(result["models"]) == {(0, "return"), (1, "return")}
    assert all(len(models) == 1 for models in result["models"].values())
    models, scaler, window = walk_forward.last_window_models(result, "return")
    assert window["window"] == 1
    assert len(models) == 1
    assert scaler is None


def test_last_window_models_names_available_rewards(data, config, tmp_path):
    windows = walk_forward.expanding_windows(
        len(data.dates), lookback=20, train_days=200, test_days=40, folds=1
    )
    result = walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "wf",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=lambda d, c, t, s, o: (StubPolicy(len(d.assets)), None),
    )
    with pytest.raises(ValueError, match="sharpe"):
        walk_forward.last_window_models(result, "sharpe")


def test_stitched_path_is_continuous_across_windows(data, config, tmp_path):
    """구간별 테스트를 이어붙인 경로가 연속이어야 한다 (자본이 1.0 으로 안 돌아감)."""
    windows = walk_forward.expanding_windows(
        len(data.dates), lookback=20, train_days=200, test_days=40, folds=2
    )
    result = walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "wf",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=lambda d, c, t, s, o: (StubPolicy(len(d.assets)), None),
    )
    stitched = result["stitched"]["ppo"]["return"]
    per_window = [result["histories"][(index, "return", "ppo")] for index in (0, 1)]
    assert len(stitched) == sum(len(frame) for frame in per_window)
    assert stitched.date.is_monotonic_increasing
    assert stitched.date.duplicated().sum() == 0
    expected = np.cumprod(1 + stitched.portfolio_return.to_numpy(float))
    assert stitched.portfolio_value.to_numpy(float) == pytest.approx(expected)
    assert stitched.max_drawdown.is_monotonic_increasing
    # 모든 전략이 같은 날짜 집합을 가져야 비교표를 만들 수 있다.
    frame = strategies.returns_frame(
        {"ppo": stitched, **result["stitched"]["baseline"]}
    )
    assert len(frame) == len(stitched)


# --------------------------------------------------------------------------- #
# 시장 국면
# --------------------------------------------------------------------------- #
def test_regimes_follow_the_benchmark_trend():
    rising = np.full(200, 0.003)
    falling = np.full(200, -0.003)
    flat = np.zeros(200)
    assert market_regime.classify(rising).iloc[-1] == "bull"
    assert market_regime.classify(falling).iloc[-1] == "bear"
    assert market_regime.classify(flat).iloc[-1] == "sideways"


def test_regime_summary_covers_every_strategy():
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "ppo": rng.normal(0.001, 0.01, 120),
            "benchmark": np.r_[np.full(60, 0.004), np.full(60, -0.004)],
        },
        index=index,
    )
    labelled = market_regime.regime_frame(frame)
    summary = market_regime.regime_summary(labelled)
    assert set(summary.strategy) == {"ppo", "benchmark"}
    assert summary.regime.nunique() >= 2


# --------------------------------------------------------------------------- #
# 통계 검증
# --------------------------------------------------------------------------- #
def test_one_way_detects_a_clear_difference():
    rng = np.random.default_rng(5)
    groups = {
        "a": rng.normal(0.0, 0.01, 40),
        "b": rng.normal(0.0, 0.01, 40),
        "c": rng.normal(0.05, 0.01, 40),
    }
    result = anova.one_way(groups, label="테스트")
    assert result["significant"] is True
    assert result["eta_squared"] > 0.14
    assert result["effect_size_label"] == "큼"


def test_identical_groups_are_not_significant():
    values = np.linspace(-0.02, 0.02, 30)
    result = anova.one_way({"a": values, "b": values.copy(), "c": values.copy()})
    assert result["significant"] is False
    assert result["p_value"] == pytest.approx(1.0)
    assert "동일" in anova.interpret(result)


def test_posthoc_runs_only_when_significant():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        {
            "period_return": np.r_[
                rng.normal(0.0, 0.01, 40), rng.normal(0.08, 0.01, 40)
            ],
            "strategy": ["ppo"] * 40 + ["mvo"] * 40,
        }
    )
    result = anova.run_with_posthoc(frame, "period_return", "strategy", label="전략")
    assert result["posthoc"] is not None
    assert result["posthoc"]["pairs"][0]["significant"] is True
    assert not anova.tukey_table([result]).empty


def test_two_way_reports_effect_sizes():
    rng = np.random.default_rng(7)
    rows = []
    for strategy, lift in (("ppo", 0.05), ("mvo", 0.0), ("equal_weight", 0.0)):
        for regime, shift in (("bull", 0.02), ("bear", -0.02), ("sideways", 0.0)):
            for value in rng.normal(lift + shift, 0.01, 10):
                rows.append(
                    {"period_return": value, "strategy": strategy, "regime": regime}
                )
    result = anova.two_way(
        pd.DataFrame(rows), "period_return", "strategy", "regime", label="2요인"
    )
    assert result["interaction_included"] is True
    assert result["terms"]["strategy"]["significant"] is True
    assert result["terms"]["regime"]["eta_squared"] > 0.0
    assert not anova.summary_table([result]).empty


def test_period_returns_compound_within_each_month():
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    returns = pd.DataFrame({"ppo": np.full(40, 0.001)}, index=index)
    monthly = anova.period_returns(returns)
    january = returns.loc["2024-01"]
    assert monthly.iloc[0]["ppo"] == pytest.approx(1.001 ** len(january) - 1)


# --------------------------------------------------------------------------- #
# SHAP 피처 압축/복원
# --------------------------------------------------------------------------- #
def test_reduce_and_rebuild_preserve_the_targeted_features(data, config):
    explainer = shap_analysis.PolicyExplainer(data, config, [StubPolicy(len(ASSETS))])
    observation = explainer.env.observation_at(100, np.full(len(ASSETS), 0.25))
    reduced = explainer.reduce(observation, 1)
    rebuilt = explainer.rebuild(observation, 1, reduced)
    assert explainer.reduce(rebuilt, 1) == pytest.approx(reduced, abs=1e-5)


def test_rebuild_only_touches_the_selected_asset(data, config):
    explainer = shap_analysis.PolicyExplainer(data, config, [StubPolicy(len(ASSETS))])
    observation = explainer.env.observation_at(100, np.full(len(ASSETS), 0.25))
    changed = explainer.rebuild(observation, 0, np.zeros(6))
    other = explainer.layout.return_indices(1)
    assert changed[other] == pytest.approx(observation[other])
    assert changed[explainer.layout.rsi0 + 1] == pytest.approx(
        observation[explainer.layout.rsi0 + 1]
    )


def test_policy_batching_matches_single_observations(data, config):
    explainer = shap_analysis.PolicyExplainer(data, config, [StubPolicy(len(ASSETS))])
    rows = np.vstack(
        [
            explainer.env.observation_at(index, np.full(len(ASSETS), 0.25))
            for index in (100, 120, 140)
        ]
    )
    batched = explainer.target_weights(rows)
    assert batched.shape == (3, len(ASSETS))
    assert batched.sum(axis=1) == pytest.approx(np.ones(3))
    single = explainer.target_weights(rows[1])
    assert single[0] == pytest.approx(batched[1])


# --------------------------------------------------------------------------- #
# 저장 / API
# --------------------------------------------------------------------------- #
def test_clean_converts_nan_and_numpy_types():
    cleaned = report.clean(
        {"a": np.float64("nan"), "b": np.int64(3), "c": [np.bool_(True)]}
    )
    assert cleaned == {"a": None, "b": 3, "c": [True]}


def test_backtest_payload_has_one_entry_per_strategy_and_day(data, config):
    histories = strategies.run_all(
        data, config, 250, 290, [StubPolicy(len(ASSETS))], None
    )
    returns = strategies.returns_frame(histories)
    table = metrics.metrics_table(
        {name: returns[name] for name in returns.columns}, benchmark_key="benchmark"
    )
    payload = report.backtest_payload(table, returns)
    assert len(payload["strategies"]) == 5
    assert len(payload["path"]) == len(returns)
    json.dumps(payload, allow_nan=False)


def test_api_reports_missing_results_then_serves_them(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from backend.performance import api

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    api._read.cache_clear()
    app = fastapi.FastAPI()
    app.include_router(api.router)
    client = TestClient(app)

    assert client.get("/performance/health").json()["ready"] is False
    assert client.get("/performance/backtest").status_code == 503

    (tmp_path / "api").mkdir()
    report.write_json(
        tmp_path / "api" / "backtest.json",
        {"strategies": [{"strategy": "ppo", "metrics": {}}], "path": []},
    )
    response = client.get("/performance/backtest", params={"strategy": "ppo"})
    assert response.status_code == 200
    assert response.json()["strategies"][0]["strategy"] == "ppo"
    assert (
        client.get("/performance/backtest", params={"strategy": "x"}).status_code == 404
    )


# --------------------------------------------------------------------------- #
# 리뷰 지적사항 회귀 테스트
# --------------------------------------------------------------------------- #
def test_drawdown_counts_a_loss_on_the_very_first_day():
    """고점에 최초 투자원금을 포함해야 첫날 손실이 낙폭으로 잡힌다."""
    index = pd.date_range("2024-01-01", periods=2, freq="B")
    returns = pd.DataFrame({"x": [-0.10, 0.05]}, index=index)
    drawdowns = strategies.drawdown_frame(returns)
    assert drawdowns.x.tolist() == pytest.approx([0.10, 0.055])
    # metrics 의 MDD 와 어긋나지 않아야 한다.
    assert drawdowns.x.max() == pytest.approx(metrics.max_drawdown(returns.x))


def test_stitched_drawdown_matches_metrics_on_an_opening_loss(data, config):
    frames = []
    for start, end in ((250, 270), (270, 290)):
        frame = strategies.run_strategy(data, config, start, end, "equal_weight")
        frames.append(frame)
    stitched = strategies.stitch_histories(frames)
    returns = stitched.portfolio_return.to_numpy(float)
    assert stitched.max_drawdown.iloc[-1] == pytest.approx(
        metrics.max_drawdown(returns)
    )
    assert stitched.drawdown.min() >= -1e-12


def test_raw_benchmark_ignores_the_safeguard(data, config):
    """순수 시장 기준은 15% 낙폭에서 현금화되지 않아야 한다."""
    tight = {**config, "max_drawdown": 0.02}
    guarded = strategies.run_strategy(data, tight, 250, 380, "benchmark")
    raw = strategies.run_strategy(data, tight, 250, 380, "benchmark_raw")
    assert guarded.safeguard_triggered.any()
    assert not raw.safeguard_triggered.any()
    assert (raw.cash_weight == 1).sum() == 0
    # 안전장치가 걸린 쪽은 청산 후 수익률이 0 으로 고정된다.
    assert (guarded.portfolio_return == 0).sum() > 0


def test_regime_classification_uses_the_raw_market_column():
    """Safe-Guard 로 현금이 된 벤치마크를 쓰면 하락장이 횡보장으로 잘못 분류된다."""
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    falling = np.full(120, -0.004)
    guarded = np.r_[falling[:40], np.zeros(80)]  # 40일 뒤 청산되어 수익률 0
    frame = pd.DataFrame(
        {"ppo": np.zeros(120), "benchmark": guarded, "benchmark_raw": falling},
        index=index,
    )
    labelled = market_regime.regime_frame(frame)
    assert labelled.regime.iloc[-1] == "bear"
    # 기준을 잘못 고르면 횡보장으로 분류된다 (회귀 방지용 대조).
    wrong = market_regime.regime_frame(frame, benchmark="benchmark")
    assert wrong.regime.iloc[-1] == "sideways"


def test_rolling_windows_actually_restrict_the_training_data(data, config, tmp_path):
    """--rolling 이 실제로 학습 데이터를 자르는지 확인한다.

    investment_ai.train_model 은 train_start 를 받지 않으므로, 데이터를 잘라
    넘기지 않으면 rolling 을 지정해도 가장 이른 날짜부터 학습한다.
    """
    seen = []

    def recording_train_fn(train_data, config_, train_end, seed, output):
        seen.append(
            {
                "days": len(train_data.dates),
                "first": str(train_data.dates[0])[:10],
                "train_end": train_end,
            }
        )
        return StubPolicy(len(train_data.assets)), None

    windows = walk_forward.expanding_windows(
        len(data.dates),
        lookback=20,
        train_days=200,
        test_days=40,
        folds=2,
        expanding=False,
    )
    walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "rolling",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=recording_train_fn,
    )
    assert [row["days"] for row in seen] == [200, 200], "학습 길이가 고정되어야 한다"
    assert seen[0]["first"] != seen[1]["first"], "학습 시작일이 이동해야 한다"
    assert [row["train_end"] for row in seen] == [199, 199]


def test_expanding_windows_keep_the_full_history(data, config, tmp_path):
    seen = []

    def recording_train_fn(train_data, config_, train_end, seed, output):
        seen.append(len(train_data.dates))
        return StubPolicy(len(train_data.assets)), None

    windows = walk_forward.expanding_windows(
        len(data.dates), lookback=20, train_days=200, test_days=40, folds=2
    )
    walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "expanding",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=recording_train_fn,
    )
    # 확장 윈도우: 학습 구간이 window 마다 test_days 만큼 길어진다.
    assert seen == [200, 240]


def test_slice_market_data_keeps_panels_aligned(data):
    sliced = walk_forward.slice_market_data(data, 100, 199)
    assert len(sliced.dates) == 100
    assert sliced.returns.shape == (100, len(data.assets))
    assert sliced.rsi.shape == (100, len(data.assets))
    assert sliced.macro.shape[0] == 100
    assert sliced.returns[0] == pytest.approx(data.returns[100])
    assert sliced.news_metadata.date.min() == pd.Timestamp(sliced.dates[0])
    assert sliced.news_metadata.date.max() == pd.Timestamp(sliced.dates[-1])


def test_mvo_strict_window_waits_for_252_days(data):
    early = mvo.weights_at(data, 30, strict_window=True)
    assert early == pytest.approx(mvo.equal_weights(len(data.assets)))
    late = mvo.weights_at(data, 380, strict_window=True)
    assert late.sum() == pytest.approx(1.0)
    assert not np.allclose(late, mvo.equal_weights(len(data.assets)))


def test_reward_samples_read_the_investment_ai_validation_histories(tmp_path):
    """walk-forward 없이 --run 만으로도 Reward ANOVA 표본을 만들 수 있어야 한다."""
    from backend.performance import run_analysis

    rng = np.random.default_rng(11)
    for reward, lift in (("return", 0.0), ("sharpe", 0.001), ("return_mdd", -0.001)):
        for fold in range(2):
            directory = tmp_path / "validation_models" / reward / f"fold_{fold}_seed_42"
            directory.mkdir(parents=True)
            dates = pd.date_range("2024-01-01", periods=80, freq="B")
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "portfolio_return": rng.normal(lift, 0.01, len(dates)),
                }
            ).to_csv(directory / "validation_history.csv", index=False)
    samples = run_analysis.reward_samples_from_run(tmp_path, "ME")
    assert samples is not None
    assert set(samples.reward) == {"return", "sharpe", "return_mdd"}
    assert samples.group.nunique() == 6
    result = anova.run_with_posthoc(samples, "period_return", "reward", label="보상")
    assert result["p_value"] is not None


def test_reward_samples_are_none_without_validation_histories(tmp_path):
    from backend.performance import run_analysis

    assert run_analysis.reward_samples_from_run(tmp_path, "ME") is None


def test_target_check_reports_met_and_unmet(data, config):
    histories = strategies.run_all(
        data, config, 250, 320, [StubPolicy(len(ASSETS))], None
    )
    returns = strategies.returns_frame(histories)
    table = metrics.metrics_table(
        {name: returns[name] for name in returns.columns},
        benchmark_key="benchmark_raw",
    )
    result = report.target_check(table, "ppo")
    assert set(result["criteria"]) == {"sharpe", "cumulative_return"}
    assert result["criteria"]["sharpe"]["threshold"] == 0.2
    assert isinstance(result["all_met"], bool)
    json.dumps(report.clean(result), allow_nan=False)


def test_targets_compare_each_metric_with_its_own_reference():
    """Sharpe 는 동일가중, 누적수익률은 S&P500 기준이어야 한다."""
    table = pd.DataFrame(
        {
            "sharpe": [1.20, 1.10, 0.80, 0.80],
            "cumulative_return": [0.30, 0.25, 0.15, 0.15],
            "days": [100, 100, 100, 100],
        },
        index=["ppo", "equal_weight", "benchmark_raw", "market_index"],
    )
    result = report.target_check(table, "ppo")
    sharpe = result["criteria"]["sharpe"]
    assert sharpe["reference"] == "equal_weight"
    # 동일가중 대비 +0.10 이므로 +0.2 목표에 미달이다.
    assert sharpe["difference"] == pytest.approx(0.10)
    assert sharpe["met"] is False
    # 시장과 비교했다면 +0.40 이라 잘못 "달성"이 되었을 것이다.
    assert sharpe["difference_vs"]["benchmark_raw"] == pytest.approx(0.40)

    cumulative = result["criteria"]["cumulative_return"]
    assert cumulative["reference"] == "benchmark_raw"
    assert cumulative["difference"] == pytest.approx(0.15)
    assert cumulative["met"] is True
    assert result["all_met"] is False


def test_targets_fall_back_when_a_reference_is_missing():
    table = pd.DataFrame(
        {"sharpe": [1.0, 0.5], "cumulative_return": [0.2, 0.1]},
        index=["ppo", "benchmark"],
    )
    result = report.target_check(table, "ppo")
    assert result["criteria"]["sharpe"]["reference"] == "benchmark"
    assert result["criteria"]["sharpe"]["requested_reference"] == "equal_weight"


def test_market_index_has_no_cost_or_execution_delay(data, config):
    """지수 수익률은 첫날부터 시장을 그대로 따라가야 한다."""
    history = strategies.run_strategy(data, config, 250, 300, "benchmark_raw")
    dates = pd.to_datetime(history.date)
    index = strategies.market_index_returns(data, dates)
    assert len(index) == len(history)
    # Buy & Hold 의 첫날은 현금 상태 + 진입 수수료(7bps)라 시장과 무관하게
    # -0.0007 이다. 지수는 그날의 실제 시장 수익률을 그대로 쓴다.
    entry_cost = -(config["transaction_cost_bps"] + config["slippage_bps"]) / 10000
    assert history.portfolio_return.iloc[0] == pytest.approx(entry_cost, abs=1e-9)
    assert index.iloc[0] != pytest.approx(entry_cost, abs=1e-6)
    # 진입 이후에는 거의 같지만 비용 때문에 완전히 같지는 않다.
    assert index.iloc[1:].to_numpy() == pytest.approx(
        history.portfolio_return.to_numpy()[1:], abs=5e-4
    )
    # 첫날 시장이 하락했다면 현금 상태였던 Buy & Hold 가 오히려 앞설 수도 있다.
    # 방향을 단정할 수 없으므로, 두 계열이 같지 않다는 점만 확인한다.
    assert metrics.cumulative_return(index) != pytest.approx(
        metrics.cumulative_return(history.portfolio_return), abs=1e-6
    )


def test_market_index_rejects_unknown_dates(data, config):
    with pytest.raises(ValueError, match="없는 날짜"):
        strategies.market_index_returns(data, pd.to_datetime(["1990-01-02"]))


def test_calendar_rebalancing_lands_on_real_month_ends(data, config):
    history = strategies.run_mvo(data, config, 250, 340, frequency="month_end")
    traded = pd.to_datetime(history.loc[history.rebalanced, "date"])
    assert len(traded) >= 2
    monthly = strategies.run_mvo(data, config, 250, 340, frequency="monthly")
    # 근사 주기(21거래일)와 실제 달력 말일은 날짜가 다르다.
    approximate = pd.to_datetime(monthly.loc[monthly.rebalanced, "date"])
    assert set(traded) != set(approximate)
    # 리밸런싱하지 않는 날에는 거래도 비용도 없다.
    quiet = history[~history.rebalanced]
    assert quiet.turnover.abs().max() == pytest.approx(0.0)
    assert quiet.cost_amount.abs().max() == pytest.approx(0.0)


def test_calendar_frequency_rejects_unknown_values(data, config):
    with pytest.raises(ValueError, match="frequency must be"):
        strategies.run_mvo(data, config, 250, 300, frequency="weekly")


def test_paired_difference_detects_a_consistent_edge():
    index = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 0.02, 60)  # 같은 시장 충격
    wide = pd.DataFrame({"ppo": shared + 0.004, "equal_weight": shared}, index=index)
    result = anova.paired_difference(wide, "ppo", "equal_weight")
    assert result["n"] == 60
    assert result["mean_difference"] == pytest.approx(0.004, abs=1e-9)
    assert result["significant"] is True
    assert result["bootstrap"]["excludes_zero"] is True
    assert "대응 비교" in anova.interpret_paired(result)


def test_paired_difference_is_more_sensitive_than_one_way():
    """같은 시장 충격을 공유하면 독립표본 ANOVA 는 차이를 놓치기 쉽다."""
    rng = np.random.default_rng(5)
    shared = rng.normal(0, 0.05, 40)
    left, right = shared + 0.01, shared
    paired = anova.paired_difference(pd.DataFrame({"a": left, "b": right}), "a", "b")
    independent = anova.one_way({"a": left, "b": right})
    assert paired["p_value"] < 0.05
    assert independent["p_value"] > 0.05


def test_block_bootstrap_reports_an_interval_around_the_mean():
    rng = np.random.default_rng(7)
    sample = rng.normal(0.01, 0.02, 80)
    result = anova.block_bootstrap_ci(sample, block=5, resamples=500)
    assert result["available"] is True
    assert result["block_size"] == 5
    assert result["lower"] < result["mean"] < result["upper"]


def test_overlap_diagnostics_flags_shared_dates():
    frame = pd.DataFrame(
        {
            "period_return": [0.01, 0.02, 0.03, 0.04],
            "strategy": ["ppo", "mvo", "ppo", "mvo"],
            "period_end": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]
            ),
        }
    )
    diagnostics = anova.overlap_diagnostics(frame, "strategy")
    assert diagnostics["dates_shared_across_groups"] == 2
    assert diagnostics["duplicated_group_date_rows"] == 0


def test_notable_decisions_picks_the_largest_moves():
    payload = {
        "features": ["a"],
        "assets": {
            "SPY": {
                "name": "x",
                "observations": [],
                "decisions": [
                    {"date": "2025-01-02", "base_value": 0.2, "final_value": 0.21},
                    {"date": "2025-01-03", "base_value": 0.2, "final_value": 0.35},
                    {"date": "2025-01-06", "base_value": 0.2, "final_value": 0.05},
                ],
            }
        },
    }
    assert shap_analysis.notable_decisions(payload, "SPY", 2) == [
        "2025-01-06",
        "2025-01-03",
    ]


def test_headline_chart_drops_the_safeguarded_benchmark():
    frame = pd.DataFrame(
        columns=[
            "ppo",
            "mvo",
            "equal_weight",
            "benchmark",
            "benchmark_raw",
            "market_index",
        ]
    )
    columns = plots.headline_columns(frame)
    assert "benchmark" not in columns
    assert "benchmark_raw" not in columns
    assert columns == ["ppo", "mvo", "equal_weight", "market_index"]


def test_api_strategy_filter_also_trims_the_daily_path(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from backend.performance import api

    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    api._read.cache_clear()
    (tmp_path / "api").mkdir()
    report.write_json(
        tmp_path / "api" / "backtest.json",
        {
            "strategies": [
                {"strategy": "ppo", "metrics": {}},
                {"strategy": "mvo", "metrics": {}},
            ],
            "path": [
                {
                    "date": "2025-01-02",
                    "regime": "bull",
                    "ppo": {"return": 0.01},
                    "mvo": {"return": 0.02},
                }
            ],
        },
    )
    app = fastapi.FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    payload = client.get("/performance/backtest", params={"strategy": "ppo"}).json()
    assert len(payload["strategies"]) == 1
    assert set(payload["path"][0]) == {"date", "regime", "ppo"}


def test_performance_api_imports_without_analysis_dependencies():
    """API 컨테이너에는 scipy/shap/matplotlib 를 설치하지 않는다.

    `backend/investment_ai/api.py` 가 이 라우터를 등록하므로, 무거운 모듈을
    import 하면 API 이미지가 통째로 커지거나 기동에 실패한다.
    """
    import subprocess
    import sys

    script = """
import sys, importlib.abc
BLOCKED = {"scipy", "shap", "matplotlib", "statsmodels", "sklearn"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(name)
        return None

sys.meta_path.insert(0, Blocker())
from backend.performance.api import router
print(len(router.routes))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) == 7


def test_reward_samples_fall_back_to_validation_csv(tmp_path):
    """validation_models 일별 기록이 없으면 validation.csv 로 대체한다."""
    from backend.performance import run_analysis

    pd.DataFrame(
        {
            "reward": ["return"] * 2 + ["sharpe"] * 2 + ["return_mdd"] * 2,
            "fold": [0, 1] * 3,
            "seed": [42, 73] * 3,
            "total_return": [0.06, 0.02, 0.04, 0.05, 0.01, 0.03],
        }
    ).to_csv(tmp_path / "validation.csv", index=False)

    assert run_analysis.reward_samples_from_run(tmp_path, "ME") is None
    table, samples, source = run_analysis.reward_tables(None, tmp_path, "ME")
    assert samples is not None
    assert set(samples.reward) == {"return", "sharpe", "return_mdd"}
    assert len(samples) == 6
    assert "validation.csv" in source
    result = anova.run_with_posthoc(samples, "period_return", "reward", label="보상")
    assert result["p_value"] is not None


def test_reward_tables_reports_when_nothing_is_available(tmp_path):
    from backend.performance import run_analysis

    table, samples, source = run_analysis.reward_tables(None, tmp_path, "ME")
    assert samples is None
    assert "미수행" in source


def test_holm_correction_matches_a_hand_calculation():
    """4개 비교에서 p=0.0381 은 보정 후 0.1524 가 되어 유의하지 않다."""
    adjusted = anova.holm_adjust([0.2416, 0.0381, 0.4557, 0.1307])
    assert adjusted == pytest.approx([0.4832, 0.1524, 0.4832, 0.3921], abs=1e-4)
    assert adjusted[1] > 0.05, "보정 후에는 유의하지 않아야 한다"
    # 단조성: 보정값은 원래 순위를 뒤집지 않는다.
    order = np.argsort([0.2416, 0.0381, 0.4557, 0.1307])
    ranked = [adjusted[i] for i in order]
    assert ranked == sorted(ranked)


def test_holm_handles_missing_p_values():
    assert anova.holm_adjust([None, 0.01]) == [None, pytest.approx(0.01)]
    assert anova.holm_adjust([]) == []


def test_apply_holm_marks_results_and_interpretation():
    index = pd.date_range("2024-01-01", periods=40, freq="B")
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 0.02, 40)
    # 세 계열의 차이가 상수가 되지 않도록 각각 다른 잡음을 더한다
    # (차이의 분산이 0 이면 검정 자체가 불가능해진다).
    wide = pd.DataFrame(
        {
            "a": shared + 0.006 + rng.normal(0, 0.004, 40),
            "b": shared + rng.normal(0, 0.004, 40),
            "c": shared - 0.001 + rng.normal(0, 0.004, 40),
        },
        index=index,
    )
    results = [
        anova.paired_difference(wide, "a", "b"),
        anova.paired_difference(wide, "a", "c"),
        anova.paired_difference(wide, "b", "c"),
    ]
    anova.apply_holm(results)
    assert all(r["comparisons"] == 3 for r in results)
    assert all(r["p_value_holm"] >= r["p_value"] for r in results)
    text = anova.interpret_paired(results[0])
    assert "Holm 보정" in text
    table = anova.paired_table(results)
    assert "p_value_holm" in table.columns


def _run_in_layout(tmp_path, mutate, snippet):
    """backend/ 를 복사한 임시 레이아웃에서 코드를 실행한다 (컨테이너 흉내)."""
    import shutil
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    workspace = tmp_path / "layout"
    workspace.mkdir()
    shutil.copytree(
        root / "backend",
        workspace / "backend",
        ignore=shutil.ignore_patterns("__pycache__", "verification"),
    )
    mutate(workspace)
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", snippet],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Minimal env: inherited flags such as PYTHONOPTIMIZE would strip the
        # snippet's asserts. Windows needs SystemRoot for socket initialization.
        env={
            "PYTHONPATH": ".",
            "PATH": os.environ.get("PATH", ""),
            **{k: os.environ[k] for k in ("SystemRoot",) if k in os.environ},
        },
    )


def test_api_starts_when_the_performance_package_is_absent(tmp_path):
    """모듈을 넣지 않은 이미지에서도 서버가 떠야 한다.

    하위 모듈을 바로 find_spec 하면 부모가 없을 때 ModuleNotFoundError 가 나서
    오히려 기동이 깨진다. 그 회귀를 막는 테스트다.
    """
    import shutil

    result = _run_in_layout(
        tmp_path,
        lambda ws: shutil.rmtree(ws / "backend" / "performance"),
        "from backend.investment_ai.api import optional_performance_router\n"
        "assert optional_performance_router() is None\n"
        "print('started')",
    )
    assert result.returncode == 0, result.stderr
    assert "started" in result.stdout


def test_api_surfaces_errors_from_inside_the_performance_module(tmp_path):
    """모듈이 있는데 안에서 실패하면 조용히 숨기지 말고 그대로 올려야 한다."""

    def break_module(workspace):
        path = workspace / "backend" / "performance" / "api.py"
        path.write_text(
            "from __future__ import annotations\n"
            "import module_that_does_not_exist_xyz\n"
        )

    result = _run_in_layout(
        tmp_path,
        break_module,
        "from backend.investment_ai.api import optional_performance_router\n"
        "optional_performance_router()",
    )
    assert result.returncode != 0
    assert "module_that_does_not_exist_xyz" in result.stderr


def test_calendar_rebalancing_supports_month_start_and_end(data, config):
    dates = data.dates
    starts = sorted(mvo.calendar_rebalance_dates(dates, "month_start"))
    ends = sorted(mvo.calendar_rebalance_dates(dates, "month_end"))
    assert len(starts) == len(ends)
    assert starts != ends
    # 각 월의 첫 거래일은 그 달 마지막 거래일보다 앞선다.
    assert all(a < b for a, b in zip(starts, ends))
    first = pd.Timestamp(str(dates[starts[1]])[:10])
    previous = pd.Timestamp(str(dates[starts[1] - 1])[:10])
    assert first.month != previous.month, "월이 바뀐 첫 거래일이어야 한다"


def test_quarter_start_picks_four_dates_per_year(data):
    picks = sorted(mvo.calendar_rebalance_dates(data.dates, "quarter_start"))
    months = {pd.Timestamp(str(data.dates[i])[:10]).month for i in picks}
    assert months <= {1, 4, 7, 10}


def test_mvo_runs_with_month_start_rebalancing(data, config):
    history = strategies.run_mvo(data, config, 250, 340, frequency="month_start")
    assert history.rebalanced.sum() >= 2
    quiet = history[~history.rebalanced]
    assert quiet.turnover.abs().max() == pytest.approx(0.0)


def test_two_way_runs_posthoc_on_a_significant_factor():
    """Two-way 에서 유의한 주효과는 어느 수준끼리 다른지까지 알려줘야 한다."""
    rng = np.random.default_rng(3)
    rows = []
    for strategy in ("ppo", "mvo", "equal_weight"):
        for regime, shift in (("bull", 0.05), ("bear", -0.05), ("sideways", 0.0)):
            for value in rng.normal(shift, 0.01, 10):
                rows.append(
                    {"period_return": value, "strategy": strategy, "regime": regime}
                )
    result = anova.two_way(
        pd.DataFrame(rows), "period_return", "strategy", "regime", label="검증 3"
    )
    assert result["terms"]["regime"]["significant"] is True
    assert result["terms"]["strategy"]["significant"] is False
    posthoc = result["posthoc"]
    # 유의한 요인만 사후검정한다.
    assert set(posthoc) == {"regime"}
    pairs = {(p["group1"], p["group2"]) for p in posthoc["regime"]["pairs"]}
    assert pairs == {("bear", "bull"), ("bear", "sideways"), ("bull", "sideways")}
    assert all(p["significant"] for p in posthoc["regime"]["pairs"])
    assert "독립이 아닙니다" in result["posthoc_note"]


def test_two_way_posthoc_covers_exactly_the_significant_factors():
    """사후검정은 유의한 주효과에만 붙어야 한다 (우연에 의존하지 않는 성질 검증)."""
    rng = np.random.default_rng(11)
    rows = []
    for strategy in ("ppo", "mvo", "equal_weight"):
        for regime, shift in (("bull", 0.05), ("bear", -0.05), ("sideways", 0.0)):
            for value in rng.normal(shift, 0.01, 10):
                rows.append(
                    {"period_return": value, "strategy": strategy, "regime": regime}
                )
    result = anova.two_way(pd.DataFrame(rows), "period_return", "strategy", "regime")
    significant = {
        name for name in ("strategy", "regime") if result["terms"][name]["significant"]
    }
    assert set(result["posthoc"] or {}) == significant


def test_two_way_has_no_posthoc_when_no_factor_is_significant():
    """모든 그룹이 같은 분포면 사후검정이 없어야 한다."""
    value = np.tile(np.linspace(-0.02, 0.02, 10), 4)
    rows = []
    index = 0
    for strategy in ("ppo", "mvo"):
        for regime in ("bull", "bear"):
            for _ in range(10):
                rows.append(
                    {
                        "period_return": value[index % len(value)],
                        "strategy": strategy,
                        "regime": regime,
                    }
                )
                index += 1
    result = anova.two_way(pd.DataFrame(rows), "period_return", "strategy", "regime")
    assert not any(t["significant"] for t in result["terms"].values())
    assert result["posthoc"] is None


def test_tukey_table_handles_both_oneway_and_twoway(tmp_path):
    rng = np.random.default_rng(5)
    one_way_frame = pd.DataFrame(
        {
            "period_return": np.r_[
                rng.normal(0.0, 0.01, 30), rng.normal(0.09, 0.01, 30)
            ],
            "strategy": ["ppo"] * 30 + ["mvo"] * 30,
        }
    )
    one = anova.run_with_posthoc(
        one_way_frame, "period_return", "strategy", label="전략"
    )
    rows = []
    for strategy in ("ppo", "mvo"):
        for regime, shift in (("bull", 0.05), ("bear", -0.05), ("sideways", 0.0)):
            for value in rng.normal(shift, 0.01, 10):
                rows.append(
                    {"period_return": value, "strategy": strategy, "regime": regime}
                )
    two = anova.two_way(pd.DataFrame(rows), "period_return", "strategy", "regime")
    table = anova.tukey_table([one, two])
    assert not table.empty
    assert "factor" in table.columns
    assert set(table.factor) >= {"strategy", "regime"}


def test_walk_forward_uses_the_index_as_the_alpha_beta_reference(data, config):
    """구간별 Alpha/Beta 기준이 전체 지표와 같은 정의여야 한다.

    예전에는 baselines 안에서 market_index 를 찾았는데 그것은 전략이 아니라서
    절대 잡히지 않고 늘 Safe-Guard 벤치마크로 떨어졌다.
    """
    baselines = strategies.run_all(
        data,
        config,
        250,
        300,
        None,
        None,
        strategies=("equal_weight", "benchmark", "benchmark_raw"),
    )
    name, returns = walk_forward.window_market_reference(data, baselines)
    assert name == strategies.MARKET_REFERENCE
    assert returns is not None
    expected = strategies.market_index_returns(
        data, pd.to_datetime(baselines["benchmark"].date)
    )
    assert returns == pytest.approx(expected.to_numpy())
    # 안전장치 벤치마크와 달라야 한다 (비용·체결지연 차이).
    assert returns != pytest.approx(
        baselines["benchmark"].portfolio_return.to_numpy(float)
    )


def test_walk_forward_records_which_reference_it_used(data, config, tmp_path):
    windows = walk_forward.expanding_windows(
        len(data.dates), lookback=20, train_days=200, test_days=40, folds=1
    )
    result = walk_forward.run_walk_forward(
        data,
        {**config, "seeds": [1]},
        tmp_path / "wf",
        windows=windows,
        rewards=("return",),
        seeds=[1],
        train_fn=lambda d, c, t, s, o: (StubPolicy(len(d.assets)), None),
    )
    assert set(result["metrics"].market_reference) == {strategies.MARKET_REFERENCE}


def test_market_reference_falls_back_without_spy(data, config):
    stripped = walk_forward.replace(data, assets=["AAA", "BBB", "CCC", "DDD"])
    baselines = {
        "benchmark": strategies.run_strategy(data, config, 250, 280, "benchmark")
    }
    name, returns = walk_forward.window_market_reference(stripped, baselines)
    assert name == "benchmark"
    assert returns is not None
