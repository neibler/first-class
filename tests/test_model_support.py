"""Model plumbing only. Any real .learn() call fails this test module."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from backend.investment_ai.model_factory import (
    MODEL_TYPES,
    create_model,
    model_class,
    model_parameters,
    validate_model_config,
)
from backend.investment_ai.prediction import PredictionSession
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.training import load_bundle, schema_for
from backend.investment_ai.preprocessing import FeatureScaler
from backend.investment_ai.settings import write_json
from backend.investment_ai import model_runs, train
from backend.investment_ai.model_comparison import (
    compare_completed,
    metrics_from_returns,
)
from backend.investment_ai.backtest import run_backtest
from backend.investment_ai.infer import predict_weights
from tests.test_contract_v3 import data, config, service, state


@pytest.fixture(autouse=True)
def never_learn(monkeypatch):
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        pytest.fail("Training is forbidden in plumbing tests")

    for kind in MODEL_TYPES:
        monkeypatch.setattr(model_class(kind), "learn", forbidden)


def small_config(kind):
    c = config()
    c.update(
        model_type=kind,
        n_steps=8,
        batch_size=4,
        total_timesteps=16,
        checkpoint_steps=8,
        device="cpu",
        models={
            "ppo": {"policy_kwargs": {"net_arch": [8]}},
            "recurrent_ppo": {
                "lstm_hidden_size": 8,
                "policy_kwargs": {"net_arch": [8]},
            },
            "sac": {
                "buffer_size": 32,
                "learning_starts": 0,
                "batch_size": 4,
                "policy_kwargs": {"net_arch": [8]},
            },
            "td3": {
                "buffer_size": 32,
                "learning_starts": 0,
                "batch_size": 4,
                "policy_kwargs": {"net_arch": [8]},
            },
        },
    )
    return c


@pytest.mark.parametrize("kind", MODEL_TYPES)
def test_factory_predict_save_load_and_spaces_without_learning(kind, tmp_path):
    c, d = small_config(kind), data()
    validate_model_config(c)
    scaler = FeatureScaler.fit(d, 2)
    env = PortfolioEnv(d, c, scaler=scaler)
    model = create_model(kind, env, c, 42)
    assert model.num_timesteps == 0
    obs, _ = env.reset()
    session = PredictionSession(model, kind)
    weights = session.predict(obs, env)
    assert weights.shape == (len(d.assets),)
    assert np.all(weights >= 0) and np.isclose(weights.sum(), 1)
    assert model.action_space == env.action_space
    assert model.observation_space == env.observation_space
    if kind == "recurrent_ppo":
        assert session.lstm_states is not None and not session.episode_start
    model.save(tmp_path / "model.zip")
    scaler.save(tmp_path / "scaler.json")
    write_json(tmp_path / "config.json", c)
    write_json(tmp_path / "schema.json", schema_for(d, c))
    restored, _, _ = load_bundle(tmp_path, d)
    actual = PredictionSession(restored, kind).predict(obs, env)
    np.testing.assert_allclose(actual, weights)
    session.reset()
    assert session.lstm_states is None and session.episode_start
    env.close()


def test_legacy_ppo_factory_parameters_are_unchanged():
    c = config()
    assert model_parameters("ppo", c) == {
        "n_steps": c["n_steps"],
        "batch_size": c["batch_size"],
    }
    model = create_model("ppo", PortfolioEnv(data(), c), c, 42)
    assert model.learning_rate == 0.0003 and model.seed == 42


class RecurrentSpy:
    def __init__(self):
        self.calls = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls.append((state, bool(episode_start[0])))
        return np.zeros(2), (np.array([len(self.calls)]), np.array([len(self.calls)]))


def test_backtest_preserves_memory_and_resets_each_run():
    model = RecurrentSpy()
    c = small_config("recurrent_ppo")
    d = data()
    run_backtest(d, c, 0, 4, [model])
    assert model.calls[0] == (None, True)
    assert all(not start and memory is not None for memory, start in model.calls[1:])
    n = len(model.calls)
    run_backtest(d, c, 0, 2, [model])
    assert model.calls[n] == (None, True)


def test_recurrent_advances_on_hold_days_and_safeguard_stops_calls():
    c, d = small_config("recurrent_ppo"), data()
    c["rebalance_days"] = 5
    model = RecurrentSpy()
    sessions = [PredictionSession(model, "recurrent_ppo")]
    s = state(d)
    s.update(
        max_drawdown=0.0, portfolio_value=1.0, peak_value=1.0, days_since_rebalance=1
    )
    result = predict_weights(d, c, [model], None, s, sessions)
    assert result["status"] == "hold_until_rebalance" and len(model.calls) == 1
    s["max_drawdown"] = 0.2
    result = predict_weights(d, c, [model], None, s, sessions)
    assert result["cash_weight"] == 1 and len(model.calls) == 1


def test_recurrent_service_isolates_episodes_and_repeated_requests(tmp_path):
    obj = service(tmp_path)
    obj.config["model_type"] = "recurrent_ppo"
    obj.models = [RecurrentSpy()]
    obj.recurrent_sessions = {}
    first = obj.optimize()
    assert obj.optimize() == first
    assert len(obj.models[0].calls) == 1
    s = state(obj.data)
    s.update(
        episode_id="other",
        reset_episode=True,
        weights={a: 0.0 for a in obj.data.assets},
        max_drawdown=0.0,
        portfolio_value=1.0,
        peak_value=1.0,
        stopped=False,
    )
    obj.optimize(state=s)
    assert obj.models[0].calls[-1] == (None, True)
    assert set(obj.recurrent_sessions) == {"default", "other"}
    s["reset_episode"] = False
    with pytest.raises(ValueError, match="consecutive"):
        obj.optimize(state=s)


def test_calendar_windows_and_insufficient_data():
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    windows = model_runs.calendar_windows(dates, 4, 1)
    assert len(windows) == 2
    first, second = windows
    assert dates[first["split"]] == pd.Timestamp("2024-01-01")
    assert first["stop"] == second["split"]
    assert dates[second["start"]] == pd.Timestamp("2021-01-01")
    with pytest.raises(ValueError, match="Insufficient"):
        model_runs.calendar_windows(pd.bdate_range("2023-01-01", "2025-12-31"))


def test_cli_and_paths():
    args = train.build_parser().parse_args(
        [
            "--model",
            "td3",
            "--reward",
            "sharpe",
            "--timesteps",
            "100000",
            "--seeds",
            "42",
            "123",
            "2026",
            "--dry-run",
        ]
    )
    assert args.model == "td3" and args.seeds == [42, 123, 2026]
    path = model_runs.bundle_path(Path("runtime"), "sac", "return_mdd", 42, "test")
    assert path == Path("runtime/models/sac/test/holdout/sac_return_mdd_seed42")
    with pytest.raises(ValueError):
        model_runs.bundle_path(Path("runtime"), "sac", "return", 42, "../bad")


def test_dry_run_calls_no_training_and_writes_no_results(tmp_path, monkeypatch):
    c = small_config("sac")
    monkeypatch.setattr(model_runs, "ROOT", tmp_path)
    monkeypatch.setattr(
        model_runs, "load_data", lambda *args: (data(np.zeros((20, 2))), None)
    )
    monkeypatch.setattr(
        model_runs, "train_model", lambda *a, **kw: pytest.fail("must not train")
    )
    paths = model_runs.run_seed_experiments(
        c, seeds=[42, 123], run_id="test", dry_run=True
    )
    assert len(paths) == 2 and not any(tmp_path.iterdir())


def test_comparison_empty_does_not_make_up_results(tmp_path, capsys):
    output = tmp_path / "result.csv"
    assert compare_completed(tmp_path, output).empty
    assert "No completed model runs found." in capsys.readouterr().out
    assert not output.exists()


def test_metrics_conventions():
    # Mathematical unit fixture, not model performance.
    values = np.array([0.1, -0.1, 0.05])
    metrics = metrics_from_returns(values)
    assert metrics["mdd"] == pytest.approx(0.1)
    assert metrics["cagr"] == pytest.approx(np.prod(1 + values) ** 84 - 1)
    assert metrics["calmar"] == pytest.approx(metrics["cagr"] / 0.1)
    assert metrics_from_returns([0, 0])["sortino"] is None


def test_config_files_support_all_algorithms():
    c = json.loads(Path("config/multi_model.json").read_text(encoding="utf-8"))
    for kind in MODEL_TYPES:
        validate_model_config({**c, "model_type": kind})
    assert len(c["assets"]) == 15


def test_single_bundle_service_uses_factory_metadata(tmp_path, monkeypatch):
    from backend.investment_ai import serving

    c, d = small_config("td3"), data()
    write_json(tmp_path / "config.json", c)
    write_json(tmp_path / "schema.json", schema_for(d, c))
    write_json(tmp_path / "manifest.json", {"completed": True})
    write_json(tmp_path / "run_spec.json", {"synthetic": False})
    monkeypatch.setattr(serving, "ROOT", tmp_path)
    monkeypatch.setattr(serving, "load_data", lambda *args: (d, None))
    model = create_model("td3", PortfolioEnv(d, c), c, 42)
    monkeypatch.setattr(serving, "load_bundle", lambda *args: (model, None, c))
    obj = serving.InvestmentService(tmp_path)
    result = obj.optimize()
    assert len(obj.models) == 1
    assert result["cash_weight"] == pytest.approx(0)
    assert sum(result["target_weights"].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="evaluate"):
        obj.backtest()


def test_compare_seed_summary_from_unit_fixtures(tmp_path):
    # These short arithmetic fixtures are temporary; they are not model results.
    for seed in [42, 123]:
        bundle = tmp_path / str(seed)
        (bundle / "evaluation").mkdir(parents=True)
        write_json(bundle / "manifest.json", {"completed": True})
        write_json(bundle / "config.json", {"risk_free_annual": 0.0})
        write_json(
            bundle / "evaluation.json",
            {
                "completed": True,
                "model_type": "sac",
                "seed": seed,
                "reward_type": "return",
                "window_id": "holdout",
                "comparison_group": "unit_fixture",
                "test_start": "2024-01-02",
                "test_end": "2024-01-03",
            },
        )
        pd.DataFrame(
            {"date": ["2024-01-02", "2024-01-03"], "portfolio_return": [0.01, -0.005]}
        ).to_csv(bundle / "evaluation/ppo_history.csv", index=False)
    output = tmp_path / "output/comparison.csv"
    table = compare_completed(tmp_path, output)
    assert len(table) == 2
    summary = pd.read_csv(output.with_name("comparison_seed_summary.csv"))
    assert summary.cagr_count.iloc[0] == 2
    assert summary.cagr_std.iloc[0] == pytest.approx(0)


def test_training_dispatch_preserves_windows_and_seed_without_learning(
    tmp_path, monkeypatch
):
    c = small_config("recurrent_ppo")
    c["data_path"] = "fixture.parquet"
    d = data(np.zeros((20, 2)))
    calls = []

    def training_stub(local, config, end, seed, output):
        calls.append((local, config, end, seed))
        output.mkdir(parents=True)

    monkeypatch.setattr(model_runs, "ROOT", tmp_path)
    monkeypatch.setattr(model_runs, "load_data", lambda *args: (d, None))
    monkeypatch.setattr(model_runs, "train_model", training_stub)
    paths = model_runs.run_seed_experiments(c, seeds=[42, 123], run_id="unit")
    assert [call[3] for call in calls] == [42, 123]
    assert all(call[1]["model_type"] == "recurrent_ppo" for call in calls)
    assert all(call[2] == 15 for call in calls)
    assert all(
        json.loads((path / "run_spec.json").read_text())["test_start_index"] == 16
        for path in paths
    )
    assert not (tmp_path / "runs/latest.json").exists()


def test_serving_config_selects_bundle_and_rejects_type_mismatch(tmp_path, monkeypatch):
    from backend.investment_ai import serving

    project = tmp_path / "app"
    (project / "config").mkdir(parents=True)
    bundle = tmp_path / "runtime/models/sac/unit"
    bundle.mkdir(parents=True)
    c, d = small_config("sac"), data()
    for name, value in {
        "config.json": c,
        "schema.json": schema_for(d, c),
        "manifest.json": {"completed": True},
        "run_spec.json": {"synthetic": False},
    }.items():
        write_json(bundle / name, value)
    write_json(
        project / "config/serving.json",
        {"model_type": "sac", "model_path": "models/sac/unit"},
    )
    monkeypatch.setattr(
        serving, "__file__", str(project / "backend/investment_ai/serving.py")
    )
    monkeypatch.setattr(serving, "ROOT", tmp_path / "runtime")
    monkeypatch.delenv("INVESTMENT_AI_RUN", raising=False)
    monkeypatch.setattr(serving, "load_data", lambda *args: (d, None))
    model = create_model("sac", PortfolioEnv(d, c), c, 42)
    monkeypatch.setattr(serving, "load_bundle", lambda *args: (model, None, c))
    obj = serving.InvestmentService()
    assert obj.run == bundle and obj.config["model_type"] == "sac"
    write_json(
        project / "config/serving.json",
        {"model_type": "td3", "model_path": "models/sac/unit"},
    )
    with pytest.raises(ValueError, match="does not match"):
        serving.InvestmentService()


@pytest.mark.parametrize("mode", ["walk_forward", "typo", "", None, [], 7])
def test_config_mode_rejected_before_dispatch(tmp_path, monkeypatch, mode):
    c = small_config("ppo")
    c["mode"] = mode
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    monkeypatch.setattr(
        train, "run_experiment", lambda *a: pytest.fail("legacy called")
    )
    monkeypatch.setattr(
        model_runs,
        "run_seed_experiments",
        lambda *a, **k: pytest.fail("training called"),
    )
    with pytest.raises(ValueError, match="Unsupported mode"):
        train.main(["--config", str(path)])


@pytest.mark.parametrize("mode", ["single", "walk-forward", "legacy"])
def test_valid_config_modes_keep_dispatch(monkeypatch, mode):
    c = small_config("recurrent_ppo")
    c["mode"] = mode
    monkeypatch.setattr(train, "read_config", lambda *a: c)
    calls = []
    monkeypatch.setattr(train, "run_experiment", lambda *a: calls.append("legacy"))
    monkeypatch.setattr(
        model_runs,
        "run_seed_experiments",
        lambda *a, **k: calls.append(k["walk_forward"]) or [],
    )
    train.main([])
    assert calls == ["legacy" if mode == "legacy" else mode == "walk-forward"]


def test_current_calendar_coverage_boundary():
    dates = pd.bdate_range("2021-01-04", "2025-12-31")
    with pytest.raises(ValueError, match="Insufficient data"):
        model_runs.calendar_windows(dates, 4, 1)
    assert len(model_runs.calendar_windows(dates, 3, 1)) == 1
    assert len(model_runs.calendar_windows(dates, 2, 1)) == 2
    extended = pd.bdate_range("2021-01-04", "2026-01-02")
    assert len(model_runs.calendar_windows(extended, 4, 1)) == 1
