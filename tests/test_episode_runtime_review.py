"""Episode-clock, portable runtime and resumed-bundle regression tests.

All training here is tiny, temporary test training, never the user's model runs.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from backend.investment_ai import train, training, model_runs, serving
from backend.investment_ai.data_loader import load_data, fingerprint
from backend.investment_ai.episode_clock import episode_clock, rebalance_phase
from backend.investment_ai.infer import predict_weights, main as infer_main
from backend.investment_ai.model_factory import validate_model_config
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.settings import write_json, bundle_input_root
from scripts import build_trained_models as builder
from scripts.install_trained_models import install
from tests.test_contract_v3 import data, inputs, state
from tests.test_model_support import small_config
from tests.test_review_model_sharing import packages  # noqa: F401


def specification(d, kind, synthetic=False, seed=42, data_seed=17):
    return {
        "start_date": str(d.dates[0])[:10],
        "end_date": str(d.dates[-1])[:10],
        "test_start_index": 20,
        "model_type": kind,
        "reward_type": "return_mdd",
        "seed": seed,
        "window_id": "holdout",
        "data_hash": fingerprint(d),
        "data_seed": data_seed,
        "synthetic": synthetic,
    }


@pytest.mark.parametrize("period", [1, 5, 21])
def test_training_inference_observations_match_despite_trades(period):
    d = data(np.zeros((40, 2)))
    d.risk[6, 0] = 0.8
    c = {**small_config("ppo"), "rebalance_days": period}
    env = PortfolioEnv(d, c, start=1)
    obs, _ = env.reset()
    reasons = set()

    class Capture:
        def predict(self, observation, _env):
            np.testing.assert_allclose(observation, obs, rtol=0, atol=0)
            return np.array([0.9, 0.1])

    for _ in range(25):
        local = model_runs.subset(d, 0, env.t + 1)
        s = {
            **state(local),
            "weights": dict(zip(d.assets, env.weights)),
            "portfolio_value": env.value,
            "peak_value": env.peak,
            "max_drawdown": env.mdd,
            "days_since_rebalance": 0,
            "episode_start_date": str(d.dates[env.episode_start])[:10],
            "episode_elapsed_bars": env.t - env.episode_start,
        }
        proposal = predict_weights(local, c, [], None, s, [Capture()])
        assert proposal["rebalance_phase"] == rebalance_phase(
            env.t - env.episode_start, period
        )
        obs, _, _, _, info = env.step(
            np.array([1.0, -1.0]) if env.t % 2 else np.array([-1.0, 1.0])
        )
        reasons.update(info["rebalance_reasons"])
    assert {"monthly", "risk_score", "portfolio_weight_distance"} <= reasons


def test_clock_does_not_infer_episode_age_from_last_trade():
    d = data()
    s = state(d)
    s.pop("episode_start_date")
    s.pop("episode_elapsed_bars")
    with pytest.raises(ValueError, match="episode_start_date"):
        episode_clock(d.dates, s)
    s.update(episode_start_date=str(d.dates[0])[:10], episode_elapsed_bars=4)
    old = episode_clock(d.dates, s)
    assert episode_clock(d.dates, {**s, "days_since_rebalance": 99}, old) == old
    with pytest.raises(ValueError, match="mismatch"):
        episode_clock(d.dates, {**s, "episode_elapsed_bars": 0}, old)


@pytest.mark.parametrize("kind", ["sac", "td3"])
def test_offpolicy_resume_under_warmup_extra_steps(tmp_path, kind):
    c = small_config(kind)
    c.update(
        total_timesteps=1008,
        checkpoint_steps=1008,
        data_path=None,
        news_path=None,
        macro_columns=[],
    )
    c["models"][kind]["learning_starts"] = 1000
    d = data(np.zeros((30, 2)))
    source, result = tmp_path / "source", tmp_path / "resumed"
    training.train_model(
        d, c, 19, 42, source, run_spec=specification(d, kind, synthetic=True)
    )
    with pytest.raises(ValueError, match="learning_starts"):
        validate_model_config({**c, "total_timesteps": 500})
    validate_model_config({**c, "total_timesteps": 500}, completed_steps=1008)
    train.main(
        ["--resume", str(source), "--output", str(result), "--extra-steps", "500"]
    )
    metadata = json.loads((result / "manifest.json").read_text())
    assert metadata["initial_timesteps"] == 1008
    assert metadata["actual_timesteps"] == 1508 and metadata["completed"]


@pytest.mark.parametrize("kind", ["ppo", "recurrent_ppo", "sac", "td3"])
def test_synthetic_resume_evaluate_infer_and_serving(tmp_path, monkeypatch, kind):
    c = small_config(kind)
    c.update(
        data_path=None,
        news_path=None,
        news_mode="development",
        macro_columns=[],
        dummy_macro_columns=[],
        dummy_start="2024-01-02",
        dummy_end="2024-02-16",
        scenario="bull",
        rebalance_days=5,
    )
    d, _ = load_data({**c, "seed": 17}, tmp_path)
    spec = specification(d, kind, synthetic=True)
    source, result = tmp_path / "source", tmp_path / "resumed"
    training.train_model(d, c, 19, 42, source, run_spec=spec)
    monkeypatch.setattr(train, "ROOT", tmp_path / "unused_default")
    monkeypatch.setattr(serving, "ROOT", tmp_path / "service_runtime")
    monkeypatch.delenv("INVESTMENT_AI_DATA", raising=False)
    monkeypatch.delenv("INVESTMENT_AI_NEWS", raising=False)
    train.main(["--resume", str(source), "--output", str(result), "--extra-steps", "8"])
    assert (result / "synthetic_market.parquet").is_file()
    assert (result / "news_metadata.parquet").is_file()
    assert json.loads((result / "run_spec.json").read_text())["data_seed"] == 17
    assert model_runs.evaluate_bundle(result)
    infer_main(["--bundle", str(result), "--output", str(tmp_path / "prediction")])
    assert (tmp_path / "prediction/latest_weights.json").is_file()
    obj = serving.InvestmentService(result)
    first = obj.optimize()
    restarted = serving.InvestmentService(result)
    assert restarted.optimize() == first
    assert first["episode_elapsed_bars"] == 0
    # Advance one known bar with actual holdings, then retry and restart.
    frame, news = inputs(np.zeros((45, 2)))
    frame = frame[frame.date >= str(d.dates[0])[:10]]
    news = news[news.date >= str(d.dates[0])[:10]]
    next_date = str(frame.date.unique()[len(d.dates)])
    panel = frame[frame.date <= next_date]
    news_panel = news[news.date <= next_date]
    s = {
        "date": next_date,
        "weights": first["target_weights"],
        "portfolio_value": 1.0,
        "peak_value": 1.0,
        "max_drawdown": 0.0,
        "days_since_rebalance": 0,
    }
    second = restarted.optimize(
        panel.to_dict("records"), news_panel.to_dict("records"), s
    )
    assert second["episode_elapsed_bars"] == 1 and second["rebalance_phase"] == 0.2
    again = serving.InvestmentService(result)
    assert (
        again.optimize(panel.to_dict("records"), news_panel.to_dict("records"), s)
        == second
    )
    stopped = again.optimize(
        panel.to_dict("records"),
        news_panel.to_dict("records"),
        {**s, "max_drawdown": 0.2, "portfolio_value": 0.8},
    )
    assert stopped["cash_weight"] == 1 and stopped["episode_elapsed_bars"] == 1
    recovered = serving.InvestmentService(result)
    still_stopped = recovered.optimize(
        panel.to_dict("records"), news_panel.to_dict("records"), s
    )
    assert (
        still_stopped["cash_weight"] == 1 and still_stopped["episode_elapsed_bars"] == 1
    )


def test_real_resume_keeps_external_input_root_and_rejects_tampering(
    tmp_path, monkeypatch
):
    external = tmp_path / "external"
    (external / "data").mkdir(parents=True)
    frame, news = inputs(np.zeros((30, 2)))
    frame.to_parquet(external / "data/market.parquet", index=False)
    news.to_parquet(external / "data/news.parquet", index=False)
    c = {
        **small_config("ppo"),
        "data_path": "data/market.parquet",
        "news_path": "data/news.parquet",
        "macro_columns": [],
    }
    d, _ = load_data(c, external)
    source = external / "models/source"
    training.train_model(d, c, 19, 42, source, run_spec=specification(d, "ppo"))
    write_json(source / "installed_inputs.json", {"relative_root": "../.."})
    wrong = tmp_path / "wrong"
    (wrong / "data").mkdir(parents=True)
    frame.assign(log_return=0.5).to_parquet(wrong / "data/market.parquet", index=False)
    news.to_parquet(wrong / "data/news.parquet", index=False)
    monkeypatch.setattr(train, "ROOT", wrong)
    monkeypatch.setattr(model_runs, "ROOT", wrong)
    monkeypatch.setattr(serving, "ROOT", wrong)
    result = tmp_path / "separate/results/resumed"
    train.main(["--resume", str(source), "--output", str(result), "--extra-steps", "8"])
    assert bundle_input_root(result, wrong) == external.resolve()
    assert (
        json.loads((result / "installed_inputs.json").read_text())["relative_root"]
        != "../.."
    )
    assert model_runs.evaluate_bundle(result)
    obj = serving.InvestmentService(result)
    assert np.all(obj.data.returns == 0)
    assert obj.optimize()["episode_elapsed_bars"] == 0
    frame.assign(log_return=0.25).to_parquet(
        external / "data/market.parquet", index=False
    )
    with pytest.raises(ValueError, match="checksum"):
        model_runs.evaluate_bundle(result)
    with pytest.raises(ValueError, match="checksum"):
        serving.InvestmentService(result)


def test_unverified_phase_bundle_is_rejected(tmp_path):
    c = {**small_config("ppo"), "data_path": None, "news_path": None}
    d = data(np.zeros((30, 2)))
    training.train_model(d, c, 19, 42, tmp_path / "old")
    path = tmp_path / "old/schema.json"
    schema = json.loads(path.read_text())
    schema.pop("phase_definition")
    write_json(path, schema)
    with pytest.raises(ValueError, match="phase definition"):
        training.load_bundle(path.parent)
    with pytest.raises(ValueError, match="phase definition"):
        training.train_model(d, c, 19, 42, tmp_path / "new", path.parent)


def test_env_runtime_used_by_comparison_and_packager(tmp_path):
    env = {**os.environ, "INVESTMENT_AI_HOME": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from backend.investment_ai.model_comparison import ROOT; "
                "from scripts.build_trained_models import RUNTIME; "
                "print(ROOT); print(RUNTIME)"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines() == [str(tmp_path.resolve())] * 2


def test_missing_comparison_directory_preserves_outputs(tmp_path):
    from backend.investment_ai.model_comparison import compare_completed

    output = tmp_path / "comparison.csv"
    output.write_text("keep")
    with pytest.raises(FileNotFoundError):
        compare_completed(tmp_path / "missing", output)
    assert output.read_text() == "keep"


def make_directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable; covered by Linux CI")
        raise


def remove_directory_link(link):
    link.unlink()


def test_symlink_inputs_preserve_archive_names(packages, tmp_path):  # noqa: F811
    import zipfile

    runtime = packages / "runtime"
    alias = runtime / "data"
    prepared = runtime / "prepared"
    assert alias.resolve().parent == runtime.resolve()
    alias.rename(prepared)
    contents = (prepared / "market.parquet").read_bytes()
    make_directory_link(alias, prepared)
    try:
        builder.build(
            seed=42,
            selections={
                kind: str(runtime / "models" / kind / "seed42")
                for kind in builder.KINDS
            },
        )
        with zipfile.ZipFile(packages / "shared/trained_models.zip") as archive:
            assert "data/market.parquet" in archive.namelist()
            assert "prepared/market.parquet" not in archive.namelist()
        install(tmp_path / "installed", packages / "shared")
        assert (tmp_path / "installed/data/market.parquet").read_bytes() == contents
    finally:
        remove_directory_link(alias)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "market.parquet").write_bytes(contents)
    (outside / "news.parquet").write_bytes(b"news")
    make_directory_link(alias, outside)
    try:
        with pytest.raises(ValueError, match="runtime"):
            builder.build(seed=42)
    finally:
        remove_directory_link(alias)


def test_api_import_keeps_optional_analysis_dependencies_separate(tmp_path):
    code = (
        "import sys; import backend.main; "
        "assert 'shap' not in sys.modules; "
        "assert 'statsmodels' not in sys.modules; print('API import OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "INVESTMENT_AI_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "API import OK" in result.stdout


def test_archive_alias_and_boundary_without_os_link_privilege(
    request, tmp_path, monkeypatch
):  # noqa: F811
    from pathlib import Path
    import zipfile

    package_root = request.getfixturevalue("packages")
    runtime = package_root / "runtime"
    alias = runtime / "data/market.parquet"
    prepared = runtime / "data/prepared/market.parquet"
    prepared.parent.mkdir()
    prepared.write_bytes(alias.read_bytes())
    original = Path.resolve
    target = prepared

    def resolve(path, *args, **kwargs):
        if path == alias:
            return original(target, *args, **kwargs)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    builder.build(seed=42)
    with zipfile.ZipFile(package_root / "shared/trained_models.zip") as archive:
        assert "data/market.parquet" in archive.namelist()
        assert "data/prepared/market.parquet" not in archive.namelist()
    install(tmp_path / "installed", package_root / "shared")
    assert (
        tmp_path / "installed/data/market.parquet"
    ).read_bytes() == alias.read_bytes()
    target = tmp_path / "outside.parquet"
    with pytest.raises(ValueError, match="runtime"):
        builder.build(seed=42)


def test_cross_drive_style_binding_requires_valid_hashes(tmp_path, monkeypatch):
    from backend.investment_ai import bundle_inputs

    root = tmp_path / "external"
    (root / "data").mkdir(parents=True)
    (root / "data/market.parquet").write_bytes(b"market")
    source = root / "models/source"
    source.mkdir(parents=True)
    write_json(source / "installed_inputs.json", {"relative_root": "../.."})
    output = tmp_path / "other_drive"
    output.mkdir()
    c = {"data_path": "data/market.parquet", "news_path": None}
    write_json(output / "config.json", c)

    def different_drive(*args):
        raise ValueError("different Windows drives")

    monkeypatch.setattr(bundle_inputs.os.path, "relpath", different_drive)
    bundle_inputs.preserve_inputs(output, None, c, None, source)
    assert bundle_input_root(output) == root.resolve()
    marker = output / "installed_inputs.json"
    contents = json.loads(marker.read_text())
    assert contents["absolute_root"] == str(root.resolve())
    contents.pop("sha256")
    write_json(marker, contents)
    with pytest.raises(ValueError, match="absolute input"):
        bundle_input_root(output)
