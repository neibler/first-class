import json
from copy import deepcopy
import numpy as np
import pytest
from backend.investment_ai import training, train
from backend.investment_ai.checkpoints import publish_checkpoint, resolve_checkpoint
from backend.investment_ai.model_factory import validate_model_config
from backend.investment_ai.rebalancing import REBALANCE_POLICY
from backend.investment_ai.infer import predict_weights
from backend.investment_ai.settings import effective_mode, write_json
from tests.test_contract_v3 import config, data, state, service
from tests.test_model_support import small_config
from tests.test_shared_models_installer import package
from scripts.install_trained_models import install
from frontend.shap_presentation import select_factors
from frontend.weight_format import format_weight


@pytest.mark.parametrize("kind", ["sac", "td3"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("gradient_steps", 1.5),
        ("gradient_steps", True),
        ("gradient_steps", 0),
        ("learning_starts", -1),
        ("learning_starts", 1.5),
        ("learning_starts", False),
    ],
)
def test_offpolicy_types(kind, key, value):
    c = small_config(kind)
    c["models"][kind][key] = value
    with pytest.raises(ValueError, match=key):
        validate_model_config(c)


class SavingModel:
    num_timesteps = 8
    fail = False

    def save(self, path):
        path.write_bytes(str(self.num_timesteps).encode())

    def save_replay_buffer(self, path):
        path.write_bytes(b"partial")
        if self.fail:
            raise OSError("interrupted")
        path.write_bytes(str(self.num_timesteps).encode())


def test_atomic_checkpoint_survives_partial_buffer(tmp_path):
    m = SavingModel()
    publish_checkpoint(m, tmp_path, {"model_type": "sac", "actual_timesteps": 8})
    previous, meta = resolve_checkpoint(tmp_path)
    m.num_timesteps = 16
    m.fail = True
    with pytest.raises(OSError):
        publish_checkpoint(m, tmp_path, {"model_type": "sac", "actual_timesteps": 16})
    current, meta = resolve_checkpoint(tmp_path)
    assert current == previous and meta["actual_timesteps"] == 8
    assert (current / "model.zip").read_bytes() == (
        current / "replay_buffer.pkl"
    ).read_bytes()


def test_policy_rejected_without_data_before_model_load(tmp_path, monkeypatch):
    write_json(tmp_path / "config.json", config())
    write_json(tmp_path / "schema.json", {"version": 3})
    monkeypatch.setattr(
        training,
        "load_model",
        lambda *a, **k: pytest.fail("loaded incompatible policy"),
    )
    with pytest.raises(ValueError, match="execution policy"):
        training.load_bundle(tmp_path)


@pytest.mark.parametrize("kind", ["ppo", "recurrent_ppo"])
def test_resume_defaults_subset_seed_device(tmp_path, monkeypatch, kind):
    c = small_config(kind)
    c.update(
        n_steps=256, batch_size=64, checkpoint_steps=512, total_timesteps=512, seed=22
    )
    b = tmp_path / "saved"
    b.mkdir()
    write_json(b / "config.json", c)
    d = data(np.zeros((20, 2)))
    spec = {
        "start_date": str(d.dates[2])[:10],
        "end_date": str(d.dates[15])[:10],
        "test_start_index": 10,
        "data_seed": 17,
    }
    write_json(b / "run_spec.json", spec)
    monkeypatch.setattr(
        "backend.investment_ai.checkpoints.resolve_checkpoint",
        lambda p: (b, {"train_end_index": 9, "seed": 22}),
    )
    seen = {}

    def load(c, root):
        assert c["seed"] == 17
        return d, None

    def learn(local, cfg, end, seed, output, resume, run_spec=None):
        seen.update(cfg)
        assert len(local.dates) == 14 and local.dates[0] == d.dates[2]
        assert end == 9 and seed == 22 and run_spec == spec

    monkeypatch.setattr(train, "load_data", load)
    monkeypatch.setattr(train, "train_model", learn)
    train.main(
        ["--resume", str(b), "--output", str(tmp_path / "new"), "--device", "cpu"]
    )
    assert seen["total_timesteps"] == 5120 and seen["device"] == "cpu"


@pytest.mark.parametrize("missing", ["model_type", "include_risk_score"])
def test_resume_missing_effective_key_rejected_before_mkdir(tmp_path, missing):
    c = small_config("sac")
    c["include_risk_score"] = False
    b = tmp_path / "old"
    b.mkdir()
    write_json(b / "config.json", c)
    write_json(b / "schema.json", {"execution_policy": REBALANCE_POLICY})
    m = SavingModel()
    publish_checkpoint(m, b, {"model_type": "sac"})
    changed = dict(c)
    changed.pop(missing)
    with pytest.raises(ValueError, match="configuration"):
        training.train_model(data(), changed, 2, 42, tmp_path / "new", b)
    assert not (tmp_path / "new").exists()


def test_recurrent_empty_predictors_and_safeguard_restart(tmp_path):
    d = data()
    c = {**config(), "model_type": "recurrent_ppo"}
    with pytest.raises(ValueError, match="PredictionSession"):
        predict_weights(d, c, [], None, predictors=[])
    obj = service(tmp_path)
    obj.config = c
    obj.recurrent_sessions = {}
    obj.latches = {"default": True}
    s = state(d, mdd=0.2)
    first = obj.optimize(state=s)
    second = obj.optimize(state=s)
    assert first == second
    assert first["cash_weight"] == 1 and first["safeguard_triggered"]
    assert not obj.recurrent_sessions


def test_near_one_holdings_distance_bounded():
    d = data()
    s = state(d, mdd=0)
    s.update(portfolio_value=1.0, peak_value=1.0)
    s["weights"] = {"SPY": 1.000005, "QQQ": 0.0}

    class Session:
        def predict(self, *args):
            return np.array([0.0, 1.0])

    result = predict_weights(d, config(), [], None, s, [Session()])
    assert result["portfolio_weight_distance"] == pytest.approx(1.0)


def test_shap_baseline_not_holdings_and_balanced_hidden():
    factors = [
        dict(key="negative", absolute_sum=0.1, direction="decrease"),
        dict(key="balanced", absolute_sum=0.8, direction="balanced"),
    ]
    before = deepcopy(factors)
    for direction in ["increase", "decrease", "hold", "unknown"]:
        primary, _ = select_factors(factors, direction)
        assert [f["key"] for f in primary] == ["negative"]
    assert factors == before


def test_install_parent_conflict_no_partial_write(tmp_path):
    shared = package(tmp_path, {"first.txt": b"first", "blocked/file.txt": b"second"})
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "blocked").write_text("file")
    with pytest.raises(ValueError, match="parent"):
        install(dest, shared)
    assert not (dest / "first.txt").exists()


@pytest.mark.parametrize(
    "value,expected",
    [(0.104, "10%"), (0.105, "11%"), (0.0999, "10%"), (0.0, "0%"), (1.0, "100%")],
)
def test_weight_rounding(value, expected):
    assert format_weight(value) == expected


def test_shared_mode_resolution():
    assert effective_mode({"model_type": "ppo"}) == "single"
    assert effective_mode({}) == "legacy"
    with pytest.raises(ValueError):
        effective_mode({"mode": "typo"})


@pytest.mark.parametrize("kind", ["ppo", "recurrent_ppo", "sac", "td3"])
def test_short_training_and_resume_roundtrip(tmp_path, kind):
    c = small_config(kind)
    d = data(np.zeros((20, 2)))
    first = tmp_path / "first"
    training.train_model(d, c, 15, 42, first)
    folder, manifest = resolve_checkpoint(first)
    assert manifest["completed"] and manifest["actual_timesteps"] == 16
    assert manifest["execution_policy"] == REBALANCE_POLICY
    second = tmp_path / "second"
    training.train_model(d, c, 15, 42, second, first)
    _, resumed = resolve_checkpoint(second)
    assert resumed["actual_timesteps"] == 32
    training.load_bundle(second, d, exact_data=True)


def test_package_fingerprint_detects_changed_data(tmp_path, monkeypatch):
    from scripts import build_trained_models as builder
    from backend.investment_ai import data_loader
    from backend.investment_ai.data_loader import fingerprint

    d = data()
    monkeypatch.setattr(data_loader, "load_data", lambda *a: (d, None))
    spec = {"start_date": str(d.dates[0])[:10], "end_date": str(d.dates[-1])[:10]}
    expected = fingerprint(d)
    builder.verify_inputs({"seed": 42}, spec, expected)
    d.risk[0, 0] += 0.01
    with pytest.raises(ValueError, match="fingerprint"):
        builder.verify_inputs({"seed": 42}, spec, expected)


def test_normalized_input_path_and_custom_destination(tmp_path):
    from backend.investment_ai.settings import bundle_input_root

    files = {
        "models/ppo/config.json": json.dumps(
            {"data_path": "./data/market.parquet", "news_path": "./data/news.parquet"}
        ).encode(),
        "data/market.parquet": b"market",
        "data/news.parquet": b"news",
    }
    shared = package(tmp_path, files)
    manifest_path = shared / "trained_models_manifest.json"
    meta = json.loads(manifest_path.read_text())
    meta["models"] = {"ppo": {"path": "models/ppo"}}
    write_json(manifest_path, meta)
    dest = tmp_path / "custom"
    install(dest, shared)
    assert bundle_input_root(dest / "models/ppo") == dest


def test_fresh_training_input_archive(tmp_path):
    from scripts.install_training_inputs import install as install_inputs
    from backend.investment_ai.settings import read_config
    from backend.investment_ai.data_loader import load_data
    from backend.investment_ai.model_runs import plan_runs

    install_inputs(tmp_path)
    c = read_config("config/multi_model.json")
    d, _ = load_data(c, tmp_path)
    assert len(plan_runs(c, d, [22], True)) == 3
    assert len(d.macros) == 13


def test_shap_uses_data_seed_and_exact_saved_subset(tmp_path, monkeypatch):
    from backend.performance import model_shap
    from backend.investment_ai.data_loader import fingerprint

    c = small_config("ppo")
    c["seed"] = 22
    d = data(np.zeros((20, 2)))
    b = tmp_path / "model"
    spec = dict(
        start_date=str(d.dates[0])[:10],
        end_date=str(d.dates[-1])[:10],
        test_start_index=16,
        data_seed=17,
        model_type="ppo",
        data_hash=fingerprint(d),
    )
    training.train_model(d, c, 15, 22, b, run_spec=spec)

    def load(cfg, root):
        assert cfg["seed"] == 17
        return d, None

    monkeypatch.setattr(model_shap, "load_data", load)
    path = model_shap.explain_bundle(b, sample=1, permutations=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["data_hash"] == fingerprint(d)
    assert payload["model_type"] == "ppo"


def test_bundle_infer_writes_output(tmp_path, monkeypatch):
    from backend.investment_ai import infer, serving

    class Service:
        def __init__(self, path):
            pass

        def optimize(self, **kwargs):
            return dict(
                as_of="2025-01-01",
                target_weights={"SPY": 1.0},
                status="target_proposal",
            )

    monkeypatch.setattr(serving, "InvestmentService", Service)
    output = tmp_path / "result"
    infer.main(["--bundle", "fixture", "--output", str(output)])
    assert (output / "latest_weights.json").is_file()
    assert (output / "latest_weights.csv").is_file()


def test_preflight_effective_years_and_default_mode(tmp_path, monkeypatch):
    import sys
    from backend.investment_ai import preflight
    from backend.investment_ai.data_loader import fingerprint

    d = data(np.zeros((20, 2)))
    c = small_config("ppo")
    c.pop("mode", None)
    monkeypatch.setattr(preflight, "read_config", lambda *a: c)
    monkeypatch.setattr(preflight, "load_data", lambda *a: (d, None))
    monkeypatch.setattr(preflight, "check_env", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    called = []

    def plan(cfg, panel, seeds, wf, years, test):
        called.append((wf, years, test))
        return [dict(start=0, split=16, stop=20)]

    monkeypatch.setattr(preflight, "plan_runs", plan)
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight", "--config", "fixture", "--train-years", "2", "--test-years", "1"],
    )
    preflight.main()
    assert called == [(False, 2, 1)]
    assert json.loads((tmp_path / "preflight.json").read_text())[
        "fingerprint"
    ] == fingerprint(d)


def test_deployment_correlation_uses_installed_training_inputs(tmp_path, monkeypatch):
    from scripts.install_training_inputs import install as install_inputs
    from backend.investment_ai import settings
    from backend.performance.correlation import (
        load_correlation_data,
        analyze_correlations,
    )

    install_inputs(tmp_path)
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    panel = load_correlation_data()
    result = analyze_correlations(panel, "SPY", window=60)
    assert result["status"] == "ok" and result["relationships"]
