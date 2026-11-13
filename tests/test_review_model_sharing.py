import json

import pytest

from backend.investment_ai import infer, serving
from backend.investment_ai.model_comparison import compare_completed
from backend.investment_ai.settings import bundle_input_root
from scripts import build_trained_models as builder
from scripts.install_trained_models import install


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def packages(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "verify_inputs", lambda *args: None)
    root = tmp_path / "source"
    runtime = root / "runtime"
    monkeypatch.setattr(builder, "PROJECT", root)
    monkeypatch.setattr(builder, "RUNTIME", runtime)
    for kind in builder.KINDS:
        for seed in (42, 73):
            bundle = runtime / "models" / kind / f"seed{seed}"
            common = dict(model_type=kind, seed=seed, data_hash="same")
            spec = dict(
                common,
                start_date="2021-01-04",
                end_date="2025-12-31",
                test_start_index=1004,
            )
            for name, value in {
                "config.json": dict(
                    model_type=kind,
                    data_path="data/market.parquet",
                    news_path="data/news.parquet",
                ),
                "manifest.json": dict(
                    common,
                    completed=True,
                    actual_timesteps=100000,
                    train_start="2021-01-04",
                    train_end="2024-12-30",
                ),
                "run_spec.json": spec,
                "evaluation.json": dict(
                    spec,
                    completed=True,
                    test_start="2024-12-31",
                    test_end="2025-12-31",
                    comparison_group="same",
                ),
                "schema.json": {},
                "scaler.json": {},
            }.items():
                write(bundle / name, value)
            (bundle / "model.zip").write_bytes(b"test-model")
            (bundle / "evaluation").mkdir()
            (bundle / "evaluation/ppo_history.csv").write_text(
                "date,portfolio_return\n2024-12-31,0\n2025-12-31,0\n"
            )
    (runtime / "data").mkdir()
    (runtime / "data/market.parquet").write_bytes(b"market")
    (runtime / "data/news.parquet").write_bytes(b"news")
    return root


def test_multi_seed_requires_selection_and_installs_portably(packages, tmp_path):
    with pytest.raises(ValueError, match="--seed"):
        builder.build()
    builder.build(seed=42)
    destination = tmp_path / "custom"
    manifest = install(destination, packages / "shared")
    install(destination, packages / "shared")
    moved = tmp_path / "relocated"
    destination.rename(moved)
    for model in manifest["models"].values():
        bundle = moved / model["path"]
        assert bundle_input_root(bundle) == moved
        config = json.loads((bundle / "config.json").read_text())
        assert (
            bundle_input_root(bundle) / config["data_path"]
        ).read_bytes() == b"market"
        assert json.loads((bundle / "manifest.json").read_text())["seed"] == 42


@pytest.mark.parametrize(
    "field,value",
    [
        ("comparison_group", "different"),
        ("test_start", "2025-01-01"),
        ("completed", False),
    ],
)
def test_mixed_evaluation_rejected_without_replacing_archive(packages, field, value):
    builder.build(seed=42)
    archive = packages / "shared/trained_models.zip"
    previous = archive.read_bytes()
    path = packages / "runtime/models/td3/seed42/evaluation.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    write(path, meta)
    with pytest.raises(ValueError):
        builder.build(seed=42)
    assert archive.read_bytes() == previous


def test_explicit_bundle_selection(packages):
    builder.build(selections={kind: f"models/{kind}/seed73" for kind in builder.KINDS})
    manifest = json.loads(
        (packages / "shared/trained_models_manifest.json").read_text()
    )
    assert all(v["path"].endswith("seed73") for v in manifest["models"].values())


def test_missing_bundled_input_rejected_before_install(packages, tmp_path):
    # Builder rejects a missing file before producing an archive.
    (packages / "runtime/data/news.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        builder.build(seed=42)
    assert not (packages / "shared/trained_models.zip").exists()


def test_empty_comparison_preserves_existing_outputs(tmp_path):
    output = tmp_path / "comparison.csv"
    summary = tmp_path / "comparison_seed_summary.csv"
    output.write_text("old")
    summary.write_text("old")
    unrelated = tmp_path / "model.zip"
    unrelated.write_bytes(b"keep")
    assert compare_completed(tmp_path, output).empty
    assert output.read_text() == "old" and summary.read_text() == "old"
    assert unrelated.read_bytes() == b"keep"


def test_missing_model_errors_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="--activate"):
        infer.resolve_run()
    with pytest.raises(FileNotFoundError, match="Selected model directory"):
        serving.InvestmentService(tmp_path / "missing")


@pytest.mark.parametrize(
    "kind,seeds,state,allowed",
    [
        ("recurrent_ppo", [42], False, True),
        ("recurrent_ppo", [42, 73], False, False),
        ("recurrent_ppo", [42], True, False),
        ("ppo", [42, 73], False, True),
    ],
)
def test_legacy_cli_recurrent_session(
    tmp_path, monkeypatch, kind, seeds, state, allowed
):
    write(tmp_path / "selection.json", {"seeds": seeds})
    for seed in seeds:
        write(tmp_path / f"final/seed_{seed}/config.json", {"model_type": kind})
        write(tmp_path / f"final/seed_{seed}/schema.json", {"macros": []})
    monkeypatch.setattr(infer, "load_data", lambda *a: (None, None))
    monkeypatch.setattr(infer, "load_bundle", lambda *a: (object(), None))
    received = []

    def predict(*args, **kwargs):
        received.append(kwargs["predictors"])
        return {}

    monkeypatch.setattr(infer, "predict_weights", predict)
    monkeypatch.setattr(infer, "save_prediction", lambda *a: None)
    args = ["--run", str(tmp_path)]
    if state:
        write(tmp_path / "state.json", {})
        args += ["--state", str(tmp_path / "state.json")]
    if not allowed:
        with pytest.raises(ValueError, match="fresh cash episode"):
            infer.main(args)
        assert not received
    else:
        infer.main(args)
        if kind == "recurrent_ppo":
            assert received[0][0].model_type == kind
            assert received[0][0].episode_start
        else:
            assert received == [None]


@pytest.mark.parametrize("input_path", ["data/missing.parquet", "../outside.parquet"])
def test_installer_checks_model_input_paths_before_writes(
    packages, tmp_path, input_path
):
    import hashlib
    import zipfile

    builder.build(seed=42)
    archive = packages / "shared/trained_models.zip"
    with zipfile.ZipFile(archive) as z:
        payload = {name: z.read(name) for name in z.namelist()}
    name = "models/ppo/seed42/config.json"
    config = json.loads(payload[name])
    config["data_path"] = input_path
    payload[name] = json.dumps(config).encode()
    with zipfile.ZipFile(archive, "w") as z:
        for name, content in payload.items():
            z.writestr(name, content)
    manifest_path = packages / "shared/trained_models_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest["files"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in payload.items()
    }
    write(manifest_path, manifest)
    target = tmp_path / "invalid_install"
    with pytest.raises(ValueError, match="Missing or unsafe bundled input"):
        install(target, packages / "shared")
    assert not target.exists()


def test_explicit_single_model_excludes_other_models(packages, tmp_path):
    builder.build(selections={"ppo": "models/ppo/seed73"})
    manifest = install(tmp_path / "ppo_only", packages / "shared")
    assert set(manifest["models"]) == {"ppo"}
    assert manifest["models"]["ppo"]["path"] == "models/ppo/seed73"
    assert all(
        not name.startswith(("models/td3/", "models/sac/", "models/recurrent_ppo/"))
        for name in manifest["files"]
    )
