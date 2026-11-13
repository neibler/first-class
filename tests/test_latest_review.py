"""Regression coverage for the follow-up review (November 13)."""

import json
from pathlib import Path

import httpx
import pytest

from backend.investment_ai import checkpoints, preflight, train
from frontend.api_client import APIError
from frontend.shap_client import load_model_shap
from frontend.shap_state import rsi_summary
from scripts import build_trained_models as builder
from scripts import install_training_inputs as inputs
from tests.test_contract_v3 import config, data, service, state
from tests.test_model_shap import payload
from tests.test_review_model_sharing import packages  # noqa: F401
from tests.test_review_regressions import SavingModel


@pytest.mark.parametrize("kind", ["ppo", "recurrent_ppo", "sac", "td3"])
def test_retention_recovery_and_failed_save(tmp_path, kind):
    model = SavingModel()
    for step in range(1, 6):
        model.num_timesteps = step
        checkpoints.publish_checkpoint(
            model, tmp_path, {"model_type": kind, "step": step}
        )
        folder, manifest = checkpoints.resolve_checkpoint(tmp_path)
        assert manifest["step"] == step
        assert (folder / "model.zip").read_bytes() == str(step).encode()
        assert len(list((tmp_path / "checkpoint_generations").iterdir())) == min(
            step, 2
        )
    pointer = json.loads((tmp_path / "checkpoint.json").read_text())
    previous = tmp_path / pointer["previous"]["generation"]
    assert previous.exists() and folder.exists()
    # The previous intact model+buffer pair remains recoverable after corruption.
    (folder / "model.zip").write_bytes(b"broken")
    recovered, meta = checkpoints.resolve_checkpoint(tmp_path)
    assert recovered == previous and meta["step"] == 4
    if kind in {"sac", "td3"}:
        assert (recovered / "replay_buffer.pkl").read_bytes() == b"4"
        model.fail = True
        with pytest.raises(OSError):
            checkpoints.publish_checkpoint(model, tmp_path, {"model_type": kind})
        assert checkpoints.resolve_checkpoint(tmp_path)[0] == previous
    model.fail = False
    model.num_timesteps = 6
    checkpoints.publish_checkpoint(model, tmp_path, {"model_type": kind, "step": 6})
    assert checkpoints.resolve_checkpoint(tmp_path)[1]["step"] == 6
    assert previous.exists()
    assert len(list((tmp_path / "checkpoint_generations").iterdir())) == 2


def test_checkpoint_publication_rejects_concurrent_writer(tmp_path):
    with checkpoints.publication_lock(tmp_path):
        with pytest.raises(OSError):
            checkpoints.publish_checkpoint(
                SavingModel(), tmp_path, {"model_type": "ppo"}
            )
    assert not (tmp_path / "checkpoint.json").exists()


def test_packaging_excludes_completed_checkpoint_manifests(packages):  # noqa: F811
    for kind in builder.KINDS:
        root = packages / "runtime/models" / kind / "seed42"
        for parent in ("checkpoint_generations/100_abcd", "checkpoints/step100"):
            folder = root / parent
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text((root / "manifest.json").read_text())
    builder.build(seed=42)
    import zipfile

    with zipfile.ZipFile(packages / "shared/trained_models.zip") as archive:
        assert not any("checkpoint" in name for name in archive.namelist())


@pytest.mark.parametrize(
    "option,value", [("--train-years", "4"), ("--test-years", "1")]
)
def test_resume_rejects_explicit_period_options(tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        train.main(
            [
                "--resume",
                str(tmp_path),
                "--output",
                str(tmp_path / "new"),
                option,
                value,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


def test_preflight_rejects_legacy_multi_seed_before_loading(monkeypatch):
    cfg = {
        **config(),
        "mode": "legacy",
        "model_type": "recurrent_ppo",
        "seeds": [17, 22],
    }
    monkeypatch.setattr(preflight, "read_config", lambda _: cfg)
    monkeypatch.setattr("sys.argv", ["preflight", "--config", "unused"])
    monkeypatch.setattr(
        preflight, "load_data", lambda *a: pytest.fail("loaded before validation")
    )
    with pytest.raises(ValueError, match="exactly one seed"):
        preflight.main()


@pytest.mark.parametrize("flag", ["stopped", "safeguard_triggered", "reset_episode"])
@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None, [], {}])
def test_service_rejects_non_boolean_flags(tmp_path, flag, invalid):
    obj = service(tmp_path)
    supplied = {**state(data(), mdd=0.2), flag: invalid}
    with pytest.raises(ValueError, match="must be a boolean"):
        obj.optimize(state=supplied)
    assert obj.latches == {} and not obj.latch_path.exists()


@pytest.mark.parametrize("assets", [[], [1], "SPY", 1, None, {}])
def test_shap_assets_must_be_nonempty_mapping(monkeypatch, assets):
    record = payload()
    record["assets"] = assets
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: httpx.Response(
            200, json=record, request=httpx.Request("GET", "http://test")
        ),
    )
    with pytest.raises(APIError):
        load_model_shap("ppo")


@pytest.mark.parametrize(
    "value,display,description",
    [
        (29.99, "30", "비교적 낮은"),
        (44.99, "45", "중립"),
        (55.01, "55", "중립"),
        (70.01, "70", "비교적 높은"),
        (70.06, "70.1", "과매수"),
        (29.94, "29.9", "과매도"),
    ],
)
def test_rsi_rounding_and_classification_agree(value, display, description):
    result = rsi_summary(value)
    assert f"RSI는 {display}로" in result and description in result


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_input_install_failure_leaves_no_final_partial_files(
    tmp_path, monkeypatch, failure
):
    destination = tmp_path / "runtime"
    if failure == "write":
        monkeypatch.setattr(
            inputs.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk full"))
        )
    else:
        original = Path.replace
        count = 0

        def fail_second(self, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError):
        inputs.install(destination)
    assert not [p for p in destination.rglob("*") if p.is_file()]


def test_installed_data_matches_all_training_configs(tmp_path):
    import hashlib
    import pandas as pd

    inputs.install(tmp_path)
    manifest = json.loads(Path("shared/training_inputs_manifest.json").read_text())
    for filename in (
        "investment_ai",
        "multi_model",
        "multi_model_no_risk",
        "final_2026",
    ):
        cfg = json.loads(Path(f"config/{filename}.json").read_text())
        for key in ("data_path", "news_path"):
            path = cfg[key]
            assert (
                hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
                == manifest["files"][path]
            )
        market = pd.read_parquet(tmp_path / cfg["data_path"])
        assert str(market.date.min())[:10] == "2019-02-05"
        assert str(market.date.max())[:10] == "2026-09-30"
        assert len(cfg["macro_columns"]) == 13
    assert "2019-02-05~2026-09-30" in Path("docs/CORRELATION.md").read_text(
        encoding="utf-8"
    )


def test_correlation_error_keeps_footer_and_retry(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import api_client, correlation_client

    calls = []

    def unavailable(*a):
        calls.append(a)
        raise APIError("test API unavailable")

    monkeypatch.setattr(correlation_client, "load_correlations", unavailable)
    monkeypatch.setattr(
        api_client,
        "get_health",
        lambda: api_client.HealthStatus(status="ok", model_loaded=False),
    )
    page = AppTest.from_file("frontend/app.py", default_timeout=20).run()
    page.switch_page("app_pages/correlation.py").run()
    assert not page.exception
    assert any("교육용 데모" in item.value for item in page.caption)
    assert page.button(key="refresh_health")
    assert page.error and not page.metric
    page.button(key="retry_correlation").click().run()
    assert len(calls) >= 2 and not page.exception


def test_empty_shap_summary_and_unmatched_replay_date(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import shap_client

    record = payload()
    for asset in record["assets"].values():
        for row in asset["decisions"]:
            row["shap_values"] = [0.0, 0.0]
            row["final_value"] = row["base_value"]
    monkeypatch.setattr(shap_client, "load_model_shap", lambda _: record)
    page = AppTest.from_file("frontend/app_pages/shap.py", default_timeout=20).run()
    assert not page.exception
    assert any("상세 분석 보기" in item.value for item in page.info)
    page.button(key="shap_details_button").click().run()
    assert not page.exception and page.dataframe
    captions = " ".join(item.value for item in page.caption)
    assert "Replay 기준 날짜" in captions and "평가 기록 시점" not in captions


def test_docker_weight_formatter_and_allocation():
    from frontend.charts import allocation

    assert "!frontend/weight_format.py" in Path(".dockerignore").read_text()
    chart = allocation([{"ticker": "SPY", "weight": 0.105}]).to_dict()
    assert chart["layer"][0]["data"]["values"][0]["weight_label"] == "11%"


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_model_install_io_failure_preserves_existing_files(
    tmp_path, monkeypatch, failure
):
    from scripts import install_trained_models as installer
    from tests.test_shared_models_installer import package

    shared = package(
        tmp_path, {"first": b"first", "second": b"second", "existing": b"keep"}
    )
    dest = tmp_path / "runtime"
    dest.mkdir()
    (dest / "existing").write_bytes(b"keep")
    if failure == "write":
        monkeypatch.setattr(
            installer.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk full"))
        )
    else:
        original = Path.replace
        count = 0

        def fail_second(self, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("rename failed")
            return original(self, target)

        monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(OSError):
        installer.install(dest, shared)
    assert list(dest.iterdir()) == [dest / "existing"]
    assert (dest / "existing").read_bytes() == b"keep"


def test_checkpoint_pointer_failure_preserves_previous_pair(tmp_path, monkeypatch):
    model = SavingModel()
    checkpoints.publish_checkpoint(model, tmp_path, {"model_type": "sac"})
    old = checkpoints.resolve_checkpoint(tmp_path)[0]
    original = Path.replace

    def fail_pointer(self, target):
        if target.name == "checkpoint.json":
            raise OSError("pointer write failed")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", fail_pointer)
    model.num_timesteps += 1
    with pytest.raises(OSError):
        checkpoints.publish_checkpoint(model, tmp_path, {"model_type": "sac"})
    assert checkpoints.resolve_checkpoint(tmp_path)[0] == old
    assert list((tmp_path / "checkpoint_generations").iterdir()) == [old]
