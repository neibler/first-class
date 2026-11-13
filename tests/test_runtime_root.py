"""Direct startup and installer activation share one portable runtime root."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from scripts import install_trained_models as installer

PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = PROJECT / "backend/investment_ai/settings.py"


@pytest.mark.parametrize(
    "configured", [None, "relative-runtime", "~/portfolio-runtime", "absolute"]
)
def test_fresh_settings_preserve_explicit_roots(tmp_path, monkeypatch, configured):
    monkeypatch.chdir(tmp_path)
    if configured is None:
        monkeypatch.delenv("INVESTMENT_AI_HOME", raising=False)
        expected = PROJECT / "runtime"
    else:
        value = str(tmp_path / "custom") if configured == "absolute" else configured
        monkeypatch.setenv("INVESTMENT_AI_HOME", value)
        expected = Path(value).expanduser().resolve()
    assert runpy.run_path(str(SETTINGS))["ROOT"] == expected


@pytest.mark.parametrize("configured", [None, "deployment-runtime"])
def test_activation_path_resolves_to_installed_bundle(
    tmp_path, monkeypatch, configured
):
    copied_settings = tmp_path / "backend/investment_ai/settings.py"
    copied_settings.parent.mkdir(parents=True)
    copied_settings.write_bytes(SETTINGS.read_bytes())
    (tmp_path / "config").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(installer, "PROJECT", tmp_path)
    if configured is None:
        monkeypatch.delenv("INVESTMENT_AI_HOME", raising=False)
        destination = tmp_path / "runtime"
    else:
        monkeypatch.setenv("INVESTMENT_AI_HOME", configured)
        destination = tmp_path / configured
    relative = "models/ppo/example"
    monkeypatch.setattr(
        installer, "install", lambda path: {"models": {"ppo": {"path": relative}}}
    )
    monkeypatch.setattr(
        sys, "argv", ["install", "--destination", str(destination), "--activate", "ppo"]
    )
    installer.main()
    selected = json.loads((tmp_path / "config/serving.json").read_text())["model_path"]
    assert selected == relative
    root = runpy.run_path(str(copied_settings))["ROOT"]
    assert (root / selected).resolve() == destination / relative


def test_direct_all_shared_uses_installer_runtime_without_wrapper():
    env = os.environ.copy()
    env.pop("INVESTMENT_AI_HOME", None)
    script = """
import json
import sys
from backend.investment_ai.settings import ROOT
from backend.performance import model_shap
paths = []
model_shap.explain_bundle = lambda bundle, *args: paths.append(str(bundle))
sys.argv = ["model_shap", "--all-shared"]
model_shap.main()
print(json.dumps({"root": str(ROOT), "paths": paths}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    payload = json.loads(result.stdout.splitlines()[-1])
    expected = installer.PROJECT / "runtime"
    assert Path(payload["root"]) == expected
    manifest = json.loads(
        (PROJECT / "shared/trained_models_manifest.json").read_text(encoding="utf-8")
    )
    expected_paths = [expected / model["path"] for model in manifest["models"].values()]
    assert expected_paths
    assert [Path(path) for path in payload["paths"]] == expected_paths
    assert all(Path(path).is_relative_to(expected) for path in payload["paths"])
