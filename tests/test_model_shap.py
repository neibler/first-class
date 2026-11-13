from pathlib import Path
import numpy as np
import httpx
import pytest

from backend.performance.model_shap import policy_function
from frontend import shap_client
from frontend.api_client import APIError


def payload(model="ppo"):
    return {
        "model_type": model,
        "run_id": "unit",
        "features": ["rsi", "macro_vix"],
        "method": "SHAP permutation",
        "permutations": 2,
        "background_end": "2024-12-30",
        "note": "실제 결과 테스트",
        "assets": {
            a: {
                "decisions": [
                    {
                        "date": date,
                        "execution_date": "2025-01-03",
                        "base_value": 0.1,
                        "final_value": 0.1 + value,
                        "feature_values_raw": [0.5, 2.0],
                        "shap_values": [value, 0.0],
                    }
                    for date, value in [
                        ("2025-01-02", 0.03 * i),
                        ("2025-01-03", 0.02 * i),
                    ]
                ]
            }
            for i, a in enumerate(["SPY", "QQQ"], 1)
        },
    }


def test_recurrent_perturbations_use_fixed_copied_memory():
    class Model:
        def predict(self, rows, state, episode_start, deterministic):
            assert not episode_start.any()
            result = rows + state[0][0]
            state[0][:] = 999  # The explainer must not let this mutate future calls.
            return result, state

    class Env:
        config = {"min_weight_threshold": 0.03}

        def action_weights(self, row):
            return row / row.sum()

    state = (np.ones((1, 1, 2)), np.ones((1, 1, 2)))
    predict = policy_function(Model(), "recurrent_ppo", Env(), state, False)
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        predict(rows), (rows + 1) / (rows + 1).sum(axis=1, keepdims=True)
    )
    np.testing.assert_allclose(
        predict(rows), (rows + 1) / (rows + 1).sum(axis=1, keepdims=True)
    )
    np.testing.assert_equal(state[0], 1)


def test_client_validates_real_payload(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(
            200, json=payload(), request=httpx.Request("GET", "http://test")
        ),
    )
    assert shap_client.load_model_shap("ppo")["features"] == ["rsi", "macro_vix"]
    with pytest.raises(APIError):
        shap_client.load_model_shap("td3")


def test_client_rejects_nonadditive_explanation(monkeypatch):
    data = payload()
    data["assets"]["SPY"]["decisions"][0]["final_value"] = 0.9
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(
            200, json=data, request=httpx.Request("GET", "http://test")
        ),
    )
    with pytest.raises(APIError):
        shap_client.load_model_shap("ppo")


def test_api_refuses_missing_or_stale_result(tmp_path, monkeypatch):
    import json
    import hashlib
    from fastapi import HTTPException
    from backend.performance.api import model_shap
    from backend.investment_ai import settings

    monkeypatch.setattr(
        "backend.performance.api.SERVING_CONFIG", tmp_path / "serving.json"
    )
    monkeypatch.delenv("INVESTMENT_AI_RUN", raising=False)
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    registry = json.loads(Path("shared/trained_models_manifest.json").read_text())
    bundle = tmp_path / registry["models"]["ppo"]["path"]
    with pytest.raises(HTTPException) as error:
        model_shap("ppo")
    assert error.value.status_code == 503
    bundle.mkdir(parents=True)
    (bundle / "model.zip").write_bytes(b"test")
    result = payload()
    result["model_sha256"] = hashlib.sha256(b"test").hexdigest()
    (bundle / "shap.json").write_text(json.dumps(result), encoding="utf-8")
    assert model_shap("ppo")["model_type"] == "ppo"
    (bundle / "model.zip").write_bytes(b"changed")
    with pytest.raises(HTTPException) as error:
        model_shap("ppo")
    assert error.value.status_code == 409


def test_active_model_shap_overrides_shared_registry(tmp_path, monkeypatch):
    import json
    import hashlib
    from backend.performance import api
    from backend.investment_ai import settings
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.delenv("INVESTMENT_AI_RUN", raising=False)
    selected = tmp_path / "models/ppo/new"
    selected.mkdir(parents=True)
    (selected / "config.json").write_text(json.dumps({"model_type": "ppo"}))
    (selected / "model.zip").write_bytes(b"new-model")
    record = payload()
    record["run_id"] = "new-run"
    record["model_sha256"] = hashlib.sha256(b"new-model").hexdigest()
    (selected / "shap.json").write_text(json.dumps(record), encoding="utf-8")
    config = tmp_path / "serving.json"
    config.write_text(json.dumps({"model_path": "models/ppo/new"}))
    monkeypatch.setattr(api, "SERVING_CONFIG", config)
    assert api.model_shap(None)["run_id"] == "new-run"
    with pytest.raises(HTTPException) as error:
        api.model_shap("td3")
    assert error.value.status_code == 404
    (selected / "shap.json").unlink()
    with pytest.raises(HTTPException) as error:
        api.model_shap(None)
    assert error.value.status_code == 503  # Never silently fall back to old SHAP.
    config.write_text(json.dumps({"model_path": "missing"}))
    (selected / "shap.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("INVESTMENT_AI_RUN", str(selected))
    assert api.model_shap(None)["run_id"] == "new-run"


def test_client_requests_active_shap_without_model_filter(monkeypatch):
    def get(*args, **kwargs):
        assert kwargs["params"] == {}
        return httpx.Response(
            200, json=payload(), request=httpx.Request("GET", "http://test")
        )

    monkeypatch.setattr(httpx, "get", get)
    assert shap_client.load_model_shap()["model_type"] == "ppo"
