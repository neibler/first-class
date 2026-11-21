import hashlib
import json
import numpy as np
import pytest
from backend.investment_ai import baseline_execution as b, serving, session_registry
from backend.investment_ai.training import schema_for
from tests.test_baseline_review_fixes import make_service, cfg, Model
from tests.test_confirmation_history import request
from tests.test_contract_v3 import data
from frontend import investment_client


def test_zero_target_incumbent_does_not_reserve_slot():
    desired, actual, _ = b.executable_target(
        [0.0, 1.0],
        [1.0, 0.0],
        cfg(max_holdings=1, sell_threshold=0.0, max_rebalance_turnover=1.0),
    )
    np.testing.assert_array_equal(desired, [0, 1])
    np.testing.assert_array_equal(actual, [0, 1])


@pytest.mark.parametrize(
    "name",
    [
        "session_tokens",
        "execution_history",
        "execution_confirmations",
        "execution_references",
        "clocks",
        "latches",
        "recurrent",
        "version",
    ],
)
@pytest.mark.parametrize("latched", [False, True])
def test_real_startup_preserves_flat_legacy_latches(
    tmp_path, monkeypatch, name, latched
):
    d = data()
    c = cfg(data_path="unused-market.parquet")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for filename, value in [
        ("config.json", c),
        ("schema.json", schema_for(d, c)),
        ("run_spec.json", {"synthetic": False}),
        ("manifest.json", {"completed": True}),
    ]:
        (bundle / filename).write_text(json.dumps(value), encoding="utf-8")
    (bundle / "model.zip").write_bytes(b"test-model")
    monkeypatch.setattr(serving, "ROOT", tmp_path)
    monkeypatch.setattr(serving, "load_data", lambda *a, **kw: (d, None))
    monkeypatch.setattr(serving, "load_bundle", lambda *a, **kw: (Model(), None, c))
    state_file = (
        tmp_path
        / "service_state"
        / (hashlib.sha256(str(bundle.resolve()).encode()).hexdigest() + ".json")
    )
    state_file.parent.mkdir()
    legacy = {name: latched, "invested": True}
    state_file.write_text(json.dumps(legacy))
    obj = serving.InvestmentService(bundle)
    assert obj.latches == legacy
    assert not obj.session_tokens and not obj.execution_history
    assert not obj.execution_references and not obj.episode_clocks


@pytest.mark.parametrize("recurrent", [False, True])
def test_reset_and_confirmation_rejected_atomically(tmp_path, recurrent):
    obj = make_service(tmp_path, recurrent)
    s, c = request(obj, "cash-fill", [0.0, 0.0])
    if recurrent:
        obj.optimize(state={**s, "reset_episode": True})
        obj.data = data(np.zeros((6, 2)))
        s, c = request(obj, "cash-fill", [0.0, 0.0])
    obj.optimize(state=s, execution_confirmation=c)
    before = obj.latch_path.read_bytes()
    s["reset_episode"] = True
    s["episode_start_date"] = s["date"]
    s["episode_elapsed_bars"] = 0
    with pytest.raises(ValueError, match="separate"):
        obj.optimize(state=s, execution_confirmation=c)
    assert obj.latch_path.read_bytes() == before
    assert "cash-fill" in obj.execution_history["test"]
    obj.optimize(state=s)
    assert "test" not in obj.execution_history
    assert "test" not in obj.execution_references


def test_new_named_episode_cannot_adopt_preview(tmp_path):
    obj = make_service(tmp_path, True)
    preview = obj.optimize()
    obj.data = data(np.zeros((6, 2)))
    s, _ = request(obj, "unused", [0.0, 0.0], "new-name")
    s.update(episode_start_date=preview["episode_start_date"], episode_elapsed_bars=1)
    before = obj.latch_path.read_bytes()
    with pytest.raises(ValueError, match="reset"):
        obj.optimize(state=s)
    assert obj.latch_path.read_bytes() == before
    assert "new-name" not in obj.recurrent_sessions


def test_failed_expiry_cleanup_retries_without_reviving_token(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(session_registry, "now", lambda: clock[0])
    monkeypatch.setenv("INVESTMENT_AI_SESSION_TTL_SECONDS", "10")
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    obj.optimize(session_token=token)
    before = obj.latch_path.read_bytes()
    original = obj._save_episode_state

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(obj, "_save_episode_state", fail)
    clock[0] = 110
    with pytest.raises(OSError):
        obj.optimize(session_token=token)
    assert obj.latch_path.read_bytes() == before
    with pytest.raises(ValueError, match="Unknown"):
        obj._session_scope(token)
    monkeypatch.setattr(obj, "_save_episode_state", original)
    with pytest.raises(ValueError, match="Unknown"):
        obj.optimize(session_token=token)
    saved = json.loads(obj.latch_path.read_text())
    assert saved["session_tokens"] == [] and saved["recurrent"] == {}


@pytest.mark.parametrize("token", [None, "", "  ", False, 12, [], {}])
def test_invalid_creation_token_is_client_error(monkeypatch, token):
    monkeypatch.setattr(
        investment_client,
        "_request",
        lambda *a, **kw: {"schema_version": "3.0", "session_token": token},
    )
    with pytest.raises(investment_client.InvestmentAPIError):
        investment_client.create_inference_session()


def test_missing_creation_token_is_client_error(monkeypatch):
    monkeypatch.setattr(
        investment_client, "_request", lambda *a, **kw: {"schema_version": "3.0"}
    )
    with pytest.raises(investment_client.InvestmentAPIError):
        investment_client.create_inference_session()
