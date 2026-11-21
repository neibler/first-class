import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.investment_ai.api import create_app
from tests.test_baseline_review_fixes import make_service
from tests.test_confirmation_history import request, confirm
from tests.test_contract_v3 import data


def client_for(obj):
    app = create_app()
    app.state.service = obj
    return TestClient(app)


def test_rejected_anonymous_requests_allocate_nothing(tmp_path, monkeypatch):
    obj = make_service(tmp_path)
    client = client_for(obj)
    writes = []
    original = obj._save_episode_state

    def save():
        writes.append(1)
        original()

    monkeypatch.setattr(obj, "_save_episode_state", save)
    for _ in range(10):
        assert client.post("/optimize", json={"news": []}).status_code == 422
    assert not getattr(obj, "session_tokens", set())
    assert not writes
    assert not obj.latch_path.exists()


def test_both_issuance_paths_enforce_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("INVESTMENT_AI_MAX_SESSIONS", "2")
    obj = make_service(tmp_path)
    client = client_for(obj)
    assert client.post("/inference-sessions").status_code == 200
    assert client.post("/optimize", json={}).status_code == 200
    assert client.post("/inference-sessions").status_code == 429
    assert client.post("/optimize", json={}).status_code == 429
    assert len(obj.session_tokens) == 2


def prepare_recurrent(tmp_path):
    obj = make_service(tmp_path, True)
    obj.data = data(np.zeros((4, 2)))
    s, _ = request(obj, "unused", [0.0, 0.0])
    s["reset_episode"] = True
    obj.optimize(state=s)
    obj.data = data(np.zeros((5, 2)))
    confirm(obj, "A", [0.6, 0.4])
    obj.data = data(np.zeros((6, 2)))
    return obj


def test_same_day_recommendation_confirmation_retry(tmp_path):
    obj = prepare_recurrent(tmp_path)
    s, c = request(obj, "B", [0.8, 0.2])
    recommendation = obj.optimize(state=s)
    assert recommendation["should_rebalance"]
    calls = obj.models[0].calls
    lstm = obj.recurrent_sessions["test"]["sessions"][0].lstm_states
    confirmed = obj.optimize(state=s, execution_confirmation=c)
    assert confirmed["portfolio_weight_distance"] == 0
    assert not confirmed["should_rebalance"]
    assert confirmed["last_executed_weights"] == s["weights"]
    assert confirmed["raw_target_weights"] == recommendation["raw_target_weights"]
    assert obj.models[0].calls == calls
    assert obj.recurrent_sessions["test"]["sessions"][0].lstm_states is lstm
    assert obj.optimize(state=s, execution_confirmation=c) == confirmed
    assert obj.optimize(state=s) == confirmed
    assert obj.models[0].calls == calls


def restore(obj, tmp_path):
    saved = json.loads(obj.latch_path.read_text())
    restored = make_service(tmp_path, True)
    restored.data = obj.data
    restored._restore_sessions(saved)
    restored.latches = saved["latches"]
    restored.episode_clocks = saved["clocks"]
    restored.execution_references = saved["execution_references"]
    restored._restore_execution_history(saved)
    restored._restore_recurrent(saved["recurrent"])
    return restored


def test_expiry_rejects_token_cleans_only_owner_and_frees_capacity(
    tmp_path, monkeypatch
):
    import hashlib
    from backend.investment_ai import session_registry as registry

    clock = [100.0]
    monkeypatch.setattr(registry, "now", lambda: clock[0])
    monkeypatch.setenv("INVESTMENT_AI_SESSION_TTL_SECONDS", "10")
    monkeypatch.setenv("INVESTMENT_AI_MAX_SESSIONS", "2")
    obj = make_service(tmp_path, True)
    first = obj.create_session()
    s, _ = request(obj, "unused", [0.0, 0.0])
    s["reset_episode"] = True
    obj.optimize(state=s, session_token=first)
    clock[0] = 105
    second = obj.create_session()
    obj.optimize(state=s, session_token=second)
    second_key = json.dumps(
        [hashlib.sha256(second.encode()).hexdigest(), "episode", "test"]
    )
    active = obj.recurrent_sessions[second_key]
    calls = obj.models[0].calls
    clock[0] = 110
    with pytest.raises(ValueError, match="Unknown inference session"):
        obj.optimize(session_token=first)
    assert len(obj.session_tokens) == 1
    assert len(obj.episode_clocks) == len(obj.recurrent_sessions) == 1
    assert obj.recurrent_sessions[second_key]["sessions"] is active["sessions"]
    obj.optimize(state=s, session_token=second)
    assert obj.models[0].calls == calls
    replacement = obj.create_session()
    assert replacement not in (first, second)
    assert len(obj.session_tokens) == 2
    assert len(json.loads(obj.latch_path.read_text())["session_tokens"]) == 2


def test_expiry_is_persisted_not_extended_by_restart(tmp_path, monkeypatch):
    from backend.investment_ai import session_registry as registry

    clock = [100.0]
    monkeypatch.setattr(registry, "now", lambda: clock[0])
    monkeypatch.setenv("INVESTMENT_AI_SESSION_TTL_SECONDS", "10")
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    first = obj.optimize(session_token=token)
    clock[0] = 109
    restored = restore(obj, tmp_path)
    assert restored.optimize(session_token=token) == first
    assert restored.models[0].calls == 0
    clock[0] = 110
    with pytest.raises(ValueError, match="Unknown"):
        restored.optimize(session_token=token)
    assert not restored.session_tokens and not restored.recurrent_sessions


def test_existing_registry_legacy_migration_keeps_tokens_and_fixed_deadline(
    tmp_path, monkeypatch
):
    from backend.investment_ai import session_registry as registry

    clock = [100.0]
    monkeypatch.setattr(registry, "now", lambda: clock[0])
    monkeypatch.setenv("INVESTMENT_AI_SESSION_TTL_SECONDS", "10")
    obj = make_service(tmp_path)
    token = obj.create_session()
    saved = json.loads(obj.latch_path.read_text())
    saved.pop("session_expires_at")
    assert obj._restore_sessions(saved)
    obj._save_episode_state()
    clock[0] = 109
    saved = json.loads(obj.latch_path.read_text())
    assert not obj._restore_sessions(saved)
    assert obj._session_scope(token)
    clock[0] = 110
    with pytest.raises(ValueError, match="Unknown"):
        obj._session_scope(token)


def test_capacity_does_not_evict_active_session_or_write(tmp_path, monkeypatch):
    monkeypatch.setenv("INVESTMENT_AI_MAX_SESSIONS", "1")
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    first = obj.optimize(session_token=token)
    before = obj.latch_path.read_bytes()
    client = client_for(obj)
    for _ in range(5):
        assert client.post("/inference-sessions").status_code == 429
        assert client.post("/optimize", json={}).status_code == 429
    assert obj.latch_path.read_bytes() == before
    assert obj.optimize(session_token=token) == first
    assert obj.models[0].calls == 1


@pytest.mark.parametrize("kind", ["explicit", "implicit"])
def test_failed_publication_discards_new_token(tmp_path, monkeypatch, kind):
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    first = obj.optimize(session_token=token)
    tokens = obj.session_tokens.copy()
    before = obj.latch_path.read_bytes()

    def fail():
        raise OSError("disk unavailable")

    monkeypatch.setattr(obj, "_save_episode_state", fail)
    with pytest.raises(OSError):
        if kind == "explicit":
            obj.create_session()
        else:
            obj.optimize_new_session()
    assert obj.session_tokens == tokens
    assert obj.latch_path.read_bytes() == before
    assert len(obj.recurrent_sessions) == 1
    assert (
        next(iter(obj.recurrent_sessions.values()))["prediction"]["weights"]
        == first["target_weights"]
    )


def test_same_day_confirmation_after_restart_and_next_day(tmp_path):
    obj = prepare_recurrent(tmp_path)
    s, c = request(obj, "B", [0.55, 0.45])
    obj.optimize(state=s)
    obj = restore(obj, tmp_path)
    result = obj.optimize(state=s, execution_confirmation=c)
    assert result["last_executed_weights"] == s["weights"]
    assert result["portfolio_weight_distance"] == 0
    assert obj.models[0].calls == 0
    obj = restore(obj, tmp_path)
    assert obj.optimize(state=s, execution_confirmation=c) == result
    assert obj.optimize(state=s) == result
    assert obj.models[0].calls == 0
    obj.data = data(np.zeros((7, 2)))
    next_state, _ = request(obj, "unused", [0.55, 0.45])
    obj.optimize(state=next_state)
    assert obj.models[0].calls == 1


@pytest.mark.parametrize("change", ["market", "holdings", "value"])
def test_same_day_changed_observation_cannot_reuse_output(tmp_path, change):
    obj = prepare_recurrent(tmp_path)
    s, c = request(obj, "B", [0.55, 0.45])
    obj.optimize(state=s)
    calls = obj.models[0].calls
    before = obj.latch_path.read_bytes()
    if change == "market":
        obj.data.returns[-1, 0] = 0.01
    elif change == "holdings":
        s["weights"] = c["weights"] = {"SPY": 0.8, "QQQ": 0.2}
    else:
        s.update(portfolio_value=0.99, max_drawdown=0.01)
    with pytest.raises(ValueError, match="consecutive"):
        obj.optimize(state=s, execution_confirmation=c)
    assert obj.models[0].calls == calls
    assert obj.latch_path.read_bytes() == before
    assert obj.execution_references["test"] == {"SPY": 0.6, "QQQ": 0.4}


def test_same_day_confirmation_save_failure_preserves_cache(tmp_path, monkeypatch):
    obj = prepare_recurrent(tmp_path)
    s, c = request(obj, "B", [0.8, 0.2])
    proposal = obj.optimize(state=s)
    before = obj.latch_path.read_bytes()
    signature = obj.recurrent_sessions["test"]["signature"]
    calls = obj.models[0].calls
    original = obj._save_episode_state

    def fail():
        raise OSError("disk unavailable")

    monkeypatch.setattr(obj, "_save_episode_state", fail)
    with pytest.raises(OSError):
        obj.optimize(state=s, execution_confirmation=c)
    assert obj.recurrent_sessions["test"]["signature"] == signature
    assert (
        obj.recurrent_sessions["test"]["prediction"]["weights"]
        == proposal["target_weights"]
    )
    assert obj.execution_references["test"] == {"SPY": 0.6, "QQQ": 0.4}
    assert obj.latch_path.read_bytes() == before
    assert obj.models[0].calls == calls
    monkeypatch.setattr(obj, "_save_episode_state", original)
    assert (
        obj.optimize(state=s, execution_confirmation=c)["portfolio_weight_distance"]
        == 0
    )
    assert obj.models[0].calls == calls


def test_failed_anonymous_requests_preserve_existing_registry(tmp_path, monkeypatch):
    obj = make_service(tmp_path)
    token = obj.create_session()
    before = obj.latch_path.read_bytes()

    def unexpected():
        raise AssertionError("Rejected request must not write registry")

    monkeypatch.setattr(obj, "_save_episode_state", unexpected)
    client = client_for(obj)
    for _ in range(10):
        assert client.post("/optimize", json={"news": []}).status_code == 422
    assert obj._session_scope(token)
    assert len(obj.session_tokens) == 1 and obj.latch_path.read_bytes() == before


def test_successful_anonymous_request_publishes_once(tmp_path, monkeypatch):
    obj = make_service(tmp_path)
    writes = []
    original = obj._save_episode_state

    def save():
        writes.append(1)
        original()

    monkeypatch.setattr(obj, "_save_episode_state", save)
    result = client_for(obj).post("/optimize", json={})
    assert result.status_code == 200 and len(writes) == 1
    assert obj._session_scope(result.json()["session_token"])
    assert len(json.loads(obj.latch_path.read_text())["session_expires_at"]) == 1


def test_legacy_same_day_confirmation_does_not_advance_lstm(tmp_path, monkeypatch):
    from backend.investment_ai.rebalancing import REBALANCE_POLICY

    obj = make_service(tmp_path, True)
    obj.config["execution_policy"] = REBALANCE_POLICY
    obj.data = data(np.zeros((4, 2)))
    s, _ = request(obj, "unused", [0.0, 0.0])
    s["reset_episode"] = True
    obj.optimize(state=s)
    obj.data = data(np.zeros((5, 2)))
    confirm(obj, "A", [0.6, 0.4])
    obj.data = data(np.zeros((6, 2)))
    s, c = request(obj, "B", [0.55, 0.45])
    before = obj.optimize(state=s)
    calls = obj.models[0].calls

    def no_second_filter(*args, **kwargs):
        raise AssertionError("Legacy cached threshold decision must be preserved")

    monkeypatch.setattr(
        "backend.investment_ai.serving.predict_weights", no_second_filter
    )
    confirmed = obj.optimize(state=s, execution_confirmation=c)
    assert confirmed["raw_target_weights"] == before["raw_target_weights"]
    assert confirmed["last_executed_weights"] == s["weights"]
    assert obj.optimize(state=s, execution_confirmation=c) == confirmed
    assert obj.optimize(state=s) == confirmed
    assert obj.models[0].calls == calls


def test_http_recurrent_recommendation_then_confirmation(tmp_path):
    obj = make_service(tmp_path, True)
    obj.data = data(np.zeros((4, 2)))
    client = client_for(obj)
    token = client.post("/inference-sessions").json()["session_token"]
    s, _ = request(obj, "unused", [0.0, 0.0])
    start = s["date"]
    s["reset_episode"] = True
    assert (
        client.post("/optimize", json={"session_token": token, "state": s}).status_code
        == 200
    )
    for days, weights, identifier in [(5, [0.6, 0.4], "A"), (6, [0.55, 0.45], "B")]:
        obj.data = data(np.zeros((days, 2)))
        s, c = request(obj, identifier, weights)
        s.update(episode_start_date=start, episode_elapsed_bars=days - 4)
        body = {"session_token": token, "state": s}
        if identifier == "B":
            assert client.post("/optimize", json=body).status_code == 200
            calls = obj.models[0].calls
        body["execution_confirmation"] = c
        result = client.post("/optimize", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["portfolio_weight_distance"] == 0
        if identifier == "B":
            assert client.post("/optimize", json=body).json() == result.json()
            assert obj.models[0].calls == calls
