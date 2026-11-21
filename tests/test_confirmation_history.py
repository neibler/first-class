"""Confirmation history and continuing all-cash reference regressions."""

from copy import deepcopy
import json
import numpy as np
import pytest
from tests.test_baseline_review_fixes import make_service
from tests.test_contract_v3 import state, data


def request(obj, execution_id, weights, episode="test"):
    s = state(obj.data, mdd=0)
    s.update(
        portfolio_value=1.0,
        weights=dict(zip(["SPY", "QQQ"], weights)),
        episode_id=episode,
    )
    clock = getattr(obj, "episode_clocks", {}).get(episode)
    if clock:
        s["episode_start_date"] = clock["episode_start_date"]
        dates = [str(d)[:10] for d in obj.data.dates]
        s["episode_elapsed_bars"] = (
            len(dates) - 1 - dates.index(clock["episode_start_date"])
        )
    confirmation = {
        "execution_id": execution_id,
        "date": s["date"],
        "weights": deepcopy(s["weights"]),
    }
    return s, confirmation


def confirm(obj, execution_id, weights, episode="test"):
    s, c = request(obj, execution_id, weights, episode)
    return obj.optimize(state=s, execution_confirmation=c)


def test_old_id_cannot_be_reused_after_another_confirmation(tmp_path):
    obj = make_service(tmp_path)
    confirm(obj, "A", [0.6, 0.4])
    confirm(obj, "B", [0.5, 0.5])
    before = obj.latch_path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        confirm(obj, "A", [0.8, 0.2])
    assert obj.latch_path.read_bytes() == before
    assert obj.execution_references["test"] == {"SPY": 0.5, "QQQ": 0.5}


def test_old_identical_retry_never_rolls_reference_back(tmp_path):
    obj = make_service(tmp_path)
    first = confirm(obj, "A", [0.6, 0.4])
    confirm(obj, "B", [0.5, 0.5])
    before = obj.latch_path.read_bytes()
    assert confirm(obj, "A", [0.6, 0.4]) == first
    assert obj.execution_references["test"] == {"SPY": 0.5, "QQQ": 0.5}
    assert obj.latch_path.read_bytes() == before


def test_continuing_cash_restores_confirmed_reference(tmp_path):
    obj = make_service(tmp_path)
    confirm(obj, "A", [0.6, 0.4])
    s, _ = request(obj, "unused", [0.0, 0.0])
    result = obj.optimize(state=s)
    assert result["last_executed_weights"] == {"SPY": 0.6, "QQQ": 0.4}
    assert result["portfolio_weight_distance"] == pytest.approx(0.5)
    assert "portfolio_weight_distance" in result["rebalance_reasons"]
    assert result["should_rebalance"]
    assert obj.execution_references["test"] == {"SPY": 0.6, "QQQ": 0.4}


def restart(obj, tmp_path, recurrent=False):
    saved = json.loads(obj.latch_path.read_text())
    restored = make_service(tmp_path, recurrent)
    restored.data = obj.data
    restored.session_tokens = set(saved.get("session_tokens", []))
    restored.latches = saved["latches"]
    restored.episode_clocks = saved["clocks"]
    restored.execution_references = saved["execution_references"]
    restored._restore_execution_history(saved)
    restored._restore_recurrent(saved["recurrent"])
    return restored


@pytest.mark.parametrize("changed", ["weights", "date"])
@pytest.mark.parametrize("restarted", [False, True])
def test_nonlatest_id_conflicts_survive_restart(tmp_path, changed, restarted):
    obj = make_service(tmp_path)
    confirm(obj, "A", [0.6, 0.4])
    confirm(obj, "B", [0.5, 0.5])
    if restarted:
        obj = restart(obj, tmp_path)
    if changed == "date":
        obj.data = data(np.zeros((6, 2)))
    before = obj.latch_path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        confirm(obj, "A", [0.8, 0.2] if changed == "weights" else [0.6, 0.4])
    assert obj.latch_path.read_bytes() == before
    assert obj.execution_references["test"] == {"SPY": 0.5, "QQQ": 0.5}
    assert set(obj.execution_history["test"]) == {"A", "B"}


def test_latest_duplicate_and_distinct_ids(tmp_path):
    obj = make_service(tmp_path)
    first = confirm(obj, "A", [0.6, 0.4])
    assert confirm(obj, "A", [0.6, 0.4]) == first
    assert len(obj.execution_history["test"]) == 1
    confirm(obj, "B", [0.5, 0.5])
    assert len(obj.execution_history["test"]) == 2


def test_historical_identical_event_after_newer_date_is_read_only(tmp_path):
    obj = make_service(tmp_path)
    s, c = request(obj, "A", [0.6, 0.4])
    first = obj.optimize(state=s, execution_confirmation=c)
    obj.data = data(np.zeros((6, 2)))
    confirm(obj, "B", [0.5, 0.5])
    obj = restart(obj, tmp_path)
    before = obj.latch_path.read_bytes()
    assert obj.optimize(state=s, execution_confirmation=c) == first
    assert obj.latch_path.read_bytes() == before
    assert obj.execution_references["test"] == {"SPY": 0.5, "QQQ": 0.5}
    assert obj.episode_clocks["test"]["episode_elapsed_bars"] == 1
    assert obj.models[0].calls == 0


def test_unseen_backdated_event_still_rejected(tmp_path):
    obj = make_service(tmp_path)
    old_state, old_confirmation = request(obj, "new-but-old", [0.6, 0.4])
    confirm(obj, "A", [0.6, 0.4])
    obj.data = data(np.zeros((6, 2)))
    confirm(obj, "B", [0.5, 0.5])
    with pytest.raises(ValueError, match="Stale"):
        obj.optimize(state=old_state, execution_confirmation=old_confirmation)
    assert set(obj.execution_history["test"]) == {"A", "B"}


def test_episode_and_owner_histories_are_independent(tmp_path):
    obj = make_service(tmp_path)
    confirm(obj, "A", [0.6, 0.4], "one")
    confirm(obj, "A", [0.8, 0.2], "two")
    for token, weights in [
        (obj.create_session(), [0.2, 0.8]),
        (obj.create_session(), [0.3, 0.7]),
    ]:
        s, c = request(obj, "A", weights, "one")
        obj.optimize(state=s, execution_confirmation=c, session_token=token)
    assert len(obj.execution_history) == 4
    assert obj.execution_references["one"] == {"SPY": 0.6, "QQQ": 0.4}
    assert obj.execution_references["two"] == {"SPY": 0.8, "QQQ": 0.2}


def test_legacy_latest_confirmation_migration(tmp_path):
    obj = make_service(tmp_path)
    first = confirm(obj, "A", [0.6, 0.4])
    saved = json.loads(obj.latch_path.read_text())
    saved.pop("execution_history")
    obj._restore_execution_history(saved)
    assert obj.execution_history["test"]["A"]["response"] is None
    assert confirm(obj, "A", [0.6, 0.4]) == first
    assert obj.execution_history["test"]["A"]["response"] == first
    confirm(obj, "B", [0.5, 0.5])
    with pytest.raises(ValueError, match="Conflicting"):
        confirm(obj, "A", [0.8, 0.2])


def test_legacy_historical_retry_uses_latest_saved_view(tmp_path):
    obj = make_service(tmp_path)
    s, a = request(obj, "A", [0.6, 0.4])
    obj.optimize(state=s, execution_confirmation=a)
    saved = json.loads(obj.latch_path.read_text())
    saved.pop("execution_history")
    obj._restore_execution_history(saved)
    latest = confirm(obj, "B", [0.5, 0.5])
    before = obj.latch_path.read_bytes()
    assert obj.optimize(state=s, execution_confirmation=a) == latest
    assert obj.latch_path.read_bytes() == before


def test_failed_save_rolls_back_history_reference_and_clock(tmp_path, monkeypatch):
    obj = make_service(tmp_path)
    confirm(obj, "A", [0.6, 0.4])
    before = obj.latch_path.read_bytes()
    history = deepcopy(obj.execution_history)
    clock = deepcopy(obj.episode_clocks)

    def fail():
        raise OSError("save failed")

    monkeypatch.setattr(obj, "_save_episode_state", fail)
    with pytest.raises(OSError, match="save failed"):
        confirm(obj, "B", [0.5, 0.5])
    assert obj.execution_history == history
    assert obj.episode_clocks == clock
    assert obj.execution_references["test"] == {"SPY": 0.6, "QQQ": 0.4}
    assert obj.latch_path.read_bytes() == before


def test_recommendation_does_not_approve_execution(tmp_path):
    obj = make_service(tmp_path)
    obj.optimize()
    assert not getattr(obj, "execution_history", {})
    confirm(obj, "A", [0.6, 0.4])
    s, _ = request(obj, "not-an-event", [0.0, 0.0])
    history = deepcopy(obj.execution_history)
    obj.optimize(state=s)
    obj.optimize(state=s)
    obj.optimize()
    assert obj.execution_history == history
    assert obj.execution_confirmations["test"]["execution_id"] == "A"


def test_client_recommendation_does_not_create_confirmation(monkeypatch):
    from frontend import investment_client

    sent = []
    monkeypatch.setattr(
        investment_client, "_request", lambda *args, **kw: sent.append(kw) or {}
    )
    investment_client.optimize(session_token="session")
    assert sent == [{"json": {"session_token": "session"}}]


def test_recurrent_cash_continuation_retry_and_conflict_are_isolated(tmp_path):
    obj = make_service(tmp_path, True)
    obj.data = data(np.zeros((4, 2)))
    s, _ = request(obj, "unused", [0.0, 0.0])
    s["reset_episode"] = True
    obj.optimize(state=s)
    obj.data = data(np.zeros((5, 2)))
    first = confirm(obj, "A", [0.6, 0.4])
    obj.data = data(np.zeros((6, 2)))
    confirm(obj, "B", [0.5, 0.5])
    obj = restart(obj, tmp_path, True)
    before = obj.latch_path.read_bytes()
    old_state, c = request(obj, "A", [0.8, 0.2])
    with pytest.raises(ValueError, match="Conflicting"):
        obj.optimize(state=old_state, execution_confirmation=c)
    assert obj.models[0].calls == 0 and obj.latch_path.read_bytes() == before
    obj.data = data(np.zeros((7, 2)))
    s, _ = request(obj, "unused", [0.0, 0.0])
    result = obj.optimize(state=s)
    assert result["last_executed_weights"] == {"SPY": 0.5, "QQQ": 0.5}
    assert result["portfolio_weight_distance"] == pytest.approx(0.5)
    assert result["should_rebalance"]
    assert obj.optimize(state=s) == result
    assert obj.models[0].calls == 1
    active = obj.recurrent_sessions["test"]
    clock = deepcopy(obj.episode_clocks["test"])
    obj.optimize()
    assert obj.recurrent_sessions["test"] is active
    assert obj.episode_clocks["test"] == clock
    assert obj.execution_references["test"] == {"SPY": 0.5, "QQQ": 0.5}
    assert first["last_executed_weights"] == {"SPY": 0.6, "QQQ": 0.4}


def test_cash_fallback_only_without_reference_and_reset_clears_history(tmp_path):
    obj = make_service(tmp_path)
    s, _ = request(obj, "unused", [0.0, 0.0])
    assert obj.optimize(state=s)["last_executed_weights"] == s["weights"]
    confirm(obj, "A", [0.6, 0.4])
    s["reset_episode"] = True
    assert obj.optimize(state=s)["last_executed_weights"] == s["weights"]
    assert "test" not in obj.execution_history
    assert "test" not in obj.execution_references
    confirm(obj, "A", [0.8, 0.2], "new-episode")


def test_http_conflict_is_422_and_reference_unchanged(tmp_path):
    from backend.investment_ai.api import create_app
    from fastapi.testclient import TestClient

    obj = make_service(tmp_path)
    app = create_app()
    app.state.service = obj
    client = TestClient(app)
    token = client.post("/inference-sessions").json()["session_token"]
    for identifier, weights in [("A", [0.6, 0.4]), ("B", [0.5, 0.5])]:
        s, c = request(obj, identifier, weights)
        response = client.post(
            "/optimize",
            json={"session_token": token, "state": s, "execution_confirmation": c},
        )
        assert response.status_code == 200
    before = obj.latch_path.read_bytes()
    s, c = request(obj, "A", [0.8, 0.2])
    response = client.post(
        "/optimize",
        json={"session_token": token, "state": s, "execution_confirmation": c},
    )
    assert response.status_code == 422 and "Conflicting" in response.json()["detail"]
    assert obj.latch_path.read_bytes() == before


def test_latest_confirmation_retry_after_later_recommendation_is_read_only(tmp_path):
    obj = make_service(tmp_path)
    s, c = request(obj, "A", [0.6, 0.4])
    first = obj.optimize(state=s, execution_confirmation=c)
    obj.data = data(np.zeros((6, 2)))
    next_state, _ = request(obj, "unused", [0.55, 0.45])
    obj.optimize(state=next_state)
    before = obj.latch_path.read_bytes()
    assert obj.optimize(state=s, execution_confirmation=c) == first
    assert obj.latch_path.read_bytes() == before
    assert obj.episode_clocks["test"]["episode_elapsed_bars"] == 1
