from copy import deepcopy
import numpy as np
import pytest
from tests.test_baseline_review_fixes import cfg, make_service
from tests.test_contract_v3 import data, state
from backend.investment_ai import baseline_execution as b
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.infer import predict_weights


def test_partial_exits_admit_available_slot():
    _, target, _ = b.executable_target(
        [0.01, 0.49, 0.3, 0.2], [0.5, 0.5, 0, 0], cfg(max_holdings=3)
    )
    assert target[2] > 0 and target[3] == 0
    assert np.count_nonzero(target > b.EPS) == 3
    assert 0.5 * np.abs(target - [0.5, 0.5, 0, 0]).sum() <= 0.15 + 1e-12


@pytest.mark.parametrize("threshold", [0.03, 0.05])
def test_legacy_original_threshold(threshold):
    e = PortfolioEnv(
        data(),
        cfg(
            execution_policy="monthly_risk_distance_filtered_v2",
            min_weight_threshold=threshold,
        ),
        start=1,
    )
    e.reset()
    row = e.step_weights([threshold, 1 - threshold + 0.000005], force=True)[-1]
    assert row["filtered_target_weights"]["SPY"] > 0


def test_baseline_keeps_small_cash():
    d = data()
    s = state(d, mdd=0)
    s.update(
        portfolio_value=1.0,
        weights={"SPY": 0.999995, "QQQ": 0.0},
        last_executed_weights={"SPY": 0.999995, "QQQ": 0.0},
    )

    class P:
        def predict(self, *args):
            return np.array([0.6, 0.4])

    result = predict_weights(d, cfg(), [], None, s, [P()])
    assert result["weights"]["SPY"] == 0.999995
    assert result["cash_weight"] == pytest.approx(0.000005)


def test_sessions_isolate_same_episode(tmp_path):
    obj = make_service(tmp_path, True)
    a, btoken = obj.create_session(), obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    first = obj.optimize(state=s, session_token=a)
    obj.optimize(state=s, session_token=btoken)
    assert obj.models[0].calls == 2
    assert obj.optimize(state=s, session_token=a) == first
    assert obj.models[0].calls == 2


def test_cash_does_not_mutate_active_episode(tmp_path):
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    first = obj.optimize(state=s, session_token=token)
    clocks = deepcopy(obj.episode_clocks)
    keys = set(obj.recurrent_sessions)
    cash = obj.optimize(session_token=token)
    assert all(obj.episode_clocks[k] == v for k, v in clocks.items())
    assert keys < set(obj.recurrent_sessions)
    assert obj.optimize(session_token=token) == cash
    assert obj.optimize(state=s, session_token=token) == first
    assert obj.models[0].calls == 2


def test_unconfirmed_reference_not_persisted(tmp_path):
    obj = make_service(tmp_path)
    obj.execution_references = {"test": {"SPY": 0.6, "QQQ": 0.4}}
    s = state(obj.data, mdd=0.05)
    s.update(
        weights={"SPY": 0.5, "QQQ": 0.5}, last_executed_weights={"SPY": 0.5, "QQQ": 0.5}
    )
    obj.optimize(state=s)
    assert obj.execution_references["test"] == {"SPY": 0.6, "QQQ": 0.4}


def restore_service(obj, tmp_path):
    import json

    saved = json.loads(obj.latch_path.read_text())
    restored = make_service(tmp_path, True)
    restored.session_tokens = set(saved["session_tokens"])
    restored.execution_references = saved["execution_references"]
    restored._restore_execution_history(saved)
    restored.episode_clocks = saved["clocks"]
    restored.latches = saved["latches"]
    restored._restore_recurrent(saved["recurrent"])
    return restored


def test_continuation_interleaved_cash_and_restart(tmp_path):
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    first = obj.optimize(state=s, session_token=token)
    # Acknowledged holdings on the next bar continue the active LSTM.
    obj.optimize(session_token=token)
    restored = restore_service(obj, tmp_path)
    assert restored.optimize(state=s, session_token=token) == first
    assert restored.models[0].calls == 0
    restored.data = data(np.zeros((6, 2)))
    s.update(
        date=str(restored.data.dates[-1])[:10],
        reset_episode=False,
        episode_elapsed_bars=1,
        weights={"SPY": 0.6, "QQQ": 0.4},
    )
    confirmation = {
        "execution_id": "fill-1",
        "date": s["date"],
        "weights": s["weights"],
    }
    second = restored.optimize(
        state=s, session_token=token, execution_confirmation=confirmation
    )
    assert second["episode_elapsed_bars"] == 1
    assert restored.models[0].calls == 1
    assert (
        restored.optimize(
            state=s, session_token=token, execution_confirmation=confirmation
        )
        == second
    )
    assert restored.models[0].calls == 1
    restarted = restore_service(restored, tmp_path)
    restarted.data = restored.data
    assert (
        restarted.optimize(
            state=s, session_token=token, execution_confirmation=confirmation
        )
        == second
    )
    assert restarted.models[0].calls == 0
    # A safety latch persists independently of cash previews and restart.
    stopped = {**s, "max_drawdown": 0.2, "portfolio_value": 0.8}
    assert restarted.optimize(state=stopped, session_token=token)["cash_weight"] == 1
    restarted.optimize(session_token=token)
    recovered = restore_service(restarted, tmp_path)
    recovered.data = restarted.data
    assert recovered.optimize(state=s, session_token=token)["cash_weight"] == 1
    assert recovered.models[0].calls == 0


def test_unknown_session_rejected_without_state_change(tmp_path):
    obj = make_service(tmp_path, True)
    with pytest.raises(ValueError, match="session"):
        obj.optimize(session_token="someone-elses-token")
    assert not obj.recurrent_sessions and not obj.latches


def test_old_cache_missing_reference_keeps_lstm(tmp_path):
    obj = make_service(tmp_path, True)
    first = obj.optimize()
    record = next(iter(obj.recurrent_sessions.values()))
    record["prediction"].pop("last_executed_weights")
    lstm = record["sessions"][0].lstm_states
    assert obj.optimize() == first
    assert obj.models[0].calls == 1
    assert (
        next(iter(obj.recurrent_sessions.values()))["sessions"][0].lstm_states is lstm
    )


def test_invalid_confirmation_is_atomic(tmp_path):
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    obj.optimize(state=s, session_token=token)
    before = obj.latch_path.read_bytes()
    # Confirmations follow initialization; reset + confirmation is a separate error.
    s["reset_episode"] = False
    with pytest.raises(ValueError, match="Confirmed"):
        obj.optimize(
            state=s,
            session_token=token,
            execution_confirmation={
                "execution_id": "bad",
                "date": s["date"],
                "weights": {"SPY": 1.0, "QQQ": 0.0},
            },
        )
    assert obj.latch_path.read_bytes() == before
    assert obj.models[0].calls == 1


def test_dust_cleanup_preserves_cap_and_counts():
    current = np.array([0.5, 0.5 - 1e-12, 1e-12, 0])
    _, target, _ = b.executable_target([0.1, 0.1, 0, 0.8], current, cfg(max_holdings=3))
    assert target[2] == 0
    assert np.count_nonzero(target) == np.count_nonzero(target > b.EPS) <= 3
    assert 0.5 * np.abs(target - current).sum() <= 0.15 + 1e-12


def test_fee_stop_keeps_original_reason():
    e = PortfolioEnv(data(), cfg(transaction_cost_bps=4900, max_drawdown=0.01), start=1)
    e.reset()
    row = e.step_weights([0.6, 0.4], force=True)[-1]
    assert row["cash_weight"] == 1
    assert row["rebalance_reasons"] == ["forced", "mdd_stop"]
    assert row["transaction_cost"] > 0


def test_invalid_fill_before_environment_mutation(monkeypatch):
    e = PortfolioEnv(data(), cfg(), start=1)
    e.reset()

    def invalid(*args):
        raise ValueError("fill rejected")

    monkeypatch.setattr(b, "execute", invalid)
    with pytest.raises(ValueError, match="fill rejected"):
        e.step_weights([0.6, 0.4], force=True)
    assert e.t == 1 and e.value == 1 and not e.history
    np.testing.assert_array_equal(e.weights, [0, 0])


def test_old_history_holdings_fallback_and_unavailable():
    import pandas as pd
    from backend.investment_ai.backtest import performance

    e = PortfolioEnv(data(), cfg(), start=1)
    e.reset()
    e.step_weights([0.6, 0.4], force=True)
    frame = pd.DataFrame(e.history).drop(columns=["holdings_count"])
    assert performance(frame)["average_holdings_count"] == 2
    frame = frame.drop(columns=[c for c in frame if c.startswith("held_")])
    result = performance(frame)
    assert all(
        result[key] is None
        for key in (
            "average_holdings_count",
            "min_holdings_count",
            "max_holdings_count",
        )
    )


def test_equivalent_policy_hash_and_distinct_policy(tmp_path, monkeypatch):
    import json
    from backend.investment_ai import model_runs, bundle_inputs
    from backend.investment_ai.rebalancing import REBALANCE_POLICY

    d = data()
    spec = {
        "start_date": str(d.dates[0])[:10],
        "end_date": str(d.dates[-1])[:10],
        "test_start_index": 2,
    }
    (tmp_path / "run_spec.json").write_text(json.dumps(spec))
    (tmp_path / "manifest.json").write_text('{"completed":true}')
    c = cfg()
    c.pop("execution_policy")
    monkeypatch.setattr(bundle_inputs, "load_bundle_inputs", lambda *args: (d, None))
    monkeypatch.setattr(
        model_runs, "load_bundle", lambda *args, **kwargs: (None, None, c)
    )
    monkeypatch.setattr(model_runs, "compare", lambda *args: ({}, None))

    def evaluate():
        (tmp_path / "config.json").write_text(json.dumps(c))
        model_runs.evaluate_bundle(tmp_path)
        return json.loads((tmp_path / "evaluation.json").read_text())[
            "comparison_group"
        ]

    default = evaluate()
    c["execution_policy"] = REBALANCE_POLICY
    assert evaluate() == default
    c["execution_policy"] = b.POLICY
    assert evaluate() != default
    baseline = evaluate()
    c["max_rebalance_turnover"] = 0.2
    assert evaluate() != baseline


def test_episode_name_cannot_collide_with_cash_namespace(tmp_path):
    obj = make_service(tmp_path, True)
    token = obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(
        episode_id="cash-preview",
        weights={"SPY": 0.0, "QQQ": 0.0},
        portfolio_value=1.0,
        reset_episode=True,
    )
    obj.optimize(state=s, session_token=token)
    obj.optimize(session_token=token)
    assert len(obj.recurrent_sessions) == 2 and obj.models[0].calls == 2


def test_api_tokens_anonymous_isolation_and_retry(tmp_path):
    from backend.investment_ai.api import create_app, OptimizeRequest

    obj = make_service(tmp_path, True)
    app = create_app()
    app.state.service = obj
    endpoint = next(r.endpoint for r in app.routes if r.path == "/optimize")
    first = endpoint(OptimizeRequest())
    other = endpoint(OptimizeRequest())
    assert first["session_token"] != other["session_token"]
    assert obj.models[0].calls == 2
    assert endpoint(OptimizeRequest(session_token=first["session_token"])) == first
    assert obj.models[0].calls == 2


@pytest.mark.parametrize("threshold", [0.03, 0.05])
def test_legacy_inference_filters_original_boundary(threshold):
    d = data()
    d.risk[:] = 0.8

    class P:
        def predict(self, *args):
            return np.array([threshold, 1 - threshold + 0.000005])

    result = predict_weights(
        d,
        cfg(
            execution_policy="monthly_risk_distance_filtered_v2",
            min_weight_threshold=threshold,
        ),
        [],
        None,
        None,
        [P()],
    )
    assert result["filtered_target_weights"]["SPY"] > 0


def test_owner_safety_latch_isolated(tmp_path):
    obj = make_service(tmp_path, True)
    a, btoken = obj.create_session(), obj.create_session()
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    obj.optimize(state=s, session_token=a)
    obj.optimize(state=s, session_token=btoken)
    stopped = {**s, "reset_episode": False, "max_drawdown": 0.2, "portfolio_value": 0.8}
    assert obj.optimize(state=stopped, session_token=a)["safeguard_triggered"]
    assert not obj.optimize(state=s, session_token=btoken)["safeguard_triggered"]
    assert obj.optimize(state={**s, "reset_episode": False}, session_token=a)[
        "safeguard_triggered"
    ]


def test_reset_drops_prior_execution_reference(tmp_path):
    obj = make_service(tmp_path, True)
    obj.execution_references = {"test": {"SPY": 0.6, "QQQ": 0.4}}
    obj.execution_confirmations = {"test": {"execution_id": "old"}}
    s = state(obj.data, mdd=0)
    s.update(weights={"SPY": 0.0, "QQQ": 0.0}, portfolio_value=1.0, reset_episode=True)
    result = obj.optimize(state=s)
    assert result["last_executed_weights"] == s["weights"]
    assert "test" not in obj.execution_references
    assert "test" not in obj.execution_confirmations


def test_fill_is_computed_once_before_commit(monkeypatch):
    calls = []
    original = b.execute

    def once(*args):
        calls.append(args)
        assert len(calls) == 1
        return original(*args)

    monkeypatch.setattr(b, "execute", once)
    e = PortfolioEnv(data(), cfg(), start=1)
    e.reset()
    e.step_weights([0.6, 0.4], force=True)
    assert len(calls) == 1 and e.t == 2


def test_old_signature_without_reference_is_migrated_once(tmp_path):
    import hashlib
    import json
    from backend.investment_ai.data_loader import fingerprint

    obj = make_service(tmp_path, True)
    first = obj.optimize()
    record = next(iter(obj.recurrent_sessions.values()))
    record["prediction"].pop("last_executed_weights")
    legacy = {
        "date": first["as_of"],
        "weights": {"SPY": 0.0, "QQQ": 0.0},
        "portfolio_value": 1.0,
        "peak_value": 1.0,
        "max_drawdown": 0.0,
        "days_since_rebalance": 0,
        "stopped": False,
        "episode_start_date": first["episode_start_date"],
        "episode_elapsed_bars": 0,
    }
    record["signature"] = hashlib.sha256(
        (fingerprint(obj.data) + json.dumps(legacy, sort_keys=True)).encode()
    ).hexdigest()
    assert obj.optimize() == first
    assert obj.optimize() == first
    assert obj.models[0].calls == 1


def test_http_session_contract_and_rejected_token(tmp_path):
    from fastapi.testclient import TestClient
    from backend.investment_ai.api import create_app

    obj = make_service(tmp_path, True)
    app = create_app()
    app.state.service = obj
    client = TestClient(app)
    created = client.post("/inference-sessions")
    assert created.status_code == 200
    token = created.json()["session_token"]
    first = client.post("/optimize", json={"session_token": token})
    assert first.status_code == 200
    assert (
        client.post("/optimize", json={"session_token": token}).json() == first.json()
    )
    assert obj.models[0].calls == 1
    assert client.post("/optimize", json={"session_token": "x" * 43}).status_code == 422
    assert obj.models[0].calls == 1
