import json
from pathlib import Path

import numpy as np
import pytest

from backend.investment_ai import experiment as ex
from backend.investment_ai.data_loader import dummy_data
from backend.investment_ai.preprocessing import FeatureScaler


@pytest.mark.parametrize("kind,seeds", [("ppo", [42, 73]), ("recurrent_ppo", [42])])
@pytest.mark.parametrize("data_path", [None, "", "market.parquet"])
def test_experiment_records_actual_input_kind_without_training(
    tmp_path, monkeypatch, data_path, kind, seeds
):
    config = json.loads(Path("config/dummy_only.json").read_text(encoding="utf-8"))
    config.update(
        data_path=data_path,
        assets=["SPY", "QQQ"],
        lookback=5,
        dummy_start="2023-01-02",
        dummy_end="2023-05-31",
        initial_train_days=30,
        validation_days=10,
        folds=1,
        min_episode_days=10,
        seeds=seeds,
        model_type=kind,
        reward_candidates=["return"],
    )
    if data_path:
        dummy_data(config).to_parquet(tmp_path / data_path, index=False)
    monkeypatch.setattr(ex, "ROOT", tmp_path)

    class FixedPolicy:
        def predict(
            self, observation, deterministic=True, state=None, episode_start=None
        ):
            return np.zeros(2), (
                (np.ones(1), np.ones(1)) if episode_start is not None else None
            )

    # Exercise real loading, backtests, report and summary; replace PPO fitting only.
    def fake_train(data, config, end, seed, directory):
        directory.mkdir(parents=True)
        return FixedPolicy(), FeatureScaler.fit(data, end)

    monkeypatch.setattr(ex, "train_model", fake_train)
    monkeypatch.setattr(
        ex,
        "load_bundle",
        lambda directory, data, exact_data: (
            FixedPolicy(),
            FeatureScaler.fit(data, 30),
        ),
    )
    run = ex.run_experiment(config, run_id="input_kind")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic"] is (not data_path)
    assert (run / "synthetic_market.parquet").exists() is (not data_path)

    prediction = json.loads((run / "latest_weights.json").read_text())
    assert prediction["mode"] == "new_portfolio_cash"
    assert sum(prediction["weights"].values()) == pytest.approx(1)


def test_legacy_recurrent_ensemble_rejected_before_io_or_training(
    tmp_path, monkeypatch
):
    config = json.loads(Path("config/dummy_only.json").read_text(encoding="utf-8"))
    config.update(model_type="recurrent_ppo", seeds=[42, 73])
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(ex, "load_data", lambda *a: pytest.fail("must not load data"))
    monkeypatch.setattr(ex, "train_model", lambda *a: pytest.fail("must not train"))
    with pytest.raises(
        ValueError, match="Legacy recurrent_ppo supports exactly one seed"
    ):
        ex.run_experiment(config, "unsupported")
    assert list(tmp_path.iterdir()) == []


def test_failed_inference_never_marks_run_complete(tmp_path, monkeypatch):
    def fail_prediction(*args, **kwargs):
        raise ValueError("inference unavailable")

    monkeypatch.setattr(ex, "predict_weights", fail_prediction)
    pointer = tmp_path / "runs/latest.json"
    pointer.parent.mkdir()
    pointer.write_text('{"run_id": "previous"}')
    with pytest.raises(ValueError, match="inference unavailable"):
        test_experiment_records_actual_input_kind_without_training(
            tmp_path, monkeypatch, None, "recurrent_ppo", [42]
        )
    assert not (tmp_path / "runs/input_kind/summary.json").exists()
    assert json.loads(pointer.read_text()) == {"run_id": "previous"}
