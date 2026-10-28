import json
from pathlib import Path

import numpy as np
import pytest

from backend.investment_ai import experiment as ex
from backend.investment_ai.data_loader import dummy_data
from backend.investment_ai.preprocessing import FeatureScaler


@pytest.mark.parametrize("data_path", [None, "", "market.parquet"])
def test_experiment_records_actual_input_kind_without_training(
    tmp_path, monkeypatch, data_path
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
        seeds=[42],
        reward_candidates=["return"],
    )
    if data_path:
        dummy_data(config).to_parquet(tmp_path / data_path, index=False)
    monkeypatch.setattr(ex, "ROOT", tmp_path)

    class FixedPolicy:
        def predict(self, observation, deterministic=True):
            return np.zeros(2), None

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
