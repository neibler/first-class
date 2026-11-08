"""Full observation SHAP for saved single-model bundles. No training."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.investment_ai.data_loader import load_data, fingerprint
from backend.investment_ai.model_runs import subset
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.prediction import PredictionSession
from backend.investment_ai.settings import bundle_input_root, ROOT
from backend.investment_ai.training import load_bundle, feature_names


def policy_function(model, kind, env, state=None, episode_start=True):
    """Freeze pre-decision memory; every perturbation starts from that same state."""
    frozen = deepcopy(state)

    def predict(rows):
        rows = np.atleast_2d(rows).astype(np.float32)
        kwargs = {"deterministic": True}
        if kind == "recurrent_ppo":
            kwargs.update(
                state=(
                    None
                    if frozen is None
                    else tuple(np.repeat(s, len(rows), axis=1) for s in frozen)
                ),
                episode_start=np.full(len(rows), episode_start, dtype=bool),
            )
        actions, _ = model.predict(rows, **kwargs)
        return np.vstack([env.action_weights(a) for a in np.atleast_2d(actions)])

    return predict


def replay(data, config, model, scaler, start, end):
    env = PortfolioEnv(data, config, start=start, end=end, scaler=scaler)
    obs, _ = env.reset(seed=config["seed"])
    session = PredictionSession(model, config["model_type"])
    rows = []
    while True:
        row = {
            "date": str(data.dates[env.t])[:10],
            "observation": obs.copy(),
            "state": deepcopy(session.lstm_states),
            "episode_start": session.episode_start,
        }
        target = session.predict(obs, env)
        obs, _, terminated, truncated, info = env.step_weights(target)
        row.update(target=target, execution_date=str(data.dates[env.t])[:10])
        if info["rebalanced"] and not info["safeguard_triggered"]:
            rows.append(row)
        if terminated or truncated:
            break
    env.close()
    return rows, env


def explain_bundle(bundle, sample=10, permutations=8):
    import shap

    if sample < 1 or permutations < 1:
        raise ValueError("sample/permutations must be positive")
    torch.set_num_threads(1)
    bundle = Path(bundle).resolve()
    meta = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if not meta.get("completed"):
        raise ValueError("Model training is incomplete")
    spec = json.loads((bundle / "run_spec.json").read_text(encoding="utf-8"))
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    data, _ = load_data(config, bundle_input_root(bundle, ROOT))
    dates = pd.DatetimeIndex(data.dates)
    data = subset(
        data,
        int(dates.searchsorted(spec["start_date"])),
        int(dates.searchsorted(spec["end_date"], side="right")),
    )
    model, scaler, config = load_bundle(bundle, data, exact_data=True)
    split = spec["test_start_index"]
    train_rows, _ = replay(
        data, config, model, scaler, config["lookback"] - 1, split - 1
    )
    if not train_rows:
        raise ValueError("No training reference observations")
    background = np.mean([r["observation"] for r in train_rows], axis=0, keepdims=True)
    rows, env = replay(data, config, model, scaler, split - 1, len(data.dates) - 1)
    if not rows:
        raise ValueError("No executed test decisions before safeguard")
    chosen = np.unique(np.linspace(0, len(rows) - 1, min(sample, len(rows)), dtype=int))
    names = feature_names(data, config)
    payload = {
        "format_version": 1,
        "model_type": config["model_type"],
        "run_id": config["run_id"],
        "features": names,
        "feature_keys": names,
        "data_hash": fingerprint(data),
        "model_sha256": hashlib.sha256((bundle / "model.zip").read_bytes()).hexdigest(),
        "method": "SHAP permutation",
        "permutations": permutations,
        "background": "mean of executed training decisions before safeguard",
        "background_end": str(data.dates[split - 1])[:10],
        "output_unit": "policy target weight before safeguard",
        "note": (
            "실제 모델의 전체 관측 입력 해석입니다. 기억 상태는 결정 직전 값으로 고정하며 "
            "과거 기억 자체의 기여는 설명하지 않습니다. 기여도는 인과관계가 아닙니다. "
            "배경 평균과 입력 교란은 실제 시장 상태와 다를 수 있습니다."
        ),
        "assets": {a: {"name": a, "decisions": []} for a in data.assets},
    }
    for index in chosen:
        row = rows[index]
        predict = policy_function(
            model, config["model_type"], env, row["state"], row["episode_start"]
        )
        actual = predict(row["observation"])[0]
        np.testing.assert_allclose(actual, row["target"], atol=1e-6)
        explainer = shap.PermutationExplainer(predict, background, seed=42)
        result = explainer(
            row["observation"][None, :],
            max_evals=(2 * len(names) + 1) * permutations,
            batch_size=256,
            silent=True,
        )
        values, base = result.values[0], result.base_values[0]
        np.testing.assert_allclose(base + values.sum(axis=0), actual, atol=2e-5)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite SHAP values")
        for i, asset in enumerate(data.assets):
            payload["assets"][asset]["decisions"].append(
                {
                    "date": row["date"],
                    "execution_date": row["execution_date"],
                    "feature_values_raw": row["observation"].tolist(),
                    "shap_values": values[:, i].tolist(),
                    "base_value": float(base[i]),
                    "final_value": float(actual[i]),
                }
            )
        print(config["model_type"], row["date"], "SHAP verified", flush=True)
    output = bundle / "shap.json"
    output.with_suffix(".pending").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    output.with_suffix(".pending").replace(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--all-shared", action="store_true")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=8)
    args = parser.parse_args()
    if bool(args.bundle) == args.all_shared:
        parser.error("Choose --bundle or --all-shared")
    if args.all_shared:
        manifest = (
            Path(__file__).resolve().parents[2] / "shared/trained_models_manifest.json"
        )
        items = json.loads(manifest.read_text(encoding="utf-8"))["models"].values()
        bundles = [ROOT / item["path"] for item in items]
    else:
        bundles = [args.bundle]
    for bundle in bundles:
        print(explain_bundle(bundle, args.sample, args.permutations), flush=True)


if __name__ == "__main__":
    main()
