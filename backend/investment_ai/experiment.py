"""Expanding-window selection followed by a frozen, untouched final holdout."""

import platform
import shutil
import importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from .settings import ROOT, SOURCE_DIR, write_json
from .data_loader import load_data, prepare_data, fingerprint
from .synthetic import generate
from .portfolio_env import PortfolioEnv
from .preprocessing import FeatureScaler
from .training import train_model, load_bundle, source_hash
from .splits import walk_forward_splits
from .backtest import run_backtest, performance, compare
from .infer import predict_weights, save_prediction
from .report import build_report
from .records import save_asset_history
from .rewards import canonical_reward


def run_experiment(config, run_id=None):
    run_id = run_id or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    if not run_id or any(
        x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for x in run_id
    ):
        raise ValueError("run_id allows letters, numbers, underscores and hyphens")
    config = {
        **config,
        "run_id": run_id,
        "reward": canonical_reward(config["reward"]),
        "reward_candidates": [canonical_reward(x) for x in config["reward_candidates"]],
    }
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    shutil.copytree(
        SOURCE_DIR, run / "source", ignore=shutil.ignore_patterns("__pycache__")
    )
    data, frame = load_data(config, ROOT)
    if not config["data_path"]:
        frame.to_parquet(run / "synthetic_market.parquet", index=False)
    data.news_metadata.to_parquet(run / "news_metadata.parquet", index=False)
    folds, test_start = walk_forward_splits(len(data.dates), config)
    write_json(
        run / "splits.json",
        {
            "folds": folds,
            "test_start_index": test_start,
            "test_start_date": str(data.dates[test_start])[:10],
        },
    )
    env = PortfolioEnv(
        data,
        config,
        end=folds[0]["train_end"],
        scaler=FeatureScaler.fit(data, folds[0]["train_end"]),
    )
    check_env(env, warn=True)
    rows = []
    for reward in config["reward_candidates"]:
        candidate = {**config, "reward": reward}
        for fold in folds:
            for seed in config["seeds"]:
                candidate = {**candidate, "window_id": f"validation_{fold['fold']}"}
                directory = (
                    run
                    / "validation_models"
                    / reward
                    / f"fold_{fold['fold']}_seed_{seed}"
                )
                print(
                    f"TRAIN reward={reward} fold={fold['fold']} seed={seed}", flush=True
                )
                model, scaler = train_model(
                    data, candidate, fold["train_end"], seed, directory
                )
                history = run_backtest(
                    data,
                    candidate,
                    fold["validation_start"] - 1,
                    fold["validation_end"],
                    [model],
                    scaler,
                )
                history.to_csv(directory / "validation_history.csv", index=False)
                save_asset_history(
                    history, data.assets, directory / "validation_asset_history.csv"
                )
                stats = performance(history, candidate["risk_free_annual"])
                score = (
                    stats["total_return"]
                    - config["selection_mdd_penalty"] * stats["max_drawdown"]
                )
                rows.append(
                    {
                        "reward": reward,
                        "fold": fold["fold"],
                        "seed": seed,
                        "score": score,
                        **stats,
                    }
                )
                pd.DataFrame(rows).to_csv(run / "validation.csv", index=False)
    validation = pd.DataFrame(rows)
    ranking = (
        validation.groupby("reward")
        .score.agg(["mean", "std", "count"])
        .sort_values(["mean"], ascending=False)
    )
    ranking.to_csv(run / "reward_ranking.csv")
    selected = str(ranking.index[0])
    # Freeze selection before the holdout; do not cherry-pick seeds.
    selection = {
        "reward": selected,
        "seeds": config["seeds"],
        "criterion": "mean(validation_return - penalty * validation_mdd)",
        "mdd_penalty": config["selection_mdd_penalty"],
        "test_used_for_selection": False,
        "data_hash": fingerprint(data),
        "source_hash": source_hash(),
    }
    write_json(run / "selection.json", selection)
    final_config = {**config, "reward": selected, "window_id": "final_holdout"}
    bundles = []
    for seed in config["seeds"]:
        directory = run / "final" / f"seed_{seed}"
        print(f"FINAL TRAIN reward={selected} seed={seed}", flush=True)
        train_model(data, final_config, test_start - 1, seed, directory)
        bundles.append(load_bundle(directory, data, exact_data=True))
    models, scaler = [bundle[0] for bundle in bundles], bundles[0][1]
    metrics, histories = compare(
        data, final_config, test_start - 1, len(data.dates) - 1, models, scaler, run
    )
    seed_stats = []
    for seed, model in zip(config["seeds"], models):
        history = run_backtest(
            data, final_config, test_start - 1, len(data.dates) - 1, [model], scaler
        )
        history.to_csv(run / "final" / f"seed_{seed}" / "test_history.csv", index=False)
        save_asset_history(
            history,
            data.assets,
            run / "final" / f"seed_{seed}" / "test_asset_history.csv",
        )
        seed_stats.append(
            {"seed": seed, **performance(history, config["risk_free_annual"])}
        )
    pd.DataFrame(seed_stats).to_csv(run / "test_seed_comparison.csv", index=False)
    stress = []
    for scenario in ("bull", "bear", "sideways", "crash", "volatile"):
        stress_config = {
            **final_config,
            "seed": config["seed"] + 10000,
            "dummy_macro_columns": data.macros,
        }
        scenario_data = prepare_data(
            generate(stress_config, scenario),
            data.assets,
            macro_columns=data.macros,
            news_mode="development",
        )
        directory = run / "stress" / scenario
        stats, _ = compare(
            scenario_data,
            {**final_config, "window_id": f"stress_{scenario}"},
            final_config["lookback"] - 1 + final_config["observation_lag"],
            len(scenario_data.dates) - 1,
            models,
            scaler,
            directory,
        )
        stress.extend(
            {"scenario": scenario, "strategy": strategy, **values}
            for strategy, values in stats.items()
        )
    pd.DataFrame(stress).to_csv(run / "stress.csv", index=False)
    # Sensitivity is descriptive and does not affect reward/model selection.
    sensitivity = []
    for bps in (0.0, 7.0, 20.0):
        cost_config = {**final_config, "transaction_cost_bps": bps, "slippage_bps": 0.0}
        history = run_backtest(
            data, cost_config, test_start - 1, len(data.dates) - 1, models, scaler
        )
        sensitivity.append(
            {
                "combined_cost_bps": bps,
                **performance(history, config["risk_free_annual"]),
            }
        )
    pd.DataFrame(sensitivity).to_csv(run / "cost_sensitivity.csv", index=False)
    prediction = predict_weights(data, final_config, models, scaler)
    save_prediction(prediction, run)
    summary = {
        "run_id": run_id,
        "synthetic": not config["data_path"],
        "dates": len(data.dates),
        "assets": len(data.assets),
        "macros": data.macros,
        "folds": len(folds),
        "seeds": len(config["seeds"]),
        "candidate_count": len(config["reward_candidates"]),
        "selected_reward": selected,
        "test_start": str(data.dates[test_start])[:10],
        "test_end": str(data.dates[-1])[:10],
        "models_trained": len(rows) + len(models),
        "steps_per_model": config["total_timesteps"],
        "total_training_steps": (len(rows) + len(models)) * config["total_timesteps"],
        "data_hash": fingerprint(data),
        "source_hash": source_hash(),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "torch",
                "gymnasium",
                "stable-baselines3",
                "pyarrow",
            ]
        },
        "final_test": metrics,
        "completed": True,
    }
    write_json(run / "summary.json", summary)
    build_report(run, summary, metrics, histories, rows, stress)
    write_json(ROOT / "runs/latest.json", {"run_id": run_id})
    print(f"COMPLETE {run}", flush=True)
    return run
