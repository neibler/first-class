"""Explicit single-model/seed workflows. Importing this module never trains."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import hashlib
import json
import pandas as pd
from .settings import bundle_input_root, ROOT, validate_config, write_json
from .data_loader import load_data, fingerprint
from .rebalancing import REBALANCE_POLICY
from .training import train_model, load_bundle
from .backtest import compare
from .rewards import canonical_reward


def subset(data, start, stop):
    dates = data.dates[start:stop]
    return replace(
        data,
        dates=dates,
        **{
            name: getattr(data, name)[start:stop]
            for name in ("returns", "rsi", "macd", "macro", "risk")
        },
        news_metadata=data.news_metadata[data.news_metadata.date.isin(dates)].copy(),
    )


def calendar_windows(dates, train_years=4, test_years=1):
    if (
        type(train_years) is not int
        or type(test_years) is not int
        or min(train_years, test_years) < 1
    ):
        raise ValueError("Train/test years must be positive integers")
    dates = pd.DatetimeIndex(dates)
    anchor = dates[0]
    windows = []
    while True:
        test_date = anchor + pd.DateOffset(years=train_years)
        stop_date = test_date + pd.DateOffset(years=test_years)
        if stop_date > dates[-1] + pd.offsets.BDay(1):
            break
        start, split, stop = map(
            int, dates.searchsorted([anchor, test_date, stop_date])
        )
        if split <= start or stop <= split:
            break
        windows.append({"start": start, "split": split, "stop": stop})
        anchor += pd.DateOffset(years=test_years)
    if not windows:
        raise ValueError(
            f"Insufficient data for Train {train_years} years / "
            f"Test {test_years} years; no training started"
        )
    return windows


def bundle_path(root, model_type, reward_type, seed, run_id, window_id="holdout"):
    from .model_factory import MODEL_TYPES

    if model_type not in MODEL_TYPES:
        raise ValueError("Unknown model type")
    for value in (run_id, window_id):
        if not value or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in value
        ):
            raise ValueError("Unsafe run/window identifier")
    return (
        Path(root)
        / "models"
        / model_type
        / run_id
        / window_id
        / f"{model_type}_{canonical_reward(reward_type)}_seed{seed}"
    )


def plan_runs(config, data, seeds, walk_forward=False, train_years=4, test_years=1):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide unique seeds")
    validate_config({**config, "seeds": seeds})
    if walk_forward:
        windows = calendar_windows(data.dates, train_years, test_years)
    else:
        split = int(len(data.dates) * (1 - config["test_fraction"]))
        windows = [{"start": 0, "split": split, "stop": len(data.dates)}]
    if any(
        w["split"] - w["start"] <= config["lookback"] or w["stop"] - w["split"] < 2
        for w in windows
    ):
        raise ValueError("Not enough training/lookback or holdout observations")
    return windows


def run_seed_experiments(
    config,
    model_type=None,
    reward_type=None,
    seeds=None,
    run_id=None,
    walk_forward=False,
    train_years=4,
    test_years=1,
    dry_run=False,
):
    kind = model_type or config.get("model_type", "ppo")
    reward = canonical_reward(reward_type or config["reward"])
    seeds = list(seeds if seeds is not None else config["seeds"])
    config = {
        **config,
        "model_type": kind,
        "reward": reward,
        "reward_candidates": [reward],
        "seeds": seeds,
    }
    data, frame = load_data(config, ROOT)
    windows = plan_runs(config, data, seeds, walk_forward, train_years, test_years)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plans = []
    for i, window in enumerate(windows):
        window_id = f"window_{i}" if walk_forward else "holdout"
        for seed in seeds:
            path = bundle_path(ROOT, kind, reward, seed, run_id, window_id)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            plans.append({**window, "window_id": window_id, "seed": seed, "path": path})
    if dry_run:
        for plan in plans:
            print(
                f"PLAN {kind} {reward} seed={plan['seed']} "
                f"train={str(data.dates[plan['start']])[:10]}"
                f"..{str(data.dates[plan['split']-1])[:10]} "
                f"test={str(data.dates[plan['split']])[:10]}"
                f"..{str(data.dates[plan['stop']-1])[:10]} -> {plan['path']}"
            )
        return [p["path"] for p in plans]
    for plan in plans:
        local = subset(data, plan["start"], plan["stop"])
        split = plan["split"] - plan["start"]
        run_config = {
            **config,
            "run_id": run_id,
            "window_id": plan["window_id"],
            "seed": plan["seed"],
        }
        train_model(local, run_config, split - 1, plan["seed"], plan["path"])
        write_json(
            plan["path"] / "run_spec.json",
            {
                "start_date": str(local.dates[0])[:10],
                "end_date": str(local.dates[-1])[:10],
                "test_start_index": split,
                "model_type": kind,
                "reward_type": reward,
                "seed": plan["seed"],
                "window_id": plan["window_id"],
                "data_hash": fingerprint(local),
                "data_seed": config["seed"],
                "synthetic": not config["data_path"],
            },
        )
        if not config["data_path"]:
            local.news_metadata.to_parquet(
                plan["path"] / "news_metadata.parquet", index=False
            )
            selected = pd.to_datetime(frame.date).isin(local.dates)
            frame[selected].to_parquet(
                plan["path"] / "synthetic_market.parquet", index=False
            )
    return [p["path"] for p in plans]


def evaluate_bundle(path):
    """Explicit evaluation only, never called automatically by training."""
    path = Path(path).resolve()
    spec = json.loads((path / "run_spec.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise ValueError("Training bundle is incomplete")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    data, _ = load_data(
        {**config, "seed": spec.get("data_seed", config["seed"])},
        bundle_input_root(path, ROOT),
    )
    dates = pd.DatetimeIndex(data.dates)
    data = subset(
        data,
        int(dates.searchsorted(spec["start_date"])),
        int(dates.searchsorted(spec["end_date"], side="right")),
    )
    model, scaler, config = load_bundle(path, data, exact_data=True)
    metrics, _ = compare(
        data,
        config,
        spec["test_start_index"] - 1,
        len(data.dates) - 1,
        [model],
        scaler,
        path / "evaluation",
    )
    # Model/hyperparameter/seed changes are allowed; environment/data/split must match.
    common = {
        k: v
        for k, v in config.items()
        if k
        not in {
            "model_type",
            "models",
            "device",
            "seed",
            "seeds",
            "run_id",
            "window_id",
            "n_steps",
            "batch_size",
            "checkpoint_steps",
            "mode",
            "reward_candidates",
        }
    }
    group = hashlib.sha256(
        json.dumps(
            {
                "config": common,
                "rebalance_policy": REBALANCE_POLICY,
                "data": fingerprint(data),
                "split": spec["test_start_index"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    write_json(
        path / "evaluation.json",
        {
            **spec,
            "completed": True,
            "comparison_group": group,
            "rebalance_policy": REBALANCE_POLICY,
            "final_test": metrics,
            "test_start": str(data.dates[spec["test_start_index"]])[:10],
            "test_end": str(data.dates[-1])[:10],
        },
    )
    return metrics
