"""Validated configuration and versioned experiment paths."""

import json
import os
from pathlib import Path
import numpy as np

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = (
    Path(os.environ.get("INVESTMENT_AI_HOME", SOURCE_DIR.parent)).expanduser().resolve()
)
SCHEMA_VERSION = 3
REWARDS = ("return", "sharpe", "return_mdd", "portfolio_return", "rolling_sharpe")


def read_config(path=None):
    config = json.loads(
        Path(path or ROOT / "config.json").read_text(encoding="utf-8-sig")
    )
    validate_config(config)
    return config


def validate_config(c):
    if c.get("observation_lag", 0) != 0 or c.get("execution_delay", 1) != 1:
        raise ValueError("Schema v3 requires observation_lag=0 and execution_delay=1")
    if c.get("news_mode") not in {"required", "development"}:
        raise ValueError("Explicit news_mode required")
    for key in (
        "lookback",
        "total_timesteps",
        "n_steps",
        "batch_size",
        "rebalance_days",
        "reward_window",
        "checkpoint_steps",
        "initial_train_days",
        "validation_days",
        "folds",
        "min_episode_days",
    ):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["observation_lag"] < 0 or not isinstance(c["observation_lag"], int):
        raise ValueError("observation_lag must be a nonnegative integer")
    if c["batch_size"] < 2 or c["n_steps"] < 2 or c["reward_window"] < 2:
        raise ValueError("batch_size, n_steps, reward_window must be >= 2")
    if c["total_timesteps"] % c["n_steps"] or c["n_steps"] % c["batch_size"]:
        raise ValueError("Invalid PPO rollout/batch divisibility")
    if c["checkpoint_steps"] % c["n_steps"]:
        raise ValueError("checkpoint_steps must be a multiple of n_steps")
    for key in (
        "max_drawdown",
        "transaction_cost_bps",
        "slippage_bps",
        "logit_scale",
        "mdd_lambda",
        "risk_free_annual",
        "selection_mdd_penalty",
    ):
        if not np.isfinite(c[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < c["max_drawdown"] < 1 or not 0 < c["test_fraction"] < 0.5:
        raise ValueError("Invalid max_drawdown/test_fraction")
    if (
        min(
            c["transaction_cost_bps"],
            c["slippage_bps"],
            c["mdd_lambda"],
            c["selection_mdd_penalty"],
        )
        < 0
    ):
        raise ValueError("Costs and mdd_lambda must be nonnegative")
    if (c["transaction_cost_bps"] + c["slippage_bps"]) >= 5000 or not 0 < c[
        "logit_scale"
    ] <= 10:
        raise ValueError("Invalid cost/logit_scale")
    if (
        c["risk_free_annual"] <= -1
        or c["reward"] not in REWARDS
        or not set(c["reward_candidates"]).issubset(REWARDS)
    ):
        raise ValueError("Invalid risk-free rate/reward")
    if (
        not c["reward_candidates"]
        or not c["seeds"]
        or len(set(c["seeds"])) != len(c["seeds"])
    ):
        raise ValueError("Provide unique seeds and reward candidates")
    from .rewards import canonical_reward

    if len({canonical_reward(x) for x in c["reward_candidates"]}) != len(
        c["reward_candidates"]
    ):
        raise ValueError("Duplicate reward candidates")
    if any(
        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32
        for s in [c["seed"], *c["seeds"]]
    ):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if not c["assets"] or len(c["assets"]) != len(set(c["assets"])):
        raise ValueError("Provide a nonempty unique asset list")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
