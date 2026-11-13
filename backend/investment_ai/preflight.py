"""Validate the exact training inputs and environment without learning."""

import argparse
import json
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
from .settings import ROOT, read_config
from .data_loader import load_data, fingerprint
from .portfolio_env import PortfolioEnv
from .splits import walk_forward_splits
from .model_runs import plan_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "walk-forward", "legacy"))
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--test-years", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    from .settings import effective_mode

    mode = effective_mode(config, args.mode)
    if mode == "legacy":
        from .experiment import validate_legacy_config

        validate_legacy_config(config)
    data, _ = load_data(config, ROOT)
    windows = None
    if mode in ("single", "walk-forward"):
        windows = plan_runs(
            config,
            data,
            config["seeds"],
            mode == "walk-forward",
            args.train_years,
            args.test_years,
        )
        folds, test_start = windows, windows[-1]["split"]
    else:
        folds, test_start = walk_forward_splits(len(data.dates), config)
    env = PortfolioEnv(data, config)
    check_env(env, warn=True)
    report = {
        "training_executed": False,
        "market_source": (
            "REAL_INPUT" if config["data_path"] else "SYNTHETIC_NOT_REAL_PRICES"
        ),
        "news_source": config["news_path"],
        "dates": len(data.dates),
        "rows": len(data.dates) * len(data.assets),
        "start": str(data.dates[0])[:10],
        "end": str(data.dates[-1])[:10],
        "assets": data.assets,
        "macro_columns": data.macros,
        "risk_min": float(data.risk.min()),
        "risk_max": float(data.risk.max()),
        "observation_size": env.observation_space.shape[0],
        "action_size": env.action_space.shape[0],
        "folds": len(folds),
        "holdout_days": (
            windows[-1]["stop"] - test_start
            if windows
            else len(data.dates) - test_start
        ),
        "fingerprint": fingerprint(data),
        "max_holdings": config.get("max_holdings"),
        "training_windows": (
            [
                {
                    "train_start": str(data.dates[w["start"]])[:10],
                    "train_end": str(data.dates[w["split"] - 1])[:10],
                    "test_start": str(data.dates[w["split"]])[:10],
                    "test_end": str(data.dates[w["stop"] - 1])[:10],
                    "test_days": w["stop"] - w["split"],
                }
                for w in windows
            ]
            if windows is not None
            else None
        ),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
