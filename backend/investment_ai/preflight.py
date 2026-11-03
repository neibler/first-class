"""Validate the exact training inputs and environment without learning."""

import argparse
import json
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
from .settings import ROOT, read_config
from .data_loader import load_data, fingerprint
from .portfolio_env import PortfolioEnv
from .splits import walk_forward_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    data, _ = load_data(config, ROOT)
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
        "holdout_days": len(data.dates) - test_start,
        "fingerprint": fingerprint(data),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
