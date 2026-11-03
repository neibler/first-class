"""CLI: full research run, or continued training from a saved bundle."""

import argparse
import json
from pathlib import Path
from .settings import ROOT, read_config, validate_config
from .experiment import run_experiment
from .data_loader import load_data
from .training import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Existing training bundle; resume starts a fresh episode",
    )
    parser.add_argument("--extra-steps", type=int, default=5000)
    parser.add_argument("--output", type=Path, help="New directory for resumed bundle")
    args = parser.parse_args()
    if args.resume:
        if not args.output:
            parser.error("--resume requires --output (a new directory)")
        config = json.loads((args.resume / "config.json").read_text())
        config["total_timesteps"] = args.extra_steps
        validate_config(config)
        manifest = json.loads((args.resume / "manifest.json").read_text())
        data, _ = load_data(config, ROOT)
        train_model(
            data,
            config,
            manifest["train_end_index"],
            manifest["seed"],
            args.output,
            args.resume,
        )
        print(f"Resumed bundle saved: {args.output}")
    else:
        run_experiment(read_config(args.config), args.run_id)


if __name__ == "__main__":
    main()
