"""CLI: full research run, or continued training from a saved bundle."""

import argparse
import json
from pathlib import Path
from .settings import ROOT, read_config, validate_config
from .experiment import run_experiment
from .data_loader import load_data
from .training import train_model


def build_parser():
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
    parser.add_argument("--model", choices=("ppo", "recurrent_ppo", "sac", "td3"))
    parser.add_argument("--reward", choices=("return", "sharpe", "return_mdd"))
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--mode", choices=("single", "walk-forward", "legacy"))
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is not None and args.seeds is not None:
        parser.error("Choose --seed or --seeds")
    if args.resume:
        if (
            args.dry_run
            or args.model
            or args.reward
            or args.timesteps
            or args.seed is not None
            or args.seeds
            or args.mode
        ):
            parser.error(
                "Resume uses saved settings; use only --extra-steps and --output"
            )
        if not args.output:
            parser.error("--resume requires --output (a new directory)")
        config = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
        config["total_timesteps"] = args.extra_steps
        validate_config(config)
        manifest = json.loads(
            (args.resume / "manifest.json").read_text(encoding="utf-8")
        )
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
        config = read_config(args.config)
        for argument, key in [
            (args.model, "model_type"),
            (args.reward, "reward"),
            (args.timesteps, "total_timesteps"),
            (args.device, "device"),
        ]:
            if argument is not None:
                config[key] = argument
        mode = (
            args.mode
            if args.mode is not None
            else config.get("mode", "single" if "model_type" in config else "legacy")
        )
        if mode not in ("single", "walk-forward", "legacy"):
            raise ValueError(
                f"Unsupported mode {mode!r}; choose single, walk-forward or legacy"
            )
        if mode == "legacy" and (
            args.model
            or args.reward
            or args.seed is not None
            or args.seeds
            or args.dry_run
        ):
            parser.error(
                "Use --mode single or --mode walk-forward with model/seed overrides"
            )
        if mode == "legacy":
            validate_config(config)
            run_experiment(config, args.run_id)
        else:
            from .model_runs import run_seed_experiments

            seeds = [args.seed] if args.seed is not None else args.seeds
            paths = run_seed_experiments(
                config,
                seeds=seeds,
                run_id=args.run_id,
                walk_forward=mode == "walk-forward",
                train_years=args.train_years,
                test_years=args.test_years,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                for path in paths:
                    print(f"Trained bundle: {path}")


if __name__ == "__main__":
    main()
