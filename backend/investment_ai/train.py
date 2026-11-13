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
    parser.add_argument("--extra-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, help="New directory for resumed bundle")
    parser.add_argument("--model", choices=("ppo", "recurrent_ppo", "sac", "td3"))
    parser.add_argument("--reward", choices=("return", "sharpe", "return_mdd"))
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--mode", choices=("single", "walk-forward", "legacy"))
    parser.add_argument("--train-years", type=int, default=None)
    parser.add_argument("--test-years", type=int, default=None)
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
            or args.train_years is not None
            or args.test_years is not None
        ):
            parser.error(
                "Resume uses saved settings; use only --extra-steps and --output"
            )
        if not args.output:
            parser.error("--resume requires --output (a new directory)")
        config = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
        from .model_factory import model_parameters
        from .checkpoints import resolve_checkpoint
        from .settings import bundle_input_root
        from .model_runs import subset
        import pandas as pd

        kind = config.get("model_type", "ppo")
        steps = args.extra_steps
        if steps is None:
            n = model_parameters(kind, config).get("n_steps", 1)
            steps = ((5000 + n - 1) // n) * n
        config["total_timesteps"] = steps
        if args.device:
            config["device"] = args.device
        validate_config(config)
        _, manifest = resolve_checkpoint(args.resume)
        spec_path = args.resume / "run_spec.json"
        spec = (
            json.loads(spec_path.read_text(encoding="utf-8"))
            if spec_path.exists()
            else None
        )
        input_config = (
            {**config, "seed": spec.get("data_seed", config["seed"])}
            if spec
            else config
        )
        data, _ = load_data(input_config, bundle_input_root(args.resume, ROOT))
        if spec:
            dates = pd.DatetimeIndex(data.dates)
            if (
                pd.Timestamp(spec["start_date"]) not in dates
                or pd.Timestamp(spec["end_date"]) not in dates
            ):
                raise ValueError("Resume window boundaries are absent from inputs")
            data = subset(
                data,
                int(dates.searchsorted(spec["start_date"])),
                int(dates.searchsorted(spec["end_date"], side="right")),
            )
            if spec["test_start_index"] - 1 != manifest["train_end_index"]:
                raise ValueError("Resume split mismatch")
        train_model(
            data,
            config,
            manifest["train_end_index"],
            manifest["seed"],
            args.output,
            args.resume,
            run_spec=spec,
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
        from .settings import effective_mode

        mode = effective_mode(config, args.mode)
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
                train_years=4 if args.train_years is None else args.train_years,
                test_years=1 if args.test_years is None else args.test_years,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                for path in paths:
                    print(f"Trained bundle: {path}")


if __name__ == "__main__":
    main()
