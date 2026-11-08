"""Compare only completed saved holdouts; never train or select a serving model."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .accounting import equity_curve, sharpe_ratio

METRICS = ["cagr", "sharpe", "sortino", "mdd", "calmar"]


def metrics_from_returns(returns, annual_rf=0.0):
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Need finite daily simple returns > -1")
    equity = equity_curve(r)
    cagr = float(equity[-1] ** (252 / len(r)) - 1)
    mdd = float(np.max(1 - equity / np.maximum.accumulate(equity)))
    excess = r - np.expm1(np.log1p(annual_rf) / 252)
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    return {
        "cagr": cagr,
        "sharpe": sharpe_ratio(r, annual_rf),
        "sortino": (
            float(excess.mean() / downside * np.sqrt(252)) if downside > 1e-12 else None
        ),
        "mdd": mdd,
        "calmar": cagr / mdd if mdd > 1e-12 else None,
    }


def compare_completed(root, output):
    rows, seen = [], set()
    for path in sorted(Path(root).rglob("evaluation.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not meta.get("completed"):
            continue
        bundle = path.parent
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("completed"):
            continue
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(bundle / "evaluation/ppo_history.csv")
        if (
            frame.empty
            or not frame.date.is_unique
            or not frame.date.is_monotonic_increasing
        ):
            raise ValueError(f"Invalid evaluation dates: {bundle}")
        if (
            str(frame.date.iloc[0]) != meta["test_start"]
            or str(frame.date.iloc[-1]) != meta["test_end"]
        ):
            raise ValueError("Evaluation dates do not match metadata")
        key = (meta["comparison_group"], meta["model_type"], meta["seed"])
        if key in seen:
            raise ValueError(
                "Duplicate model/seed/holdout; select one experiment directory"
            )
        seen.add(key)
        rows.append(
            {
                "model": meta["model_type"],
                "reward": meta["reward_type"],
                "seed": meta["seed"],
                "window_id": meta["window_id"],
                "comparison_group": meta["comparison_group"],
                "test_start": meta["test_start"],
                "test_end": meta["test_end"],
                **metrics_from_returns(
                    frame.portfolio_return, config["risk_free_annual"]
                ),
            }
        )
    if not rows:
        output = Path(output)
        # Only remove outputs owned by this comparison, never saved model results.
        output.unlink(missing_ok=True)
        output.with_name(output.stem + "_seed_summary.csv").unlink(missing_ok=True)
        print("No completed model runs found. Stale comparison outputs removed.")
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    summary = table.groupby(["comparison_group", "model", "reward"])[METRICS].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output.with_name(output.stem + "_seed_summary.csv"))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runtime/models"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/model_comparison.csv")
    )
    args = parser.parse_args(argv)
    compare_completed(args.runs, args.output)


if __name__ == "__main__":
    main()
