"""Generate portfolio proposals; never places orders."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .settings import ROOT, write_json
from .data_loader import load_data
from .portfolio_env import PortfolioEnv
from .training import load_bundle
from .accounting import safeguard_breached
from .prediction import PredictionSession
from .episode_clock import episode_clock, rebalance_phase
from .rebalancing import (
    get_rebalance_reasons,
    filter_target_weights,
    portfolio_weight_distance,
    MIN_WEIGHT_THRESHOLD,
)


def resolve_run(path=None):
    if path:
        return Path(path).resolve()
    pointer_path = ROOT / "runs/latest.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(
            f"No selected model: {pointer_path}. Install and activate a model with "
            "python scripts/install_trained_models.py --activate ppo, "
            "or supply --bundle."
        )
    pointer = json.loads(pointer_path.read_text())
    return ROOT / "runs" / pointer["run_id"]


def predict_weights(data, config, models, scaler, state=None, predictors=None):
    env = PortfolioEnv(data, config, scaler=scaler)
    date = str(data.dates[-1])[:10]
    clock = episode_clock(data.dates, state)
    if state is None:
        state = {
            "date": date,
            "weights": {a: 0.0 for a in data.assets},
            "portfolio_value": 1.0,
            "peak_value": 1.0,
            "max_drawdown": 0.0,
            "days_since_rebalance": 0,
            "stopped": False,
        }
        mode = "new_portfolio_cash"
    else:
        mode = "provided_portfolio_state"
    if state["date"] != date or set(state["weights"]) != set(data.assets):
        raise ValueError("State date/assets must match latest input")
    held = np.array([state["weights"][a] for a in data.assets], dtype=float)
    if (
        not np.isfinite(held).all()
        or np.any(held < 0)
        or not (np.isclose(held.sum(), 1) or np.isclose(held.sum(), 0))
    ):
        raise ValueError("Holdings must be fully invested or entirely cash")
    if held.sum() > 0 and np.isclose(held.sum(), 1):
        held /= held.sum()
    else:
        held[:] = 0
    value, peak, mdd = (
        state["portfolio_value"],
        state["peak_value"],
        state["max_drawdown"],
    )
    if (
        not np.isfinite([value, peak, mdd]).all()
        or not 0 < value <= peak
        or not 0 <= mdd <= 1
    ):
        raise ValueError("Invalid portfolio value/peak/MDD")
    dd = 1 - value / peak
    if mdd + 1e-10 < dd:
        raise ValueError("MDD cannot be smaller than current drawdown")
    elapsed = state["days_since_rebalance"]
    if not isinstance(state.get("safeguard_triggered", False), bool):
        raise ValueError("safeguard_triggered must be boolean")
    if not isinstance(state.get("stopped", False), bool):
        raise ValueError("stopped must be boolean")
    if not isinstance(elapsed, int) or elapsed < 0:
        raise ValueError("days_since_rebalance must be a nonnegative integer")
    phase = rebalance_phase(clock["episode_elapsed_bars"], config["rebalance_days"])
    obs = env.observation_at(len(data.dates) - 1, held, dd, mdd, phase)
    recurrent = config.get("model_type", "ppo") == "recurrent_ppo"
    stopped = (
        state.get("stopped", False)
        or state.get("safeguard_triggered", False)
        or safeguard_breached(mdd, config["max_drawdown"])
    )
    if recurrent and not stopped and not predictors:
        raise ValueError(
            "Recurrent inference requires explicit PredictionSession state"
        )
    if predictors is None:
        predictors = [
            PredictionSession(m, config.get("model_type", "ppo")) for m in models
        ]
    if not stopped and not predictors:
        raise ValueError("At least one prediction session is required")
    # All policies compute a fresh target every day, even when no trade is due.
    # A latched safety stop needs no further model calls until explicit reset.
    raw_target = (
        np.zeros(len(data.assets))
        if stopped
        else np.mean([p.predict(obs, env) for p in predictors], axis=0)
    )
    target = (
        raw_target.copy()
        if stopped
        else filter_target_weights(
            raw_target,
            config.get("min_weight_threshold", MIN_WEIGHT_THRESHOLD),
            config.get("max_holdings"),
        )
    )
    distance = portfolio_weight_distance(held, target)
    reasons = get_rebalance_reasons(
        data.dates[-1],
        data.dates[-2] if len(data.dates) > 1 else None,
        held,
        target,
        data.risk[-1],
        mdd,
        config,
        stopped=stopped,
    )
    if reasons == ["mdd_stop"]:
        weights, status = np.zeros(len(data.assets)), "cash_safeguard"
    elif reasons == ["hold"]:
        weights, status = held.copy(), "hold_until_rebalance"
    else:
        weights, status = target.copy(), "target_proposal"
    # A normal signal queues a t+1 target even if today's weights already match:
    # prices can drift before the fill. A latched cash stop never queues an exit.
    should_trade = reasons != ["hold"] and (
        reasons != ["mdd_stop"] or bool(np.any(held > 1e-12))
    )
    return {
        "as_of_date": date,
        "episode_start_date": clock["episode_start_date"],
        "episode_elapsed_bars": clock["episode_elapsed_bars"],
        "rebalance_phase": phase,
        "model_target_weights": dict(zip(data.assets, map(float, raw_target))),
        "raw_target_weights": dict(zip(data.assets, map(float, raw_target))),
        "filtered_target_weights": dict(zip(data.assets, map(float, target))),
        "portfolio_weight_distance": distance,
        "max_risk_score": float(data.risk[-1].max()),
        "rebalance_reasons": reasons,
        "should_rebalance": should_trade,
        "mode": mode,
        "status": status,
        "safeguard_triggered": status == "cash_safeguard",
        "observation_date": date,
        "decision_date": date,
        "execution_date": None,
        "first_return_end_date": None,
        "first_return_end_delay_trading_bars": 1 if stopped else 2,
        "execution_delay_trading_bars": 0 if stopped else 1,
        "note": (
            "Immediate cash safeguard proposal. No order submitted."
            if stopped
            else "Target proposal for the next eligible close; "
            "reconcile holdings at execution. No order submitted."
        ),
        "cash_weight": float(np.clip(1 - weights.sum(), 0, 1)),
        "weights": dict(zip(data.assets, map(float, weights))),
    }


def save_prediction(prediction, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "latest_weights.json", prediction)
    pd.DataFrame(
        [
            {
                "date": prediction["as_of_date"],
                "ticker": asset,
                "target_weight": weight,
                "status": prediction["status"],
            }
            for asset, weight in prediction["weights"].items()
        ]
    ).to_csv(output / "latest_weights.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--news", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path, help="Single-model bundle")
    args = parser.parse_args(argv)
    if args.bundle:
        if args.run:
            parser.error("Choose --bundle or --run")
        from .serving import InvestmentService

        service = InvestmentService(args.bundle)
        state = (
            json.loads(args.state.read_text(encoding="utf-8-sig"))
            if args.state
            else None
        )
        if args.data or args.news:
            parser.error("Use API market/news inputs for an explicit single bundle")
        result = service.optimize(state=state)
        if args.output:
            save_prediction(
                {
                    **result,
                    "as_of_date": result["as_of"],
                    "weights": result["target_weights"],
                },
                args.output,
            )
        print(json.dumps(result, indent=2))
        return
    run = resolve_run(args.run)
    selected = json.loads((run / "selection.json").read_text())
    paths = [run / "final" / f"seed_{seed}" for seed in selected["seeds"]]
    config = json.loads((paths[0] / "config.json").read_text())
    config["macro_columns"] = json.loads((paths[0] / "schema.json").read_text())[
        "macros"
    ]
    if args.data:
        config["data_path"] = str(args.data.resolve())
        config["news_path"] = str(args.news.resolve()) if args.news else None
    elif args.news:
        raise ValueError("--news requires --data")
    recurrent = config.get("model_type") == "recurrent_ppo"
    if recurrent and (len(paths) != 1 or args.state):
        raise ValueError(
            "Legacy recurrent CLI requires one model and a fresh cash episode; "
            "use the persistent API for an existing recurrent episode."
        )
    data, _ = load_data(config, ROOT)
    bundles = [load_bundle(path, data) for path in paths]
    state = (
        json.loads(args.state.read_text(encoding="utf-8-sig")) if args.state else None
    )
    result = predict_weights(
        data,
        config,
        [b[0] for b in bundles],
        bundles[0][1],
        state,
        predictors=(
            [PredictionSession(bundles[0][0], "recurrent_ppo")] if recurrent else None
        ),
    )
    save_prediction(result, args.output or run / "inference")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
