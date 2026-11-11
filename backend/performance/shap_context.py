"""Attach verified market context to saved SHAP without recalculating SHAP."""

from copy import deepcopy
import json
import re
import logging

import pandas as pd


def context_at(data, features, observation, index):
    if len(features) != len(observation):
        raise ValueError("Feature/observation length mismatch")
    values = {}
    assets = {a.lower(): i for i, a in enumerate(data.assets)}
    macros = {name: i for i, name in enumerate(data.macros)}
    for feature, model_value in zip(features, observation):
        lag = re.fullmatch(r"log_return_lag_(\d+)_([a-z0-9]+)", feature)
        if lag and lag[2] in assets and index >= int(lag[1]):
            values[feature] = float(data.returns[index - int(lag[1]), assets[lag[2]]])
            continue
        for prefix, array, scale in [
            ("rsi_div_100_", data.rsi, 100),
            ("macd_", data.macd, 1),
            ("risk_score_", data.risk, 1),
        ]:
            if feature.startswith(prefix) and feature[len(prefix) :] in assets:
                values[feature] = float(
                    array[index, assets[feature[len(prefix) :]]] / scale
                )
                break
        if feature.startswith("macro_") and feature[6:] in macros:
            values[feature] = float(data.macro[index, macros[feature[6:]]])
        if feature.startswith("current_weight_") or feature in {
            "drawdown",
            "mdd",
            "rebalance_phase",
        }:
            values[feature] = float(model_value)
    previous_index = max(0, index - 5)
    previous = (
        {
            f: float(data.macro[previous_index, macros[f[6:]]])
            for f in values
            if f.startswith("macro_") and f[6:] in macros
        }
        if previous_index < index
        else {}
    )
    return {
        "source": "verified_market_data",
        "as_of": str(data.dates[index])[:10],
        "values": values,
        "previous_values": previous,
        "comparison_date": str(data.dates[previous_index])[:10] if previous else None,
        "comparison_trading_days": index - previous_index if previous else None,
    }


def attach_market_context(payload, bundle):
    result = deepcopy(payload)  # Never mutate the cached original SHAP response.
    try:
        from backend.investment_ai.settings import ROOT, bundle_input_root
        from backend.investment_ai.data_loader import load_data, fingerprint
        from backend.investment_ai.model_runs import subset
        from backend.investment_ai.training import feature_names

        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        spec = json.loads((bundle / "run_spec.json").read_text(encoding="utf-8"))
        data, _ = load_data(
            {**config, "seed": spec.get("data_seed", config["seed"])},
            bundle_input_root(bundle, ROOT),
        )
        dates = pd.DatetimeIndex(data.dates)
        data = subset(
            data,
            int(dates.searchsorted(spec["start_date"])),
            int(dates.searchsorted(spec["end_date"], side="right")),
        )
        if (
            fingerprint(data) != payload.get("data_hash")
            or feature_names(data, config) != payload["features"]
        ):
            raise ValueError("SHAP market data or feature order mismatch")
        indices = {str(date)[:10]: i for i, date in enumerate(data.dates)}
        contexts = {}
        for asset in result["assets"].values():
            for row in asset["decisions"]:
                index = indices[row["date"]]
                if row["date"] not in contexts:
                    contexts[row["date"]] = context_at(
                        data, payload["features"], row["feature_values_raw"], index
                    )
                context = deepcopy(contexts[row["date"]])
                # Portfolio state belongs to this precise observation.
                for f, v in zip(payload["features"], row["feature_values_raw"]):
                    if f.startswith("current_weight_") or f in {
                        "drawdown",
                        "mdd",
                        "rebalance_phase",
                    }:
                        context["values"][f] = float(v)
                row["market_context"] = context
        result["market_context_status"] = "verified"
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        logging.getLogger(__name__).warning(
            "Verified market context unavailable for SHAP", exc_info=True
        )
        for asset in result["assets"].values():
            for row in asset["decisions"]:
                row.pop("market_context", None)
        result["market_context_status"] = "unavailable"
    return result
