"""Preserve and validate the input location of a completed or resumed bundle."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from .settings import ROOT, bundle_input_root, write_json
from .data_loader import load_data, fingerprint


def load_bundle_inputs(bundle, config, default=ROOT, data_seed=None):
    bundle = Path(bundle).resolve()
    spec_path = bundle / "run_spec.json"
    spec = (
        json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    )
    local = {
        **config,
        "seed": spec.get(
            "data_seed", data_seed if data_seed is not None else config["seed"]
        ),
    }
    if spec.get("synthetic", False):
        local["data_path"] = str(bundle / "synthetic_market.parquet")
        local["news_path"] = str(bundle / "news_metadata.parquet")
    return load_data(local, bundle_input_root(bundle, default))


def preserve_inputs(output, data, config, run_spec, resume=None):
    output = Path(output).resolve()
    synthetic = not config["data_path"]
    if run_spec is not None and run_spec.get("synthetic", synthetic) != synthetic:
        raise ValueError("run_spec synthetic flag disagrees with input configuration")
    if synthetic and run_spec is not None:
        count = len(data.assets)
        frame = pd.DataFrame(
            {
                "date": np.repeat(data.dates, count),
                "ticker": np.tile(data.assets, len(data.dates)),
                "log_return": data.returns.ravel(),
                "rsi": data.rsi.ravel(),
                "macd": data.macd.ravel(),
            }
        )
        for i, name in enumerate(data.macros):
            frame[name] = np.repeat(data.macro[:, i], count)
        frame.to_parquet(output / "synthetic_market.parquet", index=False)
        data.news_metadata.to_parquet(output / "news_metadata.parquet", index=False)
    elif resume and config["data_path"]:
        root = bundle_input_root(resume, ROOT).resolve()
        if data is not None:
            from .model_runs import subset

            original, _ = load_data(config, root)
            dates = pd.DatetimeIndex(original.dates)
            boundaries = pd.DatetimeIndex([data.dates[0], data.dates[-1]])
            if not boundaries.isin(dates).all():
                raise ValueError("Bound input window is missing")
            original = subset(
                original,
                int(dates.get_loc(boundaries[0])),
                int(dates.get_loc(boundaries[1])) + 1,
            )
            if fingerprint(original) != fingerprint(data):
                raise ValueError("Bound input differs from resumed data")
        hashes = {}
        for key in ("data_path", "news_path"):
            if config.get(key):
                name = config[key]
                path = (root / name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError(
                        "Resume input must be within its declared input root"
                    )
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Rebase to the new bundle; never copy the source's relative marker blindly.
        try:
            location = {"relative_root": os.path.relpath(root, output)}
        except ValueError:  # Different Windows drives cannot have a relative path.
            location = {"absolute_root": str(root)}
        write_json(output / "installed_inputs.json", {**location, "sha256": hashes})
