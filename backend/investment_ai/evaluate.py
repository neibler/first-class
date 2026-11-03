"""Re-evaluate frozen final models, rejecting changed historical data."""

import argparse
import json
from pathlib import Path
from .settings import ROOT, write_json
from .infer import resolve_run
from .data_loader import load_data
from .training import load_bundle
from .backtest import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = resolve_run(args.run)
    selection = json.loads((run / "selection.json").read_text())
    paths = [run / "final" / f"seed_{seed}" for seed in selection["seeds"]]
    config = json.loads((paths[0] / "config.json").read_text())
    data, _ = load_data(config, ROOT)
    bundles = [load_bundle(path, data, exact_data=True) for path in paths]
    split = json.loads((run / "splits.json").read_text())["test_start_index"]
    output = args.output or run / "reevaluation"
    metrics, _ = compare(
        data,
        config,
        split - 1,
        len(data.dates) - 1,
        [b[0] for b in bundles],
        bundles[0][1],
        output,
    )
    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
