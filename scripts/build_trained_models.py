"""Package completed, evaluated models and exact inputs; never train."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from backend.investment_ai.settings import ROOT  # noqa: E402

RUNTIME = ROOT
KINDS = ("ppo", "recurrent_ppo", "sac", "td3")


def verify_inputs(config, spec, expected):
    import sys

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    import pandas as pd
    from backend.investment_ai.data_loader import load_data, fingerprint
    from backend.investment_ai.model_runs import subset

    for key in ("data_path", "news_path"):
        if not (RUNTIME / config[key]).resolve().is_relative_to(RUNTIME.resolve()):
            raise ValueError("Input resolves outside runtime")
    data, _ = load_data(
        {**config, "seed": spec.get("data_seed", config.get("seed", 42))}, RUNTIME
    )
    dates = pd.DatetimeIndex(data.dates)
    if any(pd.Timestamp(spec[k]) not in dates for k in ("start_date", "end_date")):
        raise ValueError("Bundle input window is missing")
    data = subset(
        data,
        int(dates.searchsorted(spec["start_date"])),
        int(dates.searchsorted(spec["end_date"], side="right")),
    )
    if fingerprint(data) != expected:
        raise ValueError("Bundle input fingerprint mismatch")


def build(seed=None, selections=None):
    selections = selections or {}
    if set(selections) - set(KINDS):
        raise ValueError("Unknown model selection")
    periods = set()
    payload = {}
    models = {}

    def add(path):
        # Preserve configured archive names; resolve only for boundary validation.
        path = Path(path).absolute()
        if not path.resolve().is_relative_to(RUNTIME.resolve()) or ".." in path.parts:
            raise ValueError("Only runtime files may be shared")
        name = path.relative_to(RUNTIME).as_posix()
        payload[name] = path.read_bytes()

    # Explicit selections define the entire package; never add unselected models.
    for kind in selections or dict.fromkeys(KINDS):
        candidates = [
            p.parent
            for p in (RUNTIME / "models" / kind).rglob("manifest.json")
            if not {"checkpoint_generations", "checkpoints"}.intersection(
                p.relative_to(RUNTIME / "models" / kind).parts
            )
            and json.loads(p.read_text(encoding="utf-8")).get("completed")
        ]
        if kind in selections:
            selected = (RUNTIME / selections[kind]).absolute()
            if not selected.resolve().is_relative_to(RUNTIME.resolve()):
                raise ValueError("Selected bundle must be inside runtime")
            candidates = [p for p in candidates if p.resolve() == selected]
        if seed is not None:
            candidates = [
                p
                for p in candidates
                if json.loads((p / "manifest.json").read_text(encoding="utf-8")).get(
                    "seed"
                )
                == seed
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"Select exactly one completed {kind} bundle using --seed "
                f"or --bundle {kind}=PATH"
            )
        bundle = candidates[0]
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if config.get("model_type") != kind:
            raise ValueError("Model type mismatch")
        spec = json.loads((bundle / "run_spec.json").read_text(encoding="utf-8"))
        evaluation = json.loads(
            (bundle / "evaluation.json").read_text(encoding="utf-8")
        )
        if not evaluation.get("completed"):
            raise ValueError("Evaluation is incomplete")
        for key in ("model_type", "seed", "data_hash"):
            if meta.get(key) != spec.get(key) or spec.get(key) != evaluation.get(key):
                raise ValueError(f"Bundle metadata mismatch: {key}")
        for key in ("start_date", "end_date", "test_start_index"):
            if spec[key] != evaluation[key]:
                raise ValueError(f"Evaluation period mismatch: {key}")
        with (bundle / "evaluation/ppo_history.csv").open(
            encoding="utf-8", newline=""
        ) as f:
            dates = [row["date"] for row in csv.DictReader(f)]
        if (
            not dates
            or dates != sorted(set(dates))
            or dates[0] != evaluation["test_start"]
            or dates[-1] != evaluation["test_end"]
        ):
            raise ValueError("Evaluation history dates mismatch")
        periods.add(
            (
                spec["start_date"],
                spec["end_date"],
                spec["test_start_index"],
                meta["train_start"],
                meta["train_end"],
                evaluation["test_start"],
                evaluation["test_end"],
                evaluation["comparison_group"],
            )
        )
        verify_inputs(config, spec, meta["data_hash"])
        for name in (
            "model.zip",
            "config.json",
            "schema.json",
            "scaler.json",
            "manifest.json",
            "run_spec.json",
            "evaluation.json",
        ):
            add(bundle / name)
        if (bundle / "shap.json").exists():
            add(bundle / "shap.json")
        for path in (bundle / "evaluation").glob("*.csv"):
            add(path)
        for key in ("data_path", "news_path"):
            path = Path(config[key])
            if path.is_absolute():
                raise ValueError("Shared model inputs must use relative paths")
            add(RUNTIME / path)
        provenance = (RUNTIME / config["data_path"]).parent / "provenance.json"
        if provenance.exists():
            # Local machine paths are not needed by recipients.
            report = json.loads(provenance.read_text(encoding="utf-8"))
            for key in ("market_source", "news_source"):
                report[key] = Path(report[key]).name
            shared_provenance = provenance.with_name("shared_provenance.json")
            payload[shared_provenance.relative_to(RUNTIME).as_posix()] = json.dumps(
                report, indent=2
            ).encode()
        models[kind] = {
            "path": bundle.relative_to(RUNTIME).as_posix(),
            "steps": meta["actual_timesteps"],
            "data_hash": meta["data_hash"],
        }
    if len({m["data_hash"] for m in models.values()}) != 1:
        raise ValueError("Models use different input datasets")
    if len(periods) != 1:
        raise ValueError(
            "Models use different training/evaluation periods or comparison groups"
        )
    output = PROJECT / "shared/trained_models.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(payload.items()):
            z.writestr(name, content)
    manifest = {
        "format_version": 1,
        "models": models,
        "purpose": "inference and evaluation; replay buffers excluded",
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in payload.items()},
    }
    (output.parent / "trained_models_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packaged {len(payload)} files, {output.stat().st_size:,} bytes: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bundle", action="append", default=[], metavar="MODEL=PATH")
    args = parser.parse_args()
    choices = {}
    for value in args.bundle:
        kind, separator, path = value.partition("=")
        if not separator or not path or kind in choices:
            parser.error("Use one --bundle MODEL=PATH per model")
        choices[kind] = path
    build(seed=args.seed, selections=choices)
