"""Package completed, evaluated models and exact inputs; never train."""

import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / "runtime"
KINDS = ("ppo", "recurrent_ppo", "sac", "td3")


def build():
    payload = {}
    models = {}

    def add(path):
        path = path.resolve()
        if not path.is_relative_to(RUNTIME):
            raise ValueError("Only runtime files may be shared")
        name = path.relative_to(RUNTIME).as_posix()
        payload[name] = path.read_bytes()

    for kind in KINDS:
        candidates = [
            p.parent
            for p in (RUNTIME / "models" / kind).rglob("manifest.json")
            if json.loads(p.read_text(encoding="utf-8")).get("completed")
        ]
        if len(candidates) != 1:
            raise ValueError(f"Select exactly one completed {kind} bundle")
        bundle = candidates[0]
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if config.get("model_type") != kind:
            raise ValueError("Model type mismatch")
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
    output = PROJECT / "shared/trained_models.zip"
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
    build()
