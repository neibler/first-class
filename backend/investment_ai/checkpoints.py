"""Publish a complete model/buffer/manifest generation with one atomic pointer."""

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_checkpoint(bundle):
    bundle = Path(bundle)
    pointer = bundle / "checkpoint.json"
    if not pointer.exists():
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_type") in {"sac", "td3"}:
            raise ValueError(
                "Legacy off-policy resume has no verified model/buffer pair"
            )
        return bundle, manifest
    info = json.loads(pointer.read_text(encoding="utf-8"))
    folder = (bundle / info["generation"]).resolve()
    if not folder.is_relative_to(bundle.resolve() / "checkpoint_generations"):
        raise ValueError("Unsafe checkpoint generation")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    required = {"model.zip", "manifest.json"}
    if manifest.get("model_type") in {"sac", "td3"}:
        required.add("replay_buffer.pkl")
    if set(info["sha256"]) != required:
        raise ValueError("Incomplete checkpoint generation")
    for name, expected in info["sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError("Checkpoint checksum mismatch")
    return folder, manifest


def publish_checkpoint(model, output, manifest):
    output = Path(output)
    generation = (
        output
        / "checkpoint_generations"
        / (str(model.num_timesteps) + "_" + uuid4().hex)
    )
    generation.mkdir(parents=True)
    model.save(generation / "model.zip")
    if manifest["model_type"] in {"sac", "td3"}:
        model.save_replay_buffer(generation / "replay_buffer.pkl")
    (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    info = {
        "generation": generation.relative_to(output).as_posix(),
        "sha256": {p.name: digest(p) for p in generation.iterdir()},
    }
    pending = output / "checkpoint.pending.json"
    pending.write_text(json.dumps(info), encoding="utf-8")
    pending.replace(output / "checkpoint.json")
    # Compatibility exports for evaluation/sharing. Resume only reads the pointer.
    for name in ("model.zip", "manifest.json"):
        temp = output / (name + ".pending")
        shutil.copyfile(generation / name, temp)
        temp.replace(output / name)
