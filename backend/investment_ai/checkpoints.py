"""Atomic model/buffer pairs with current and previous generation retention."""

import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def publication_lock(output):
    """OS lock is released even when the writer process exits unexpectedly."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoint.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import os

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verified_generation(bundle, info):
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
    try:
        return verified_generation(bundle, info)
    except (ValueError, OSError, KeyError):
        if not info.get("previous"):
            raise
        # Recovery always verifies the entire previous model/buffer pair.
        return verified_generation(bundle, info["previous"])


def prune_generations(output, info):
    """Caller holds publication lock; never follow directory links for deletion."""
    root = output.resolve() / "checkpoint_generations"
    keep = {(output / info["generation"]).resolve()}
    if info.get("previous"):
        keep.add((output / info["previous"]["generation"]).resolve())
    for folder in root.iterdir():
        if (
            re.fullmatch(r"\d+_[0-9a-f]{32}", folder.name)
            and folder.is_dir()
            and not folder.is_symlink()
            and not getattr(folder, "is_junction", lambda: False)()
            and folder.resolve().parent == root
            and folder.resolve() not in keep
        ):
            shutil.rmtree(folder)


def publish_checkpoint(model, output, manifest):
    output = Path(output).resolve()
    with publication_lock(output):
        previous = None
        pointer = output / "checkpoint.json"
        if pointer.exists():
            old = json.loads(pointer.read_text(encoding="utf-8"))
            folder, _ = resolve_checkpoint(output)
            source = (
                old
                if folder == (output / old["generation"]).resolve()
                else old["previous"]
            )
            previous = {k: source[k] for k in ("generation", "sha256")}
        generation = (
            output
            / "checkpoint_generations"
            / (str(model.num_timesteps) + "_" + uuid4().hex)
        )
        generation.mkdir(parents=True)
        committed = False
        pending = output / "checkpoint.pending.json"
        try:
            model.save(generation / "model.zip")
            if manifest["model_type"] in {"sac", "td3"}:
                model.save_replay_buffer(generation / "replay_buffer.pkl")
            (generation / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            info = {
                "generation": generation.relative_to(output).as_posix(),
                "sha256": {p.name: digest(p) for p in generation.iterdir()},
            }
            if previous:
                info["previous"] = previous
            pending.write_text(json.dumps(info), encoding="utf-8")
            pending.replace(pointer)
            committed = True
            # Evaluation exports; resume reads the atomic generation pointer only.
            for name in ("model.zip", "manifest.json"):
                temp = output / (name + ".pending")
                shutil.copyfile(generation / name, temp)
                temp.replace(output / name)
            prune_generations(output, info)
        finally:
            pending.unlink(missing_ok=True)
            if not committed:
                # Generated absolute path is a direct child of the intended root.
                assert generation.parent == output / "checkpoint_generations"
                shutil.rmtree(generation)
