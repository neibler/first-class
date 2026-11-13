"""Install checksummed training inputs; no models, training, or config changes."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def install(destination, shared=None):
    destination = Path(destination).resolve()
    shared = Path(shared) if shared else PROJECT / "shared"
    meta = json.loads(
        (shared / "training_inputs_manifest.json").read_text(encoding="utf-8")
    )
    archive = shared / "training_inputs.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Training archive checksum mismatch")
    pending = []
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(
            meta["files"]
        ):
            raise ValueError("Unexpected training input members")
        for name, digest in meta["files"].items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe input path")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe input path")
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Input checksum mismatch")
            for parent in target.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError("Input parent path conflict")
            if target.exists():
                if not target.is_file() or target.read_bytes() != content:
                    raise ValueError("Existing input differs")
            else:
                pending.append((target, content))
    # Stage all bytes before exposing any final path. Each rename is atomic.
    staged = []
    installed = []
    try:
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, target))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, target in staged:
            temporary.replace(target)
            installed.append(target)
    except BaseException:
        for target in installed:
            target.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.environ.get("INVESTMENT_AI_HOME", PROJECT / "runtime")),
    )
    print(install(parser.parse_args().destination))
