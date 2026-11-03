"""Install the checked-in trained demo without overwriting local training/data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def install(destination):
    destination = Path(destination).resolve()
    archive = PROJECT / "shared/investment_demo.zip"
    manifest = json.loads((PROJECT / "shared/manifest.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(manifest["files"]):
            raise ValueError("Unexpected archive members")
        pending = []
        for name, digest in manifest["files"].items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path")
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
            if target.exists():
                if name == "runs/latest.json":
                    continue  # Never switch an existing user's active run.
                if target.read_bytes() != content:
                    raise ValueError(
                        f"Existing file differs: {target}; use a fresh destination"
                    )
            else:
                pending.append((target, content))
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"Installed trained demo {manifest['run_id']} at {destination}")
    print("Existing latest.json is preserved. No training was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT / "runtime")
    install(parser.parse_args().destination)
