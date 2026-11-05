"""Install verified final models without replacing local datasets or models."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def install(destination, shared=None):
    destination = Path(destination).resolve()
    shared = Path(shared) if shared else PROJECT / "shared"
    archive = shared / "trained_models.zip"
    manifest = json.loads(
        (shared / "trained_models_manifest.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    pending = []
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(
            manifest["files"]
        ):
            raise ValueError("Unexpected archive members")
        for name, digest in manifest["files"].items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path")
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(
                        f"Existing file differs: {name}; use a fresh destination"
                    )
            else:
                pending.append((target, content))
        # Validate the whole archive and all conflicts before writing any files.
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"Installed four trained models at {destination}; no training performed.")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=PROJECT / "runtime")
    parser.add_argument("--activate", choices=("ppo", "recurrent_ppo", "sac", "td3"))
    args = parser.parse_args()
    manifest = install(args.destination)
    if args.activate:
        bundle = args.destination.resolve() / manifest["models"][args.activate]["path"]
        default = (PROJECT / "runtime").resolve()
        model_path = (
            bundle.relative_to(default).as_posix()
            if bundle.is_relative_to(default)
            else str(bundle)
        )
        config = {"model_type": args.activate, "model_path": model_path}
        path = PROJECT / "config/serving.json"
        if path.exists():
            backup = path.with_name("serving.before_shared_install.json")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Selected {args.activate} for serving. Restart the API.")


if __name__ == "__main__":
    main()
