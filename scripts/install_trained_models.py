"""Install verified final models without replacing local datasets or models."""

import argparse
import hashlib
import json
import os
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
                if not target.is_file():
                    raise ValueError(f"Installation target is not a file: {name}")
                if target.read_bytes() != content:
                    raise ValueError(
                        f"Existing file differs: {name}; use a fresh destination"
                    )
            else:
                pending.append((target, content))
        for model in manifest.get("models", {}).values():
            name = model["path"]
            rel = PurePosixPath(name)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or "\\" in name
                or ":" in name
                or not rel.parts
            ):
                raise ValueError("Unsafe model path")
            bundle = (destination / name).resolve()
            if not bundle.is_relative_to(destination):
                raise ValueError("Unsafe model path")
            config = json.loads(z.read(f"{name}/config.json"))
            for key in ("data_path", "news_path"):
                value = config[key]
                path = PurePosixPath(value)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in value
                    or ":" in value
                    or path.as_posix() not in manifest["files"]
                ):
                    raise ValueError(f"Missing or unsafe bundled input: {key}")
            marker = bundle / "installed_inputs.json"
            if marker.relative_to(destination).as_posix() in manifest["files"]:
                raise ValueError("Archive must not supply installation metadata")
            content = (
                json.dumps(
                    {"relative_root": os.path.relpath(destination, bundle)}, indent=2
                )
                + "\n"
            ).encode()
            if marker.exists() and marker.read_bytes() != content:
                raise ValueError("Existing installation metadata differs")
            if not marker.exists():
                pending.append((marker, content))
        # Validate the whole archive and all conflicts before writing any files.
        for target, _ in pending:
            if target.exists() and not target.is_file():
                raise ValueError(f"Installation target is not a file: {target}")
            for parent in target.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(
                        f"Installation parent is not a directory: {parent}"
                    )
        for target, content in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"Installed four trained models at {destination}; no training performed.")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.environ.get("INVESTMENT_AI_HOME", PROJECT / "runtime")),
    )
    parser.add_argument("--activate", choices=("ppo", "recurrent_ppo", "sac", "td3"))
    args = parser.parse_args()
    manifest = install(args.destination)
    if args.activate:
        bundle = args.destination.resolve() / manifest["models"][args.activate]["path"]
        default = (
            Path(os.environ.get("INVESTMENT_AI_HOME", PROJECT / "runtime"))
            .expanduser()
            .resolve()
        )
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
