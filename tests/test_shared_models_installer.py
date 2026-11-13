import hashlib
import json
import zipfile

import pytest

from scripts.install_trained_models import install


def package(tmp_path, files):
    shared = tmp_path / "shared"
    shared.mkdir()
    archive = shared / "trained_models.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    manifest = {
        "files": {n: hashlib.sha256(c).hexdigest() for n, c in files.items()},
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (shared / "trained_models_manifest.json").write_text(json.dumps(manifest))
    return shared


def test_repeat_install_preserves_identical_files(tmp_path):
    shared = package(tmp_path, {"models/model.zip": b"fixture"})
    target = tmp_path / "runtime"
    install(target, shared)
    install(target, shared)
    assert (target / "models/model.zip").read_bytes() == b"fixture"


def test_conflict_rejected_before_any_writes(tmp_path):
    shared = package(tmp_path, {"new.txt": b"new", "existing.txt": b"incoming"})
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "existing.txt").write_bytes(b"local")
    with pytest.raises(ValueError, match="Existing file differs"):
        install(target, shared)
    assert not (target / "new.txt").exists()
    assert (target / "existing.txt").read_bytes() == b"local"


def test_traversal_rejected(tmp_path):
    shared = package(tmp_path, {"../outside": b"bad"})
    with pytest.raises(ValueError, match="Unsafe archive path"):
        install(tmp_path / "runtime", shared)
    assert not (tmp_path / "outside").exists()


def test_corrupt_archive_rejected(tmp_path):
    shared = package(tmp_path, {"file.txt": b"content"})
    with (shared / "trained_models.zip").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        install(tmp_path / "runtime", shared)
