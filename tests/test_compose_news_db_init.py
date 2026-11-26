"""news-db-init shell script from docker-compose.yml, run against temp folders.

The real container copies the host news DB into a volume; here /source and
/news/chroma are swapped for temp paths and chown is stubbed (not root).
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def find_bash():
    """Linux bash, or Git Bash on Windows (System32's bash is WSL, other paths)."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        for candidate in (
            Path(git).parents[1] / "bin" / "bash.exe",
            Path(git).parents[1] / "usr" / "bin" / "bash.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    return None


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def script(source, target):
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    text = compose["services"]["news-db-init"]["command"][0]
    # Placeholders first so one substituted path can never be rewritten by the
    # next replace; quoted so temp paths with spaces stay one shell word.
    # Suffixes such as /. and /* stay outside the quotes and keep working.
    return (
        text.replace("$$", "$")
        .replace("/news/chroma", "@TARGET@")
        .replace("/source", "@SOURCE@")
        .replace("@TARGET@", shlex.quote(target.as_posix()))
        .replace("@SOURCE@", shlex.quote(source.as_posix()))
        .replace("chown -R 10001:10001", "true")
    )


def run(source, target, path_prefix=None):
    text = script(source, target)
    if path_prefix:
        # Set inside bash: Windows PATH uses ";" and Git Bash converts it.
        text = f'PATH={shlex.quote(path_prefix)}:"$PATH"; {text}'
    return subprocess.run(
        [BASH, "-c", text], capture_output=True, text=True, timeout=60
    )


def posix(path):
    """Paths bash understands on Windows (Git Bash) and Linux alike."""
    text = path.as_posix()
    if len(text) > 1 and text[1] == ":":
        return Path(f"/{text[0].lower()}{text[2:]}")
    return path


@pytest.fixture
def dirs(tmp_path):
    # A space in every path: the substituted paths must stay one shell word.
    source, target = tmp_path / "news db" / "source", tmp_path / "docker vol"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "chroma.sqlite3").write_text("db", encoding="utf-8")
    (source / "index.bin").write_text("index", encoding="utf-8")
    return source, target


def test_copies_once_then_reports_up_to_date(dirs):
    source, target = dirs
    first = run(posix(source), posix(target))
    assert first.returncode == 0, first.stdout + first.stderr
    assert "copying news DB" in first.stdout
    assert (target / "index.bin").read_text(encoding="utf-8") == "index"
    assert (target / ".source").exists()
    second = run(posix(source), posix(target))
    assert second.returncode == 0 and "up to date" in second.stdout


def test_failed_copy_fails_the_initializer(dirs, tmp_path):
    source, target = dirs
    shim = tmp_path / "fake bin"
    shim.mkdir()
    (shim / "cp").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (shim / "cp").chmod(0o755)
    result = run(posix(source), posix(target), path_prefix=posix(shim).as_posix())
    # api depends on service_completed_successfully, so it must not start.
    assert result.returncode == 1
    assert "copy failed" in result.stdout
    assert not (target / ".source").exists()


def test_missing_source_clears_the_old_copy_and_still_succeeds(dirs):
    source, target = dirs
    assert run(posix(source), posix(target)).returncode == 0
    (source / "chroma.sqlite3").unlink()
    result = run(posix(source), posix(target))
    # Exit 0 so api starts and reports the DB as missing instead of using it.
    assert result.returncode == 0
    assert not (target / "chroma.sqlite3").exists()
    assert not (target / ".source").exists()
