"""Shared helpers for the news collectors: secrets and month ranges."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


def env_value(name: str, env_path: Path = Path(".env")) -> str:
    """Read a secret from the environment, falling back to a KEY=value .env file."""
    value = os.getenv(name, "").strip()
    if not value and env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, raw = line.partition("=")
            if sep and key.strip() == name:
                value = raw.strip().strip("'\"")
    return value


def iter_months(start: str, end: str) -> Iterator[tuple[int, int]]:
    """Inclusive month range: ("2024-11", "2025-02") -> 2024-11 ... 2025-02."""
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
