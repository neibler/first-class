"""Whole-percent display without modifying portfolio values."""

from decimal import Decimal, ROUND_HALF_UP


def format_weight(value):
    return (
        str((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        + "%"
    )
