"""Whole-percent display without modifying portfolio values."""

from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP


def format_weight(value):
    return (
        str((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        + "%"
    )


def round_weights(values):
    """Whole percents whose sum equals the rounded total (100 for full weights).

    Rounding each weight on its own can show 23+23+23+23+6 = 98% for weights that
    sum to exactly 100%. This floors every value, then gives the missing points to
    the largest remainders (ties keep the input order). Display only: the
    underlying weights are not changed.
    """
    percents = [Decimal(str(value)) * 100 for value in values]
    floors = [p.to_integral_value(rounding=ROUND_FLOOR) for p in percents]
    target = sum(percents, Decimal(0)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    missing = int(target - sum(floors, Decimal(0)))
    order = sorted(range(len(percents)), key=lambda i: (-(percents[i] - floors[i]), i))
    for i in order[:missing]:
        floors[i] += 1
    return [int(value) for value in floors]


def format_weights(values):
    """`round_weights` as "N%" labels, in the input order."""
    return [f"{value}%" for value in round_weights(values)]
