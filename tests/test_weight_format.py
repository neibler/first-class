"""Display rounding keeps shown weights adding up to the real total."""

import pytest

from frontend.weight_format import format_weight, format_weights, round_weights


def test_each_weight_rounded_alone_can_miss_100():
    weights = [0.2339, 0.2339, 0.2339, 0.2339, 0.0644]
    assert sum(int(format_weight(w)[:-1]) for w in weights) == 98


def test_round_weights_gives_missing_points_to_largest_remainders():
    weights = [0.2339, 0.2339, 0.2339, 0.2339, 0.0644]
    # Floors are 23×4 + 6 = 98; the two points go to 6.44 (.44) and the first .39.
    assert round_weights(weights) == [24, 23, 23, 23, 7]
    assert sum(round_weights(weights)) == 100
    assert format_weights(weights) == ["24%", "23%", "23%", "23%", "7%"]


@pytest.mark.parametrize(
    "weights",
    [
        [0.3439, 0.3439, 0.2720, 0.0402],
        [0.7121, 0.2417, 0.0462],
        [1.0],
        [1 / 3, 1 / 3, 1 / 3],
        [0.0, 0.5, 0.5],
    ],
)
def test_full_weights_always_show_100(weights):
    shown = round_weights(weights)
    assert sum(shown) == 100
    # Never more than one point away from the plain rounding of each value.
    assert all(abs(s - w * 100) < 1 for s, w in zip(shown, weights))


def test_partial_totals_round_to_their_own_total_and_keep_input_order():
    assert round_weights([0.155, 0.155]) == [16, 15]
    assert round_weights([]) == []
