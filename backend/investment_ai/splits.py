import numpy as np


def walk_forward_splits(n_dates, config):
    test_start = int(n_dates * (1 - config["test_fraction"]))
    first, width, count = (
        config["initial_train_days"],
        config["validation_days"],
        config["folds"],
    )
    minimum = config["lookback"] + config["observation_lag"] + 1
    if (
        first < minimum
        or first + width * count > test_start
        or test_start >= n_dates - 1
    ):
        raise ValueError(
            "Insufficient dates for disjoint validation folds and final test"
        )
    starts = np.linspace(first, test_start - width, count, dtype=int)
    folds = [
        {
            "fold": i,
            "train_end": int(start - 1),
            "validation_start": int(start),
            "validation_end": int(start + width - 1),
        }
        for i, start in enumerate(starts)
    ]
    return folds, test_start
