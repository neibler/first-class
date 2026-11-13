"""Episode clock shared by training, inference and durable serving state."""

PHASE_DEFINITION = "episode_elapsed_bars_v1"


def rebalance_phase(elapsed, period):
    if type(elapsed) is not int or elapsed < 0:
        raise ValueError("episode_elapsed_bars must be a nonnegative integer")
    if type(period) is not int or period < 1:
        raise ValueError("rebalance_days must be a positive integer")
    return (elapsed % period) / period


def episode_clock(dates, state=None, previous=None, reset=False):
    dates = [str(d)[:10] for d in dates]
    today = dates[-1]
    state = state or {}
    start = state.get("episode_start_date")
    elapsed = state.get("episode_elapsed_bars")
    if previous and not reset:
        if today < previous["date"] or previous["date"] not in dates:
            raise ValueError(
                "Episode clock needs ordered bars including the previous date"
            )
        result = {
            "date": today,
            "episode_start_date": previous["episode_start_date"],
            "episode_elapsed_bars": previous["episode_elapsed_bars"]
            + len(dates)
            - 1
            - dates.index(previous["date"]),
        }
    elif reset or not state:
        result = {"date": today, "episode_start_date": today, "episode_elapsed_bars": 0}
    elif start is not None and elapsed is not None:
        rebalance_phase(elapsed, 1)
        if start not in dates or dates.index(start) + elapsed != len(dates) - 1:
            raise ValueError("Episode start/date/bar count mismatch")
        result = {
            "date": today,
            "episode_start_date": start,
            "episode_elapsed_bars": elapsed,
        }
    else:
        raise ValueError(
            "Existing portfolio needs episode_start_date and episode_elapsed_bars"
        )
    for key in ("episode_start_date", "episode_elapsed_bars"):
        if state.get(key) is not None and state[key] != result[key]:
            raise ValueError(f"Episode clock mismatch: {key}")
    rebalance_phase(result["episode_elapsed_bars"], 1)
    return result
