"""Explicit per-model, per-episode recurrent state; no environment changes."""

import numpy as np


def predict_weights(
    model,
    observation,
    env,
    model_type="ppo",
    state=None,
    episode_start=True,
    deterministic=True,
):
    if model_type == "recurrent_ppo":
        if not episode_start and state is None:
            raise ValueError("Continuing recurrent prediction requires LSTM state")
        action, next_state = model.predict(
            observation,
            state=None if episode_start else state,
            episode_start=np.array([episode_start], dtype=bool),
            deterministic=deterministic,
        )
    else:
        action, _ = model.predict(observation, deterministic=deterministic)
        next_state = None
    return {"weights": env.action_weights(action), "state": next_state}


class PredictionSession:
    def __init__(self, model, model_type="ppo"):
        self.model, self.model_type = model, model_type
        self.reset()

    def reset(self):
        self.lstm_states = None
        self.episode_start = True

    def predict(self, observation, env):
        result = predict_weights(
            self.model,
            observation,
            env,
            self.model_type,
            self.lstm_states,
            self.episode_start,
        )
        self.lstm_states = result["state"]
        self.episode_start = False
        return result["weights"]
