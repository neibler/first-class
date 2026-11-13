"""One algorithm factory; every model uses the unchanged PortfolioEnv."""

from copy import deepcopy
import inspect
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

MODEL_TYPES = ("ppo", "recurrent_ppo", "sac", "td3")


def model_class(model_type):
    if model_type == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type: {model_type}")
    return {"ppo": PPO, "sac": SAC, "td3": TD3}[model_type]


def model_parameters(model_type, config):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type: {model_type}")
    if model_type in ("ppo", "recurrent_ppo"):
        params = {"n_steps": config["n_steps"], "batch_size": config["batch_size"]}
    else:
        params = {
            "buffer_size": 10000,
            "learning_starts": 1000,
            "batch_size": 256,
            "train_freq": 1,
            "gradient_steps": 1,
        }
    params.update(deepcopy(config.get("models", {}).get(model_type, {})))
    if model_type == "recurrent_ppo":
        policy = params.setdefault("policy_kwargs", {})
        for name in ("lstm_hidden_size", "n_lstm_layers"):
            if name in params:
                policy[name] = params.pop(name)
    return params


def validate_model_config(config, *, completed_steps=0):
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    kind = config.get("model_type", "ppo")
    params = model_parameters(kind, config)
    reserved = {"policy", "env", "seed", "device", "verbose"}
    allowed = set(inspect.signature(model_class(kind).__init__).parameters) - reserved
    unknown = set(params) - allowed - ({"action_noise_std"} if kind == "td3" else set())
    if unknown:
        raise ValueError(f"Unsupported {kind} parameters: {sorted(unknown)}")
    for name in ("n_steps", "batch_size", "buffer_size", "train_freq"):
        if name in params and (type(params[name]) is not int or params[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    if kind in ("ppo", "recurrent_ppo"):
        n, batch = params["n_steps"], params["batch_size"]
        if n < 2 or batch < 2 or n % batch:
            raise ValueError("Invalid on-policy rollout/batch divisibility")
        if config["total_timesteps"] % n or config["checkpoint_steps"] % n:
            raise ValueError("Steps/checkpoints must be multiples of effective n_steps")
    else:
        for name, minimum in (("learning_starts", 0), ("gradient_steps", 1)):
            if type(params[name]) is not int or params[name] < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if params["learning_starts"] < 0 or params["gradient_steps"] < 1:
            raise ValueError("Invalid off-policy warmup/gradient_steps")
        if params["batch_size"] > params["buffer_size"]:
            raise ValueError("batch_size exceeds replay buffer")
        if completed_steps + config["total_timesteps"] <= params["learning_starts"]:
            raise ValueError("total_timesteps must exceed learning_starts")
    if (
        not np.isfinite(params.get("action_noise_std", 0.1))
        or params.get("action_noise_std", 0.1) < 0
    ):
        raise ValueError("action_noise_std must be finite and nonnegative")


def create_model(model_type, env, config, seed=42):
    params = model_parameters(model_type, config)
    if model_type == "td3":
        std = params.pop("action_noise_std", 0.1)
        params["action_noise"] = NormalActionNoise(
            mean=np.zeros(env.action_space.shape),
            sigma=std * np.ones(env.action_space.shape),
        )
    policy = "MlpLstmPolicy" if model_type == "recurrent_ppo" else "MlpPolicy"
    return model_class(model_type)(
        policy, env, seed=seed, device=config.get("device", "cpu"), verbose=0, **params
    )


def load_model(path, model_type="ppo", env=None, device="cpu"):
    return model_class(model_type).load(path, env=env, device=device)
