"""Reusable PPO training with versioned bundles and bounded checkpoints."""

import csv
import json
import hashlib
from pathlib import Path
import torch
from .model_factory import create_model, load_model
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from .settings import SCHEMA_VERSION, SOURCE_DIR, write_json, risk_score_enabled
from .data_loader import fingerprint
from .portfolio_env import PortfolioEnv
from .preprocessing import FeatureScaler
from .records import asset_rows
from .rebalancing import REBALANCE_POLICY
from .checkpoints import publish_checkpoint, resolve_checkpoint
from .settings import validate_config
from .episode_clock import PHASE_DEFINITION


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(SOURCE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class HistoryCallback(BaseCallback):
    def __init__(self, path, assets):
        super().__init__()
        self.path, self.episode = path, 0
        self.assets = assets
        self.asset_writer = None
        self.handle = self.writer = None

    def _on_training_start(self):
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.asset_path = self.path.with_name("training_asset_history.csv")
        self.asset_handle = self.asset_path.open("a", newline="", encoding="utf-8")

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            row = {
                k: v
                for k, v in info.items()
                if k not in ("terminal_observation", "episode", "TimeLimit.truncated")
            }
            row = {"training_step": self.num_timesteps, "episode": self.episode, **row}
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                if self.path.stat().st_size == 0:
                    self.writer.writeheader()
            self.writer.writerow(row)
            for asset_row in asset_rows(row, self.assets):
                if self.asset_writer is None:
                    self.asset_writer = csv.DictWriter(
                        self.asset_handle, fieldnames=list(asset_row)
                    )
                    if self.asset_path.stat().st_size == 0:
                        self.asset_writer.writeheader()
                self.asset_writer.writerow(asset_row)
            self.episode += int(done)
        return True

    def _on_training_end(self):
        self.handle.close()
        self.asset_handle.close()
        self.asset_writer = None
        self.writer = None


def feature_names(data, config):
    names = [
        f"log_return_lag_{lag}_{a.lower()}"
        for lag in range(config["lookback"] - 1, -1, -1)
        for a in data.assets
    ]
    for kind in ["rsi_div_100", "macd", "current_weight"]:
        names.extend(f"{kind}_{a.lower()}" for a in data.assets)
    names.extend(f"macro_{name}" for name in data.macros)
    if risk_score_enabled(config):
        names.extend(f"risk_score_{a.lower()}" for a in data.assets)
    return names + ["drawdown", "mdd", "rebalance_phase"]


def schema_for(data, config):
    return {
        "version": SCHEMA_VERSION,
        "phase_definition": PHASE_DEFINITION,
        "execution_policy": REBALANCE_POLICY,
        "feature_names": feature_names(data, config),
        "execution_delay": 1,
        "timing": "observe_close_t_execute_close_t_plus_1",
        "news_columns": [
            "date",
            "ticker",
            "risk_score",
            "risk_type",
            "has_news",
            "days_since_news",
        ],
        "scaler_file": "scaler.json",
        "assets": data.assets,
        "macros": data.macros,
        "lookback": config["lookback"],
        "observation_lag": config["observation_lag"],
        "observation_size": len(feature_names(data, config)),
        "order": [
            "log_return_history",
            "rsi_div_100",
            "macd",
            "weights",
            "macro",
            *(["risk_score"] if risk_score_enabled(config) else []),
            "drawdown",
            "mdd",
            "rebalance_phase",
        ],
    }


def train_model(data, config, train_end, seed, output, resume=None, run_spec=None):
    output = Path(output)
    if resume:
        checkpoint_path, old = resolve_checkpoint(resume)
    validate_config(
        config, completed_steps=old.get("actual_timesteps", 0) if resume else 0
    )
    torch.set_num_threads(1)
    scaler = FeatureScaler.fit(data, train_end)
    data_hash = fingerprint(data)
    if resume:
        checkpoint_path, old = resolve_checkpoint(resume)
        saved_schema = json.loads(
            (Path(resume) / "schema.json").read_text(encoding="utf-8")
        )
        if saved_schema.get("execution_policy") != REBALANCE_POLICY:
            raise ValueError("Incompatible execution policy; retrain the model")
        if saved_schema.get("phase_definition") != PHASE_DEFINITION:
            raise ValueError("Unverified phase definition; retrain the model")
        old_config = json.loads(
            (Path(resume) / "config.json").read_text(encoding="utf-8")
        )
        effective = dict(config)
        for key, default in {
            "model_type": "ppo",
            "include_risk_score": True,
            "device": "cpu",
            "max_holdings": None,
        }.items():
            old_config.setdefault(key, default)
            effective.setdefault(key, default)
        ignored = {"total_timesteps", "checkpoint_steps", "device"}
        if any(
            old_config.get(k) != effective.get(k)
            for k in old_config.keys() | effective.keys()
            if k not in ignored
        ):
            raise ValueError(
                "Resume requires unchanged configuration except "
                "steps/checkpoint interval"
            )
        if (
            old["data_hash"] != data_hash
            or old["train_end_index"] != train_end
            or old["source_hash"] != source_hash()
            or old["seed"] != seed
        ):
            raise ValueError("Resume data, source, seed or split mismatch")
        scaler = FeatureScaler.load(Path(resume) / "scaler.json")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    if run_spec is not None:
        write_json(output / "run_spec.json", run_spec)
    from .bundle_inputs import preserve_inputs

    preserve_inputs(output, data, config, run_spec, resume)
    env = Monitor(
        PortfolioEnv(
            data,
            {**config, "window_id": config.get("window_id", "standalone") + "_train"},
            end=train_end,
            scaler=scaler,
            training=True,
        ),
        str(output / "monitor.csv"),
    )
    if resume:
        model = load_model(
            checkpoint_path / "model.zip",
            config.get("model_type", "ppo"),
            env=env,
            device=config.get("device", "cpu"),
        )
        if config.get("model_type") in {"sac", "td3"}:
            model.load_replay_buffer(checkpoint_path / "replay_buffer.pkl")
    else:
        model = create_model(config.get("model_type", "ppo"), env, config, seed)
    model.set_logger(configure(str(output), ["csv"]))
    scaler.save(output / "scaler.json")
    write_json(output / "config.json", config)
    write_json(output / "schema.json", schema_for(data, config))
    callback = HistoryCallback(output / "training_portfolio_history.csv", data.assets)
    initial_steps = model.num_timesteps
    target = initial_steps + config["total_timesteps"]
    while model.num_timesteps < target:
        steps = min(config["checkpoint_steps"], target - model.num_timesteps)
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=False)
        # Commit only a complete model + buffer generation.
        publish_checkpoint(
            model,
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "execution_policy": REBALANCE_POLICY,
                "model_type": config.get("model_type", "ppo"),
                "reward_type": config["reward"],
                "timesteps": config["total_timesteps"],
                "train_start": str(data.dates[0])[:10],
                "train_end": str(data.dates[train_end])[:10],
                "seed": seed,
                "data_hash": data_hash,
                "source_hash": source_hash(),
                "train_end_index": train_end,
                "train_end_date": str(data.dates[train_end])[:10],
                "actual_timesteps": model.num_timesteps,
                "initial_timesteps": initial_steps,
                "completed": model.num_timesteps >= target,
                "resume_source": str(resume) if resume else None,
            },
        )
    env.close()
    model.logger.dump(step=model.num_timesteps)
    model.logger.close()
    return model, scaler


def load_bundle(path, data=None, exact_data=False):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "schema.json").read_text(encoding="utf-8"))
    if schema.get("execution_policy") != REBALANCE_POLICY:
        raise ValueError("Incompatible execution policy; retrain the model")
    if schema.get("phase_definition") != PHASE_DEFINITION:
        raise ValueError("Unverified phase definition; retrain the model")
    if schema["version"] != SCHEMA_VERSION:
        raise ValueError("Incompatible model schema version")
    if data is not None and schema != schema_for(data, config):
        raise ValueError("Model assets/features/order do not match input")
    if exact_data and data is not None:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["data_hash"] != fingerprint(data):
            raise ValueError("Evaluation dataset changed since training")
    scaler = FeatureScaler.load(path / "scaler.json")
    return (
        load_model(path / "model.zip", config.get("model_type", "ppo")),
        scaler,
        config,
    )
