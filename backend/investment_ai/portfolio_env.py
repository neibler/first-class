"""Daily close-to-close accounting with delayed observations and explicit cash stop."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .rewards import calculate_reward, canonical_reward
from .accounting import safeguard_breached


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, config, start=None, end=None, scaler=None, training=False):
        super().__init__()
        self.data, self.config, self.scaler = data, config, scaler
        self.training = training
        self.lookback = int(config["lookback"])
        self.lag = 0
        if (
            config.get("observation_lag", 0) != 0
            or config.get("execution_delay", 1) != 1
        ):
            raise ValueError("Use schema v3 timing: observe t, execute t+1")
        self.minimum_start = self.lookback - 1 + self.lag
        self.start = self.minimum_start if start is None else int(start)
        self.end = len(data.dates) - 1 if end is None else int(end)
        if (
            self.lookback < 1
            or self.lag < 0
            or not self.minimum_start <= self.start < self.end < len(data.dates)
        ):
            raise ValueError("Invalid lookback/start/end")
        self.period = int(config.get("rebalance_days", 1))
        self.rate = (config["transaction_cost_bps"] + config["slippage_bps"]) / 10000
        if (
            self.period < 1
            or not 0 <= self.rate < 0.5
            or not 0 < config["max_drawdown"] < 1
        ):
            raise ValueError("Invalid costs, rebalance period or drawdown")
        self.n_assets = len(data.assets)
        self.action_space = spaces.Box(-1, 1, (self.n_assets,), dtype=np.float32)
        # Three state features: current drawdown, running MDD, rebalance phase.
        size = self.lookback * self.n_assets + 4 * self.n_assets + len(data.macros) + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.start
        if self.training and self.config.get("random_start", True):
            upper = max(
                self.start + 1, self.end - self.config.get("min_episode_days", 60)
            )
            self.t = int(self.np_random.integers(self.start, upper))
        self.episode_start = self.t
        self.weights = np.zeros(
            self.n_assets
        )  # initial capital is cash; charge first entry
        self.value = self.peak = 1.0
        self.drawdown = self.mdd = 0.0
        self.safeguard_triggered = False
        self.history = []
        self.done = False
        return self._get_observation(), {
            "portfolio_value": self.value,
            "safeguard_triggered": False,
        }

    def observation_at(self, feature_index, weights, drawdown=0.0, mdd=0.0, phase=0.0):
        d, t = self.data, int(feature_index)
        if not self.lookback - 1 <= t < len(d.dates):
            raise ValueError("Not enough lookback for observation")
        returns = d.returns[t - self.lookback + 1 : t + 1]
        macd, macro = d.macd[t], d.macro[t]
        if self.scaler is not None:
            returns = self.scaler.transform("return", returns)
            macd = self.scaler.transform("macd", macd)
            macro = self.scaler.transform("macro", macro)
        result = np.concatenate(
            [
                returns.ravel(),
                d.rsi[t] / 100,
                macd,
                weights,
                macro,
                d.risk[t],
                [drawdown, mdd, phase],
            ]
        ).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite observation")
        return result

    def _get_observation(self):
        phase = ((self.t - self.episode_start) % self.period) / self.period
        return self.observation_at(
            self.t - self.lag, self.weights, self.drawdown, self.mdd, phase
        )

    def action_weights(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n_assets,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        logits = np.clip(action, -1, 1) * self.config.get("logit_scale", 5.0)
        weights = np.exp(logits - logits.max())
        return weights / weights.sum()

    def step(self, action):
        return self.step_weights(self.action_weights(action))

    def step_weights(self, target, force=False, execute=True):
        """Observe at t, accrue existing holdings to t+1, then execute at t+1 close."""
        if self.done:
            raise RuntimeError("Call reset before stepping")
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (self.n_assets,)
            or not np.isfinite(target).all()
            or np.any(target < 0)
            or not np.isclose(target.sum(), 1)
        ):
            raise ValueError(
                "Target weights must be finite, nonnegative and sum to one"
            )
        due = bool(
            execute and ((self.t - self.episode_start) % self.period == 0 or force)
        )
        decision_date = str(self.data.dates[self.t])[:10]
        start_value = self.value
        self.t += 1
        gross = np.expm1(self.data.returns[self.t])
        gross_return = float(self.weights @ gross)
        self.value *= 1 + gross_return
        self.weights = self.weights * (1 + gross) / (1 + gross_return)
        previous = self.weights.copy()  # Drifted holdings immediately before execution.
        self.peak = max(self.peak, self.value)
        self.drawdown = 1 - self.value / self.peak
        self.mdd = max(self.mdd, self.drawdown)
        self.safeguard_triggered |= safeguard_breached(
            self.mdd, self.config["max_drawdown"]
        )
        turnover = cost = cost_amount = exit_turnover = exit_cost = 0.0
        if not self.safeguard_triggered and due:
            turnover = float(np.abs(target - previous).sum())
            cost = turnover * self.rate
            cost_amount = self.value * cost
            self.value *= 1 - cost
            self.weights = target.copy()
            self.drawdown = 1 - self.value / self.peak
            self.mdd = max(self.mdd, self.drawdown)
            self.safeguard_triggered |= safeguard_breached(
                self.mdd, self.config["max_drawdown"]
            )
        if self.safeguard_triggered:
            # Risk liquidation is immediate at the breach close.
            # Cancel queued buys.
            exit_turnover = float(self.weights.sum())
            exit_cost = exit_turnover * self.rate
            cost_amount += self.value * exit_cost
            self.value *= 1 - exit_cost
            self.weights = np.zeros(self.n_assets)
        final_target = self.weights.copy()
        self.drawdown = 1 - self.value / self.peak
        self.mdd = max(self.mdd, self.drawdown)
        net_return = self.value / start_value - 1
        terminated = bool(self.safeguard_triggered)
        truncated = bool(self.t >= self.end and not terminated)
        self.done = terminated or truncated
        execution_date = str(self.data.dates[self.t])[:10]
        info = {
            "observation_date": decision_date,
            "decision_date": decision_date,
            "execution_date": execution_date,
            "date": execution_date,
            "return_start_date": decision_date,
            "new_weights_return_start_date": execution_date,
            "new_weights_return_end_date": (
                str(self.data.dates[self.t + 1])[:10]
                if self.t + 1 < len(self.data.dates)
                else ""
            ),
            "portfolio_return": net_return,
            "gross_return": gross_return,
            "portfolio_value": self.value,
            "drawdown": self.drawdown,
            "max_drawdown": self.mdd,
            "turnover": turnover,
            "exit_turnover": exit_turnover,
            "cost": cost,
            "exit_cost": exit_cost,
            "cost_amount": cost_amount,
            "rebalanced": bool(due and turnover > 0),
            "cash_weight": float(np.clip(1 - self.weights.sum(), 0, 1)),
            "safeguard_triggered": terminated,
            "stop_reason": (
                "max_drawdown" if terminated else "data_end" if truncated else ""
            ),
            "reward_type": canonical_reward(self.config["reward"]),
            "run_id": self.config.get("run_id", "standalone"),
            "window_id": self.config.get("window_id", "standalone"),
        }
        for i, asset in enumerate(self.data.assets):
            a = asset.lower()
            info[f"previous_weight_{a}"] = float(previous[i])
            info[f"current_weight_{a}"] = float(self.weights[i])
            info[f"target_weight_{a}"] = float(final_target[i])
            info[f"weight_change_{a}"] = float(self.weights[i] - previous[i])
            info[f"weight_{a}"] = float(final_target[i])
            info[f"held_{a}"] = float(self.weights[i])
        self.history.append(info.copy())
        reward = calculate_reward(
            self.config["reward"],
            net_return,
            drawdown=self.drawdown,
            mdd=self.mdd,
            history=self.history,
            window=self.config.get("reward_window", 20),
            mdd_lambda=self.config.get("mdd_lambda", 0.1),
            annual_rf=self.config.get("risk_free_annual", 0.0),
        )
        self.history[-1]["reward"] = info["reward"] = reward
        return self._get_observation(), reward, terminated, truncated, info
