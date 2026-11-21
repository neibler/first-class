"""Daily close-to-close accounting with delayed observations and explicit cash stop."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from . import baseline_execution as baseline
from .rewards import calculate_reward, canonical_reward
from .accounting import safeguard_breached
from .settings import risk_score_enabled
from .episode_clock import rebalance_phase
from .rebalancing import (
    get_rebalance_reasons,
    validate_rebalancing,
    filter_target_weights,
    portfolio_weight_distance,
    MIN_WEIGHT_THRESHOLD,
)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, config, start=None, end=None, scaler=None, training=False):
        super().__init__()
        self.data, self.config, self.scaler = data, config, scaler
        self.training = training
        self.include_risk_score = risk_score_enabled(config)
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
        validate_rebalancing(config)
        # Preserve the trained observation feature; it no longer gates trading.
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
        size = (
            (self.lookback + 3 + int(self.include_risk_score)) * self.n_assets
            + len(data.macros)
            + 3
        )
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
        self.last_executed_weights = self.weights.copy()
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
                *([d.risk[t]] if self.include_risk_score else []),
                [drawdown, mdd, phase],
            ]
        ).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite observation")
        return result

    def _get_observation(self):
        phase = rebalance_phase(self.t - self.episode_start, self.period)
        return self.observation_at(
            self.t - self.lag, self.weights, self.drawdown, self.mdd, phase
        )

    def action_weights(self, action):
        """Raw softmax weights; filter only after any ensemble averaging."""
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n_assets,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        logits = np.clip(action, -1, 1) * self.config.get("logit_scale", 5.0)
        weights = np.exp(logits - logits.max())
        return weights / weights.sum()

    def step(self, action):
        return self.step_weights(self.action_weights(action))

    def step_weights(self, target, force=False, execute=True, apply_filter=True):
        """Observe at t, accrue existing holdings to t+1, then execute at t+1 close."""
        if self.done:
            raise RuntimeError("Call reset before stepping")
        original_target = np.array(target, dtype=float, copy=True)
        target = baseline.normalize_target(original_target)
        if target.shape != (self.n_assets,):
            raise ValueError("Target weights shape differs from asset universe")
        raw_target = target.copy()
        use_baseline = baseline.enabled(self.config) and apply_filter
        if apply_filter and not use_baseline:
            target = filter_target_weights(
                original_target,
                self.config.get("min_weight_threshold", MIN_WEIGHT_THRESHOLD),
                self.config.get("max_holdings"),
            )
        decision_weights = self.weights.copy()
        last_before = self.last_executed_weights.copy()
        reference = last_before if use_baseline else target
        distance = portfolio_weight_distance(decision_weights, reference)
        max_risk_score = float(self.data.risk[self.t].max())
        reason_fn = baseline.reasons if use_baseline else get_rebalance_reasons
        reasons = reason_fn(
            self.data.dates[self.t],
            self.data.dates[self.t - 1] if self.t else None,
            self.weights,
            reference,
            self.data.risk[self.t],
            self.mdd,
            self.config,
            stopped=self.safeguard_triggered,
        )
        if reasons != ["mdd_stop"]:
            if not execute:
                reasons = ["hold"]  # Explicit buy-and-hold benchmark override.
            elif force:
                reasons = ([] if reasons == ["hold"] else reasons) + ["forced"]
        due = reasons not in (["hold"], ["mdd_stop"])
        if use_baseline and due:
            # Fail invalid holdings/config before advancing time or accounting.
            baseline.executable_target(raw_target, self.weights, self.config)
        decision_date = str(self.data.dates[self.t])[:10]
        start_value = self.value
        # Validate the next-close portfolio and fill before mutating episode state.
        gross = np.expm1(self.data.returns[self.t + 1])
        gross_return = float(self.weights @ gross)
        if (
            not np.isfinite(gross).all()
            or not np.isfinite(gross_return)
            or gross_return <= -1
        ):
            raise ValueError("Invalid next-close market return")
        drifted = self.weights * (1 + gross) / (1 + gross_return)
        planned = None
        if use_baseline:
            baseline.weights(drifted)
            if due and not self.safeguard_triggered:
                planned = baseline.executable_target(raw_target, drifted, self.config)
                planned_fill = baseline.execute(drifted, planned[1], self.rate)
        self.t += 1
        self.value *= 1 + gross_return
        self.weights = drifted
        previous = self.weights.copy()  # Drifted holdings immediately before execution.
        self.peak = max(self.peak, self.value)
        self.drawdown = 1 - self.value / self.peak
        self.mdd = max(self.mdd, self.drawdown)
        self.safeguard_triggered |= safeguard_breached(
            self.mdd, self.config["max_drawdown"]
        )
        turnover = cost = cost_amount = exit_turnover = exit_cost = 0.0
        desired_turnover = executed_turnover = 0.0
        executable = previous.copy()
        normal_post = previous.copy()
        entries = exits = 0
        if not self.safeguard_triggered and due:
            if use_baseline:
                target, executable, desired_turnover = planned
                turnover, cost = planned_fill
            else:
                executable = target.copy()
                desired_turnover = portfolio_weight_distance(previous, target)
                turnover = float(np.abs(target - previous).sum())
                cost = turnover * self.rate
            cost_amount = self.value * cost
            self.value *= 1 - cost
            self.weights = executable.copy()
            normal_post = self.weights.copy()
            executed_turnover = portfolio_weight_distance(previous, self.weights)
            entries = int(
                np.count_nonzero(
                    (previous <= baseline.EPS) & (self.weights > baseline.EPS)
                )
            )
            exits = int(
                np.count_nonzero(
                    (previous > baseline.EPS) & (self.weights <= baseline.EPS)
                )
            )
            if turnover > 0:
                self.last_executed_weights = self.weights.copy()
            self.drawdown = 1 - self.value / self.peak
            self.mdd = max(self.mdd, self.drawdown)
            self.safeguard_triggered |= safeguard_breached(
                self.mdd, self.config["max_drawdown"]
            )
        if self.safeguard_triggered:
            reasons = (reasons + ["mdd_stop"]) if turnover > 0 else ["mdd_stop"]
            # Risk liquidation is immediate at the breach close.
            # Cancel queued buys.
            exit_turnover = float(self.weights.sum())
            exit_cost = exit_turnover * self.rate
            cost_amount += self.value * exit_cost
            self.value *= 1 - exit_cost
            executed_turnover += 0.5 * exit_turnover
            exits += int(np.count_nonzero(self.weights > baseline.EPS))
            self.weights = np.zeros(self.n_assets)
            if exit_turnover > 0:
                self.last_executed_weights = self.weights.copy()
        final_target = self.weights.copy()
        self.drawdown = 1 - self.value / self.peak
        self.mdd = max(self.mdd, self.drawdown)
        net_return = self.value / start_value - 1
        terminated = bool(self.safeguard_triggered)
        truncated = bool(self.t >= self.end and not terminated)
        self.done = terminated or truncated
        execution_date = str(self.data.dates[self.t])[:10]
        info = {
            "last_executed_weights_before": dict(
                zip(self.data.assets, map(float, last_before))
            ),
            "last_executed_weights": dict(
                zip(self.data.assets, map(float, self.last_executed_weights))
            ),
            "pre_trade_weights": dict(zip(self.data.assets, map(float, previous))),
            "post_trade_weights": dict(zip(self.data.assets, map(float, self.weights))),
            "executable_target_weights": dict(
                zip(self.data.assets, map(float, executable))
            ),
            "normal_post_trade_weights": dict(
                zip(self.data.assets, map(float, normal_post))
            ),
            "desired_turnover": desired_turnover,
            "executed_turnover": executed_turnover,
            "normal_executed_turnover": portfolio_weight_distance(
                previous, normal_post
            ),
            "transaction_cost": cost_amount,
            "holdings_count": int(np.count_nonzero(self.weights > baseline.EPS)),
            "entry_count": entries,
            "exit_count": exits,
            "rebalance_trigger": "|".join(reasons),
            "raw_target_weights": dict(zip(self.data.assets, map(float, raw_target))),
            "filtered_target_weights": dict(zip(self.data.assets, map(float, target))),
            "current_weights": dict(
                zip(self.data.assets, map(float, decision_weights))
            ),
            "portfolio_weight_distance": distance,
            "max_risk_score": max_risk_score,
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
            "rebalanced": bool(turnover > 0 or exit_turnover > 0),
            "should_rebalance": bool(due or exit_turnover > 0),
            "rebalance_reasons": reasons,
            "rebalance_reason": "|".join(reasons),
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
            info[f"model_target_weight_{a}"] = float(raw_target[i])
            info[f"raw_target_weight_{a}"] = float(raw_target[i])
            info[f"filtered_target_weight_{a}"] = float(target[i])
            info[f"decision_current_weight_{a}"] = float(decision_weights[i])
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
            executed_turnover=executed_turnover,
            turnover_penalty_lambda=(
                self.config.get("turnover_penalty_lambda", 0.003)
                if use_baseline
                else 0.0
            ),
        )
        self.history[-1]["reward"] = info["reward"] = reward
        return self._get_observation(), reward, terminated, truncated, info
