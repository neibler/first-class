"""Read-only inference service. Models/data are loaded once, never trained here."""

import json
import os
import hashlib
from threading import RLock
import numpy as np
import pandas as pd
import torch
from .settings import ROOT
from .data_loader import load_data, prepare_data
from .training import load_bundle, schema_for
from .infer import resolve_run, predict_weights
from .backtest import performance
from .accounting import equity_curve


class InvestmentService:
    def __init__(self, run=None):
        torch.set_num_threads(1)
        configured = run or os.environ.get("INVESTMENT_AI_RUN")
        self.run = resolve_run(configured)
        self.run_id = self.run.name
        self.lock = RLock()
        state_key = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        self.latch_path = ROOT / "service_state" / f"{state_key}.json"
        self.latches = (
            json.loads(self.latch_path.read_text()) if self.latch_path.exists() else {}
        )
        self.summary = json.loads(
            (self.run / "summary.json").read_text(encoding="utf-8")
        )
        if not self.summary.get("completed"):
            raise ValueError("Run is not complete")
        selection = json.loads(
            (self.run / "selection.json").read_text(encoding="utf-8")
        )
        paths = [self.run / "final" / f"seed_{seed}" for seed in selection["seeds"]]
        if not paths:
            raise ValueError("No ensemble members")
        config = json.loads((paths[0] / "config.json").read_text(encoding="utf-8"))
        self.model_config = config.copy()
        self.schema = json.loads((paths[0] / "schema.json").read_text(encoding="utf-8"))
        config["macro_columns"] = self.schema["macros"]
        external = os.environ.get("INVESTMENT_AI_DATA")
        external_news = os.environ.get("INVESTMENT_AI_NEWS")
        if external_news and not external:
            raise ValueError("INVESTMENT_AI_NEWS requires INVESTMENT_AI_DATA")
        if external:
            config["data_path"] = external
            config["news_path"] = external_news
            self.data_source = "configured_parquet"
        elif self.summary["synthetic"]:
            config["data_path"] = str(self.run / "synthetic_market.parquet")
            self.data_source = "synthetic"
        else:
            self.data_source = "configured_parquet"
        if not config["data_path"]:
            raise ValueError(
                "Saved input or explicit parquet path required; "
                "API never generates data"
            )
        self.data, _ = load_data(config, ROOT)
        self.config = self.model_config.copy()
        bundles = [load_bundle(path, self.data) for path in paths]
        if any(bundle[2] != self.model_config for bundle in bundles):
            raise ValueError("Ensemble configuration mismatch")
        self.models = [bundle[0] for bundle in bundles]
        self.scaler = bundles[0][1]
        for _, scaler, _ in bundles[1:]:
            for key in vars(self.scaler):
                if not np.array_equal(getattr(self.scaler, key), getattr(scaler, key)):
                    raise ValueError("Ensemble preprocessing mismatch")
        self.histories = {}
        for strategy in self.summary["final_test"]:
            history = pd.read_csv(self.run / f"{strategy}_history.csv")
            if (
                history.empty
                or not history.date.is_unique
                or not history.date.is_monotonic_increasing
            ):
                raise ValueError("Invalid saved history dates")
            numeric = history.select_dtypes(include="number")
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError("Nonfinite saved history")
            np.testing.assert_allclose(
                equity_curve(history.portfolio_return)[1:],
                history.portfolio_value,
                atol=1e-9,
            )
            self.histories[strategy] = history
        if "ppo" not in self.histories:
            raise ValueError("PPO history missing")
        for history in self.histories.values():
            if not history.date.equals(self.histories["ppo"].date):
                raise ValueError("Saved strategy dates differ")

    def optimize(self, market=None, news=None, state=None):
        data = self.data
        source = self.data_source
        if market is not None:
            source = "request_payload"
            data = prepare_data(
                pd.DataFrame(market),
                self.config["assets"],
                pd.DataFrame(news) if news is not None else None,
                macro_columns=self.schema["macros"],
                news_mode=self.config.get("news_mode", "required"),
            )
            if schema_for(data, self.config) != self.schema:
                raise ValueError("Input schema differs from model")
        elif news is not None:
            raise ValueError("news requires market in the same request")
        with self.lock:
            state = dict(state) if state is not None else None
            episode_id = state.get("episode_id", "default") if state else "default"
            if state and state.get("reset_episode", False):
                if (
                    state.get("stopped", False)
                    or state.get("safeguard_triggered", False)
                    or state["max_drawdown"] != 0
                    or state["portfolio_value"] != state["peak_value"]
                    or any(state["weights"].values())
                ):
                    raise ValueError(
                        "Explicit reset requires a fresh all-cash state "
                        "with zero MDD"
                    )
            reset = bool(state and state.get("reset_episode", False))
            if self.latches.get(episode_id, False) and not reset:
                if state is None:
                    raise ValueError(
                        "Episode is safeguarded; supply state or explicitly reset"
                    )
                state["safeguard_triggered"] = True
            prediction = predict_weights(
                data, self.config, self.models, self.scaler, state
            )
            # Persist after validation; an API restart must not clear a stop.
            self.latches[episode_id] = prediction["safeguard_triggered"]
            self.latch_path.parent.mkdir(parents=True, exist_ok=True)
            pending = self.latch_path.with_suffix(".pending")
            pending.write_text(json.dumps(self.latches))
            pending.replace(self.latch_path)
        current = state["weights"] if state else {asset: 0.0 for asset in data.assets}
        return {
            "schema_version": "3.0",
            "model_version": self.run_id,
            "data_source": source,
            "model_trained_on_synthetic": self.summary["synthetic"],
            "as_of": prediction["as_of_date"],
            "status": prediction["status"],
            "safeguard_triggered": prediction["safeguard_triggered"],
            "episode_id": episode_id,
            "observation_date": prediction["observation_date"],
            "decision_date": prediction["decision_date"],
            "execution_date": prediction["execution_date"],
            "first_return_end_date": prediction["first_return_end_date"],
            "first_return_end_delay_trading_bars": 2,
            "mode": prediction["mode"],
            "target_weights": prediction["weights"],
            "current_weights": current,
            "cash_weight": prediction["cash_weight"],
            "current_cash_weight": float(np.clip(1 - sum(current.values()), 0, 1)),
            "execution_delay_trading_bars": prediction["execution_delay_trading_bars"],
            "is_order": False,
        }

    def backtest(self, benchmark="spy_buy_hold", observations=None):
        if benchmark not in self.histories or benchmark == "ppo":
            raise ValueError("Unknown benchmark")
        p = self.histories["ppo"].copy()
        b = self.histories[benchmark].copy()
        if observations is not None:
            p, b = p.tail(observations).copy(), b.tail(observations).copy()
        # Rebase currency costs as well as equity when displaying a subperiod.
        for frame in (p, b):
            capital = float(
                frame.iloc[0].portfolio_value / (1 + frame.iloc[0].portfolio_return)
            )
            frame["cost_amount"] /= capital
        rows = [
            {
                "date": str(p.iloc[0].decision_date),
                "portfolio_log": 0.0,
                "benchmark_log": 0.0,
            }
        ]
        rows += [
            {
                "date": str(d),
                "portfolio_log": float(np.log1p(pr)),
                "benchmark_log": float(np.log1p(br)),
            }
            for d, pr, br in zip(p.date, p.portfolio_return, b.portfolio_return)
        ]
        pv = equity_curve(p.portfolio_return.to_numpy())
        bv = equity_curve(b.portfolio_return.to_numpy())
        drawdowns = pv / np.maximum.accumulate(pv) - 1
        path = [
            {
                "date": row["date"],
                "portfolio": float(pv[i] - 1),
                "benchmark": float(bv[i] - 1),
                "drawdown": float(drawdowns[i]),
            }
            for i, row in enumerate(rows)
        ]
        stats = performance(p, self.config["risk_free_annual"])
        benchmark_stats = performance(b, self.config["risk_free_annual"])
        last = p.iloc[-1]
        stopped = bool((self.histories["ppo"].stop_reason == "max_drawdown").any())
        assets = [
            {
                "ticker": asset,
                "name": asset,
                "weight": float(last[f"held_{asset.lower()}"]),
            }
            for asset in self.config["assets"]
        ]
        cash = float(last.cash_weight)
        if not np.isclose(sum(a["weight"] for a in assets) + cash, 1.0, atol=1e-8):
            raise ValueError("Invalid saved holdings/cash balance")
        return {
            "schema_version": "3.0",
            "model_version": self.run_id,
            "data_source": (
                "synthetic" if self.summary["synthetic"] else "historical_parquet"
            ),
            "as_of": str(last.date),
            "benchmark": benchmark,
            "periods_per_year": 252,
            "annual_risk_free_rate": self.config["risk_free_annual"],
            "metric_convention": "simple_daily_returns_annualized_sharpe_positive_mdd",
            "assets": assets,
            "cash_weight": cash,
            "stopped": stopped,
            "safeguard_triggered": stopped,
            "allocation_kind": "backtest_end_holdings",
            "returns": rows,
            "path": path,
            "metrics": stats,
            "benchmark_metrics": benchmark_stats,
            "safeguard_applied_to_benchmarks": True,
        }
