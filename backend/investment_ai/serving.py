"""Read-only inference service. Models/data are loaded once, never trained here."""

import json
import os
import hashlib
from pathlib import Path
from copy import deepcopy
from threading import RLock
import numpy as np
import pandas as pd
import torch
from .settings import bundle_input_root, ROOT
from .data_loader import load_data, prepare_data
from .training import load_bundle, schema_for
from .infer import resolve_run, predict_weights
from .backtest import performance
from .accounting import equity_curve
from .prediction import PredictionSession
from .episode_clock import episode_clock, rebalance_phase


class InvestmentService:
    def __init__(self, run=None):
        torch.set_num_threads(1)
        serving_file = Path(__file__).resolve().parents[2] / "config/serving.json"
        serving = (
            json.loads(serving_file.read_text(encoding="utf-8"))
            if serving_file.exists()
            else {}
        )
        configured = run or os.environ.get("INVESTMENT_AI_RUN")
        selected_bundle = serving.get("model_path") if not configured else None
        if selected_bundle:
            configured = Path(selected_bundle)
            if not configured.is_absolute():
                configured = ROOT / configured
            if configured.suffix == ".zip":
                if configured.name != "model.zip":
                    raise ValueError("Select the bundle directory or its model.zip")
                configured = configured.parent
        self.run = resolve_run(configured)
        if not self.run.is_dir():
            raise FileNotFoundError(
                f"Selected model directory does not exist: {self.run}. "
                "Run python scripts/install_trained_models.py --activate ppo "
                "or select an installed bundle."
            )
        self.run_id = self.run.name
        self.lock = RLock()
        state_key = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        self.latch_path = ROOT / "service_state" / f"{state_key}.json"
        saved_state = (
            json.loads(self.latch_path.read_text()) if self.latch_path.exists() else {}
        )
        self.latches = (
            saved_state.get("latches", {})
            if saved_state.get("version") == 2
            else saved_state
        )
        self.episode_clocks = (
            saved_state.get("clocks", {}) if saved_state.get("version") == 2 else {}
        )
        self.recurrent_sessions = {}
        single = (self.run / "run_spec.json").exists()
        if single:
            manifest = json.loads(
                (self.run / "manifest.json").read_text(encoding="utf-8")
            )
            spec = json.loads((self.run / "run_spec.json").read_text(encoding="utf-8"))
            if not manifest.get("completed"):
                raise ValueError("Training bundle is incomplete")
            evaluation = self.run / "evaluation.json"
            self.summary = (
                json.loads(evaluation.read_text(encoding="utf-8"))
                if evaluation.exists()
                else {
                    "completed": True,
                    "synthetic": spec["synthetic"],
                    "final_test": {},
                }
            )
            paths = [self.run]
        else:
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
                raise ValueError("No model members")
        config = json.loads((paths[0] / "config.json").read_text(encoding="utf-8"))
        if selected_bundle and config.get("model_type", "ppo") != serving.get(
            "model_type", "ppo"
        ):
            raise ValueError("Serving model_type does not match saved bundle")
        if config.get("model_type") == "recurrent_ppo" and len(paths) != 1:
            raise ValueError("Serve one recurrent model; choose an individual bundle")
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
            if single:
                config["news_path"] = str(self.run / "news_metadata.parquet")
            self.data_source = "synthetic"
        else:
            self.data_source = "configured_parquet"
        if not config["data_path"]:
            raise ValueError(
                "Saved input or explicit parquet path required; "
                "API never generates data"
            )
        self.data, _ = load_data(
            config, ROOT if external else bundle_input_root(paths[0], ROOT)
        )
        self.config = self.model_config.copy()
        bundles = [load_bundle(path, self.data) for path in paths]
        if any(bundle[2] != self.model_config for bundle in bundles):
            raise ValueError("Ensemble configuration mismatch")
        self.models = [bundle[0] for bundle in bundles]
        self.scaler = bundles[0][1]
        self.state_model_hash = hashlib.sha256(
            b"".join((path / "model.zip").read_bytes() for path in paths)
        ).hexdigest()
        if saved_state.get("version") == 2:
            if saved_state.get("model_hash") != self.state_model_hash:
                raise ValueError("Saved episode state belongs to a different model")
            self._restore_recurrent(saved_state.get("recurrent", {}))
        for _, scaler, _ in bundles[1:]:
            for key in vars(self.scaler):
                if not np.array_equal(getattr(self.scaler, key), getattr(scaler, key)):
                    raise ValueError("Ensemble preprocessing mismatch")
        self.histories = {}
        for strategy in self.summary["final_test"]:
            history = pd.read_csv(
                (self.run / "evaluation" if single else self.run)
                / f"{strategy}_history.csv"
            )
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
        if self.histories and "ppo" not in self.histories:
            raise ValueError("PPO history missing")
        for history in self.histories.values():
            if not history.date.equals(self.histories["ppo"].date):
                raise ValueError("Saved strategy dates differ")

    def _restore_recurrent(self, records):
        for episode, record in records.items():
            if len(record["sessions"]) != len(self.models):
                raise ValueError("Recurrent state/model count mismatch")
            sessions = []
            for model, stored in zip(self.models, record["sessions"]):
                session = PredictionSession(model, "recurrent_ppo")
                session.episode_start = stored["episode_start"]
                session.lstm_states = (
                    tuple(
                        np.asarray(x, dtype=np.float32) for x in stored["lstm_states"]
                    )
                    if stored["lstm_states"] is not None
                    else None
                )
                if session.lstm_states is not None and (
                    len(session.lstm_states) != 2
                    or any(not np.isfinite(x).all() for x in session.lstm_states)
                ):
                    raise ValueError("Invalid recurrent state")
                sessions.append(session)
            self.recurrent_sessions[episode] = {**record, "sessions": sessions}

    def _save_episode_state(self):
        recurrent = {}
        for episode, record in getattr(self, "recurrent_sessions", {}).items():
            recurrent[episode] = {
                **record,
                "sessions": [
                    {
                        "episode_start": bool(p.episode_start),
                        "lstm_states": (
                            [x.tolist() for x in p.lstm_states]
                            if p.lstm_states is not None
                            else None
                        ),
                    }
                    for p in record["sessions"]
                ],
            }
        payload = {
            "version": 2,
            "model_hash": getattr(self, "state_model_hash", None),
            "latches": self.latches,
            "clocks": self.episode_clocks,
            "recurrent": recurrent,
        }
        self.latch_path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.latch_path.with_suffix(".pending")
        pending.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        pending.replace(self.latch_path)

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
            supplied = state is not None
            state = dict(state) if state is not None else None
            if state is not None:
                for flag in ("safeguard_triggered", "stopped", "reset_episode"):
                    if flag in state and type(state[flag]) is not bool:
                        raise ValueError(f"{flag} must be a boolean")
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
            from .accounting import safeguard_breached

            if state and not reset:
                state["safeguard_triggered"] = bool(
                    state.get("safeguard_triggered", False)
                    or state.get("stopped", False)
                    or safeguard_breached(
                        state["max_drawdown"], self.config["max_drawdown"]
                    )
                )
            self.episode_clocks = getattr(self, "episode_clocks", {})
            previous_clock = self.episode_clocks.get(episode_id)
            if (
                previous_clock
                and not supplied
                and previous_clock["date"] != str(data.dates[-1])[:10]
            ):
                raise ValueError("Supply actual portfolio state to advance an episode")
            clock = episode_clock(data.dates, state, previous_clock, reset)
            if state is None:
                state = {
                    "date": str(data.dates[-1])[:10],
                    "weights": {a: 0.0 for a in data.assets},
                    "portfolio_value": 1.0,
                    "peak_value": 1.0,
                    "max_drawdown": 0.0,
                    "days_since_rebalance": 0,
                    "stopped": False,
                }
            state.update(
                {k: clock[k] for k in ("episode_start_date", "episode_elapsed_bars")}
            )
            recurrent = self.config.get("model_type") == "recurrent_ppo"
            if recurrent:
                prediction = self._recurrent_prediction(
                    data,
                    state,
                    episode_id,
                    reset or (not previous_clock and not supplied),
                )
            else:
                prediction = predict_weights(
                    data, self.config, self.models, self.scaler, state
                )
            if not supplied:
                prediction["mode"] = "new_portfolio_cash"
            # Persist after validation; an API restart must not clear a stop.
            self.latches[episode_id] = prediction["safeguard_triggered"]
            self.episode_clocks[episode_id] = clock
            self._save_episode_state()
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
            "episode_start_date": clock["episode_start_date"],
            "episode_elapsed_bars": clock["episode_elapsed_bars"],
            "rebalance_phase": rebalance_phase(
                clock["episode_elapsed_bars"], self.config["rebalance_days"]
            ),
            "observation_date": prediction["observation_date"],
            "decision_date": prediction["decision_date"],
            "execution_date": prediction["execution_date"],
            "first_return_end_date": prediction["first_return_end_date"],
            "first_return_end_delay_trading_bars": prediction[
                "first_return_end_delay_trading_bars"
            ],
            "mode": prediction["mode"],
            "target_weights": prediction["weights"],
            "model_target_weights": prediction["model_target_weights"],
            "raw_target_weights": prediction["raw_target_weights"],
            "filtered_target_weights": prediction["filtered_target_weights"],
            "portfolio_weight_distance": prediction["portfolio_weight_distance"],
            "max_risk_score": prediction["max_risk_score"],
            "rebalance_reasons": prediction["rebalance_reasons"],
            "should_rebalance": prediction["should_rebalance"],
            "current_weights": current,
            "cash_weight": prediction["cash_weight"],
            "current_cash_weight": float(np.clip(1 - sum(current.values()), 0, 1)),
            "execution_delay_trading_bars": prediction["execution_delay_trading_bars"],
            "is_order": False,
        }

    def _recurrent_prediction(self, data, state, episode_id, reset):
        if state and state.get("safeguard_triggered", False):
            return predict_weights(
                data, self.config, self.models, self.scaler, state, []
            )
        date = str(data.dates[-1])[:10]
        # Include the full supplied panel: changing a repeated observation is an error.
        from .data_loader import fingerprint

        signature = hashlib.sha256(
            (fingerprint(data) + json.dumps(state, sort_keys=True)).encode()
        ).hexdigest()
        old = self.recurrent_sessions.get(episode_id)
        if old and old["date"] == date and old["signature"] == signature:
            return deepcopy(old["prediction"])
        if reset:
            old = None
        if old:
            if state is None:
                raise ValueError(
                    "Supply current portfolio state to continue recurrent episode"
                )
            dates = [str(d)[:10] for d in data.dates]
            if old["date"] not in dates or dates.index(old["date"]) != len(dates) - 2:
                raise ValueError(
                    "Recurrent episode requires consecutive daily observations; "
                    "reset to start again"
                )
            sessions = [
                PredictionSession(p.model, p.model_type) for p in old["sessions"]
            ]
            for new, previous in zip(sessions, old["sessions"]):
                new.lstm_states = deepcopy(previous.lstm_states)
                new.episode_start = previous.episode_start
        else:
            if state is not None and not reset:
                raise ValueError(
                    "Recurrent state is absent (new process/episode); "
                    "explicit reset_episode is required"
                )
            sessions = [PredictionSession(m, "recurrent_ppo") for m in self.models]
        prediction = predict_weights(
            data, self.config, self.models, self.scaler, state, sessions
        )
        self.recurrent_sessions[episode_id] = {
            "date": date,
            "signature": signature,
            "sessions": sessions,
            "prediction": deepcopy(prediction),
        }
        return prediction

    def backtest(self, benchmark="spy_buy_hold", observations=None):
        if not self.histories:
            raise ValueError("No saved backtest; run evaluate --bundle first")
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
