import json
from pathlib import Path
from threading import RLock
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from backend.investment_ai.api import OptimizeRequest, OptimizeResponse
from backend.investment_ai.data_loader import prepare_data
from backend.investment_ai.portfolio_env import PortfolioEnv
from backend.investment_ai.infer import predict_weights
from backend.investment_ai.rewards import calculate_reward
from backend.investment_ai.accounting import sharpe_ratio
from backend.investment_ai.backtest import run_backtest, performance
from backend.investment_ai.training import schema_for, HistoryCallback
from backend.investment_ai.serving import InvestmentService
from backend.investment_ai.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = json.loads((ROOT / "config/investment_ai.json").read_text())
    return {
        **c,
        "assets": ["SPY", "QQQ"],
        "lookback": 1,
        "rebalance_days": 1,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "random_start": False,
        "run_id": "test_run",
        "window_id": "test_window",
    }


def inputs(returns=None):
    returns = np.zeros((5, 2)) if returns is None else np.asarray(returns)
    dates = pd.bdate_range("2024-01-02", periods=len(returns)).strftime("%Y-%m-%d")
    frame = pd.DataFrame(
        [
            {
                "date": d,
                "ticker": a,
                "log_return": np.log1p(returns[i, j]),
                "rsi": 50.0,
                "macd": 0.0,
            }
            for i, d in enumerate(dates)
            for j, a in enumerate(["SPY", "QQQ"])
        ]
    )
    news = frame[["date", "ticker"]].copy()
    news["risk_score"] = 0.4
    news["risk_type"] = "recession"
    news["has_news"] = 0
    news["days_since_news"] = 3
    return frame, news


def data(returns=None):
    frame, news = inputs(returns)
    return prepare_data(frame, ["SPY", "QQQ"], news, macro_columns=[])


def test_news_keeps_carried_score_and_metadata():
    d = data()
    assert np.all(d.risk == 0.4)
    assert (d.news_metadata.has_news == 0).all()
    assert (d.news_metadata.days_since_news == 3).all()


@pytest.mark.parametrize(
    "change", ["missing", "extra", "columns", "flag", "days", "null_type"]
)
def test_news_invalid_contract(change):
    frame, news = inputs()
    if change == "missing":
        news = news.iloc[:-1]
    if change == "extra":
        news = pd.concat([news, news.iloc[:1]])
    if change == "columns":
        news = news.drop(columns="has_news")
    if change == "flag":
        news.loc[0, "has_news"] = 2
    if change == "days":
        news.loc[0, "days_since_news"] = -1
    if change == "null_type":
        news.loc[0, "risk_type"] = None
    with pytest.raises(ValueError):
        prepare_data(frame, ["SPY", "QQQ"], news, [])


def test_news_absence_requires_explicit_development_and_initial_state():
    frame, _ = inputs()
    with pytest.raises(ValueError):
        prepare_data(frame, ["SPY", "QQQ"], macro_columns=[])
    d = prepare_data(frame, ["SPY", "QQQ"], macro_columns=[], news_mode="development")
    assert np.all(d.risk == 0)
    assert (d.news_metadata.risk_type == "").all()
    assert (d.news_metadata.days_since_news == 20).all()
    assert (d.news_metadata.has_news == 0).all()


def test_observe_today_execute_next_close_and_keep_old_returns():
    d = data([[0, 0], [0.5, 0], [0.1, 0.2], [0.3, 0.4]])
    env = PortfolioEnv(d, config())
    obs, _ = env.reset()
    assert obs[-5:-3].tolist() == pytest.approx([0.4, 0.4])
    _, _, _, _, first = env.step_weights([1, 0])
    assert first["portfolio_return"] == 0  # No exposure to the pre-fill +50%.
    assert first["observation_date"] == str(d.dates[0])[:10]
    assert first["execution_date"] == str(d.dates[1])[:10]
    _, _, _, _, second = env.step_weights([0, 1])
    assert second["portfolio_return"] == pytest.approx(0.1)  # Old SPY, not new QQQ.
    assert second["previous_weight_spy"] == 1
    assert second["current_weight_qqq"] == 1
    assert second["weight_change_spy"] == -1
    assert second["target_weight_qqq"] == 1
    assert second["new_weights_return_end_date"] == str(d.dates[3])[:10]


def test_costs_and_drawdown_stop_cancel_reentry():
    c = {**config(), "transaction_cost_bps": 10.0}
    env = PortfolioEnv(data([[0, 0], [0, 0], [-0.2, 0], [0.8, 0]]), c)
    env.reset()
    _, _, _, _, first = env.step_weights([1, 0])
    assert first["portfolio_return"] == pytest.approx(-0.001)
    _, _, terminated, _, row = env.step_weights([0, 1])
    assert terminated and row["safeguard_triggered"]
    assert row["cash_weight"] == 1 and row["target_weight_qqq"] == 0
    assert row["current_weight_spy"] == 0 and row["previous_weight_spy"] == 1
    assert row["cost_amount"] == pytest.approx(0.999 * 0.8 * 0.001)
    with pytest.raises(RuntimeError):
        env.step_weights([1, 0])
    _, info = env.reset()
    assert not info["safeguard_triggered"]


def test_threshold_is_strictly_greater():
    c = {**config(), "max_drawdown": 0.15}
    env = PortfolioEnv(data([[0, 0], [0, 0], [-0.15, 0], [0, 0]]), c)
    env.reset()
    env.step_weights([1, 0])
    assert not env.step_weights([1, 0])[2]


def state(d, mdd=0.2):
    return {
        "date": str(d.dates[-1])[:10],
        "weights": {"SPY": 1.0, "QQQ": 0.0},
        "portfolio_value": 0.95,
        "peak_value": 1.0,
        "max_drawdown": mdd,
        "days_since_rebalance": 0,
        "episode_id": "test",
    }


class NoPredict:
    def predict(self, *args, **kwargs):
        raise AssertionError("Safeguard must bypass policy")


def test_recovered_mdd_still_stops_inference():
    result = predict_weights(data(), config(), [NoPredict()], None, state(data()))
    assert result["safeguard_triggered"] and result["cash_weight"] == 1
    assert sum(result["weights"].values()) == 0
    assert result["execution_delay_trading_bars"] == 1


def test_sharpe_common_annualization_and_warmup():
    r = np.linspace(-0.01, 0.03, 20)
    history = [{"portfolio_return": v, "drawdown": 0.2} for v in r]
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)
    for name in ["sharpe", "rolling_sharpe"]:
        assert calculate_reward(
            name, r[-1], drawdown=0.02, history=history
        ) == pytest.approx(expected)
        assert calculate_reward(name, r[-1], drawdown=0.02, history=history[:-1]) == 0
    assert sharpe_ratio(np.zeros(20)) is None
    assert calculate_reward(
        "return_mdd", 0.01, drawdown=0.02, mdd=0.2, history=history
    ) == pytest.approx(-0.01)


def test_backtest_cash_padding_and_asset_records(tmp_path):
    from backend.investment_ai.records import save_asset_history

    d = data([[0, 0], [0, 0], [-0.3, 0], [0.8, 0], [0.2, 0]])
    h = run_backtest(d, config(), 0, 4, strategy="spy_buy_hold")
    assert len(h) == 4
    assert h.safeguard_triggered.tolist() == [False, True, True, True]
    assert h.iloc[-1].portfolio_return == 0
    assert h.iloc[-1].previous_weight_spy == 0
    assert h.iloc[-1].current_weight_spy == 0
    save_asset_history(h, d.assets, tmp_path / "assets.csv")
    rows = pd.read_csv(tmp_path / "assets.csv")
    assert len(rows) == 8
    np.testing.assert_allclose(
        rows.weight_change, rows.current_weight - rows.previous_weight
    )
    assert performance(h)["max_drawdown"] == pytest.approx(0.3)


def test_feature_names_and_callback_records(tmp_path):
    d = data()
    c = config()
    schema = schema_for(d, c)
    assert len(schema["feature_names"]) == schema["observation_size"]
    assert len(set(schema["feature_names"])) == schema["observation_size"]
    assert schema["feature_names"][:2] == [
        "log_return_lag_0_spy",
        "log_return_lag_0_qqq",
    ]
    env = PortfolioEnv(d, c)
    env.reset()
    info = env.step_weights([1, 0])[-1]
    cb = HistoryCallback(tmp_path / "training_portfolio_history.csv", d.assets)
    cb._on_training_start()
    cb.locals = {"infos": [info], "dones": [False]}
    cb.num_timesteps = 1
    cb._on_step()
    cb._on_training_end()
    assert len(pd.read_csv(tmp_path / "training_asset_history.csv")) == 2


def service(tmp_path):
    obj = InvestmentService.__new__(InvestmentService)
    obj.data = data()
    obj.data_source = "test"
    obj.config = config()
    obj.schema = schema_for(obj.data, obj.config)
    obj.models = [NoPredict()]
    obj.scaler = None
    obj.lock = RLock()
    obj.latch_path = tmp_path / "latch.json"
    obj.latches = {}
    obj.summary = {"synthetic": False}
    obj.run_id = "test_run"
    return obj


def test_api_latches_recovered_episode_and_validates_news(monkeypatch, tmp_path):
    obj = service(tmp_path)
    monkeypatch.setattr("backend.investment_ai.api.InvestmentService", lambda *a: obj)
    app = create_app()
    app.state.service = obj
    endpoint = next(route.endpoint for route in app.routes if route.path == "/optimize")

    class Response:
        def __init__(self, status, payload):
            self.status_code, self.payload = status, payload
            self.text = str(payload)

        def json(self):
            return self.payload

    class Client:
        def post(self, path, json):
            try:
                result = endpoint(OptimizeRequest.model_validate(json))
                return Response(
                    200, OptimizeResponse.model_validate(result).model_dump()
                )
            except HTTPException as exc:
                return Response(exc.status_code, {"detail": exc.detail})

    client = Client()
    payload = state(obj.data)
    first = client.post("/optimize", json={"state": payload})
    assert first.status_code == 200, first.text
    assert first.json()["safeguard_triggered"]
    payload.update(portfolio_value=1.0, max_drawdown=0.0)
    result = client.post("/optimize", json={"state": payload})
    assert result.status_code == 200 and result.json()["cash_weight"] == 1
    assert json.loads(obj.latch_path.read_text())["test"]
    frame, news = inputs()
    bad = client.post(
        "/optimize",
        json={
            "market": frame.to_dict("records"),
            "news": news.iloc[:-1].to_dict("records"),
        },
    )
    assert bad.status_code == 422
    payload.update(reset_episode=True)
    assert client.post("/optimize", json={"state": payload}).status_code == 422

    class Model:
        def predict(self, *a, **kw):
            return np.zeros(2), None

    obj.models = [Model()]
    payload["weights"] = {"SPY": 0.0, "QQQ": 0.0}
    reset = client.post("/optimize", json={"state": payload})
    assert reset.status_code == 200 and not reset.json()["safeguard_triggered"]
