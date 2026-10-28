"""HTTP-only First Class adapter. The backend owns metric calculations."""

import math
import os
import httpx


class InvestmentAPIError(ValueError):
    pass


def _request(method, path, **kwargs):
    base = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.request(method, base + path, timeout=20.0, **kwargs)
        response.raise_for_status()
        data = response.json()
        if data.get("schema_version") != "3.0":
            raise ValueError("Unsupported contract")
        return data
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise InvestmentAPIError(
            "투자 분석 결과를 불러오지 못했습니다. 서버와 모델 준비 상태를 확인해 주세요."
        ) from exc


def load_portfolio(observations=None):
    params = {} if observations is None else {"observations": observations}
    data = _request("GET", "/backtest", params=params)
    try:
        cash = data["cash_weight"]
        assets = [dict(asset) for asset in data["assets"]]
        numbers = [cash, *[a["weight"] for a in assets]]
        if not all(
            math.isfinite(v) and 0 <= v <= 1 for v in numbers
        ) or not math.isclose(sum(numbers), 1, abs_tol=1e-8):
            raise ValueError("Invalid holdings")
        if cash > 0:
            assets.append({"ticker": "CASH", "name": "현금", "weight": cash})
        metrics = data["metrics"]
        data["assets"] = assets
        data["display_metrics"] = {
            "path": data["path"],
            "cumulative": metrics["total_return"],
            "benchmark": data["benchmark_metrics"]["total_return"],
            "sharpe": metrics["sharpe"],
            "mdd": -metrics["max_drawdown"],
            "current_drawdown": data["path"][-1]["drawdown"],
            "observations": metrics["days"],
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise InvestmentAPIError("투자 분석 응답 형식이 올바르지 않습니다.") from exc
    return data


def optimize(state=None, market=None, news=None):
    payload = {}
    for key, value in (("state", state), ("market", market), ("news", news)):
        if value is not None:
            payload[key] = value
    return _request("POST", "/optimize", json=payload)
