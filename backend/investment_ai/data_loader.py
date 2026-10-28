"""Strict daily panel interface. No silent forward/backward fills."""

from dataclasses import dataclass
import hashlib
import re
import numpy as np
import pandas as pd

CORE = ["log_return", "rsi", "macd"]


@dataclass
class MarketData:
    dates: np.ndarray
    assets: list[str]
    macros: list[str]
    returns: np.ndarray
    rsi: np.ndarray
    macd: np.ndarray
    macro: np.ndarray
    risk: np.ndarray
    news_metadata: pd.DataFrame | None = None


def dummy_data(config):
    from .synthetic import generate

    return generate(config)


def parse_dates(series):
    if pd.api.types.is_numeric_dtype(series.dtype) or series.isna().any():
        raise ValueError(
            "date must be a non-null YYYY-MM-DD string or naive midnight timestamp"
        )
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        if not series.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("date strings must use YYYY-MM-DD")
    dates = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("date must be timezone-naive midnight")
    return dates


def fingerprint(data):
    digest = hashlib.sha256()
    digest.update(repr((data.assets, data.macros)).encode())
    digest.update(data.dates.astype("datetime64[D]").astype("<i8").tobytes())
    for array in (data.returns, data.rsi, data.macd, data.macro, data.risk):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    if data.news_metadata is not None:
        digest.update(
            data.news_metadata.to_json(
                orient="split", date_format="iso", index=False
            ).encode()
        )
    return digest.hexdigest()


def validate_columns(frame):
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", str(c)) for c in frame.columns):
        raise ValueError("All column names must be lowercase snake_case")


def prepare_data(frame, assets, news=None, macro_columns=None, news_mode="required"):
    if news_mode not in {"required", "development"}:
        raise ValueError("news_mode must be required or development")
    frame = frame.copy()
    validate_columns(frame)
    required = {"date", "ticker", *CORE}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    frame["date"] = parse_dates(frame["date"])
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("assets must be nonempty and unique")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market date/ticker keys")
    if set(frame["ticker"]) != set(assets):
        raise ValueError("Market tickers must match configured asset universe")
    if not len(frame) or frame["ticker"].isna().any():
        raise ValueError("Empty market data/null ticker")
    macros = (
        list(macro_columns)
        if macro_columns is not None
        else [
            c
            for c in frame.columns
            if c
            not in required
            | {"risk_score", "risk_type", "has_news", "days_since_news", "close"}
        ]
    )
    if len(set(macros)) != len(macros) or set(macros) & (
        required | {"risk_score", "risk_type", "has_news", "days_since_news", "close"}
    ):
        raise ValueError("Invalid macro_columns")
    dates = np.sort(frame["date"].unique())
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "ticker"])
    frame = frame.set_index(["date", "ticker"]).reindex(index)
    numeric = CORE + macros
    if not set(numeric).issubset(frame.columns):
        raise ValueError("Configured macro columns missing")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in numeric):
        raise ValueError("Features must have numeric dtypes")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Missing panel rows or nonfinite feature values")
    if np.abs(frame["log_return"]).max() > 5:
        raise ValueError("Extreme log_return; check units and adjusted-price input")
    if not frame["rsi"].between(0, 100).all():
        raise ValueError("rsi must be in [0, 100]")
    if macros and (frame.groupby(level="date")[macros].nunique() > 1).any().any():
        raise ValueError("Macro values must be identical across assets on each date")
    news_columns = [
        "date",
        "ticker",
        "risk_score",
        "risk_type",
        "has_news",
        "days_since_news",
    ]
    if news is None:
        if news_mode != "development":
            raise ValueError(
                "Actual news input is required; "
                "use explicit development mode for absent news"
            )
        news = index.to_frame(index=False)
        news["risk_score"] = 0.0
        news["risk_type"] = ""
        news["has_news"] = 0
        news["days_since_news"] = 20
    else:
        news = news.copy()
    validate_columns(news)
    if not set(news_columns).issubset(news.columns):
        raise ValueError("News requires all six contract columns")
    news["date"] = parse_dates(news["date"])
    if news.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate news date/ticker keys")
    if not news["risk_type"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("risk_type must be a non-null string; empty string is allowed")
    for name in ["risk_score", "has_news", "days_since_news"]:
        if (
            not pd.api.types.is_numeric_dtype(news[name])
            or not np.isfinite(news[name]).all()
        ):
            raise ValueError(f"Invalid news {name}")
    if not news.risk_score.between(0, 1).all():
        raise ValueError("risk_score must be in [0, 1]")
    if not news.has_news.isin([0, 1]).all():
        raise ValueError("has_news must be 0 or 1")
    if not (news.days_since_news.ge(0) & news.days_since_news.mod(1).eq(0)).all():
        raise ValueError("days_since_news must be nonnegative integer")
    initial = news.risk_type.eq("")
    if not (
        news.loc[initial, "risk_score"].eq(0)
        & news.loc[initial, "has_news"].eq(0)
        & news.loc[initial, "days_since_news"].eq(20)
    ).all():
        raise ValueError(
            "Empty risk_type requires the initial news state: 0, empty, 0, 20"
        )
    keyed = news.set_index(["date", "ticker"])
    if len(index.difference(keyed.index)) or len(keyed.index.difference(index)):
        raise ValueError(
            "News date/ticker panel has missing or extra rows; no zero fill permitted"
        )
    keyed = keyed.reindex(index)
    frame["risk_score"] = keyed["risk_score"]
    metadata = keyed.reset_index()

    def panel(name):
        return frame[name].to_numpy(dtype=np.float64).reshape(len(dates), len(assets))

    return MarketData(
        dates,
        list(assets),
        macros,
        panel("log_return"),
        panel("rsi"),
        panel("macd"),
        frame[macros]
        .to_numpy(dtype=float)
        .reshape(len(dates), len(assets), len(macros))[:, 0, :],
        panel("risk_score"),
        metadata,
    )


def read_table(path):
    """Read a local UTF-8 CSV or parquet table."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def load_data(config, root):
    # TODO: set data_path/news_path to point-in-time daily parquet files.
    frame = (
        read_table(root / config["data_path"])
        if config["data_path"]
        else dummy_data(config)
    )
    news = read_table(root / config["news_path"]) if config["news_path"] else None
    if news is not None and not config["data_path"]:
        # Hybrid smoke runs use the supplied news calendar, never pretend real prices.
        news_dates = parse_dates(news["date"])
        if not news_dates.isin(parse_dates(frame["date"])).all():
            raise ValueError("News dates outside configured dummy date range")
        frame = frame[parse_dates(frame["date"]).isin(news_dates)].copy()
    data = prepare_data(
        frame,
        config["assets"],
        news,
        config["macro_columns"],
        news_mode=config.get("news_mode", "required"),
    )
    return data, frame
