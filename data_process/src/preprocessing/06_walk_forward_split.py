"""PPO용 Walk-Forward 분할·학습 구간 기준 Z-score·원본 벤치마크 준비."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TICKERS = "SPY QQQ SMH SCHD XLE EFA SHY IEF TLT TIP LQD GLD DBB DBA VNQ".split()
MACRO = "core_pce_yoy real_pce_mom unemp ism_pmi effr ust_2y ust_10y ust_30y tips_10y vix usd_ret_20d wti yield_spread".split()
PER_TICKER = ["adj_close", "log_return", "macd", "macd_signal", "macd_hist"]
FEATURES = ["adj_close", "log_return", "rsi_14", "macd", "macd_signal", "macd_hist", *MACRO]
FEATURE_ORDER = [f"{column}_z" for column in FEATURES]
KEYS = ["date", "ticker"]
# 이름, 전체 Train 시작일, Train 종료 연도, Test 종료일.
WINDOWS = [
    ("window_1", "2019-02-05", 2022, "2023-12-31"),
    ("window_2", "2020-01-01", 2023, "2024-12-31"),
    ("window_3", "2021-01-01", 2024, "2025-12-31"),
    ("final_2026", "2022-01-01", 2025, "2026-09-30"),
]
INPUTS = ["processed/features_unscaled.parquet", "processed/log_returns.parquet", "raw/benchmark.parquet"]


def input_hashes() -> dict:
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


def load_inputs():
    features, returns, benchmark = [pd.read_parquet(ROOT / name) for name in INPUTS]
    semantics = {key: features.attrs.get(key) for key in (
        "information_cutoff", "market_proxy_exceptions", "timing_limitations",
    )}
    for frame in (features, returns, benchmark):
        # 대형 Vintage 메타데이터의 반복 복사 방지. 파일의 메타데이터는 변경하지 않음.
        frame.attrs = {}
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("입력 파일에 날짜·종목 결측 또는 중복이 있습니다.")

    if set(features["ticker"]) != set(TICKERS):
        raise ValueError("ETF 15종목 구성이 일치하지 않습니다.")
    dates = pd.DatetimeIndex(features.loc[features["ticker"].eq("SPY"), "date"].unique()).sort_values()
    if len(features) != len(dates) * 15 or not features.groupby("date").size().eq(15).all():
        raise ValueError("SPY 거래일마다 ETF 15개가 필요합니다.")
    if np.isinf(features[FEATURES].to_numpy()).any():
        raise ValueError("입력 Feature에 무한대가 있습니다.")
    if features.groupby("date")[MACRO].nunique(dropna=False).ne(1).any().any():
        raise ValueError("같은 날짜의 거시지표가 ETF별로 다릅니다.")
    pd.testing.assert_frame_equal(
        features[KEYS + ["log_return"]].sort_values(KEYS).reset_index(drop=True),
        returns.sort_values(KEYS).reset_index(drop=True), check_exact=True,
    )
    if set(benchmark["ticker"]) != {"^SP500TR"} or not pd.DatetimeIndex(benchmark["date"].sort_values()).equals(dates):
        raise ValueError("S&P 500 총수익지수와 ETF 거래일이 일치하지 않습니다.")
    if (~np.isfinite(benchmark["close"]) | benchmark["close"].le(0)).any():
        raise ValueError("벤치마크 지수 수준에 결측·무한대·0 이하 값이 있습니다.")

    features["ticker"] = pd.Categorical(features["ticker"], categories=TICKERS, ordered=True)
    features = features.sort_values(KEYS).reset_index(drop=True)
    features["ticker"] = features["ticker"].astype(str)
    return features, returns, benchmark.sort_values("date"), semantics


def period_info(data: pd.DataFrame, requested: tuple[str, str]) -> dict:
    if data.empty:
        raise ValueError(f"구간 데이터가 없습니다: {requested}")
    return {
        "requested_start": requested[0], "requested_end": requested[1],
        "start": str(data["date"].min().date()), "end": str(data["date"].max().date()),
        "trading_days": int(data["date"].nunique()), "rows": len(data),
    }


def column_stats(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"Scaler 학습 데이터에 결측·무한대가 있습니다: {values.name}")
    mean = float(array.mean())
    std = 0.0 if np.all(array == array[0]) else float(array.std(ddof=0))
    return {"mean": mean, "std": std, "scale": std if std > 0 else 1.0, "n": len(array)}


def fit_scaler(data: pd.DataFrame, period: dict) -> dict:
    """호출자가 전달한 학습 행에만 Fit. 거시지표는 날짜당 한 번 집계."""
    by_ticker = {}
    for ticker in TICKERS:
        group = data.loc[data["ticker"].eq(ticker)]
        by_ticker[ticker] = {column: column_stats(group[column]) for column in PER_TICKER}
    shared = {"rsi_14": column_stats(data["rsi_14"])}
    daily = data.drop_duplicates("date")
    macro = {column: column_stats(daily[column]) for column in MACRO}
    zeros = [
        f"{ticker}.{column}" for ticker, columns in by_ticker.items()
        for column, stats in columns.items() if stats["std"] == 0
    ] + [column for column, stats in {**shared, **macro}.items() if stats["std"] == 0]
    return {
        "fit_period": period, "per_ticker": by_ticker, "shared": shared, "macro": macro,
        "zero_std_columns": zeros,
    }


def transform_features(data: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """저장된 통계로만 Transform. 과거 참고 행에도 같은 단계의 Scaler 적용."""
    result = data.copy()
    for column in PER_TICKER:
        means = {ticker: stats[column]["mean"] for ticker, stats in scaler["per_ticker"].items()}
        scales = {ticker: stats[column]["scale"] for ticker, stats in scaler["per_ticker"].items()}
        result[f"{column}_z"] = (data[column] - data["ticker"].map(means)) / data["ticker"].map(scales)
    for column, stats in {**scaler["shared"], **scaler["macro"]}.items():
        result[f"{column}_z"] = (data[column] - stats["mean"]) / stats["scale"]
    return result[[*data.columns, *FEATURE_ORDER]]


def history_info(start: str, feature_complete: pd.Series, returns_wide: pd.DataFrame) -> dict:
    prior = feature_complete.loc[feature_complete.index < start]
    missing = np.flatnonzero(~prior.to_numpy())
    consecutive = len(prior) - missing[-1] - 1 if len(missing) else len(prior)
    returns = returns_wide.loc[returns_wide.index < start]
    valid = returns.notna().all(axis=1)
    context = returns.tail(252)
    ready = len(context) == 252 and context.notna().all().all()
    return {
        "strictly_before": start, "complete_feature_days": int(prior.sum()),
        "consecutive_complete_feature_days": int(consecutive),
        "valid_return_days": int(valid.sum()), "mvo_252_available": bool(ready),
        "mvo_252_start": str(context.index[0].date()) if ready else None,
        "mvo_252_end": str(context.index[-1].date()) if ready else None,
    }


def build_window(features, returns_wide, feature_complete, spec, hashes, semantics):
    name, start, year, test_end = spec
    ranges = {
        "fit": (start, f"{year - 1}-12-31"),
        "validation": (f"{year}-01-01", f"{year}-12-31"),
        "train": (start, f"{year}-12-31"),
        "test": (f"{year + 1}-01-01", test_end),
    }
    periods = {key: features.loc[features["date"].between(*bounds)] for key, bounds in ranges.items()}
    info = {key: period_info(periods[key], bounds) for key, bounds in ranges.items()}
    if not (info["fit"]["end"] < info["validation"]["start"]
            and info["train"]["end"] < info["test"]["start"]
            and len(periods["fit"]) + len(periods["validation"]) == len(periods["train"])):
        raise ValueError(f"{name}: 기간 중복 또는 분할 오류가 있습니다.")

    scalers = {
        "formula": "(x - mean) / scale", "ddof": 0,
        "zero_std_policy": "std=0이면 scale=1; 학습값은 0, 이후 값은 (x-mean). 행 삭제·대체 없음.",
        "tuning": fit_scaler(periods["fit"], info["fit"]),
        "refit": fit_scaler(periods["train"], info["train"]),
    }
    outputs = {}
    for stage, bounds, boundary, labels in [
        ("tuning", ranges["train"], ranges["fit"][1], ("fit", "validation")),
        ("refit", (start, test_end), ranges["train"][1], ("train", "test")),
    ]:
        original = features.loc[features["date"].between(*bounds)].reset_index(drop=True)
        data = transform_features(original, scalers[stage])
        data.insert(2, "split", np.where(data["date"].le(pd.Timestamp(boundary)), *labels))
        if (data.duplicated(KEYS).any() or not data.groupby("date").size().eq(15).all()
                or not np.isfinite(data[FEATURE_ORDER].to_numpy()).all()):
            raise ValueError(f"{name}/{stage}: 중복·ETF 누락·정규화 결측/무한대가 있습니다.")
        pd.testing.assert_frame_equal(data[features.columns], original, check_exact=True)
        expected = len(periods["train"]) + (len(periods["test"]) if stage == "refit" else 0)
        if len(data) != expected:
            raise ValueError(f"{name}/{stage}: 데이터 손실이 있습니다.")
        data.attrs = {"window": name, "scaler": stage, "feature_order": FEATURE_ORDER,
                      "ticker_order": TICKERS, "manifest": "manifest.json", "scalers": "scalers.json"}
        outputs[f"features_{stage}.parquet"] = data

    manifest = {
        "window": name, "periods": info, "ticker_order": TICKERS, "feature_order": FEATURE_ORDER,
        "raw_columns": list(features.columns),
        "layout": "date 오름차순·ticker_order 순서, 한 행=date×ticker. PPO 입력은 feature_order의 19개 컬럼만 선택.",
        "files": {
            "tuning": "features_tuning.parquet", "refit": "features_refit.parquet", "scalers": "scalers.json",
            "returns": "../../processed/returns_wide.parquet",
            "benchmark": "../../processed/benchmark_returns.parquet",
            "history_source": "../../processed/features_unscaled.parquet",
        },
        "path_base": "이 manifest.json이 있는 폴더 기준 상대 경로.",
        "split_labels": {"tuning": ["fit", "validation"], "refit": ["train", "test"]},
        "scaler_fit": {"tuning": "내부 fit만 사용", "refit": "전체 train만 사용"},
        "normalization": {
            "per_ticker": PER_TICKER, "shared_across_tickers": ["rsi_14"],
            "macro_one_row_per_date": MACRO, "ddof": 0,
            "raw_log_return": "log_return은 원본 유지. PPO 입력에는 log_return_z 사용.",
        },
        "observation_history": {
            "lookback_sessions": None, "windows_created": False, "rows_removed": 0,
            "first_complete_feature_date": str(feature_complete.index[feature_complete][0].date()),
            "before_period_start": {key: history_info(value["start"], feature_complete, returns_wide) for key, value in info.items()},
            "usage": "의사결정일 t에는 t 이전 행 참고. Validation/Test는 같은 파일의 이전 기간을 참고할 수 있음. 이력을 참고해도 학습·평가 대상 기간은 확장하지 않음.",
            "outside_window": "Window 시작 전 자료는 history_source에서 읽고, 같은 Window·stage의 저장된 Scaler로 transform_features 적용. 다른 Window의 정규화 값 재사용 및 과거 참고 행으로 Scaler 재학습 금지.",
            "window_1_limitation": "2019-02-05 이전 usd_ret_20d는 결측. Window 1 시작 시 완전한 20~60거래일 이력 없음. 날짜 변경·행 삭제·미래 값 채우기 없이 PPO 담당자가 준비 구간 결정.",
        },
        "returns_layout": {"index": "date", "columns": TICKERS, "values": "unscaled original log_return"},
        "benchmark": {
            "ticker": "^SP500TR", "level_source": "raw/benchmark.parquet:close",
            "log_return": "ln(index_level_t / index_level_previous_session)",
            "simple_return": "index_level_t / index_level_previous_session - 1",
            "normalization": "none", "first_return": "2018-01-02는 이전 가격이 없어 NaN 유지.",
        },
        "information_semantics": semantics, "input_sha256": hashes,
        "validation": {"feature_missing": 0, "feature_infinite": 0, "duplicate_keys": 0,
                       "zero_std_columns": {stage: scalers[stage]["zero_std_columns"] for stage in ("tuning", "refit")}},
        "training": "PPO/MVO 학습·관측 윈도우 생성·전략 평가 미진행. 각 Window의 모델은 독립적으로 새로 학습. 평가 중 Scaler 통계 갱신 금지.",
    }
    return {**outputs, "scalers.json": scalers, "manifest.json": manifest}


def save_artifact(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(temporary, index=path.name == "returns_wide.parquet")
        else:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="06번 결과만 교체")
    args = parser.parse_args()
    names = ["features_tuning.parquet", "features_refit.parquet", "scalers.json", "manifest.json"]
    paths = [ROOT / "splits" / spec[0] / name for spec in WINDOWS for name in names]
    paths += [ROOT / "processed/returns_wide.parquet", ROOT / "processed/benchmark_returns.parquet"]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error("06번 결과가 이미 있습니다. 교체하려면 --overwrite를 사용하세요.")
    hashes = input_hashes()
    features, returns, benchmark, semantics = load_inputs()
    returns_wide = returns.pivot(index="date", columns="ticker", values="log_return").reindex(columns=TICKERS).sort_index()
    returns_wide.columns.name = None
    returns_wide.attrs = {"values": "unscaled original log_return", "ticker_order": TICKERS,
                          "first_row": "2018-01-02 원본 NaN 유지; 이전 252거래일 참고용 수익률 보존"}
    level = benchmark["close"]
    ratio = level / level.shift(1)
    benchmark_returns = benchmark[KEYS].copy()
    benchmark_returns["index_level"] = level
    benchmark_returns["benchmark_log_return"] = np.log(ratio)
    benchmark_returns["benchmark_simple_return"] = ratio - 1
    benchmark_returns.attrs = {"source": "raw/benchmark.parquet:close", "ticker": "^SP500TR", "normalization": "none"}
    row_valid = pd.Series(np.isfinite(features[FEATURES].to_numpy()).all(axis=1), index=features.index)
    feature_complete = row_valid.groupby(features["date"]).all()

    artifacts = {ROOT / "processed/returns_wide.parquet": returns_wide,
                 ROOT / "processed/benchmark_returns.parquet": benchmark_returns}
    manifests = []
    for spec in WINDOWS:
        products = build_window(features, returns_wide, feature_complete, spec, hashes, semantics)
        artifacts.update({ROOT / "splits" / spec[0] / name: data for name, data in products.items()})
        manifests.append(products["manifest.json"])
    if hashes != input_hashes():
        raise ValueError("작업 중 입력 파일이 변경되었습니다.")
    for path, data in artifacts.items():
        save_artifact(data, path)
    for manifest in manifests:
        counts = {key: f"{value['trading_days']}일/{value['rows']}행" for key, value in manifest["periods"].items()}
        print(manifest["window"], counts, "표준편차 0:", manifest["validation"]["zero_std_columns"])
    print(f"저장: {len(artifacts)}개 파일 / 정규화 입력 {len(FEATURE_ORDER)}개 / ETF 고정 순서 {len(TICKERS)}개")
    print("원본 값·초기 행 유지. 관측 길이 미지정, 관측 윈도우·PPO/MVO 학습·전략 평가 없음.")


if __name__ == "__main__":
    main()
