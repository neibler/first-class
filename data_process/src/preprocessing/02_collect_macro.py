"""ALFRED 11개 지표의 Vintage 이력과 ISM PMI 엑셀을 원본 Parquet으로 저장."""

import argparse
import os
import warnings
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.stlouisfed.org/fred/series/observations"
VINTAGE_CHUNK_DAYS = 1800
SERIES = {
    "PCEPILFE": "core_pce",
    "PCEC96": "real_pce",
    "UNRATE": "unemp",
    "EFFR": "effr",
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "DGS30": "ust_30y",
    "DFII10": "tips_10y",
    "VIXCLS": "vix",
    "DTWEXBGS": "broad_usd",
    "DCOILWTICO": "wti",
}
COLUMNS = [
    "series_id",
    "variable_name",
    "observation_date",
    "value",
    "realtime_start",
    "realtime_end",
    "release_date",
    "source",
]


def collect_fred(
    session: requests.Session,
    series_id: str,
    variable_name: str,
    api_key: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """관측일별 최초 값·수정값과 각 값의 Vintage 유효기간 수집."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
        "output_type": 1,
        "units": "lin",
        "sort_order": "asc",
        "limit": 100000,
    }
    observations = []
    period_start = start
    while True:
        # 달력일 수를 2,000 미만으로 제한하면 일별 Vintage도 API 한도 이내.
        period_end = min(period_start + timedelta(days=VINTAGE_CHUNK_DAYS - 1), end)
        params.update(
            realtime_start=period_start.isoformat(),
            realtime_end=period_end.isoformat(),
            offset=0,
        )
        period_rows = []
        while True:
            try:
                response = session.get(API_URL, params=params, timeout=30)
                payload = response.json()
            except requests.RequestException as error:
                # 요청 URL에는 API 키가 포함되므로 예외 원문을 출력하지 않는다.
                raise RuntimeError(
                    f"{series_id}: FRED 요청 실패 ({type(error).__name__})"
                ) from None
            if not response.ok or "error_message" in payload:
                message = payload.get("error_message", f"HTTP {response.status_code}")
                raise RuntimeError(f"{series_id}: {message.replace(api_key, '[redacted]')}")
            if "observations" not in payload or "count" not in payload:
                raise ValueError(f"{series_id}: FRED 응답에 관측값 또는 전체 건수가 없습니다.")

            page = payload["observations"]
            count = int(payload["count"])
            if not page and (count != 0 or period_rows):
                raise ValueError(f"{series_id}: Vintage 수집이 중단됐습니다.")
            period_rows.extend(page)
            if len(period_rows) == count:
                break
            if len(period_rows) > count:
                raise ValueError(f"{series_id}: FRED 응답 건수가 일치하지 않습니다.")
            params["offset"] = len(period_rows)
        observations.extend(period_rows)
        if period_end == end:
            break
        # 하루를 겹쳐 받아 요청 경계에서 나뉜 동일 유효기간을 다시 연결.
        period_start = period_end

    if not observations:
        raise ValueError(f"{series_id}: 지정 기간에 Vintage 데이터가 없습니다.")

    data = pd.DataFrame(observations).rename(columns={"date": "observation_date"})
    required = {"observation_date", "value", "realtime_start", "realtime_end"}
    if not required.issubset(data.columns):
        raise ValueError(f"{series_id}: FRED 응답에 필수 컬럼이 누락됐습니다.")
    for column in ("observation_date", "realtime_start", "realtime_end"):
        data[column] = pd.to_datetime(data[column], errors="raise")
        if data[column].isna().any():
            raise ValueError(f"{series_id}: {column} 날짜가 누락됐습니다.")
    if not data["observation_date"].between(pd.Timestamp(start), pd.Timestamp(end)).all():
        raise ValueError(f"{series_id}: 요청 기간 밖의 관측값이 있습니다.")
    if data[["realtime_start", "realtime_end"]].gt(pd.Timestamp(end)).any().any():
        raise ValueError(f"{series_id}: 종료일 이후의 Vintage 정보가 포함됐습니다.")
    if data["realtime_start"].gt(data["realtime_end"]).any():
        raise ValueError(f"{series_id}: Vintage 유효기간이 잘못됐습니다.")

    # FRED의 결측 표기 '.'만 결측값으로 변환. 행과 수정 이력은 유지.
    data["value"] = pd.to_numeric(data["value"].replace(".", float("nan")), errors="raise")
    if not data["value"].notna().any():
        raise ValueError(f"{series_id}: 관측값이 모두 결측입니다.")

    data = data.sort_values(["observation_date", "realtime_start"], kind="stable")
    previous = data.shift()
    same_value = data["value"].eq(previous["value"]) | (
        data["value"].isna() & previous["value"].isna()
    )
    overlap = (
        data["observation_date"].eq(previous["observation_date"])
        & same_value
        & data["realtime_start"].le(previous["realtime_end"])
    )
    # 같은 관측일의 서로 다른 수정값은 유지하고, 분할 요청으로 겹친 기간만 연결.
    data = data.groupby((~overlap).cumsum(), sort=False).agg(
        observation_date=("observation_date", "first"),
        value=("value", "first"),
        realtime_start=("realtime_start", "min"),
        realtime_end=("realtime_end", "max"),
    ).reset_index(drop=True)
    data["series_id"] = series_id
    data["variable_name"] = variable_name
    data["release_date"] = pd.NaT
    data["source"] = "alfred"
    return data[COLUMNS]


def read_pmi(path: Path, start: date, end: date) -> pd.DataFrame:
    """엑셀의 발표일과 값을 유지. 확인되지 않은 관측월·Vintage는 결측 유지."""
    data = pd.read_excel(path, usecols=["release_date", "ISM_PMI"], engine="openpyxl")
    if pd.api.types.is_numeric_dtype(data["release_date"]):
        raise ValueError("ISM PMI: 발표일이 숫자입니다. 엑셀 날짜 형식을 확인하세요.")
    data["release_date"] = pd.to_datetime(data["release_date"], errors="raise")
    if data["release_date"].isna().any():
        raise ValueError("ISM PMI: 발표일이 누락됐습니다.")
    data["value"] = pd.to_numeric(data["ISM_PMI"], errors="raise")
    within_period = data["release_date"].ge(pd.Timestamp(start)) & data["release_date"].lt(
        pd.Timestamp(end + timedelta(days=1))
    )
    data = data.loc[within_period].copy()
    if data.empty or not data["value"].notna().any():
        raise ValueError("ISM PMI: 지정 기간의 발표값이 없습니다.")

    weekends = data.loc[data["release_date"].dt.dayofweek.ge(5), "release_date"]
    if not weekends.empty:
        warnings.warn(
            "ISM PMI: 주말로 표기된 발표일 확인 필요 (원본 유지): "
            + ", ".join(weekends.dt.strftime("%Y-%m-%d")),
            stacklevel=2,
        )
    data["series_id"] = "ISM_PMI"
    data["variable_name"] = "ism_pmi"
    data["observation_date"] = pd.NaT
    data["realtime_start"] = pd.NaT
    data["realtime_end"] = pd.NaT
    data["source"] = "investing"
    return data[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2017, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    parser.add_argument("--pmi-file", type=Path, required=True, help="ISM_PMI_cleaned.xlsx 경로")
    parser.add_argument("--overwrite", action="store_true", help="기존 원본 교체")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("start는 end보다 이전이어야 합니다.")
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        parser.error("환경변수 FRED_API_KEY를 설정하세요.")
    output = ROOT / "raw/macro_observations.parquet"
    if output.exists() and not args.overwrite:
        parser.error("원본 파일이 이미 있습니다. 교체하려면 --overwrite를 사용하세요.")

    pmi = read_pmi(args.pmi_file, args.start, args.end)
    frames = []
    with requests.Session() as session:
        for series_id, variable_name in SERIES.items():
            frames.append(
                collect_fred(session, series_id, variable_name, api_key, args.start, args.end)
            )
    data = pd.concat([*frames, pmi], ignore_index=True)
    if set(data["variable_name"]) != set(SERIES.values()) | {"ism_pmi"}:
        raise ValueError("확정한 12개 거시경제지표가 모두 수집되지 않았습니다.")
    data = data.sort_values(
        ["variable_name", "observation_date", "realtime_start", "release_date"]
    ).reset_index(drop=True)
    data.attrs = {
        "requested_start": args.start.isoformat(),
        "requested_end": args.end.isoformat(),
        "collected_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "series": SERIES,
        "vintage_chunk_days": VINTAGE_CHUNK_DAYS,
        # Published provenance must not contain the collector's personal path.
        "pmi_file": args.pmi_file.name,
        "pmi_weekend_release_dates": pmi.loc[
            pmi["release_date"].dt.dayofweek.ge(5), "release_date"
        ].dt.strftime("%Y-%m-%d").tolist(),
        "vintage_note": "ALFRED 유효일은 별도 검증된 실제 발표일·시각이 아님.",
        "pandas_version": pd.__version__,
        "requests_version": requests.__version__,
        "pyarrow_version": version("pyarrow"),
        "openpyxl_version": version("openpyxl"),
        "python_dotenv_version": version("python-dotenv"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    try:
        data.to_parquet(temporary, engine="pyarrow", index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"거시경제지표 수집 완료: {data['variable_name'].nunique()}개 / {len(data):,}행")
    print(f"FRED/ALFRED: {len(SERIES)}개")
    print("ISM PMI: 1개")
    print(f"저장 완료: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"수집 실패: {error}") from None
