# 수집 원본

- 보존 기준: 수집 원본의 값·결측·수정 이력 유지, 후속 단계의 원본 파일 변경 없음
- 날짜 기준: ETF는 미국 거래소 현지 날짜, 거시지표는 관측일·버전 유효일·발표일 구분

**파일명: `etf_prices.parquet`**

- 출처: Yahoo Finance, `yfinance`, `auto_adjust=False`
- 기간: 2018-01-02~2026-09-30
- 데이터 형태: 세로형(Long), 날짜 × ETF 15개
- 행 × 컬럼: **32,970 × 11**
- 사용 컬럼: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `dividends`, `stock_splits`, `capital_gains`
- 생성 코드: `src/preprocessing/01_collect_etf.py`
- 활용 목적: 수정종가 기준 기술적 지표·로그수익률 계산

**파일명: `benchmark.parquet`**

- 출처: Yahoo Finance, `yfinance`, S&P 500 총수익지수 `^SP500TR`
- 기간: 2018-01-02~2026-09-30
- 데이터 형태: 세로형(Long), 날짜 × 지수 1개
- 행 × 컬럼: **2,198 × 10**
- 사용 컬럼: ETF 원본과 동일한 10개 컬럼, `capital_gains` 제외
- 생성 코드: `src/preprocessing/01_collect_etf.py`
- 활용 목적: S&P 500 지수 수준·비정규화 수익률 계산

**가격 파일 컬럼 의미**

| 컬럼 | 의미 |
|---|---|
| `date`, `ticker` | 거래일, 종목·지수 식별자 |
| `open`, `high`, `low`, `close` | 수집 시가·고가·저가·종가 |
| `adj_close` | 조정 내역 반영 수정종가, ETF 파생변수 계산 기준 |
| `volume` | 거래량 |
| `dividends` | 배당금 |
| `stock_splits` | 주식분할 내역 |
| `capital_gains` | 자본이득 분배금, ETF 원본에만 포함 |

**파일명: `macro_observations.parquet`**

- 출처: FRED API의 ALFRED 수정 이력 11개 지표 + Investing.com 기반 별도 PMI 엑셀
- 기간: 관측일 2017-01-01~2026-09-30, PMI 원본 발표일 2017-01-04~2026-09-01
- 데이터 형태: 세로형(Long), 지표 × 관측일 × 버전; PMI는 발표일 기준
- 행 × 컬럼: **31,932 × 8**, 원본 변수 12개
- 사용 컬럼: 아래 8개 컬럼
- 생성 코드: `src/preprocessing/02_collect_macro.py`
- 활용 목적: 과거 시점에 이용 가능했던 경제지표 복원·파생변수 계산

| 컬럼 | 의미 |
|---|---|
| `series_id` | 원천 지표 식별자 |
| `variable_name` | 프로젝트 원본 변수명 |
| `observation_date` | 경제지표 관측일, 월별 자료는 관측월 식별 날짜 |
| `value` | 해당 버전의 관측값 |
| `realtime_start` | 해당 버전의 유효 시작일 |
| `realtime_end` | 해당 버전의 유효 종료일 |
| `release_date` | 별도 수집 PMI의 원본 발표일 |
| `source` | `alfred` 또는 `investing` |

| 원천 식별자 | 원본 변수명 |
|---|---|
| `PCEPILFE`, `PCEC96` | `core_pce`, `real_pce` |
| `UNRATE`, `ISM_PMI` | `unemp`, `ism_pmi` |
| `EFFR` | `effr` |
| `DGS2`, `DGS10`, `DGS30` | `ust_2y`, `ust_10y`, `ust_30y` |
| `DFII10`, `VIXCLS` | `tips_10y`, `vix` |
| `DTWEXBGS`, `DCOILWTICO` | `broad_usd`, `wti` |

- Vintage 의미: 같은 관측일의 최초 발표값·수정값과 각 값의 유효기간
- 행 구분: 같은 `observation_date`의 다른 버전은 보존 대상, 일반 중복과 구분
- 시점 주의: `realtime_start`·`realtime_end`는 날짜 단위 유효기간, 정확한 발표 시각과 구분
- 수집 상한: 2026-09-30, 이후 수정 이력 미포함
- 결측 보존: FRED의 `.`는 NaN, ALFRED의 `release_date`는 미수집 상태 유지
- PMI 결측: 미확인 관측월·Vintage 날짜는 결측 유지, 발표일만 별도 보유
- PMI 원본 날짜: 05번 결합 단계의 검증·보정 대상, 원본 파일에는 보정 미반영
- 최종 변수 의미·단위: [가공 데이터 안내](../processed/README.md)
