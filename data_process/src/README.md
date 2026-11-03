# 전처리 코드·실행 순서

- 코드 위치: `src/preprocessing/`
- 경로 기준: 아래 입력·출력은 `data_process/` 기준
- 실행 환경: Python 3.12.14, 기존 `.venv/bin/python`

| 파일명 | 역할 | 입력 | 출력 | 핵심 처리 |
|---|---|---|---|---|
| `01_collect_etf.py` | ETF·비교 지수 수집 | Yahoo Finance | `raw/etf_prices.parquet`, `raw/benchmark.parquet` | ETF 15개·`^SP500TR`, 일별 가격·수정종가·분배 내역 수집 |
| `02_collect_macro.py` | 거시지표 원본 수집 | FRED API·PMI 엑셀 | `raw/macro_observations.parquet` | ALFRED 11개 지표의 최초·수정 버전, PMI 발표일·값 보존 |
| `03_clean_data.py` | 원본 품질 점검 | `raw/etf_prices.parquet`, `raw/macro_observations.parquet` | 콘솔 점검표, 데이터 파일 생성 없음 | 날짜·중복·결측·휴장일·가격·Vintage 점검 |
| `04_calculate_features.py` | 기술적 지표·거시 파생변수 계산 | `raw/etf_prices.parquet`, `raw/macro_observations.parquet` | `processed/etf_features.parquet`, `processed/log_returns.parquet`, `processed/macro_features.parquet` | 수정종가 기준 RSI·MACD·로그수익률, 같은 Vintage 기준 PCE·달러 변화율·금리차 |
| `05_merge_data.py` | 날짜별 통합 입력 구성 | `processed/etf_features.parquet`, `processed/log_returns.parquet`, `processed/macro_features.parquet`, `raw/macro_observations.parquet` | `processed/features_unscaled.parquet` | SPY 거래일·16:00 동부시간 기준 결합, PMI 날짜 검증, 유효 버전·결측 구분 |
| `06_walk_forward_split.py` | 학습 분할·정규화·비교 기준 준비 | `processed/features_unscaled.parquet`, `processed/log_returns.parquet`, `raw/benchmark.parquet` | `splits/`의 Window별 Parquet·JSON, `processed/returns_wide.parquet`, `processed/benchmark_returns.parquet` | 4개 Window, 튜닝·재학습 통계 분리, 원본 수익률 보존 |

**실행 순서: 01 → 02 → 03 → 04 → 05 → 06**

- 실행 위치: 터미널의 현재 폴더 `data_process/`
- 02번 선행 조건: 개별 로컬 인증 설정·외부 PMI 엑셀 준비
- 현재 PMI 입력 파일: `$HOME/Downloads/ISM_PMI_cleaned.xlsx`, 엑셀 열 `release_date`, `ISM_PMI`
- 아래 명령어: 최초 생성 기준, 현재 결과 파일 존재 시 01·02·04·05·06의 기본 실행 중단
- 기존 결과 재생성: 해당 단계 명령어에 `--overwrite` 추가, 해당 단계 출력 교체·01/02 원천 데이터 재요청
- 기존 데이터 사용·문서 열람: 아래 생성 명령어 실행 불필요

```sh
.venv/bin/python src/preprocessing/01_collect_etf.py
.venv/bin/python src/preprocessing/02_collect_macro.py --pmi-file "$HOME/Downloads/ISM_PMI_cleaned.xlsx"
.venv/bin/python src/preprocessing/03_clean_data.py
.venv/bin/python src/preprocessing/04_calculate_features.py
.venv/bin/python src/preprocessing/05_merge_data.py
.venv/bin/python src/preprocessing/06_walk_forward_split.py
```

| 실행 항목 | 현재 코드 기준 |
|---|---|
| 01번 기본 조회 기간 | 2018-01-01~2026-09-30, `--start`·`--end`로 지정 가능 |
| 02번 기본 조회 기간 | 2017-01-01~2026-09-30, `--pmi-file` 필수 |
| 03번 재실행 | 검사·출력만 수행, 원본·가공 데이터 변경 없음 |
| 04~06번 | 현재 수집 파일 기반 계산·결합·분할, 외부 API 재수집 없음 |
| 현재 작업 범위 | 데이터 준비까지, PPO 모델·MVO 가중치·전략 성과 계산 미포함 |

- 변수·수식: [가공 데이터 안내](../processed/README.md)
- 정규화·분할·PPO 사용 기준: [학습 데이터 안내](../splits/README.md)
- 폴더·출력 경로: [전체 안내](../README.md)
