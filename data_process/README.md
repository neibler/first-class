# 금융 데이터 전처리

- 목적: PPO 기반 ETF 자산배분 모델용 금융 데이터 수집·가공·학습 구간 분할
- 완료 범위: 01~06번 전처리, 정규화 입력 및 비교 기준용 데이터 저장
- 학습 데이터 기준 시작일: **2019-02-05**

| 구분 | 실제 데이터 기간 |
|---|---|
| 거시지표 관측일 | 2017-01-01~2026-09-30, 지표별 발표·관측 범위 차이 |
| ETF·S&P 500 | 2018-01-02~2026-09-30, 미국 ETF 거래일 2,198일 |
| 결측 없는 통합 입력 | 2019-02-05~2026-09-30 |

**ETF 15개 — 고정 순서**

`SPY, QQQ, SMH, SCHD, XLE, EFA, SHY, IEF, TLT, TIP, LQD, GLD, DBB, DBA, VNQ`

**거시경제 변수 13개**

`core_pce_yoy`, `real_pce_mom`, `unemp`, `ism_pmi`, `effr`, `ust_2y`, `ust_10y`, `ust_30y`, `tips_10y`, `vix`, `usd_ret_20d`, `wti`, `yield_spread`

변수 의미·단위: [가공 데이터 안내](processed/README.md)

**폴더 구조 및 역할**

```text
data_process/
├── README.md
├── raw/                  # 수집 원본·README
├── processed/            # 파생변수·통합 데이터·원본 수익률·README
├── splits/               # 학습 구간·정규화 결과·README
│   ├── window_1/
│   ├── window_2/
│   ├── window_3/
│   └── final_2026/
├── src/
│   ├── README.md
│   └── preprocessing/    # 01~06번 Python 코드
├── .venv/                # 로컬 Python 실행 환경
├── .cache/               # 수집 라이브러리 캐시
├── .env                  # 로컬 설정, 공유 제외
└── READMD.md/            # 기존 빈 폴더, 데이터 처리와 무관
```

**전처리 순서**

| 단계 | 역할 |
|---|---|
| 01 | ETF 15개·S&P 500 총수익지수 수집 |
| 02 | ALFRED 11개 지표·별도 PMI 원본 수집 |
| 03 | 날짜·중복·결측·거래일·가격·수정 이력 점검, 콘솔 출력 |
| 04 | ETF 기술적 지표·로그수익률·거시 파생변수 계산 |
| 05 | SPY 거래일 기준 ETF·거시지표 결합, 정규화 전 데이터 저장 |
| 06 | 학습 구간 분할·학습 자료 기준 정규화·비교 기준용 수익률 저장 |

실행 순서·명령어: [코드 안내](src/README.md)

**PPO 담당자 우선 확인 파일 — Window 1 예시**

| 용도 | 경로 |
|---|---|
| 하이퍼파라미터 튜닝 | [splits/window_1/features_tuning.parquet](splits/window_1/features_tuning.parquet) |
| 전체 학습·최종 평가 | [splits/window_1/features_refit.parquet](splits/window_1/features_refit.parquet) |
| 기간·입력 컬럼·ETF 순서 | [splits/window_1/manifest.json](splits/window_1/manifest.json) |
| 보상·동일가중·MVO용 원본 수익률 | [processed/returns_wide.parquet](processed/returns_wide.parquet) |
| S&P 500 비교 기준 | [processed/benchmark_returns.parquet](processed/benchmark_returns.parquet) |

다른 Window의 파일명·사용 방법: [학습 데이터 안내](splits/README.md)

- 데이터 형식: 세로형 = 한 행에 날짜·종목 또는 지표·버전, 가로형 = 한 날짜에 ETF별 열
- 사용 기준: 정규화된 입력과 원본 보상용 수익률 구분, 직전 거래일까지의 정보로 의사결정
- 상세 안내: [원본](raw/README.md) · [가공 데이터](processed/README.md) · [학습 분할](splits/README.md)
