# PPO 학습 데이터 사용 안내

**파일 선택**

| 작업 | 각 Window 폴더의 파일 | 대상 `split` | 정규화 통계 |
|---|---|---|---|
| 하이퍼파라미터 튜닝 | `features_tuning.parquet` | `fit`, `validation` | 내부 Fit만 사용한 `tuning` |
| 전체 학습·최종 평가 | `features_refit.parquet` | `train`, `test` | 전체 Train만 사용한 `refit` |
| 평균·표준편차 확인 | `scalers.json` | 해당 없음 | `tuning`·`refit` 별도 보관 |
| 기간·입력·ETF 순서 확인 | `manifest.json` | 해당 없음 | `periods`, `feature_order`, `ticker_order` |

폴더: [window_1](window_1/) · [window_2](window_2/) · [window_3](window_3/) · [final_2026](final_2026/)

**실제 거래일 기준 분할 — 각 manifest의 시작일·종료일·거래일 수**

| 구간 | 전체 Train | Validation | Test |
|---|---|---|---|
| Window 1 | 2019-02-05~2022-12-30 · 985일 | 2022-01-03~2022-12-30 · 251일 | 2023-01-03~2023-12-29 · 250일 |
| Window 2 | 2020-01-02~2023-12-29 · 1,006일 | 2023-01-03~2023-12-29 · 250일 | 2024-01-02~2024-12-31 · 252일 |
| Window 3 | 2021-01-04~2024-12-31 · 1,005일 | 2024-01-02~2024-12-31 · 252일 | 2025-01-02~2025-12-31 · 250일 |
| Final | 2022-01-03~2025-12-31 · 1,003일 | 2025-01-02~2025-12-31 · 250일 | 2026-01-02~2026-09-30 · 187일 |

| 내부 Fit | 실제 기간 | 거래일 수 |
|---|---|---:|
| Window 1 | 2019-02-05~2021-12-31 | 734 |
| Window 2 | 2020-01-02~2022-12-30 | 756 |
| Window 3 | 2021-01-04~2023-12-29 | 753 |
| Final | 2022-01-03~2024-12-31 | 753 |

- 전체 Train: 내부 Fit + Validation, 행 수는 각 거래일 수 × 15
- Window 1: 4년보다 짧은 Train, 시작일·초기 행 유지
- Window 2·3: 4개 연도 Train + 다음 1개 연도 Test
- Final Test: 2026년 1~9월 부분 연도 평가

**정규화 파일 구조**

- 파일명: `features_tuning.parquet`, `features_refit.parquet`
- 기간: 튜닝 파일은 내부 Fit + Validation, 재학습 파일은 전체 Train + Test
- 데이터 형태: 세로형(Long), 날짜 × ETF 15개
- 사용 컬럼: [원본 통합 데이터](../processed/README.md)의 21개 컬럼 + `split` + 정규화된 `*_z` 19개
- 컬럼 수: **41개**, PPO 입력은 그중 `feature_order`의 **19개 컬럼만 선택**
- 생성 코드: `src/preprocessing/06_walk_forward_split.py`

| 폴더 | 튜닝 파일 행 × 컬럼 | 재학습 파일 행 × 컬럼 |
|---|---:|---:|
| `window_1` | 14,775 × 41 | 18,525 × 41 |
| `window_2` | 15,090 × 41 | 18,870 × 41 |
| `window_3` | 15,075 × 41 | 18,825 × 41 |
| `final_2026` | 15,045 × 41 | 17,850 × 41 |

**Z-score 기준**

계산: `(원본 값 − 학습 평균) / 학습 표준편차`, 모집단 표준편차(`ddof=0`)

| 입력 변수 | 통계 계산 단위 |
|---|---|
| `adj_close`, `macd`, `macd_signal`, `macd_hist`, `log_return` | ETF별·컬럼별 |
| `rsi_14` | 학습 구간의 ETF 15개 공통 |
| 거시경제지표 13개 | 변수별, 날짜당 한 값 |

- 원본 가격·수익률: 기존 컬럼 유지, 입력용 `adj_close_z`·`log_return_z`와 구분
- 튜닝용 통계: 내부 Fit에서만 계산, Validation에는 저장된 통계 적용
- 재학습용 통계: 전체 Train에서 새로 계산, Test에는 저장된 통계 적용
- 평가 중 통계 갱신·다른 Window의 정규화 값 재사용: 금지
- 표준편차 0 처리: 분모 1 적용·항목 기록, 현재 4개 Window의 해당 항목 없음
- 정규화 입력 결측·무한대: 현재 저장 파일 8개에서 0건

**PPO 담당자 필수 기준**

| 항목 | 사용 기준 |
|---|---|
| 입력 컬럼 | `manifest.json`의 `feature_order`, `*_z` 19개 |
| 종목 순서 | `manifest.json`의 `ticker_order`, 15개 ETF |
| 행 순서 | `date` 오름차순 → 고정 ETF 순서, `[거래일, ETF 15개, 입력변수]` 배치용 |
| 보상용 수익률 | [../processed/returns_wide.parquet](../processed/returns_wide.parquet), 날짜·ETF 순서 일치 기준 |
| S&P 500 비교 | [../processed/benchmark_returns.parquet](../processed/benchmark_returns.parquet) |
| 의사결정일 | 해당일보다 앞선 거래일까지의 정보 사용, 당일 수익률의 당일 의사결정 입력 금지 |
| 관측 길이 | PPO 담당자 결정, 06번의 관측 윈도우·3차원 배열 생성 없음 |
| 학습·평가 대상 | `split` 기준, 과거 이력 참고 기간과 수익률 집계 기간 구분 |

- Validation·Test 첫날 이력: 같은 파일의 이전 기간 자료 참고, 해당 평가 구간의 `split`만 평가 대상
- Window 시작 전 이력: `manifest.json`의 `files.history_source` 참고, 같은 Window·단계의 Scaler로 정규화 적용
- Window 1 초기 제한: 2019-02-05 이전 `usd_ret_20d` 결측, 시작 시 완전한 20~60거래일 이력 부재, 초기 행 삭제·미래 값 보완 없음
- 다른 Window·모든 Validation·Test: 시작 전 완전한 60거래일 이력 확보
- MVO용 이력: 모든 구간 시작 전 유효한 252거래일 수익률 확보, 2018년 수익률 보존
- 경로 해석: 각 `manifest.json`이 있는 폴더 기준 상대 경로
- 기존 시점 근사: VIX·WTI 주의사항은 [가공 데이터 안내](../processed/README.md) 및 `information_semantics` 참고
