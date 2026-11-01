# 성능 분석

PPO, MVO, 동일가중, S&P 500 전략을 비교하고 Walk-Forward 백테스트, 12개 성과지표, SHAP, ANOVA 결과를 생성하는 모듈입니다.

## 실행

Python 3.12 기준입니다. 모든 명령은 프로젝트 루트에서 실행합니다.

```bash
python -m pip install -r requirements-dev.txt
python scripts/install_demo.py
```

실데이터와 학습된 모델을 사용하려면 실행 환경에 따라 경로를 설정합니다.

- macOS/Linux: `export INVESTMENT_AI_HOME="$PWD/runtime"`
- Windows PowerShell: `$env:INVESTMENT_AI_HOME=(Resolve-Path runtime).Path`

```bash
python -m backend.performance.run_analysis --run runtime/runs/20261028_005508 --output runtime/results
```

모델 없이 코드 동작만 확인하려면 다음 명령을 사용합니다.

```bash
python -m backend.performance.run_analysis --config config/dummy_only.json --smoke --skip-shap
```

## 결과 확인

| 확인할 내용 | 위치 |
| --- | --- |
| 실데이터 분석 결과 및 해석 | `backend/performance/verification/real/REAL_RESULTS.md` |
| 합성 데이터 동작 검증 기록 | `backend/performance/verification/synthetic/VERIFICATION.md` |
| 12개 성과지표 | `runtime/results/metrics/metrics.csv` |
| 누적수익률·Drawdown 그래프 | `runtime/results/backtest/` |
| SHAP 결과 | `runtime/results/shap/` |
| ANOVA·사후검정 결과 | `runtime/results/statistics/` |
| API용 JSON | `runtime/results/api/` |

결과 폴더는 `--output` 옵션으로 변경할 수 있습니다. FastAPI는 `RESULTS_DIR` 환경변수가 가리키는 결과를 읽으며 `/performance/health`, `/backtest`, `/explain`, `/statistics`, `/summary`를 제공합니다.

## 테스트

```bash
python -m pytest tests/test_performance.py -q
```

## 결과 해석 시 주의사항

- 현재 실데이터 분석은 **2023–2025년 데이터와 5,000 step 학습 모델**의 1차 결과입니다. 최종 홀드아웃은 **2025-05-28~2025-12-31(151거래일)** 입니다.
- 과제 예시인 **4년 학습·1년 테스트의 다중 Walk-Forward 검증**은 현재 데이터 기간으로는 충족되지 않습니다.
- 통계적 유의성은 다중비교 보정 후의 p값을 기준으로 해석합니다. 상세 지표, 검정 결과와 한계는 `REAL_RESULTS.md`에 있습니다.
