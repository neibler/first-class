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

결과 폴더는 `--output` 옵션으로 변경할 수 있습니다. FastAPI는 `RESULTS_DIR` 환경변수가 가리키는 결과를 읽으며 `/performance/health`, `/performance/backtest`, `/performance/explain`, `/performance/statistics`, `/performance/summary`를 제공합니다.

## 공유 모델로 성과 화면 준비

팀원이 생성한 JSON이 있으면 `runtime/results/api/backtest.json`에 연결하는 것을 우선합니다.
기존 결과가 있으면 덮어쓰지 말고 보관하거나 별도 `--output` 폴더에서 생성하세요.
아래는 공유 번들 `shared/investment_demo.zip`·`shared/manifest.json`을 설치한 뒤
저장된 모델로 **평가만** 실행하는 PowerShell 예시입니다. 재학습·외부 뉴스 수집은 하지 않습니다.

```powershell
.venv/Scripts/python.exe -m pip install -r backend/performance/requirements.txt
.venv/Scripts/python.exe -X utf8 scripts/install_demo.py
$env:INVESTMENT_AI_HOME=(Resolve-Path runtime).Path
.venv/Scripts/python.exe -X utf8 -m backend.performance.run_analysis --run runtime/runs/20261028_005508 --output runtime/results --skip-shap --risk-tag none --mvo-frequency month_end --mvo-strict-window
```

Docker는 공용 `docker-compose.yml`을 바꾸지 않고 프로젝트 루트의 로컬 파일
`docker-compose.override.yml`에 아래 내용을 추가합니다. 기존 파일이 있다면 해당 서비스 설정에
병합하며 덮어쓰지 않습니다. 이 파일은 Git에서 제외되며 일반 `docker compose up`에서 자동 적용됩니다.

```yaml
services:
  api:
    volumes:
      - type: bind
        source: ${PERFORMANCE_RESULTS_DIR:-./runtime/results}
        target: /results
        read_only: true
```

```powershell
docker compose up --build -d --wait --wait-timeout 120
Invoke-RestMethod http://localhost:8000/performance/health
```

호스트 `runtime/results`를 컨테이너 `/results`에 읽기 전용으로 연결합니다.
공용 Compose의 기존 `RESULTS_DIR=/results` 설정은 유지합니다.
다른 결과 폴더를 쓰려면 로컬 `.env`의 `PERFORMANCE_RESULTS_DIR`를 지정합니다.
설정 변경 후 `docker compose up -d --force-recreate --wait`로 반영합니다.
`-f`로 Compose 파일 목록을 직접 지정할 때는 `-f docker-compose.override.yml`도 포함해야 합니다.
공용 설정만 확인하려면 `docker compose -f docker-compose.yml config --quiet`를 사용합니다.

Python 직접 실행 또는 `run.ps1 api`는 시작 전에 `RESULTS_DIR`를 명시합니다.
공용 Python 실행기의 기본값은 변경하지 않습니다:

```powershell
$env:RESULTS_DIR=(Resolve-Path runtime/results).Path
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`/performance/health`의 `available.backtest=true`와 `/performance/backtest`의 `200`을 확인하고
`http://localhost:8501/performance`를 엽니다. 파일이 없으면 `503`이 정상입니다.
`--skip-shap` 때문에 `available.explain=false`, 전체 `ready=false`여도 백테스트 조회는 가능합니다.
투자 `/health`의 `model_loaded`와 저장 결과의 준비 상태도 별개입니다.

성과 화면은 `performance-1` 응답을 사용합니다. `path`의 `value - 1`을 백분율 차트에,
`strategies[].metrics`의 12개 서버 값을 표에 표시합니다. `null`은 `—`입니다.
서버 라벨로 벤치마크를 구분하며 `evaluation`, `window_plan`, `smoke`를 표시합니다.
이번 공유 결과는 단일 홀드아웃이며 학교의 다중 Walk-Forward 완료를 뜻하지 않습니다.

다른 팀원은 보기만 한다면 `api/backtest.json`만 받으면 됩니다. 재생성하려면 공유 번들 안의
`runs/20261028_005508/`(모델은 `final/seed_42/`, `final/seed_73/`)와 `data/`를
원래 경로 구조대로 설치해야 합니다.
정확한 파일 목록과 해시는 `shared/manifest.json`을 기준으로 확인합니다.

ANOVA 화면은 `/performance/statistics`에 연결되어 있습니다. 조회에 필요한 파일은
`<RESULTS_DIR>/statistics/anova.json`입니다. 기존 결과를 연결하는 방법은
[루트 실행 안내](../../README.md#anova-결과-연결)를 참고하세요.

## 테스트

```bash
python -m pytest tests/test_performance.py -q
```

## 결과 해석 시 주의사항

- 현재 실데이터 분석은 **2023–2025년 데이터와 5,000 step 학습 모델**의 1차 결과입니다. 최종 홀드아웃은 **2025-05-28~2025-12-31(151거래일)** 입니다.
- 과제 예시인 **4년 학습·1년 테스트의 다중 Walk-Forward 검증**은 현재 데이터 기간으로는 충족되지 않습니다.
- 통계적 유의성은 다중비교 보정 후의 p값을 기준으로 해석합니다. 상세 지표, 검정 결과와 한계는 `REAL_RESULTS.md`에 있습니다.
