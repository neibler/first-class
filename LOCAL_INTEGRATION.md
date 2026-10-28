# 로컬 통합본 사용 안내

GitHub `neibler/first-class`의 `3967b6745727ec166c98ab650d4bb0980d525107`을 내려받아 투자 AI를 통합했습니다.
현재 브랜치는 `local/investment-ai-integration`입니다. 커밋·push는 하지 않았습니다.

**이번 통합 작업에서는 학습을 실행하지 않았습니다.** `runtime/runs/research_v2_verified`는 이전에 학습한 모델·결과를 복사한 것입니다.

## 지금 컴퓨터에서 실행

이 폴더에서 PowerShell을 열고 다음 명령을 실행하세요.

API 서버:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 api
```

별도 터미널에서 화면:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 ui
```

화면: http://127.0.0.1:8501
API 문서: http://127.0.0.1:8000/docs
모델 준비 상태: http://127.0.0.1:8000/ready

API·화면 실행으로 학습이 시작되지는 않습니다. 기존 저장 모델과 입력을 사용합니다.

## 직접 학습할 때

`config/investment_ai.json`을 수정한 후 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 train
```

기본은 합성 데이터, 보상 3종 × 검증 구간 3개 × seed 2개와 최종 모델 2개입니다. 모델당 5,000 step, 전체 100,000 step이며 새 결과는 `runtime/runs/실행ID/`에 생성됩니다. 기존 실행은 덮어쓰지 않습니다.

실제 데이터가 준비되면 `runtime/data/`에 넣고 설정의 `data_path`를 `data/market.parquet`, `news_path`를 `data/news.parquet`로 지정합니다. 아직 뉴스 점수가 없으면 news_path는 null로 두세요. macro_columns는 확정된 컬럼 순서로 지정합니다. 상대 데이터 경로는 runtime 폴더 기준입니다.

학습 후 평가·추론:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 evaluate
powershell -ExecutionPolicy Bypass -File .\run.ps1 infer
```

서버가 새 모델을 사용하려면 API 서버를 종료하고 다시 시작하세요. API는 시작할 때 모델을 한 번 읽습니다.

## 폴더 구분

- backend/investment_ai/: 투자 환경·학습·검증·API 코드
- config/investment_ai.json: 직접 수정할 학습 설정
- frontend/: API에 연결한 기존 First Class 화면
- runtime/data/: 실제 입력 파일을 넣을 위치
- runtime/runs/: 저장 모델·보고서·평가 기록
- run.ps1 / local_run.py: 로컬 실행 진입점

runtime 전체는 Git 제외 경로입니다. 기존 모델의 과거 source_hash와 현재 통합 소스는 다르므로 이전 모델에 이어 학습하는 resume는 거부될 수 있습니다. 새 train 실행은 가능합니다.

## 다른 컴퓨터로 옮길 때

현재 run.ps1은 `.venv`를 먼저 사용하고 없으면 이번 작업의 전용 Python을 찾습니다. 다른 컴퓨터에서는 프로젝트에 가상환경을 만드세요.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe local_run.py api
```

## Docker

Docker가 설치된 환경에서는 이 폴더에서:

```powershell
$env:INVESTMENT_AI_RUNTIME = (Resolve-Path .\runtime).Path
docker compose -f docker-compose.yml -f docker-compose.investment.yml up --build -d
```

Docker 빌드·실행은 이번 환경에 Docker가 없어 검증하지 않았습니다. 현재 통합은 포트폴리오 API와 화면 연결이며 SHAP·리서치·ANOVA 등 다른 화면의 미구현 기능은 그대로 남아 있습니다.
## 이번 통합 검증 결과

- 저장소 테스트 81개 통과.
- backend/frontend/tests의 flake8 검사 통과.
- 새 로컬 통합본의 backend.investment_ai 패키지가 실제로 로드되는지 확인.
- 복사한 기존 모델로 /ready, /optimize, /backtest ASGI 응답 확인.
- 15개 ETF 목표 비중, 157일 평가 기록, 현금 100% 상태 확인.
- 이 검증에서는 PPO.learn 호출을 금지해 학습이 실행되지 않도록 검사했습니다.
- Docker 실행과 GitHub push는 하지 않았습니다.


## 2026-10-27 뉴스 데이터 연결
기본 학습 설정은 feature/11의 실제 시장 + 실제 뉴스로 변경되었습니다. 실행 방법과 실제 시장 데이터 전환은 [NEWS_DATA_SETUP.md](NEWS_DATA_SETUP.md)를 확인하세요.


## 계약 v3 변경
뉴스 필수 6열, 누적 MDD 안전장치, 다음 종가 체결 및 연율화 Sharpe를 적용했습니다.
기존 모델은 새 계약과 호환되지 않아 새 학습이 필요합니다. [CONTRACT_V3.md](CONTRACT_V3.md) 참고.
