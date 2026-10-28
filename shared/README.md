# Shared trained investment demo

실제 2023~2025 시장·뉴스 데이터로 학습한 v3 모델 `20261028_005508`입니다.
모델 2개, scaler·설정·feature 순서, 시장/뉴스 입력, 최종 백테스트 기록과 보고서를 포함합니다.
약 1.9 MB로 Git에 함께 저장합니다. 전체 학습 로그·중간 체크포인트·Chroma DB·API 키는 포함하지 않습니다.
가격은 배당을 추가 반영하지 않은 Close 기준입니다. 재학습 없이 포트폴리오 화면과 추론을 실행할 수 있습니다.
리서치/RAG 기능을 제공하는 묶음은 아닙니다.

## 팀원 실행 방법 (프로젝트 루트, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/install_demo.py
powershell -ExecutionPolicy Bypass -File .\run.ps1 api
```

새 터미널에서:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 ui
```

브라우저에서 http://localhost:8501 을 여세요. 최초 Streamlit Email 질문에는 빈칸으로 Enter를 누르면 됩니다.
http://localhost:8000/ready 에서 model_loaded=true를 확인하세요.

설치기는 SHA-256을 검증하며 기존 데이터가 다르면 덮어쓰지 않습니다.
기존 runtime/runs/latest.json이 있으면 유지하므로 기존 사용자의 최신 모델을 바꾸지 않습니다.
새 PC에서는 함께 제공된 모델을 자동 선택합니다. 학습은 실행하지 않습니다.

## Docker (로컬 Python으로 설치 후)

```powershell
$env:INVESTMENT_AI_RUNTIME=(Resolve-Path runtime).Path
$env:INVESTMENT_AI_RUN_ID='20261028_005508'
docker compose -f docker-compose.yml -f docker-compose.investment.yml up --build
```

기본 CI Compose는 별도의 모델 없는 서버 동작을 검사합니다.
