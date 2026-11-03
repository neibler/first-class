# 투자 기능 실행

프로젝트 루트의 Python 3.12 가상환경을 사용한다. 실행 설정은
`config/investment_ai.json`, 입력·모델·상태 저장 위치는 `runtime/`이다.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe scripts/install_demo.py
powershell -ExecutionPolicy Bypass -File .\run.ps1 api
# 별도 터미널
powershell -ExecutionPolicy Bypass -File .\run.ps1 ui
```

설치기는 [팀 공유 데모](../shared/README.md)의 체크섬을 검사하고 기존 파일이 다르면 중단한다.
기존 `runtime/runs/latest.json`은 유지한다. API·UI 시작은 새 학습을 실행하지 않는다.
`run.ps1`은 프로젝트 `.venv`만 사용한다. 다른 환경에서는 해당 Python으로 `local_run.py`를 실행한다.

- 화면: http://localhost:8501
- API 문서: http://localhost:8000/docs
- 투자 준비 상태: http://localhost:8000/ready

직접 학습·평가·추론은 명시적인 `run.ps1 train`, `evaluate`, `infer` 명령으로 실행한다.
입력 준비와 계약은 [뉴스·시장 데이터 안내](../NEWS_DATA_SETUP.md),
[투자 계약](../CONTRACT_V3.md)을 따른다. 서버는 시작 시 모델을 로드하므로 모델 교체 후 재시작한다.
개인 데이터·학습 산출물·서비스 상태는 Git에 추가하지 않는다.

## Docker

공유 데모를 설치하거나 기존 모델·입력을 별도로 준비한 뒤 실행한다.

```powershell
$env:INVESTMENT_AI_RUNTIME=(Resolve-Path runtime).Path
docker compose -f docker-compose.yml -f docker-compose.investment.yml up --build -d
```

투자 overlay는 `investment-state-init`이 성공한 뒤 API를 시작한다.
이 초기화 서비스만 root로 실행하여 호스트 `service_state` 폴더와 기존 파일의 소유자를
UID/GID 10001로 설정하고 소유자 읽기·쓰기·디렉터리 접근 권한을 부여한다.
API는 계속 `appuser`로 실행한다. 모델·데이터 전체의 권한은 변경하지 않는다.
기존 Safe-Guard 상태 파일의 내용과 경로를 유지하며 새 빈 저장소로 대체하지 않는다.
호스트 파일시스템에서 `chown`이 거부되면 초기화와 API 시작이 실패한다.
UID 10001 소유권 설정을 지원하는 저장 경로와 운영 권한을 준비해야 한다.

기본 Compose는 사용자 모델을 마운트하지 않는다. 리서치의 키·DB·임베딩 준비는
[리서치 실행 안내](../RESEARCH_INTEGRATION.md)를 따른다.
