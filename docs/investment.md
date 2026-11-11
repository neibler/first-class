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
컨테이너가 사용할 기존 `runs/<실행 ID>`를 명시한다. 아래는 `runs/latest.json`의
선택을 따르는 예시이며, 다른 모델을 쓰려면 그 디렉터리 이름으로 지정한다.
공유 데모 실행 ID는 `20261028_005508`이다.

Windows / Docker Desktop (PowerShell):

```powershell
$env:INVESTMENT_AI_RUNTIME=(Resolve-Path runtime).Path
$env:INVESTMENT_AI_RUN_ID=(Get-Content "$env:INVESTMENT_AI_RUNTIME/runs/latest.json" -Raw | ConvertFrom-Json).run_id
$env:INVESTMENT_AI_STATE_GID='10001'
if (-not (Test-Path "$env:INVESTMENT_AI_RUNTIME/runs/$env:INVESTMENT_AI_RUN_ID/summary.json")) {
    throw '선택한 모델 실행 디렉터리를 확인하세요.'
}
docker compose -f docker-compose.yml -f docker-compose.investment.yml up --build -d
```

Linux (sh):

```sh
export INVESTMENT_AI_RUNTIME="$(realpath runtime)"
export INVESTMENT_AI_RUN_ID="$(python3 -c 'import json, os; from pathlib import Path; print(json.loads((Path(os.environ["INVESTMENT_AI_RUNTIME"]) / "runs/latest.json").read_text())["run_id"])')"
export INVESTMENT_AI_STATE_GID="$(id -g)"
test -f "$INVESTMENT_AI_RUNTIME/runs/$INVESTMENT_AI_RUN_ID/summary.json" || exit 1
docker compose -f docker-compose.yml -f docker-compose.investment.yml up --build -d
```

`INVESTMENT_AI_RUN_ID`와 `INVESTMENT_AI_STATE_GID`가 없으면 Compose 설정 단계에서 중단한다.
Linux에서는 실제 로컬 API를 실행하는 사용자의 기본 GID를 지정한다. 여러 호스트 사용자가
공유한다면 모두 속한 그룹을 사용한다. Windows의 `10001`은 컨테이너 내부 그룹이며,
호스트 파일 접근은 Docker Desktop과 Windows ACL 설정을 따른다.

투자 overlay는 `investment-state-init`이 성공한 뒤 API를 시작한다.
초기화 서비스만 root로 실행하며 `service_state`의 기존 소유자 UID는 유지하고
그룹을 지정한 GID로 맞춘 뒤 소유자·그룹에 읽기·쓰기·디렉터리 접근 권한을 부여한다.
하위 디렉터리의 setgid와 API의 `umask 0002`로 Docker가 새로 만든 상태 파일에도
공통 그룹과 그룹 쓰기 권한을 유지한다. API는 UID 10001의 `appuser`로 실행하며
이 그룹을 추가로 사용한다. 모델·데이터 전체의 권한은 변경하지 않는다.
기존 초기화로 소유자가 UID 10001이 된 파일도 그룹 권한을 통해 호스트에서 갱신할 수 있다.
로컬 API는 일반적인 `umask 0022` 또는 `0002`를 사용한다. 더 엄격한 마스크로 만든
파일이 있으면 로컬 API를 중지하고 초기화 서비스를 다시 실행한 뒤 Docker로 전환한다.
로컬 API와 Docker API를 같은 상태 디렉터리에 동시에 실행하지 않는다.
기존 Safe-Guard 상태 파일의 내용과 경로를 유지하며 새 빈 저장소로 대체하지 않는다.
호스트 파일시스템에서 그룹·권한 설정이 거부되면 초기화와 API 시작이 실패한다.
해당 설정을 지원하는 저장 경로와 운영 권한을 준비해야 한다.

기본 Compose는 사용자 모델을 마운트하지 않는다. 리서치의 키·DB·임베딩 준비는
[리서치 실행 안내](../RESEARCH_INTEGRATION.md)를 따른다.
