## 학습된 모델로 바로 실행

팀 공유 모델과 입력 데이터가 포함되어 있습니다. 재학습 없이 실행하려면 [shared/README.md](shared/README.md)를 따라 주세요.

# First Class 로보어드바이저

현재 구현은 FastAPI 상태 확인과 Streamlit UI 미리보기입니다.
포트폴리오 현황·SHAP 해석은 **고정된 샘플 데이터**로 동작하며,
나머지 4개 분석 화면은 실제 분석 데이터와 API가 준비되지 않은 상태를 안내합니다.
모델 로딩은 아직 구현되지 않았으며 `/health`는
`{"status":"ok","model_loaded":false}`를 반환합니다.
실제 투자·뉴스·SHAP 결과를 제공하는 분석 API는 아직 없습니다.

## Docker 실행

Docker Desktop의 Linux 컨테이너 엔진과 Docker Compose v2가 필요합니다.
저장소 루트에서 실행합니다. 기본 실행에 `.env`나 API 키는 필요하지 않습니다.

```bash
docker compose up --build -d --wait --wait-timeout 120
docker compose ps
```

| 접속 대상 | 기본 주소 |
| --- | --- |
| Streamlit UI | http://localhost:8501 |
| API 상태 | http://localhost:8000/health |
| Swagger UI | http://localhost:8000/docs |

화면 하단의 API 연결 상태를 펼치면
실제 API 연결 상태와 모델 미로드 안내를 확인할 수 있습니다.
`연결 상태 새로고침` 버튼으로 다시 조회합니다.
컨테이너 내부에서는 dashboard가 `http://api:8000`으로 HTTP 요청합니다.

화면의 메뉴로 6개 페이지를 이동합니다.
포트폴리오의 분석 기간, SHAP의 자산·결정일을 선택할 수 있습니다.
`비행 경로 재생`은 브라우저에서만 실행되며 OS의 동작 감소 설정을 따릅니다.

## Python 직접 실행

저장소 루트에서 의존성을 설치한 뒤 두 터미널에서 실행합니다.
Python 3.12와 활성화된 가상환경을 권장합니다. 아래 설치에는 이미지 전처리용
Pillow와 개발 검사 도구가 포함됩니다. 화면 실행만 필요하면 `requirements.txt`를 사용합니다.

```powershell
python -m pip install -r requirements-dev.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
python -m streamlit run frontend/app.py --server.port 8501
```

기존 가상환경이 있다면 `python` 대신 `.venv/Scripts/python`을 사용합니다.

## 유지보수와 선택 검사

표시용 에셋이 포함되어 있으므로 실행 전에 이미지 전처리를 할 필요는 없습니다.
원본 PNG를 변경한 경우에만 개발 의존성을 설치하고 저장소 루트에서 재생성합니다.

```bash
python scripts/prepare_cockpit_assets.py
python scripts/prepare_cabin_assets.py
python -m black --check backend frontend tests scripts
python -m flake8 backend frontend tests scripts
python -m pytest -q
```

브라우저 검사는 선택 도구인 `playwright==1.63.0`과 Microsoft Edge가 필요합니다.
`python -m pip install playwright==1.63.0`으로 설치합니다. Edge가 없다면
`python -m playwright install msedge`로 설치할 수 있습니다.
API 8000과 UI 8501을 실행한 상태에서 다음 검사를 실행합니다.

```bash
python scripts/check_ui.py
python scripts/check_flight_replay.py --repeats 10
python scripts/check_cockpit_assets.py
python scripts/check_shap_cabin.py
python scripts/check_shared_cabin.py
```

샘플 로딩·빈 상태·오류의 재시도 검사까지 수행하려면 별도 터미널에서 개발 UI를 실행합니다.
다른 터미널에서 `python scripts/check_ui.py --dev http://localhost:8502`를 실행합니다.
개발 상태는 실제 API 응답이 아닌 UI 시연이며 일반 실행에서는 활성화되지 않습니다.

```powershell
$env:FIRST_CLASS_UI_DEV = "1"
python -m streamlit run frontend/app.py --server.port 8502
# 개발 서버 종료 후
Remove-Item Env:FIRST_CLASS_UI_DEV
```

검사 결과는 실행 시 생성되는 Git 제외 경로 `artifacts/ui/`에 저장됩니다.
해당 경로의 기존 파일이나 별도 디자인 문서는 앱 실행에 필요하지 않습니다.

## 환경 변수

| 변수 | 기본값 | 용도 |
| --- | --- | --- |
| `API_PORT` | `8000` | 호스트에서 접속하는 API 포트 |
| `DASHBOARD_PORT` | `8501` | 호스트에서 접속하는 화면 포트 |
| `API_BASE_URL` | Compose에서 `http://api:8000`으로 지정 | dashboard의 API 주소. Python을 직접 실행할 때 기본값은 `http://127.0.0.1:8000` |

호스트 포트를 변경하려면 `.env.example`을 `.env`로 복사하고 포트를 수정합니다.
포트 변경 후에도 컨테이너 내부 API 주소는 그대로입니다.
서비스는 호스트의 `127.0.0.1`에만 공개됩니다.
`.env`, API 키, 실제 데이터와 모델 파일은 Git이나 이미지에 포함하지 않습니다.

## 중지

```bash
docker compose down
```

미구현 학교 요구사항, CI 검사와 CD 결정 사항은
[개발 상태 문서](docs/development.md)에 기록합니다.

본 시스템은 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없습니다.
백테스팅 성과는 미래 수익을 보장하지 않습니다.
