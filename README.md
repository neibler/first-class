## 학습된 모델로 바로 실행

팀 공유 모델과 입력 데이터가 포함되어 있습니다. 재학습 없이 실행하려면 [shared/README.md](shared/README.md)를 따라 주세요.

# First Class 로보어드바이저

FastAPI 앱 생성 함수에서 투자 모델 생명주기를 관리하고 투자·리서치 API를 제공합니다.
포트폴리오는 투자 HTTP API를 사용하며, SHAP은 저장된 실제 모델 설명을 조회합니다.
리서치·ANOVA 화면은 HTTP API에 연결되어 있고 성과·리스크 화면은 준비 화면입니다.
`/health`는 서버 연결 및 투자 모델 로드 상태를, `/ready`는 투자 모델 준비 상태를 반환합니다.
모델이 없으면 포트폴리오는 준비 부족을 안내하며 샘플 성공으로 대체하지 않습니다.
`POST /research`는 기존 뉴스 에이전트와 연결되며 서버의 Claude 키, 기존 DB,
로컬 임베딩 모델이 필요합니다. 실제 Claude·DB 전체 실행과 5초 성능은 미검증입니다.
투자는 `/optimize`·`/backtest`를 사용합니다. 저장된 성과·SHAP 결과는 `/performance/*`에서
조회할 수 있으며 화면 연결 상태와 실행 방법은 [성능 분석 안내](backend/performance/README.md)를 따릅니다.

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

화면의 메뉴로 7개 페이지를 이동합니다.
포트폴리오의 분석 기간, SHAP의 자산·결정일을 선택할 수 있습니다.
`비행 경로 재생`은 브라우저에서만 실행되며 OS의 동작 감소 설정을 따릅니다.

`에이전트 리서치`에서 질문과 기준일(기본 `2025-04-10`)을 입력하고 **분석 실행**을 누릅니다.
현재 공유 DB는 2023~2025년 자료이며 범위 밖 기준일은 경고하지만 요청을 제한하지 않습니다.
기본 Compose는 키·DB·임베딩을 연결하지 않으므로 정상 질문에는 **서버 API 키 미설정** 안내가 표시됩니다.
빈 질문은 입력 오류 안내가 표시됩니다. 실제 분석 준비가 끝난 서버에 연결하면 답변·원문 출처·위험 태그와
완료 후 실행 요약이 표시됩니다. 실패를 샘플 성공 결과로 대체하지 않습니다.

## Python 직접 실행

### ANOVA 결과 연결

ANOVA 화면은 `GET /performance/statistics`로 저장된 검증 결과를 조회합니다.
서버에 `<결과 폴더>/statistics/anova.json`이 필요하며, 조회에는 모델·API 키가 필요하지 않습니다.
팀원이 생성한 기존 결과를 사용합니다. 결과를 만들거나 재학습하는 기능은 화면에 포함하지 않습니다.

Python 서버는 실행 전에 결과 폴더를 설정합니다:

```powershell
$env:RESULTS_DIR = (Resolve-Path runtime/results).Path
```

Docker는 기존 로컬 실행 설정의 `api.volumes`에 결과 폴더를 `/results`로 읽기 전용 연결합니다.
예: `./runtime/results:/results:ro`. 기존 설정과 결과 파일을 보존하고 개인 절대 경로를
공용 Compose에 넣지 않습니다. 컨테이너 내부 `RESULTS_DIR`는 `/results`입니다.

`/performance/health`의 `available.statistics=true`와 `/performance/statistics`의
HTTP 200을 확인한 뒤 `/anova` 화면을 엽니다. 파일이 없으면 503 안내가 표시됩니다.
투자 모델 미로드나 SHAP 미준비로 전체 `ready=false`여도 ANOVA 조회는 가능합니다.

One-way 2종·Two-way의 F값, p값, η²와 그룹 요약, 요인별 Tukey 결과를 표시합니다.
계산 불가는 `—`, 미제공 검증은 별도 안내하며 새로고침으로 최신 파일을 다시 조회합니다.
평가 기간은 통계 응답에 없으므로 표시하지 않습니다. 표본 출처와 가정 점검은 상세 영역에서 확인합니다.

### 실행 명령

저장소 루트에서 의존성을 설치한 뒤 두 터미널에서 실행합니다.
Python 3.12와 활성화된 가상환경을 권장합니다. 아래 설치에는 이미지 전처리용
Pillow와 개발 검사 도구가 포함됩니다. 화면 실행만 필요하면 `requirements.txt`를 사용합니다.

```powershell
python -m pip install -r requirements-dev.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

직접 Uvicorn으로 실행해도 `INVESTMENT_AI_HOME`이 없으면 저장소의 `runtime/`을
사용합니다. 공유 모델 설치·활성화, `--all-shared` SHAP 계산 및
`/performance/model-shap`은 같은 기본 경로를 사용합니다. 별도 배포 경로를
사용하면 서버 시작 전에 `INVESTMENT_AI_HOME`을 지정하세요. 명시한 절대·상대
경로는 유지되며, 상대 경로는 실행 작업 폴더 기준입니다. 다른 폴더에 모델을
설치했다면 그 경로를 지정하고 `--destination`도 동일하게 설정하세요.


```powershell
python -m streamlit run frontend/app.py --server.port 8501
```

기존 가상환경이 있다면 `python` 대신 `.venv/Scripts/python`을 사용합니다.

## 뉴스 리서치 API

저장소 루트에서 실행합니다. 이미 의존성이 설치됐다면 재설치하지 않습니다.
새 환경에서는 `backend/requirements.txt`와 `backend/news_research/requirements.txt`가 필요합니다.
루트 `.env` 또는 프로세스 환경 변수에 `ANTHROPIC_API_KEY`를 설정합니다.
키를 저장소·로그·요청 본문에 넣지 않습니다.

```powershell
# 선택: 기본 DB 경로. 프로젝트 루트를 기준으로 해석합니다.
$env:NEWS_CHROMA_PATH = "data/processed/news/chroma"
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs → `POST /research` → Try it out.

```json
{"question": "최근 QQQ 위험?", "as_of": "2025-04-10"}
```

- 질문은 앞뒤 공백 정리 후 1~2,000자, 기준일은 필수 `YYYY-MM-DD`입니다.
  현재 데모 DB 범위(2023~2025)를 API의 날짜 제한으로 고정하지 않습니다.
- 키가 없으면 `503`, `detail.code: research_key_missing`을 반환합니다.
  키·DB 없이도 `/health`, `/docs`, `/openapi.json`은 동작합니다.
- DB 디렉터리 바로 아래 `chroma.sqlite3`와 기존 인덱스 폴더가 있어야 합니다.
  `chroma/chroma`를 사용하지 않으며 API가 빈 DB·컬렉션을 만들지 않습니다.
- Chroma 기본 `all-MiniLM-L6-v2` ONNX 모델 캐시를 서버에 미리 준비해야 합니다.
  요청 중 자동 모델 다운로드는 하지 않습니다. DB·모델 파일은 재생성하지 않습니다.
- 응답은 `status`, `answer`, `sources`, `risk_tags`, `plan`, `log`,
  `question`, `as_of`, `disclaimer`를 포함합니다. `log`는 완료 후 실행 요약입니다.
- Docker 이미지는 투자·뉴스 의존성과 API 모듈을 포함합니다. 투자 추론은 CPU를 사용합니다.
  기본 Compose에는 사용자 모델·뉴스 DB·임베딩 캐시·키를 연결하지 않습니다.
  실제 실행용 데이터와 캐시 연결은 [리서치 실행 안내](RESEARCH_INTEGRATION.md)를 참고합니다.

Streamlit은 기존 `API_BASE_URL`로만 요청하며 모델이나 API 키를 다루지 않습니다.
폼 제출 시 마지막 성공 결과와 질문·기준일이 같으면 HTTP 호출 없이 재사용하고, 다르면 POST를 한 번 보냅니다.
질문은 앞뒤 공백만 제거해 정확히 비교합니다. 자동 재시도하지 않습니다. 결과·오류와 제출 당시 질문·기준일은
현재 브라우저 세션에 보존하므로 페이지 이동이나 화면 재실행만으로 분석을 다시 요청하지 않습니다.
마지막 성공 결과는 세션별 한 개뿐이며, 오류·근거 부족은 이 보관값을 덮어쓰지 않습니다.
A 성공 후 B 실패 시 화면에는 B의 실패를 표시하며, A를 다시 제출하면 보관한 A를 재사용합니다.
입력 수정 후 실행·오류 재시도는 **분석 실행**을 명시적으로 누릅니다. 처리 중에도 버튼은 비활성화하지 않지만,
Streamlit 실행 상황에 따라 재클릭이 즉시 전달되지 않을 수 있습니다. `research_busy`이면 잠시 후 다시 시도합니다.
새로고침·연결 종료·서버 재시작 이후 영구 보존은 보장하지 않습니다. API 주소 변경을 확인하면 재사용값을 비우며,
같은 주소의 데이터·모델 변경은 감지할 수 없으므로 실행 환경이 고정됐다는 전제입니다. 환경 변경 시 세션을 초기화합니다.
서버의 단계별 90초 예산을 고려해 리서치 응답 읽기 timeout은 120초로 두며, health의 3초와 분리합니다.
재클릭·클라이언트 timeout·페이지 이동은 서버 분석 취소나 재시작을 보장하지 않습니다. 실제 5초 성능은 미검증입니다.

오류 코드, 시간 제한, 비용 없는 검사와 키 준비 후 검증 순서는
[개발 문서의 리서치 API 항목](docs/development.md#리서치-api-이슈-12)에 있습니다.

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

ANOVA의 실제 저장 결과가 연결된 서버에서는 `python scripts/check_anova.py`로
데스크톱·모바일 차트, 표, 새로고침과 메뉴 이동을 검사할 수 있습니다.

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
`.env`, API 키, 개인 데이터·추가 학습 모델·실행 상태는 Git이나 이미지에 포함하지 않습니다.
팀이 명시적으로 배포하는 `shared/investment_demo.zip`은 Git 추적 예외이며 기본 이미지에는 포함하지 않습니다.
투자 실행과 Docker 권한 준비는 [투자 실행 안내](docs/investment.md)를 참고하세요.

## 중지

```bash
docker compose down
```

미구현 학교 요구사항, CI 검사와 CD 결정 사항은
[개발 상태 문서](docs/development.md)에 기록합니다.

본 시스템은 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없습니다.
백테스팅 성과는 미래 수익을 보장하지 않습니다.


## 뉴스 리서치

질문·기준일·ETF로 뉴스 근거를 검색하고 답변과 출처를 표시합니다.
[RESEARCH_INTEGRATION.md](RESEARCH_INTEGRATION.md)에서 DB·API 키 설정을 확인하세요.
기존 투자 모델 재학습은 필요하지 않습니다.


## 투자 강화학습 모델 4종 (학습은 수동 실행)

PPO / RecurrentPPO / SAC / TD3는 동일한 PortfolioEnv를 사용합니다.
기존 PPO 데모는 유지하며 새 실험은 모델·seed별 개별 bundle로 저장합니다.
설정은 `config/multi_model.json`, 서비스 선택은 `config/serving.json`입니다.

### 새 clone의 환경 및 입력 데이터 준비

아래는 clone한 저장소 루트에서 실행합니다. `python --version`으로 Python 3.12를 확인하고,
가상환경이 이미 준비되어 있다면 생성·설치 단계는 생략합니다.

```powershell
python --version
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r backend/investment_ai/requirements.txt
.\.venv312\Scripts\python.exe scripts/install_trained_models.py
.\.venv312\Scripts\python.exe local_run.py preflight --config config/multi_model.json
```

`install_trained_models.py`는 저장소의 `shared/trained_models.zip`과 manifest를 검사한 뒤
모델과 **준비된 시장·뉴스 Parquet**를 `runtime/`에 설치합니다. 학습이나 활성 모델 변경은 하지 않습니다.
현재 config가 가리키는 `runtime/data/prepared/59f95dac2a33/market.parquet` 및
`news.parquet`를 설치하므로 원본 뉴스 CSV나 유료 뉴스 API 없이 아래 Dry Run이 가능합니다.
기존 파일 내용이 다르면 덮어쓰지 않고 오류로 알립니다. `shared/investment_demo.zip`은
과거 데모이므로 이번 입력 설치를 대신하지 않습니다.
preflight에서 실제 입력 기간 2021-01-04~2025-12-31, 매크로 13개, observation 376을 확인하세요.

원본 데이터가 갱신되어 **입력을 재생성할 때만**, 별도 전달받은 뉴스 CSV를
`runtime/data/news/raw/news_risk_nyt_2021_2025.csv`에 준비한 뒤 실행합니다.
시장 원본 `data_process/processed/features_unscaled.parquet`도 필요합니다.
공유 모델 설치는 이 원본 CSV를 설치하지 않습니다. 원본이 없다면 위의 준비된 Parquet를 사용하세요.

```powershell
$env:INVESTMENT_AI_HOME = (Resolve-Path runtime).Path
.\.venv312\Scripts\python.exe -m backend.investment_ai.prepare_training_inputs
.\.venv312\Scripts\python.exe local_run.py preflight --config config/multi_model.json
```

재생성 명령은 `config/investment_ai.json`, `config/multi_model.json`의 입력 경로와 macro 목록을
갱신합니다. 검증 실패 시 Parquet·provenance·config는 변경하지 않습니다.
기존 모델은 새 입력으로 재학습되지 않으며 저장된 데이터 hash와 다르면 재평가를 거부할 수 있습니다.

### 학습 없이 계획 확인

입력 설치와 preflight가 성공한 뒤 실행하세요:

```powershell
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model ppo --reward return_mdd --timesteps 100000 --seed 42 --dry-run
```

나중에 직접 학습할 때는 위 명령에서 `--dry-run`을 제거합니다.
`--model`을 `recurrent_ppo`, `sac`, `td3`로 바꾸면 같은 환경의 다른 모델을 학습합니다.
seed 반복·평가·비교·서비스 변경과 향후 Walk-Forward의 조건은
[MULTI_MODEL_TRAINING.md](MULTI_MODEL_TRAINING.md)를 참고하세요.
현재 중간 점검은 시장·뉴스 공통 기간 2021-01-04~2025-12-31의 앞 80% 학습, 뒤 20% 평가입니다.
Train 2021-01-04~2024-12-30, Test 2024-12-31~2025-12-31로 약 4년+1년 단일 구간입니다.
이 설정은 유지하며 Walk-Forward 반복 검증은 추가 데이터 확보 후 수행합니다.
현재 날짜 범위로 달력 기준 4년/1년 Walk-Forward는 완전한 첫 윈도우를 만들 수 없습니다.

## 최신 학습 완료 모델 공유

4개 모델과 필수 데이터 설치·실행은 [공유 모델 안내](shared/TRAINED_MODELS.md)를 따릅니다.
기존 install_demo.py는 과거 PPO 데모입니다. 새 공유본은 scripts/install_trained_models.py로 설치합니다.


## 새 clone의 학습 입력 준비

학습 환경 의존성을 설치한 후 아래 순서로 실행합니다. 학습 입력은 이전 E 모델 설치 ZIP과 분리되어 있습니다.

```powershell
python scripts/install_training_inputs.py
python local_run.py preflight --config config/multi_model.json
python local_run.py train --config config/multi_model.json --model ppo --seed 22 --dry-run
```

기본 Docker 이미지도 같은 검증된 training_inputs ZIP을 설치하여 Correlation에 사용합니다. 사용자 지정 runtime을 마운트하면 `python scripts/install_training_inputs.py --destination <해당 경로>`로 먼저 설치하세요. 모델은 별도로 설치/선택해야 합니다. 거래 정책 버전이 없는 이전 공유 모델은 새 추론 코드에서 거부됩니다. 해당 모델의 새 정책 학습본으로 공유 패키지를 갱신해야 합니다.
