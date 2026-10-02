# First Class 로보어드바이저

현재 구현은 FastAPI 상태 확인과 Streamlit 연결 화면입니다.
모델 로딩은 아직 구현되지 않았으며 `/health`는
`{"status":"ok","model_loaded":false}`를 반환합니다.
투자·뉴스·SHAP 결과는 제공하지 않습니다.

## Docker 실행

Docker Desktop의 Linux 컨테이너 엔진과 Docker Compose v2가 필요합니다.
저장소 루트에서 실행합니다. 기본 실행에 `.env`나 API 키는 필요하지 않습니다.

```bash
docker compose up --build -d --wait --wait-timeout 120
docker compose ps
```

| 접속 대상 | 기본 주소 |
| --- | --- |
| Streamlit 연결 화면 | http://localhost:8501 |
| API 상태 | http://localhost:8000/health |
| Swagger UI | http://localhost:8000/docs |

화면에서 API 연결 상태와 모델 미로드 안내를 확인할 수 있습니다.
`연결 상태 새로고침` 버튼으로 다시 조회합니다.
컨테이너 내부에서는 dashboard가 `http://api:8000`으로 HTTP 요청합니다.

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
