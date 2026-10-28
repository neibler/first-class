# 뉴스 리서치 통합 실행

feature/7의 뉴스 위험 태깅·점수 생성·RAG 에이전트를 투자 통합 브랜치에 병합했습니다.
기존 PPO와 CSV는 그대로 사용합니다. 이번 변경으로 PPO 재학습은 필요하지 않습니다.

## 로컬 실행

1. `.venv`를 사용한다면 해당 Python으로 `pip install -r requirements-dev.txt`를 실행합니다.
2. 팀에서 받은 `chroma_financial_news.zip`을 `data/processed/news/`에 풀어
   `data/processed/news/chroma/chroma.sqlite3`가 있도록 합니다.
   이 PC에는 이미 DB가 있습니다. DB 801MB 원본 ZIP은 저장소에 업로드하지 않습니다.
3. 프로젝트 루트 `.env`에 `ANTHROPIC_API_KEY=...`를 설정합니다.
   키는 Git에 올리지 마세요. 다른 DB 경로는 `NEWS_CHROMA_PATH=...`로 설정할 수 있습니다.
4. 기존 API를 종료하고 재시작합니다. UI도 새로고침합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 api
# 별도 터미널
powershell -ExecutionPolicy Bypass -File .\run.ps1 ui
```

대시보드의 에이전트 리서치에서 질문·ETF·기준일을 입력하고 리서치 시작을 누릅니다.
공유 DB는 과거 데이터이므로 데모 기준일은 2023~2025년을 사용하세요.
답변, 원문 링크, 위험 태그, 완료 후 검색/검증 활동 기록을 표시합니다.
리서치 결과가 학습용 CSV나 PPO 모델을 변경하지 않습니다.
질문과 검색된 근거는 Claude로 전송되며 사용자의 API 계정에 요금이 발생할 수 있습니다.

`GET /research/status`: 패키지·DB 파일·키 설정 여부 확인. 유효한 키인지 실제 요청으로 검사하지 않습니다.
`POST /research`: `{ "question": "최근 QQQ 위험 뉴스?", "as_of": "2025-04-10", "ticker": "QQQ" }`
날짜 필수, ticker 생략 가능. 잘못된 입력 422, 미설정 503, 처리 중 409, 실행 실패 502.
설정이 없어도 투자 API/대시보드는 실행됩니다. 실패 시 가짜 리포트를 대신 표시하지 않습니다.

임베딩 모델 all-MiniLM-L6-v2는 첫 의미 검색 시 Chroma가 내려받을 수 있어 첫 실행이 느릴 수 있습니다.
Claude 모델 ID는 팀의 config/agent.yaml을 유지했습니다. 계정에서 지원하는 ID인지 확인하세요.
실제 유료 LLM 요청은 자동 검증에서 실행하지 않습니다.

## Docker

모델 공유 파일을 먼저 `python scripts/install_demo.py`로 설치하고 .env에 키를 설정하세요.
Linux bind mount에서는 API 사용자(uid 10001)가 runtime/service_state 및 Chroma 디렉터리에
쓸 수 있도록 권한이 있어야 합니다. Chroma는 DB를 여는 과정에서도 쓰기가 필요합니다.

```powershell
$env:INVESTMENT_AI_RUNTIME=(Resolve-Path runtime).Path
$env:NEWS_CHROMA_HOST_PATH=(Resolve-Path data/processed/news/chroma).Path
docker compose -f docker-compose.yml -f docker-compose.investment.yml -f docker-compose.research.yml up --build
```

기본 Compose/CI는 모델·키·대용량 DB가 없는 상태도 정상적으로 안내하는지 검사합니다.
리서치 API는 기본적으로 로컬 바인딩을 사용하며, 인터넷에 공개하려면 별도 인증/사용량 제한이 필요합니다.

## 통합 검증 결과

- 전체 테스트 163개 통과 (투자 AI·뉴스 AI·API·프론트 포함)
- Black/flake8 및 pip check 통과
- 공유 투자 모델 20261028_005508 로딩·추론·151일 백테스트 조회 유지 확인
- 실제 Chroma financial_news 컬렉션 190,033개 확인
- 유료 Claude 호출 미실행: 로컬 ANTHROPIC_API_KEY 미설정
- Docker 실행 검증은 로컬 Docker 미설치로 미실행; GitHub CI에서 확인 필요
