# 뉴스 리서치 통합 실행

투자 앱의 `create_app()`·모델 생명주기·`/ready`·투자 API를 유지하고,
`backend/research_api.py`의 `POST /research` 하나를 등록한다.
Streamlit은 `frontend/api_client.py`로 HTTP만 호출한다. 별도 준비 조회 API는 사용하지 않는다.

## 로컬 실행과 계약

1. 프로젝트 가상환경에 `requirements-dev.txt`의 의존성을 준비한다.
2. 기존 `financial_news` DB와 all-MiniLM-L6-v2(384차원, cosine) ONNX 캐시를 서버에 준비한다.
   조회 시 임베딩 다운로드와 DB/컬렉션 생성은 하지 않는다. 사용자 DB를 재생성하지 않는다.
3. 서버의 프로세스 환경 또는 프로젝트 루트 `.env`에 `ANTHROPIC_API_KEY`를 설정한다.
   DB 설정은 `NEWS_CHROMA_PATH` 하나이며 미설정 시 `data/processed/news/chroma`이다.
   상대 경로는 저장소 루트 기준이다. API·build/search CLI·투자 키워드 검색에 동일하게 적용한다.
4. 사용 중인 서버·포트를 먼저 확인하고 API와 UI를 실행한다.

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# 별도 터미널
python -m streamlit run frontend/app.py --server.port 8501
```

화면에서 질문·기준일(기본 2025-04-10)을 입력하고 **분석 실행**을 누른다.
공유 DB의 2023~2025년 범위를 안내하지만 API 입력 범위로 제한하지 않는다.
요청은 `{"question":"최근 QQQ 위험?","as_of":"2025-04-10"}`이며 ETF 선택 필드는 없다.
질문은 공백 정리 후 1~2,000자, 기준일은 필수 날짜 문자열이며 추가 필드는 금지한다.
뉴스 내부 ETF 검색·위험 태그와 별도 투자 키워드 검색의 ticker는 유지한다.

응답은 `status: complete / insufficient_evidence`, 답변·출처·위험 태그·계획·완료 로그·면책을 포함한다.
근거 부족은 위험 없음이나 0점으로 바꾸지 않는다. 오류는 `detail.code`, `detail.message` 형식이다.
입력 오류 422, 준비 부족·처리 중(`research_busy`) 503, 외부/결과 오류 502,
시간 초과 504, 기타 내부 오류 500이다. 상세 코드는 [개발 문서](docs/development.md#리서치-api-이슈-12)를 따른다.

개인용 화면이므로 영구 이력 대신 **세션별 마지막 유효한 성공 한 개**만 임시 보관한다.
명시적 재제출 때 앞뒤 공백을 정리한 질문과 기준일이 같으면 재사용해 시간·비용 낭비를 줄인다.
다른 성공은 교체하고 오류·근거 부족은 성공 보관값을 덮어쓰지 않는다.
현재 요청 표시와 보관값은 분리하므로 B 실패에 A의 답변을 표시하지 않는다.
API 주소가 달라지면 보관값을 무효화한다. 서버 데이터·모델 버전 계약은 없으므로 실행 환경 고정을
전제하며 변경 시 세션을 초기화한다. 새로고침·연결 종료 뒤 영구 보존은 보장하지 않는다.

진행 중 버튼 강제 비활성화·프런트 잠금·POST 자동 재시도는 추가하지 않았다.
동기 요청 중 재클릭은 즉시 전달되지 않을 수 있다. 서버 프로세스 내 동시 요청은 503으로 거절한다.
LLM 호출 timeout 20초·SDK 재시도 0회, 단계 전후 예산 90초, HTTP 읽기 timeout 120초를 유지한다.
Self-Correction은 근거를 개선하는 에이전트 단계이며 통신 자동 재시도와 다르다.
90초 예산·클라이언트 timeout·페이지 이동·재클릭은 진행 중 서버 작업의 강제 취소를 보장하지 않는다.
실제 단계·퍼센트를 가장하지 않으며 로그는 완료 후 반환되는 기록이다.

## Docker

```powershell
docker compose up --build -d --wait --wait-timeout 120
docker compose ps
```

기본 Compose의 dashboard는 `http://api:8000`에 연결한다. 키·DB·캐시는 연결되지 않아
유효한 질문 제출 시 키 미설정 안내가 예상된다. 투자 모델도 없으면 준비 부족을 안내한다.
두 API Dockerfile에 투자·뉴스 모듈과 의존성이 포함되며 CPU 추론용 PyTorch를 사용한다.

전체 실행용 선택 구성은 다음과 같다. 이 구성의 실제 분석은 별도 검증 대상이다.

```powershell
$env:INVESTMENT_AI_RUNTIME=(Resolve-Path runtime).Path
$env:NEWS_CHROMA_HOST_PATH=(Resolve-Path data/processed/news/chroma).Path
docker compose -f docker-compose.yml -f docker-compose.investment.yml -f docker-compose.research.yml up --build
```

리서치 overlay는 서버에 키와 `/news/chroma` DB 경로를 전달한다. **임베딩 캐시는 별도로
API 사용자(appuser)의 캐시 경로에 미리 연결해야 한다.** 모델 다운로드를 요청 처리에 맡기지 않는다.
Chroma는 기존 DB를 열 때도 내부 쓰기를 할 수 있어 권한·버전 호환성은 복사한 DB로 사전 확인한다.
실제 유료 LLM·사용자 DB 전체 실행과 학교의 요청당 5초 성능은 아직 검증하지 않았다.
이번 통합의 실제 검사 결과는 `docs/progress.md`의 최신 기록을 따른다. 과거 통과 수를 재사용하지 않는다.
