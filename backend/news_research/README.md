# 뉴스 AI (`backend/news_research`)

뉴스와 공시를 모아 15개 ETF의 **일별 뉴스 위험 점수(risk_score)**를 만들어 투자 AI(PPO)에 넘기고,
같은 자료로 **출처를 인용하는 리서치 에이전트**(`POST /research`)를 제공한다.

```
수집 (NYT · MarketWatch · SEC 8-K)
  ├─ 위험 태깅 (LLM) → risk_score → 투자 AI
  └─ RAG (ChromaDB) → LangGraph 에이전트 → research() → POST /research
```

모든 명령은 배치 작업이며 API 서버 시작 시 실행되지 않는다. 결과는 `data/` 아래에 저장되고 Git에 올리지 않는다.

## 1. 수집기

| 출처 | 모듈 | 필요한 키 | 저장 내용 | 2019~2025 수집 결과 |
|---|---|---|---|---|
| NYT | `nyt_archive.py` (Archive API, 월 1회 요청) | `NYT_API_KEY` | 제목, 요약(300자 이내), 게시 시각, URL, NYT 분류 태그 | ETF 관련 19,881건 |
| MarketWatch | `mediacloud_collector.py` (Media Cloud 검색) | `MEDIACLOUD_API_KEY` | 제목, 게시 날짜, URL (본문·시각 미제공) | 40,827건 |
| SEC 8-K | `sec_filings.py` (EDGAR, 1.1초 간격) | `SEC_USER_AGENT` | 제목, 요약(300자 이내), 접수 시각, URL, 공시 본문 일부 | 미국 보유 종목 41개, 3,875건 |

```bash
cp .env.example .env   # 키 값 입력 (.env는 Git 제외)
python -m backend.news_research.nyt_archive --start 2025-01 --end 2025-12
python -m backend.news_research.mediacloud_collector --start 2019-01 --end 2025-12
python -m backend.news_research.sec_filings --start 2019-01-01 --end 2025-12-31
```

- 기사 ↔ ETF 연결: `config/etf_news_map.yaml`의 보유 종목 별칭, ETF 키워드, NYT 태그, 위험 유형별 노출도
- 출처별 가중치: `config/news_sources.yaml`
- 응답은 `data/raw/...` 아래에 캐시해 다시 실행해도 API를 다시 부르지 않고, 중단된 곳부터 이어서 받는다.

## 2. 위험 태깅 → risk_score

```bash
# ① 모델 입력 만들기 (URL·날짜는 모델에 넣지 않음)
python -m backend.news_research.risk_tagger prepare --articles "data/raw/news/nyt/articles/*.jsonl" \
    --since 2021-01-01 --until 2025-12-31 --prompt config/prompts/risk_tagging_v4.md --out RUN_DIR
# ② Claude Batch API로 태깅 (ANTHROPIC_API_KEY, 중단 후 이어서 실행 가능)
python -m backend.news_research.claude_runner --run RUN_DIR --model claude-sonnet-5-5 --tag sonnet
# ③ 결과 검사·병합
python -m backend.news_research.risk_tagger merge --run RUN_DIR --outputs outputs_sonnet.jsonl \
    --labels data/processed/news/labels/labels_v4_sonnet.jsonl --tagger claude-sonnet-5-5
# ④ 일별·ETF별 점수 (LLM 없이 규칙으로 계산)
python -m backend.news_research.risk_scoring --labels data/processed/news/labels/labels_v4_sonnet.jsonl \
    --articles "data/raw/news/nyt/articles/*.jsonl" --calendar data/processed/news/trading_days_2021_2025.csv \
    --out data/processed/news/news_risk_nyt_2021_2025.csv
```

- 기준표: `config/prompts/risk_tagging_v4.md` (팀 위험 유형 10종, 위험 수준 0~1, 방향, 확신도, 근거 인용).
  v1~v3는 검증 기록용으로 남겨 둔다.
- 거래일 달력(`--calendar`): `date` 열 하나인 CSV로, 점수를 낼 기간의 거래일이다. SPY 가격 데이터의 거래일을 쓰며,
  투자 AI 입력과 같은 거래일이어야 한다. 데모(2023~2025)용 파일도 2021년부터 계산한 뒤 잘라야
  2023년 초에 2022년 말 위험이 규칙대로 이어진다.
- 점수 계산 (`risk_scoring.py`):
  1. 기사 점수 = 위험 수준 × 확신도 × 출처 가중치
  2. ETF 연결 강도 = max(보유 종목·키워드 직접 연결, 위험 유형별 노출도). 두 번째 이후 위험 유형은 ×0.5, ETF가 헤지하는 위험은 0
  3. ETF·위험 유형·거래일별 상위 3건 평균
  4. 뉴스 없는 날은 위험 유형별로 전날 값 유지, 마지막 뉴스 후 20거래일이 지나면 만료
  5. ETF `risk_score` = 위험 유형 값 중 최댓값, `risk_type` = 그 유형 (최고값의 80% 이상인 유형까지 `;`로 연결)
- 출력 열 (팀 공통 규칙 4·6번): `date, ticker, risk_score, risk_type, has_news, days_since_news` + 근거(`evidence_url, evidence`)
  - 근거 열에는 기사 인용이 들어 있으므로 투자 AI에 넘길 때는 빼고, 어디에도 공개하지 않는다.

### 태깅 모델 검증

`label_compare.py`로 기준 라벨 200건과 비교한 뒤 통과한 모델만 전체 태깅에 쓴다.

```bash
python -m backend.news_research.label_compare --reference REFERENCE.jsonl --candidate LABELS.jsonl
```

| 모델 | 표본 | 관련 여부 κ | 첫 위험 유형 일치 | 위험 수준 Spearman | 판정 |
|---|---|---|---|---|---|
| Qwen2.5 7B (코랩, `ollama_runner.py`) | 기존 200건, 기준표 v2 | 0.68 | 59% | 0.44 | 미달 |
| Claude Haiku 5.5 | 새 표본 200건, v4 | 0.69 | 90% | 0.84 | 미달 (관련 여부) |
| **Claude Sonnet 5.5** | **새 표본 200건, v4** | **0.83** | **91%** | **0.84** | **통과** |

- 통과 기준: κ ≥ 0.8, 첫 유형 ≥ 85%, Spearman ≥ 0.7
- 새 표본 = 기준표를 고칠 때 쓰지 않은 NYT 100건 + MarketWatch 100건
- 기준 라벨은 기준표를 적용해 만든 참고 답이며 사람 정답이 아니다 (같은 라벨러가 다시 매긴 일치도 κ 0.88이 상한)
- NYT 2021~2025 14,892건 태깅 비용: Batch API로 $6.97

## 3. RAG와 리서치 에이전트

```bash
# ① 말뭉치: 정리·중복 제거·~1,000자 조각·ETF/날짜/위험 태그 메타데이터
python -m backend.news_research.rag_corpus
# ② 임베딩(all-MiniLM-L6-v2, 로컬) → ChromaDB (중단 후 이어서 실행 가능)
python -m backend.news_research.vector_store build
python -m backend.news_research.vector_store search "Nvidia China export ban" --etf SMH --until 2025-04-30
# ③ 에이전트 (ANTHROPIC_API_KEY)
python -m backend.news_research.research_agent "최근 QQQ에 위험한 뉴스가 있어?" --as-of 2025-04-10
```

- 말뭉치 출처: NYT, MarketWatch 제목, SEC 8-K, Yahoo Finance 기사 (Hugging Face `Brianferrell787/financial-news-multisource`의
  parquet, 승인 필요, `data/raw/news/yahoo/`에 둠). 2019~2025 약 19만 조각, ChromaDB 약 1.5GB
- 메타데이터: 원문 `url`(인용용), `title`, `source`, `date`, ETF별 `etf_SPY` … `etf_VNQ` 참/거짓, 위험 태깅 결과

LangGraph 흐름 (`research_agent.py`, 설정 `config/agent.yaml`):

```
Planner(하위 검색 ≤3) → Researcher(ChromaDB, 기준일 이전 뉴스만)
  → Verifier(LLM 관련성 0~1 채점) ─ 상위 3건 평균 < 0.7 → 검색어 수정·재검색 (최대 2회)
  → Analyst(근거만으로 답변, [n] 인용, Risk Tag)
```

- 한국어 질문을 받아 Planner가 영어 기사 제목 형태의 검색어로 바꾼다 (기사와 임베딩 모델이 영어).
- 기준일(`as_of`) 이전 기사만 검색하고, 프롬프트에도 기준일을 알려 이후 사건을 말하지 않게 한다.
- `sources`에는 실제 근거 목록에 있는 번호만 남기고, 근거 번호가 없는 Risk Tag는 버린다.

`research(question, as_of)`가 `/research` 응답 본문을 돌려준다:

```json
{
  "question": "...", "as_of": "2025-04-10",
  "answer": "… [1] … [2]",
  "risk_tags": [{"ticker": "QQQ", "risk_type": "regulation", "risk_score": 0.48, "risk_level": 0.6,
                 "direction": "increase", "confidence": 0.8, "date": "2025-04-10", "evidence": [1]}],
  "sources": [{"n": 1, "title": "...", "source": "nytimes", "date": "2025-04-03", "url": "https://..."}],
  "plan": [{"query": "...", "etf": "QQQ", "lookback_days": 30}],
  "log": ["[Planner] ...", "[Researcher] ...", "[Verifier] ...", "[Analyst] ..."],
  "disclaimer": "본 분석은 교육 목적으로 …"
}
```

- `risk_score` = 위험 수준 × 확신도. `log`는 대시보드에 보여 줄 에이전트 실행 로그다.
- 실행하는 서버에 ChromaDB 폴더와 `ANTHROPIC_API_KEY`가 있어야 한다. `/research` 엔드포인트와 Docker 연결은 별도 작업.

## 명세서 대비 달라진 점

| 명세서·초기 계획 | 실제 구현 | 이유 |
|---|---|---|
| 뉴스: 네이버 금융, 한경 API | 미국 언론 NYT, MarketWatch | 15개 ETF가 모두 미국 상장 자산 |
| 공시: DART | SEC EDGAR 8-K | 보유 종목이 미국 기업 |
| 초기 수집 계획: GDELT | NYT Archive + Media Cloud | GDELT가 요청을 계속 차단(429)해 수집 불가 |
| LLM: OpenAI API 또는 로컬 모델 | Claude Sonnet 5.5 (Batch API) | 로컬 Qwen2.5 7B가 검증 기준 미달 |
| 뉴스 없는 날 `risk_score = 0.0` (팀 규칙 초안) | 위험 유형별 전날 값 유지, 20거래일 후 만료 | 작은 기사가 큰 위험을 덮어쓰지 않고, 오래된 위험은 사라지게 |
| `risk_type` 리스트 | `;`로 연결한 문자열 | CSV 한 칸에 담기 위해 |
| 뉴스 → 관련 자산 1~3개 | 15개 ETF 모두에 연결 강도(0~1), ChromaDB에는 ETF별 참/거짓 | 금리 기사 하나가 TLT·IEF·VNQ 등에 동시에 영향 |
| `research(query)` | `research(question, as_of)` | 과거 시점 데모에서 이후 뉴스가 섞이지 않게 |
| Risk Tag `asset` | `ticker` | 팀 공통 규칙 1번 (종목 열 이름 통일) |

## 데이터 사용 규칙

- 미래 정보 차단: 게시 시각은 뉴욕 시간으로 저장한다. 뉴욕 16시 이후·휴장일 기사는 다음 거래일,
  날짜만 있는 기사(MarketWatch)도 다음 거래일에 반영한다. t일 점수는 t일 장 마감까지 나온 뉴스만 쓴다.
- 모델에는 기사 URL·날짜를 넣지 않는다 (사후 지식 개입을 줄이기 위해).
- SEC는 접수 시각이 불분명하면 늦은 쪽 시각을 택한다 (일찍 반영되는 일이 없도록).
- 요청 빈도: SEC 초당 1회 이하, Media Cloud 분당 2회.
- 수집 데이터는 교육 목적으로만 쓰며 재배포하지 않는다. 공개 저장소이므로 `data/`(ChromaDB 포함), `.env`는 올리지 않는다.

## 한계

- LLM은 학습 시점 이후를 알고 있어 과거 기사를 판단할 때 사후 지식이 섞일 수 있다 (URL·날짜를 빼서 줄였지만 없앨 수는 없음)
- ETF 보유 종목은 2026-10 기준 스냅샷이라 과거 구간에는 실제 구성과 다를 수 있다
- 위험 점수는 NYT만 반영했다 (MarketWatch 태깅은 비용 문제로 보류)
- 에이전트는 LLM을 여러 번 호출해 응답에 약 30초 걸린다 (명세서의 5초 기준과 차이, 화면에 로딩 표시 필요)
- Yahoo 기사는 2024년이 비어 있고 2025년 3~9월에 몰려 있어, 2024년 질문은 근거가 상대적으로 적다
- 관련성 기준 0.7은 가이드 예시 값이며 실험으로 정하지는 않았다

## 테스트

API를 부르지 않고 가짜 응답으로 검사한다.

```bash
python -m pytest -q tests
```
