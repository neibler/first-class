# 뉴스 AI (`backend/news_research`)

뉴스와 공시를 모아 15개 ETF의 **일별 뉴스 위험 점수(risk_score)**를 만들어 투자 AI(PPO)에 넘기고,
같은 자료로 **출처를 인용하는 리서치 에이전트**(`POST /research`)를 제공한다.

```
수집 (NYT · MarketWatch · SEC 8-K)          ← 이 PR
  ├─ 위험 태깅 (LLM) → risk_score → 투자 AI   ← 다음 PR
  └─ RAG (ChromaDB) → LangGraph 에이전트       ← 그다음 PR
```

## 수집기

모두 배치 명령이며 API 서버 시작 시 실행되지 않는다. 결과는 `data/` 아래에 저장되고 Git에 올리지 않는다.

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

## 명세서 대비 달라진 점

| 명세서·초기 계획 | 실제 구현 | 이유 |
|---|---|---|
| 뉴스: 네이버 금융, 한경 API | 미국 언론 NYT, MarketWatch | 15개 ETF가 모두 미국 상장 자산 |
| 공시: DART | SEC EDGAR 8-K | 보유 종목이 미국 기업 |
| 초기 수집 계획: GDELT | NYT Archive + Media Cloud | GDELT가 요청을 계속 차단(429)해 수집 불가 |

## 데이터 사용 규칙

- 미래 정보 차단: 게시 시각은 뉴욕 시간으로 저장한다. 점수 계산 시 16시 이후·휴장일 기사는 다음 거래일,
  날짜만 있는 기사(MarketWatch)도 다음 거래일에 반영한다.
- SEC는 접수 시각이 불분명하면 늦은 쪽 시각을 택한다 (일찍 반영되는 일이 없도록).
- 요청 빈도: SEC 초당 1회 이하, Media Cloud 분당 2회.
- 수집 데이터는 교육 목적으로만 쓰며 재배포하지 않는다. 공개 저장소이므로 `data/`, `.env`는 올리지 않는다.

## 테스트

API를 부르지 않고 가짜 응답으로 검사한다.

```bash
python -m pytest -q tests/test_nyt_archive.py tests/test_mediacloud_collector.py tests/test_sec_filings.py
```
