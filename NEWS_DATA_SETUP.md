# 뉴스 데이터 연결 및 학습 시작

학습은 아직 실행하지 않았습니다. 기본 설정은 **실제 시장 데이터 + 실제 뉴스**입니다.
feature/11의 종가·로그수익률·RSI·MACD를 연결했습니다. 매크로는 미제공으로 0개입니다.
기간: 2023-01-03~2025-12-31, 752거래일, 15 ETF, 11,280행. 뉴스 키가 완전히 일치합니다.
종가 수익률은 배당을 추가 반영하지 않았습니다.

## 파일 위치
- `runtime/data/news/raw/`: 원본 CSV 2개, 설명 DOCX/TXT, 원본 ZIP 보관
- `data/processed/news/chroma/`: 압축 해제한 검색 DB (190,033개 조각)
- `config/investment_ai.json`: 기본 실제 시장 + 실제 2023~2025 뉴스
- `config/dummy_only.json`: 기존 순수 더미 설정
- `config/real_market.json`: 실제 시장 데이터용 설정
- `runtime/preflight.json`: 최근 입력 및 환경 검증 결과

원본 데이터는 Git에서 제외됩니다. 뉴스의 has_news=0인 날도 기존 risk_score를 보존합니다.
뉴스 추가 열은 메타데이터로 보존하고 PPO에는 risk_score만 사용합니다.
관측 지연은 0이며 주문만 다음 거래일 종가에 체결합니다. 별도 config/hybrid_news.json에서만 더미 시장 + 실제 뉴스를 사용할 수 있습니다.

## 바로 실행 (VS Code, 프로젝트 루트 터미널)
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 preflight
powershell -ExecutionPolicy Bypass -File .\run.ps1 train
```
기본 train은 3개 보상 × 3개 검증구간 × 2개 시드 + 최종 2개 모델입니다.
모델당 5,000 step이며 총 20회/100,000 step입니다. 결과는 runtime/runs에 저장됩니다.

## 실제 시장 데이터 학습
`runtime/data/market.parquet`에 변환된 실제 전처리 파일이 준비되어 있습니다.
원본은 data/processed/etf_features.csv이며 data/raw/etf_prices.csv 및 팀 전처리 코드도 보관했습니다.
재변환: `powershell -ExecutionPolicy Bypass -File .\run.ps1 prepare_market`
현재 macro_columns=[]로 종가가 매크로로 오인되지 않도록 했습니다.
향후 매크로를 추가하면 해당 열 이름을 macro_columns에 지정하세요.
다른 데이터로 교체할 경우: 필수 열은
`date, ticker, log_return, rsi, macd`이며 매크로 열은 config의 macro_columns에 지정합니다.
전체 15 ETF의 일별 행이 필요하고, 매크로 값은 같은 날 모든 ETF에서 같아야 합니다.
뉴스와 같은 2023-01-03~2025-12-31 거래일을 사용하는 것이 권장됩니다.
시장이 더 짧은 기간이면 뉴스도 명시적으로 같은 기간으로 준비해야 하며 자동 삭제하지 않습니다.
CSV도 지원합니다. CSV를 사용하면 real_market.json의 data_path를 변경하세요.
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 preflight --config config/real_market.json
powershell -ExecutionPolicy Bypass -File .\run.ps1 train --config config/real_market.json
```
파일이 없거나 검증에 실패하면 실제 데이터 실행은 중단하며 더미로 대체하지 않습니다.

## 뉴스 검색 연결
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 news_search market --as-of 2025-04-10 --ticker QQQ
```
`backend.investment_ai.news_search.search_news()`를 다른 코드에서 호출할 수 있습니다.
제목 키워드·ETF·기준일(포함)로 읽기 전용 조회하며 제목, 날짜, 출처, URL, 본문을 반환합니다.
날짜 단위 검색이므로 장중 특정 시각 이전 공개 여부를 보증하지 않습니다.
이 기능은 의미 기반 벡터 검색이나 LLM 답변 생성이 아닙니다.

DB 컬렉션: financial_news, 임베딩: all-MiniLM-L6-v2, 차원: 384.
제공된 설명서의 완전한 리서치 에이전트는 PR #10 코드와 chromadb 1.5.9,
ANTHROPIC_API_KEY를 요구합니다. 현재 로컬 복사본에는 그 에이전트가 없으므로
자동 답변/API/UI까지 연결되었다고 볼 수 없습니다. PPO 학습에는 필요 없습니다.
원본 ZIP의 pickle 파일은 실행하거나 역직렬화하지 않았습니다.


최신 입력/보상/시간축 계약은 [CONTRACT_V3.md](CONTRACT_V3.md)를 따릅니다. 기존 v2 모델은 재학습이 필요합니다.
