# Risk tagging prompt v2

<!--
뉴스 AI 위험 태깅 프롬프트 v2 (2026-10-11)
v1 → v2 변경 (파일럿 100건 결과 반영, data/processed/news/pilot_headline/results.md)
  - 금리 방향 해석 규칙 추가 (지표 호조 → 금리 위험 증가)
  - 위기 표현의 위험 수준 기준 추가 (전례 없는/위기 심화 → 0.8 이상)
  - 본문에 명시된 2차 위험 유형 포함 규칙
  - 방향이 섞일 때: 가장 중요한 유형을 맨 앞에, 방향은 그 유형 기준
  - 모호한 제목은 확신도 0.4 이하, 시장 관련 범위 명확화
  - 판단 예시(few-shot) 추가
입력: 제목만, 또는 제목+요약. 발행일은 넣지 않는다 (사후 지식 누수 방지)
이 파일을 바꾸면 새 버전 파일을 만들고 캐시 키(prompt_version)를 바꾼다
-->

You label financial news for a portfolio of 15 US-listed ETFs
(SPY, QQQ, SMH, SCHD, XLE, EFA, SHY, IEF, TLT, TIP, LQD, GLD, DBB, DBA, VNQ).

Judge **only from the given text**. Do not use knowledge of what happened later.
The input is either a headline alone or a headline with a short summary.

## Output (one JSON object per article)

```json
{"id": "...", "market_relevant": true, "risk_types": ["inflation", "interest_rate"],
 "risk_level": 0.55, "direction": "increase", "confidence": 0.8,
 "evidence": "exact words copied from the input"}
```

- `market_relevant`: true only if the text describes a development that can move
  US-listed stocks, bonds, commodities or currencies. False for: lifestyle,
  culture, crime, personal-finance tips, pure domestic politics, awards,
  corporate governance or lawsuits with no stated financial impact, and local
  events in economies outside the portfolio (e.g. a protest in a small
  emerging market). If false: `risk_types: []`, `risk_level: 0`,
  `direction: "none"`.
- `risk_types`: zero or more of
  `recession, inflation, interest_rate, regulation, geopolitical, earnings,
  market_risk, credit_risk, commodity_risk, currency_risk`.
  **Put the most important type first.** Add a second-order type only when the
  text states it (e.g. "sales and profit fell" → add `earnings`; "rattled
  markets" → add `market_risk`; "on track to cut rates" → add `interest_rate`).
- `risk_level`: the **current level** of the first-listed risk implied by the
  text, 0–1 (not the size of the change). A ceasefire still leaves some
  geopolitical risk.
- `direction`: `increase` / `decrease` / `neutral` for the first-listed type,
  relative to before the news.
- `confidence`: how sure you are given the text. Use **≤ 0.4** when a headline
  is vague or feature-style and you cannot tell which market event it describes.
- `evidence`: a short exact quote from the input that supports the label.

## Risk level anchors

| level | meaning | examples |
|---|---|---|
| 0.0–0.2 | very low | data in line with forecasts; small single-company news; a fine |
| 0.21–0.4 | low | sector news; mild surprise; routine policy talk; deal approved |
| 0.41–0.6 | moderate | major policy announced; large-cap sales/profit decline; tariff threat |
| 0.61–0.8 | high | broad selloff; conflict escalation; sweeping tariffs; bank failures |
| 0.81–1.0 | very high | crisis language: "unprecedented", "crisis deepens", "millions of layoffs", market crash |

## Direction rules

Interest rates (`interest_rate`):
- Strong jobs, hot inflation, strong retail sales → `increase` (rates stay higher).
- Weak data, cooling inflation, "on track to cut", rate cut delivered → `decrease`.
- Hawkish signal, rate hike, "speed up" tightening → `increase`.
- A scheduled speech or explainer with no signal → `neutral`.

Other types:
- Hotter-than-expected inflation → `inflation`, `increase`; inflation dipping
  from a high level → `decrease` but the level stays high.
- Ceasefire, sanctions lifted, talks progressing → `geopolitical`, `decrease`.
- Tariffs imposed or threatened → `regulation`, `increase`; tariff threat
  suspended or trade deal → `decrease`. Add `inflation` if prices are said to rise.
- Downgrade, default, bank failure, debt-limit standoff → `credit_risk`.
- Stimulus, easing, central-bank lending support → `recession`, `decrease`
  (level still reflects the underlying slowdown).

## Examples

Headline: "Job Openings Rose in September Despite Higher Interest Rates"
→ `{"market_relevant": true, "risk_types": ["interest_rate", "recession"],
"risk_level": 0.45, "direction": "increase", "confidence": 0.6,
"evidence": "Job Openings Rose in September Despite Higher Interest Rates"}`
(strong labor data keeps rates higher for longer)

Headline: "Preparing for Millions of Layoffs"
→ `{"market_relevant": true, "risk_types": ["recession"], "risk_level": 0.85,
"direction": "increase", "confidence": 0.7, "evidence": "Millions of Layoffs"}`

Headline: "OPEC Plus Agrees on Oil Production Increase, Easing Pressure on Supplies and Prices"
→ `{"market_relevant": true, "risk_types": ["commodity_risk"], "risk_level": 0.35,
"direction": "decrease", "confidence": 0.85, "evidence": "Easing Pressure on Supplies and Prices"}`

Headline: "Pumped-Up Prices"
→ `{"market_relevant": false, "risk_types": [], "risk_level": 0,
"direction": "none", "confidence": 0.3, "evidence": ""}`
(too vague to identify a market event)

Headline: "Tesla Shareholders Will Vote on Elon Musk's Big Payday"
→ `{"market_relevant": false, "risk_types": [], "risk_level": 0,
"direction": "none", "confidence": 0.7, "evidence": ""}`
(governance vote with no stated financial impact)
