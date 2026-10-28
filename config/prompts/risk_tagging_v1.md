# Risk tagging prompt v1

<!--
뉴스 AI 위험 태깅 프롬프트 (v1, 2026-10-11)
- 입력: 기사 제목 (+ 있으면 요약). 발행일은 넣지 않는다 (사후 지식 누수 방지)
- 출력: 기사마다 JSON 한 줄
- 이 파일을 바꾸면 버전을 올리고 캐시 키(prompt_version)도 바꾼다
-->

You label financial news for a portfolio of 15 US-listed ETFs
(SPY, QQQ, SMH, SCHD, XLE, EFA, SHY, IEF, TLT, TIP, LQD, GLD, DBB, DBA, VNQ).

Judge **only from the given text**. Do not use knowledge of what happened later.

## Output (one JSON object per article)

```json
{"id": "...", "market_relevant": true, "risk_types": ["inflation"],
 "risk_level": 0.55, "direction": "increase", "confidence": 0.8,
 "evidence": "exact words copied from the input"}
```

- `market_relevant`: false for lifestyle, culture, crime, pure domestic politics,
  awards, corporate PR with no market consequence. If false, use
  `risk_types: []`, `risk_level: 0`, `direction: "none"`.
- `risk_types`: zero or more of
  `recession, inflation, interest_rate, regulation, geopolitical, earnings,
  market_risk, credit_risk, commodity_risk, currency_risk`.
- `risk_level`: the **current level** of that risk implied by the text, 0–1
  (not the size of the change). A ceasefire still leaves some geopolitical risk.
- `direction`: `increase` / `decrease` / `neutral` relative to before the news.
- `confidence`: how sure you are given the text (headline-only text is often less).
- `evidence`: a short exact quote from the input that supports the label.

## Risk level anchors

| level | meaning | examples |
|---|---|---|
| 0.0–0.2 | very low | data in line with forecasts; small single-company news |
| 0.21–0.4 | low | sector news; mild surprise; routine policy talk |
| 0.41–0.6 | moderate | major policy announced; large-cap earnings miss; notable tariff threat |
| 0.61–0.8 | high | broad selloff; conflict escalation; sweeping tariffs take effect |
| 0.81–1.0 | very high | crisis-level shock (e.g. March 2020 crash) |

## Direction anchors

- Rate cut / easing → `interest_rate`, `decrease`.
- Rate hike or "hawkish" signal → `interest_rate`, `increase`.
- Hotter-than-expected inflation → `inflation`, `increase`.
- Ceasefire / sanctions lifted → `geopolitical`, `decrease`.
- Tariffs imposed or threatened → `regulation` (+ `commodity_risk` if goods prices).
- Downgrade / default / bankruptcy → `credit_risk`, `increase`.
- Earnings miss / guidance cut → `earnings`, `increase`.
