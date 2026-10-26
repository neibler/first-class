# Risk tagging prompt v3

<!--
뉴스 AI 위험 태깅 프롬프트 v3 (2026-10-18)
v2 → v3 변경 (Claude Haiku 5.5 200건 검증 불일치 분석,
data/processed/news/validation/report_claude-haiku-5-5.md)
  - 시장 관련 범위를 포트폴리오 기준으로 명시 (미국·선진국·원자재·달러, 중국은 포함)
    → 인도·터키·아프리카 등 범위 밖 지역 사건, 암호화폐만의 사건, 인사·투자 칼럼 제외
  - 위험 유형 대응표: 관세·무역분쟁·수출통제는 regulation이 첫 유형 (geopolitical 아님),
    부채한도·신용등급은 credit_risk, market_risk는 시장 전체 움직임일 때만
  - 첫 유형 = 제목이 주로 다루는 것, 결과로 언급된 위험은 뒤에
같은 검증 표본으로 고쳤으므로, 새 표본 검증 전까지는 과대평가 가능성 있음
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

## Step 1 - Is it market relevant?

The portfolio holds: US stocks and REITs, US Treasury and corporate bonds,
stocks of developed markets outside the US (Europe, UK, Switzerland, Japan,
Australia, Hong Kong, Singapore - "EFA"), gold, industrial metals, farm
commodities, and therefore the US dollar.

`market_relevant: true` only if the text describes a development that can move
those assets. That includes:
- US or developed-market economy, central banks, fiscal policy, regulation
- companies listed in those markets, their results, deals and fines
- oil, gas, metals and farm supply or prices; wars and conflicts that affect them
- China's economy and trade policy (it drives commodity demand and US trade)

`market_relevant: false` for:
- events confined to economies outside that list (e.g. India, Turkey, Africa,
  Latin America, Russia's domestic economy) unless the text states an effect on
  global markets, commodities or US/developed-market companies
- crypto-only events (an exchange collapse, a token price) with no stated effect
  on banks, regulation or listed companies
- lifestyle, culture, crime, social-media content rules, elections and politics
  without a stated economic policy, executive appointments and pay, awards
- personal-finance tips and investing advice columns, even when they mention
  the Fed or the market
- corporate governance or lawsuits with no stated financial impact

If false: `risk_types: []`, `risk_level: 0`, `direction: "none"`.

## Step 2 - Risk types

Zero or more of `recession, inflation, interest_rate, regulation, geopolitical,
earnings, market_risk, credit_risk, commodity_risk, currency_risk`.

**The first type is what the headline is mainly about.** Risks the text names as
causes or consequences come after it (e.g. "tariffs push prices up" →
`regulation, inflation`; "prices jump because of tariffs" → `inflation,
regulation`). Add a second-order type only when the text states it.

Use these types for these subjects:

| subject | type |
|---|---|
| tariffs, trade disputes, trade deals, export controls, industrial policy, antitrust, fines, new rules for an industry | `regulation` (also between the US and China - not `geopolitical`) |
| war, military conflict, terrorism, territorial disputes, diplomatic crises, sanctions tied to them | `geopolitical` |
| central-bank rate decisions and signals, bond yields, Treasury issuance | `interest_rate` |
| consumer and producer prices, wages pushing prices | `inflation` |
| growth, jobs, layoffs, consumer spending, stimulus | `recession` |
| a company's or sector's sales, profits, guidance | `earnings` |
| bank failures, defaults, downgrades, debt-limit standoffs, lending stress | `credit_risk` |
| oil, gas, metals, farm supply and prices | `commodity_risk` |
| exchange rates, the dollar | `currency_risk` |
| broad market selloffs, volatility, investor panic or euphoria | `market_risk` (only for market-wide moves, not as a default) |

## Step 3 - Level, direction, confidence, evidence

- `risk_level`: the **current level** of the first-listed risk implied by the
  text, 0–1 (not the size of the change). A ceasefire still leaves some
  geopolitical risk.
- `direction`: `increase` / `decrease` / `neutral` for the first-listed type,
  relative to before the news.
- `confidence`: how sure you are given the text. Use **≤ 0.4** when a headline
  is vague or feature-style and you cannot tell which market event it describes.
- `evidence`: a short exact quote from the input that supports the label.

### Risk level anchors

| level | meaning | examples |
|---|---|---|
| 0.0–0.2 | very low | data in line with forecasts; small single-company news; a fine |
| 0.21–0.4 | low | sector news; mild surprise; routine policy talk; deal approved |
| 0.41–0.6 | moderate | major policy announced; large-cap sales/profit decline; tariff threat |
| 0.61–0.8 | high | broad selloff; conflict escalation; sweeping tariffs; bank failures |
| 0.81–1.0 | very high | crisis language: "unprecedented", "crisis deepens", "millions of layoffs", market crash |

### Direction rules

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
- Strong sales or profits → `earnings`, `decrease`; weak ones → `increase`.

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
