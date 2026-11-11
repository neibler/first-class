# Portfolio X-Ray 설명 레이어

SHAP 계산·모델·학습 입력은 변경하지 않는다. API 응답에는 검증된 원본 시장 문맥만 추가한다. frontend/shap_explanations.py에서 선택 ETF 기준으로 합산한다.

| 원본 규칙 | 사용자 요인 | 의미 분류 |
|---|---|---|
| `log_return_lag_*_<ETF>` | 자산명(ETF)의 최근 움직임 | 선택 ETF: 자산 자체 가격/추세, 그 외: 다른 자산 움직임 |
| `rsi_div_100, rsi, macd, macd_signal, macd_hist, macd_histogram + _<ETF>` | 자산명(ETF)의 최근 움직임 (가격 lag와 통합) | 위와 동일 |
| `macro_effr` | 미국 기준금리 | 금리 환경 |
| `macro_ust_2y` | 미국 단기금리 | 금리 환경 |
| `macro_ust_10y` | 미국 장기금리 | 금리 환경 |
| `macro_ust_30y` | 미국 장기금리 | 금리 환경 |
| `macro_tips_10y` | 미국 실질금리 | 금리 환경 |
| `macro_yield_spread` | 장기·단기 금리 차이 | 금리 환경 |
| `macro_core_pce_yoy` | 미국 물가 흐름 | 경기/물가 |
| `macro_real_pce_mom` | 미국 소비 흐름 | 경기/물가 |
| `macro_unemp` | 미국 고용 상황 | 경기/물가 |
| `macro_ism_pmi` | 미국 제조업 경기 | 경기/물가 |
| `macro_vix` | 시장 변동성 | 시장 위험 |
| `macro_usd_ret_20d` | 달러 가치의 최근 움직임 | 환율/원자재 |
| `macro_wti` | 국제 유가 | 환율/원자재 |
| `risk_score_<ETF>` | 최근 뉴스 위험 (모든 자산 통합) | 뉴스 위험 |
| `current_weight_<ETF>` | 현재 자산 배분 상태 (모든 자산 합산) | 현재 포트폴리오 |
| `drawdown, mdd` | 현재 손실과 누적 낙폭 | 운용 상태 |
| `rebalance_phase` | 비중 조정 주기 | 운용 상태 |
| 미등록 입력 | 기타 모델 입력 정보 | 기타 정보 |

동일 ETF의 수익률 lag·RSI·MACD 계열을 단일 `asset_<ETF>` 요인으로 합친다.
GLD는 금(GLD) 관련 신호로 표시하며, 동일 자산이 증가·감소 목록에 중복되지 않는다.
10년물·30년물 금리는 미국 장기금리로 합친다. 자산 이름 15개는 ASSET_NAMES에 명시한다.
현재 관측에는 MACD signal/histogram과 risk_type이 없다. MACD 별칭은 향후 호환용이며
risk_type의 기여도를 추정해 만들지 않는다.

- signed sum: 원본 SHAP 합. 양수/음수로 기여 방향을 결정한다. 절댓값 1e-12 이하는 중립.
- absolute sum: 원본 SHAP 절댓값 합. 중요도 순위와 막대 크기에 사용한다.
- 방향별 absolute sum 내림차순 Top 3. 동률은 내부 key 순으로 안정 정렬한다.
- 양쪽 막대 분모는 전체 요인 중 최대 absolute sum으로 동일하다.
- 상대 크기 75% 이상 매우 큰 영향, 40% 이상 큰 영향, 15% 이상 보통 영향, 나머지 작은 영향.
- 상대 영향은 해당 결정 내 표시이며 통계적 유의성·투자 확신을 뜻하지 않는다.
- 증감 혼합과 완전 상쇄는 상세 분석에 보존한다. 원본 합과 절댓값 합을 모두 보존한다.

상세 분석 보기 버튼: 모든 요인·분류별 수치와 raw 피처·입력 값·SHAP·signed/absolute contribution·Force Plot.
기본 화면에는 표·긴 막대·반복 설명·상쇄 문구를 표시하지 않는다.
기여 방향은 모델 배경 기준 대비이며 경제적 인과나 전일 대비 실제 비중 변화가 아니다.
SHAP 부호로 가격이 상승·하락했다는 문장을 생성하지 않는다.

SHAP: “AI는 왜 이번에 이 비중을 선택했는가?”
Rolling Correlation: “이 자산은 무엇과 같이 움직이는가?”
현재 로컬에 상관관계 결과 연결이 없어 별도 영역에서 그 상태를 안내한다. 가짜 상관계수는 제공하지 않는다.

영향도는 ▰▰▰▰ 매우 큼 / ▰▰▰▱ 큼 / ▰▰▱▱ 보통 / ▰▱▱▱ 작음으로 표시한다.
상관관계 설명은 상세 분석 안에서 SHAP과 구분하며 기본 요약에는 추가 영역을 만들지 않는다.


## 실제 시장 상태와 모델 기여도

`frontend/shap_explanations.py::group_shap_features`는 기존 집계 결과를 유지하면서
`frontend/shap_state.py::summarize_feature_state`와 `generate_shap_explanation`을 연결한다.
각 요인에 `group_name`, `asset`, `factor`, `actual_state_summary`, `shap_direction`,
`shap_importance`, `explanation`, `raw_features`를 제공한다. 외부 LLM 호출은 없다.

기본 화면은 증가/감소 Top 3 각각에 시장 상태를 덧붙인다. 요인 제목의 도움말과
상세 분석의 설명 열에는 시장 상태 + 선택 ETF에 대한 모델 기여 방향을 함께 표시한다.
기존 raw feature/model input/SHAP 수치 표와 Force Plot은 그대로 유지한다.

### 데이터 확인

기존 JSON의 `feature_values_raw`는 이름과 달리 모델에 투입한 관측값이다.
수익률·MACD·거시지표는 표준화 및 clipping되어 있어 이를 원본 값으로 해석하지 않는다.
`backend/performance/shap_context.py::attach_market_context`가 번들의 config,
run_spec, 설치 입력 경로를 사용해 원본 패널을 읽고, SHAP의 `data_hash`와
feature 순서가 일치할 때만 결정일의 `market_context`를 응답에 추가한다.
저장된 SHAP 파일이나 모델은 수정하지 않는다. 요청 중 캐시 원본도 변경하지 않는다.

원본 파일 부재·해시 불일치 시 `market_context_status=unavailable`로 반환하고,
정규화하지 않은 RSI·뉴스 점수·포트폴리오 상태만 사용한다. 설명 불가 요인은
데이터 부족으로 표시하며, SHAP 부호를 시장의 상승/하락으로 해석하지 않는다.

### 상태 설명 규칙

- 가격: 결정일을 포함한 연속 lag 0~4의 원본 로그수익률에
  `expm1(sum(log_returns))*100` 적용. 부족하면 실제 확보한 연속 거래일 수를 표시한다.
- RSI: 30 미만 과매도에 가까움, 30~45 미만 비교적 낮음,
  45~55 중립, 55 초과~70 비교적 높음, 70 초과 과매수에 가까움.
  화면 수치는 소수 첫째 자리로 표시하며, 미래 가격을 예측하지 않는다.
- MACD: 검증된 signal이 있으면 관계를 설명한다. 현재 모델은 MACD만 있으므로
  0 대비 부호만 설명한다. 같은 ETF에서 수익률·RSI·MACD 최대 3가지 사실을 사용한다.
- 금리·VIX·기타 macro: 원본 현재 입력값과 5거래일 전 값을 비교한다.
  5거래일을 확보하지 못하면 실제 비교 간격을 표시한다. 10년물과 30년물의
  방향이 다르면 각각 설명한다. 과거 값이 없으면 현재 값만 표시하며 변화는 미확인으로 안내한다.
  임의의 높음/낮음 기준이나 원본에 없는 단위는 붙이지 않는다.
- 뉴스: 그룹 내 절대 SHAP 기여가 큰 실제 risk_score 최대 3개를 ETF명과 함께 표시한다.
  risk_type별 기여도나 새 위험 등급을 만들지 않는다. risk_score 제외 모델에는 추가하지 않는다.
- 현재 비중·낙폭: 관측 당시의 원래 상태 값만 표시한다.
- 기여 방향: 기존 signed sum, 중요도: 기존 absolute sum. 시장 상태와 관계없이
  선택된 출력 ETF의 비중을 높이는/낮추는 방향이라고 설명한다.

현재 미제공 정보는 MACD signal/histogram, risk_type의 SHAP, 미등록 피처의 의미,
리밸런싱 phase의 사용자용 상태 해석이다. 원본 문맥을 확인할 수 없는 수익률·MACD·macro도
구체적 설명을 만들지 않는다. 실제 상태는 SHAP 결정일 기준이며 오늘의 실시간 시황이 아니다.
