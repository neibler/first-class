# 목표 비중 필터와 리밸런싱 v2

정책 버전: `monthly_risk_distance_filtered_v2`.
관측/action 차원, softmax 원본 계산, reward, 거래비용·슬리피지·포트폴리오 회계는 유지합니다.

## 설정

새 학습은 사용하는 config(예: `config/multi_model.json`)에서 변경합니다.

```json
{
  "min_weight_threshold": 0.03,
  "portfolio_weight_distance_threshold": 0.10,
  "risk_rebalance_threshold": 0.7,
  "max_drawdown": 0.15
}
```

- 최소 목표 비중은 0.01/0.03/0.05 등으로 비교할 수 있습니다. 0이면 필터를 끕니다.
- 전체 거리 기준은 0.05/0.10/0.15 등으로 바꿀 수 있습니다.
- 기존 `weight_drift_threshold` 개별 ETF 조건은 제거했고, 남아 있는 과거 번들의
  해당 키도 실행 조건에 사용하지 않습니다.
- 저장 모델의 추론·재평가는 해당 번들의 config를 읽습니다. 새 키가 없는 과거
  번들은 기본 0.03/0.10을 사용합니다. 기존 모델 파일은 덮어쓰지 않습니다.
- `rebalance_days`는 기존 observation의 phase 호환용이며 거래 주기를 결정하지 않습니다.

## 하루 처리 순서

1. t일 종가까지 확정된 시장 정보와 가격 변화가 반영된 실제 보유 비중을 관측합니다.
2. 모델 action을 softmax하여 raw 목표를 계산합니다. 앙상블은 raw 비중부터 평균냅니다.
3. raw 비중이 0.03 미만인 ETF를 0으로 제거하고 나머지 합을 1로 재정규화합니다.
   정확히 0.03은 유지합니다. 모두 제거되면 raw가 가장 큰 ETF 한 개를 100%로
   유지하며, 동률은 고정 자산 순서의 첫 번째를 선택합니다.
4. 당일 실제 비중과 필터 후 목표로 거리 `0.5 * sum(abs(target-current))`를 계산합니다.
5. 월 첫 거래일 / 당일 최대 뉴스 점수 >= 0.7 / 거리 >= 0.10을 확인합니다.
6. 누적 MDD > 0.15 또는 기존 안전장치 latch가 있으면 모두 무시하고 현금으로 전환합니다.
7. 정상 조건이 하나라도 맞으면 목표를 다음 거래일 종가(t+1)에 체결합니다.
   조건이 없으면 보유 수량 유지, 비용 0입니다.

정상 거래의 다음 종가 체결 시간축은 기존 정책 그대로입니다. 체결 전 t→t+1 수익은
기존 보유분으로 계산하고, 실제 거래량·비용은 t+1 가격으로 변한 비중을 기준으로
기존 회계 함수에서 계산합니다. 목표 비중 거리 계산에 미래 가격·뉴스를 사용하지 않습니다.
MDD가 체결 대기 중 악화하면 대기 매수를 취소하고 해당 종가에서 청산합니다.
이미 현금인 경우 반복 비용을 청구하지 않으며 같은 episode에서는 reset 전 재진입하지 않습니다.

거래가 없을 때 가격에 따라 실제 비중은 자연스럽게 변합니다. 현금/ETF 회계를 바꾸거나
현재 비중을 목표에 맞추는 숨은 거래는 하지 않습니다.

## 거리와 현금

ETF를 전부 보유한 포트폴리오끼리 비교하면 거리값은 one-way turnover와 같습니다.
요청한 공식은 ETF만 포함하므로 현금 100%에서 최초 진입할 때는 0.5입니다.
실제 최초 매수량은 1.0이고 비용은 실제 거래량에 부과합니다. 현금을 16번째 모델
자산으로 추가하지 않습니다. MDD 상태에서는 ETF 합 0/cash 1, 나머지 체결 목표는 ETF 합 1입니다.

## 로그·API

- `raw_target_weights`: 필터 전 모델 목표
- `filtered_target_weights`: 필터·재정규화 후 목표
- 환경 history의 `current_weights`: `decision_date` 종가의 실제 비중(거리 계산 기준)
- `portfolio_weight_distance`, `max_risk_score`
- `should_rebalance`: 목표 적용 요청 여부, `rebalanced`: 실제 거래/청산 여부
- `rebalance_reasons`: 모든 충족 사유. `monthly`, `risk_score`,
  `portfolio_weight_distance`, `mdd_stop`, `hold`.
- CSV의 `rebalance_reason`: 복수 원인을 `|`로 연결
- 기존 `turnover`, `cost`, `cost_amount`, `exit_turnover`, `exit_cost`는 그대로 유지

기존 자산별 `previous_weight`는 체결 직전, `current_weight`는 체결 완료 후 의미를
유지합니다. `decision_current_weight`, `raw_target_weight`, `filtered_target_weight`를
추가해 판단 시점과 체결 시점을 구별합니다. 일반 history의 `date`는 체결 종가입니다.

API는 실제 주문을 보내지 않습니다. `target_weights`는 조건과 안전장치 적용 후의
실행 제안(hold면 현재 비중), `model_target_weights`는 raw 목표의 호환용 별칭입니다.
현재 비중은 요청 state의 실제 보유 비중을 사용하므로 호출자는 당일 가격으로 평가한
state를 전달해야 합니다. state가 없으면 신규 현금 포트폴리오입니다.
API에는 미래 실제 체결 비용을 임의로 채워 넣지 않습니다.

## 벤치마크와 SHAP

모델 목표에만 필터를 적용합니다. 동일비중/매수 후 보유 벤치마크는 고정된 자산
구성을 유지하고 최초 매수를 반드시 실행합니다. 이 최초 진입의 예외 사유는 `forced`이며,
MDD가 우선합니다. buy-and-hold 최초 진입 후에는 가격 변화만 반영하며 추가 매수하지 않습니다.

SHAP은 필터 후 모델 목표를 설명합니다. 계산 알고리즘과 입력 feature 순서는
바꾸지 않고 설명 대상 함수에 동일한 필터를 적용합니다. 이전 SHAP 파일이나
평가 결과는 자동 갱신하지 않습니다. 새 정책 평가 및 SHAP 계산은 별도로 실행해야 합니다.

## 검증과 기존 결과

`tests/test_target_filtering.py`, `tests/test_rebalancing.py`에서 필터 경계,
모두 제거되는 경우, 거리 경계, 개별 5%p 조건 제거, 비용 0, 자연 비중 변화,
MDD 우선순위, 앙상블 처리 순서, SHAP 출력, 벤치마크 최초 진입을 확인합니다.

기존 학습 모델을 고정한 새 정책 평가도 가능하지만, 새 정책에 최적화하려면 재학습이
필요합니다. 기존 결과 보존을 위해 번들 복사본/새 run-id를 사용하세요.
새 평가 그룹에 정책 버전을 포함하므로 이전 정책 결과와 구분됩니다.
## 이번 수정 파일

- `backend/investment_ai/rebalancing.py`
- `backend/investment_ai/portfolio_env.py`
- `backend/investment_ai/infer.py`
- `backend/investment_ai/serving.py`
- `backend/investment_ai/api.py`
- `backend/investment_ai/backtest.py`
- `backend/investment_ai/records.py`
- `backend/performance/model_shap.py`
- `config/investment_ai.json`
- `config/multi_model.json`
- `config/multi_model_no_risk.json`
- `config/dummy_only.json`
- `config/hybrid_news.json`
- `config/real_market.json`
- `tests/test_target_filtering.py`
- `tests/test_rebalancing.py`
- `tests/test_model_shap.py`
- `docs/REBALANCING.md`
- `docs/RISK_INPUT_ABLATION.md`
