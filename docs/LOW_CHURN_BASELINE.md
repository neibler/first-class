# Low-churn execution baseline

새 설정: `config/low_churn_baseline.json`. 기존 multi_model.json과 저장된 모델은 변경하지 않습니다.
정책 버전: `executed_drift_capped_hysteresis_cash_v2`.
기존 `executed_drift_capped_hysteresis_v1` Bundle은 ETF-only turnover를 유지합니다. 새 PPO 실험은 새 설정으로 학습하세요.
기존 `monthly_risk_distance_filtered_v2` 모델은 기존 방식으로 평가/추론합니다.
새 baseline은 새로 학습해야 합니다. 기존 bundle 설정/스키마를 수동 수정하지 마세요.
코드 변경 전 체크포인트의 resume는 기존 source_hash 검증 때문에 거부될 수 있습니다.
이번 작업에서는 파라미터 탐색이나 대규모 학습을 실행하지 않습니다.

## 흐름

종가 t에서 raw target 계산 → 실제 비중과 마지막 체결 비중의 distance 및 월/뉴스/MDD 판단.
일반 거래는 기존처럼 t+1 종가에 실행합니다. 체결 전까지 기존 수량으로 수익률을 계산합니다.
체결 시점 실제 비중으로 hysteresis → 최대 보유 수 → cap을 다시 계산합니다.
미체결 잔량은 저장하지 않습니다. 다음 거래 조건이 충족돼야 다시 계산합니다.

- `portfolio_distance_threshold`: 0.10 (오늘 target 변화는 trigger가 아님)
- `turnover_penalty_lambda`: 0.003
- `max_rebalance_turnover`: 0.15
- `buy_threshold`: 0.05 (미보유 ETF, 경계 포함)
- `sell_threshold`: 0.03 (기존 ETF, 미만일 때 퇴출 후보)
- `max_holdings`: 7
- 뉴스 threshold 0.7, MDD 초과 기준 0.15 유지.

## 현금과 최대 보유 수

트리거 distance는 현재 실제 ETF 비중과 마지막 체결 ETF 비중 사이의 0.5*L1을 유지합니다.
거래 turnover는 `0.5 * (ETF 절대 비중 변화 합 + 현금 절대 비중 변화)`입니다.
현금은 `1 - ETF 비중 합`으로 계산하며 observation/action 차원에 추가하지 않습니다.
현금 100%에서 ETF 30%를 매수하면 turnover는 30%입니다. cap 15%에서는 ETF 총 편입 비중이 15%로 제한됩니다.
같은 정의를 desired, cap, executed, reward penalty, 평가 통계에 적용합니다. MDD 청산은 cap보다 우선합니다.
거래 후 수수료로 MDD가 발동하면 정상 거래와 긴급 청산의 turnover를 각각 계산해 합산합니다.
초기 진입도 cap 예외가 아닙니다. 월별/뉴스 조건이 없으면 초기 현금 상태를 유지합니다.
적격 ETF가 없으면 현금 목표가 가능합니다. 가장 큰 ETF를 강제 편입하지 않습니다.
기존 적격 보유 종목을 우선 유지합니다. 7종목이 모두 퇴출 기준 이상이면 신규 편입을 보류합니다.
퇴출 후보가 있어도 cap 때문에 부분 매도만 가능하면 해당 종목은 여전히 슬롯을 점유합니다.
보유 수가 7개를 초과할 신규 편입은 보류하고 매도된 자금은 현금으로 남깁니다.
추상적 체결은 매도 후 매수 순서로 해석하며 최종 양의 ETF 집합은 최대 7개입니다.

## 실제 체결과 비용

체결 목표 x와 거래 전 비중 w, 비용률 r에 대해 q=sum(abs((1-r*q)*x-w))를 풉니다.
q는 실제 매수/매도 금액 합 / 거래 직전 자산가치입니다. 비용은 자산가치*r*q입니다.
거래 후 실제 비중 x 기준 executed_turnover=0.5*sum(abs(x-w))를 보상에 사용합니다.
따라서 수수료 차감에 따른 현금/자산가치까지 일치합니다. 수수료율은 기존 5bps+2bps 그대로입니다.
reward는 기존 reward에서 lambda*executed_turnover를 추가 차감합니다.
return_mdd는 net_return - mdd_lambda*cumulative_mdd - lambda*executed_turnover입니다.
MDD 비상 청산은 cap 없이 전액 현금화합니다. 청산 turnover도 penalty에 포함합니다.
같은 날 일반 체결 비용 때문에 stop이 발생하면 일반 체결과 청산 turnover를 합산합니다.
따라서 emergency를 포함한 일일 최대 turnover는 15%를 넘을 수 있습니다.

## 상태와 서비스

PortfolioEnv.reset에서 last_executed_weights를 0으로 초기화합니다.
실제 체결 직후 post-trade 비중으로 갱신하고 NO TRADE 날에는 유지합니다.
가격 변동은 현재 비중에만 반영합니다. observation/action 크기는 변경하지 않습니다.
API는 주문을 실행하지 않습니다. held portfolio의 첫 요청은 last_executed_weights가 필요합니다.
호출자는 실제 체결을 확인한 뒤 `execution_confirmation`에 체결 ID·날짜·실제 비중을 전달합니다.
`state.last_executed_weights`만 전달한 요청은 입력으로 사용하지만 저장된 기준을 갱신하지 않습니다.
확인된 기준만 세션·에피소드별로 보존합니다. [세션 및 체결 확인 계약](INFERENCE_SESSIONS.md)을 따르세요.
부분 현금 상태를 지원합니다. 제안은 t 시점 예상값이며 실제 체결 시점 cap을 재검증해야 합니다.

## 기록

history: raw/filtered/executable target, pre/post trade, last executed 전후 비중,
desired/executed/normal executed turnover, transaction_cost, holdings_count,
entry_count, exit_count, rebalance_trigger, distance, 기존 비용 및 거래 이유.
평가: rebalance_count, total/average/max executed turnover, total_transaction_cost,
비용/초기자본, 비용/최종가치, 평균/최소/최대 보유 수, 편입/퇴출 수, 이유별 체결 횟수.
동시 이유는 각각 집계하므로 이유별 횟수 합이 거래 횟수보다 클 수 있습니다.
초기자본은 기존 환경의 1.0 정규화 단위입니다. 원화 금액은 초기 투자금액을 곱합니다.
안전장치 후 현금 유지 기간의 거래/편입/퇴출/비용은 0으로 기록합니다.
벤치마크의 기존 buy-and-hold/동일비중 정의는 유지합니다.


## PPO 실험 전 검증

```powershell
.\.venv312_e\Scripts\python.exe -m pytest tests/test_cash_turnover.py tests/test_low_churn_baseline.py -q
if ($LASTEXITCODE -ne 0) { throw "Turnover tests failed; do not start training" }
```

새 학습은 `config/low_churn_baseline.json`을 사용합니다. 기존 Bundle/체크포인트를 수정하거나 기존 학습을 새 정책으로 resume하지 않습니다.
새 정책 history의 `turnover_definition`은 `cash_inclusive_one_way_v2`입니다.
`executed_turnover`와 평가 `total_turnover`/`total_executed_turnover`는 현금 포함 one-way입니다.
기존 history의 `turnover`/`exit_turnover`는 비용 계산용 매수·매도 체결금액 비율을 유지합니다.
평가 `total_trade_notional_ratio`는 이 체결금액 비율의 합이며 one-way와 구분합니다.
비용은 실제 ETF 체결금액에 수수료+슬리피지율을 적용하고 현금 비중 변화에 별도 수수료를 부과하지 않습니다.
구버전 history/평가의 의미는 그대로 유지되므로 새 실험 표에는 같은 새 정책 결과만 비교하세요.
