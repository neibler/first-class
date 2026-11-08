# 리밸런싱 정책: 매일 계산, 조건부 거래

`monthly_risk_drift_v1` 정책은 학습 환경, 백테스트, API/CLI 추론에 공통 적용됩니다.
기존 모델의 observation/action 차원, softmax, reward, 비용 및 가치 계산은 유지합니다.

## 판단 순서

1. 누적 MDD가 `max_drawdown`(기본 0.15)을 **초과**했거나 이미 안전장치가
   발동했다면 ETF 0%, cash 100%. 같은 episode에서는 reset 전 재진입하지 않습니다.
2. 나머지 날은 모델 목표 비중을 매일 계산합니다. 아래 조건은 OR이며
   동시에 만족한 모든 사유를 기록합니다.
   - `monthly`: 직전 거래일과 당일의 월이 달라지는 월 첫 거래일.
   - `risk_score`: 당일 15개 ETF 중 하나라도 위험 점수가 0.7 이상.
   - `weight_drift`: 당일 현재 비중과 모델 목표의 차이가 하나라도 0.05 이상.
3. 조건이 없으면 `hold`. 새 거래 없이 기존 보유 수량을 유지합니다.

공통 함수는 `backend/investment_ai/rebalancing.py`의
`get_rebalance_reasons()`와 `is_monthly_rebalance_day()`입니다.
첫 입력 행에 직전 거래일이 없다면 달력상 1일만 월초로 확정합니다.
중순에서 시작한 데이터에 임의로 월초 신호를 만들지 않습니다.
15개 ETF를 대상으로 현금에서 시작할 때는 적어도 하나의 목표가 5%p보다
크므로 비중 차이 조건으로 최초 매수가 가능합니다.

## 관측과 체결 시점

- t일 종가: t일까지의 정보로 모델 목표와 조건을 판단합니다.
- 정상 리밸런싱: 기존 정책대로 t+1일 종가에서 체결합니다.
- 체결 전 수익률은 기존 보유분에 귀속됩니다. 새 비중은 t+1→t+2부터 적용됩니다.
- t+1 뉴스로 t일 결정을 바꾸지 않습니다.
- MDD는 종가 기준으로 확인하고, 해당 종가에서 즉시 청산합니다.
  체결 대기 주문보다 우선하며 반복 청산 비용을 부과하지 않습니다.
- 무거래일에는 비용이 0이지만, 가격 변동으로 실제 보유 비중은 자연스럽게
  변합니다. 매일 비중을 강제로 고정하는 거래는 하지 않습니다.

`rebalance_days`는 기존 모델의 `rebalance_phase` 입력을 유지하기 위해 남아
있으며 **거래 주기를 결정하지 않습니다**. 기존 입력 순서와 값의 계산 방식을
불필요하게 바꾸지 않았습니다. buy-and-hold 벤치마크의 `execute=False`는 일반
리밸런싱을 막지만 MDD 안전장치는 계속 적용됩니다.

## 설정

새 학습은 사용하는 config(예: `config/multi_model.json`)에서 변경합니다.

```json
{
  "risk_rebalance_threshold": 0.7,
  "weight_drift_threshold": 0.05,
  "max_drawdown": 0.15
}
```

추론·재평가는 모델 번들의 `config.json`을 사용합니다. 새 두 항목이 없는 기존
번들은 `rebalancing.py`의 상수 0.7/0.05를 기본값으로 사용합니다.
`max_drawdown`은 기존 설정을 재사용합니다. 최상위 학습 config를 바꿔도 이미
저장된 모델의 config는 자동으로 바뀌지 않습니다.
경계값에는 1e-12 수준의 부동소수점 오차만 보정합니다. 따라서
`0.55 - 0.50`은 5%p로 인정하고 `1 - 0.85`는 15% 초과로 보지 않습니다.

## 결과 필드와 호환성

API:
- `model_target_weights`: 매일 계산한 모델 원래 목표. 안전장치 상태에서는 0.
- `target_weights`: 거래 조건과 안전장치를 적용한 최종 실행 제안.
  기존 API 의미를 유지하므로 hold에서는 현재 비중입니다.
- `current_weights`: 전달된 실제 현재 보유 비중.
- `rebalance_reasons`: `monthly`, `risk_score`, `weight_drift`, `mdd_stop`, `hold` 목록.
- `should_rebalance`: 조건이 충족되어 다음 종가에 목표를 적용할지 여부.
  당일 비중이 목표와 같아도 체결일까지 가격이 변할 수 있으므로 정상 조건이
  충족되면 true입니다. 체결 시 비중 차이가 없으면 실제 거래·비용은 0입니다.
  이미 현금 100%인 안전장치 상태에서는 false로 반복 청산을 막습니다.
- API는 거래 주문을 전송하지 않습니다. `is_order=false`를 유지합니다.

환경/history:
- `rebalance_reasons`와 CSV용 `rebalance_reason`(`|`로 복수 사유 결합).
- `rebalanced`: 실제 매매 또는 안전장치 청산이 발생했는지.
- `model_target_weight_<ticker>`: 매일 계산한 목표.
- 기존 `target_weight`, `previous_weight`, `current_weight`, 비용·가치 필드는 유지.
- 자산별 CSV에도 `model_target_weight`와 `rebalance_reason`을 기록합니다.
- stop 후 백테스트의 잔여 기간은 현금, 거래/비용 0으로 채웁니다.

## 기존 결과에 대한 주의

기존 모델 파일과 과거 평가 결과는 덮어쓰지 않습니다. 저장된 성과와 SHAP은
이전 거래 정책의 결과일 수 있으므로 새 정책의 성과로 간주하지 않습니다.
새 정책 평가에는 재평가가 필요하며, 정책에 맞춘 모델 최적화에는 재학습이
필요합니다. 새 평가의 비교 그룹에는 정책 버전을 포함해 과거 정책 결과와
같은 그룹으로 묶이지 않도록 합니다.

## 검증

`tests/test_rebalancing.py`는 월초/hold, 뉴스·비중·MDD 경계값, 복수 사유,
다음 날 뉴스 누출 방지, 비용, 자연 비중 변동, 현금 반복 청산 방지,
일별 추론과 Recurrent PPO 상태 유지, API 필드 및 잘못된 설정을 검증합니다.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_rebalancing.py -q
```
