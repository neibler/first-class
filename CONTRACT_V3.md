# 투자 AI 계약 v3 — 2026-10-27

이번 변경은 학습을 실행하지 않고 적용했습니다. 기존 학습 결과는 보존됩니다.
기존 v2 모델은 새 시간축·보상 계약과 다르므로 새 코드에서 로딩을 거부합니다.
새 학습이 완료되기 전 API /ready는 기존 모델에 대해 503을 반환할 수 있습니다.

## 뉴스
- 필수 6열: date, ticker, risk_score, risk_type, has_news, days_since_news.
- 실제 뉴스(required)에서는 시장의 모든 date/ticker 키와 정확히 일치해야 합니다.
  누락·중복·초과 행, 잘못된 점수/플래그/경과일은 오류입니다. 조용한 0 채움은 없습니다.
- has_news=0이어도 전달된 risk_score를 보존합니다. 유지·만료 로직은 뉴스 AI 담당입니다.
- risk_type 빈 문자열은 초기값 (0, "", 0, 20)일 때 허용됩니다.
- 전체 뉴스 없음은 news_mode=development에서만 초기값으로 대체합니다.
- 뉴스 날짜를 재이동하지 않습니다. PPO에는 risk_score만 넣습니다.
- 검증한 모든 메타데이터는 runtime/data/news/validated_news.parquet 및 새 run의
  news_metadata.parquet에 저장됩니다. fingerprint에 메타데이터도 포함합니다.

## 관측과 체결
- observation_lag=0, execution_delay=1만 허용합니다.
- t 종가 가격/뉴스/매크로 → 결정 → 기존 보유 비중으로 t→t+1 수익 계산
  → t+1 종가 체결/비용 차감 → 새 비중은 t+1→t+2부터 적용합니다.
- 보유 비중은 수익률에 따라 변합니다. 거래 직전 비중은 이 변동을 반영한 값입니다.
- 안전장치는 예외적으로 위반 확인 종가에서 즉시 청산하고 대기 매수는 취소합니다.
- API는 주문을 보내거나 실제 체결을 주장하지 않습니다. 최신 자료 이후 거래일을
  임의 계산하지 않고 execution_date/first_return_end_date는 null로 반환합니다.
  execution_delay_trading_bars=1, first_return_end_delay_trading_bars=2를 제공합니다.
- API current_weights는 전달받은 실제 보유 상태이며 target_weights는 제안입니다.
  아래 history의 current는 모의 체결 후 비중입니다. 제안 API는 체결되지 않은
  목표를 실제 보유 비중으로 덮어쓰지 않습니다.

## Safe-Guard
- 누적 MDD > 0.15에서 ETF 비중 0, cash_weight=1, safeguard_triggered=true.
- 경계 비교에는 부동소수점 오차 1e-12 허용치를 사용합니다.
- 환경은 발동 후 종료되며 reset 전 step을 거부합니다. 백테스트 잔여 기간은 현금입니다.
- 순수 추론 함수는 max_drawdown/stopped/safeguard_triggered를 확인합니다.
- 서비스는 run+episode_id 단위로 발동 상태를 runtime/service_state에 보관합니다.
  서버 재시작만으로 해제하지 않습니다. 기본 실행처럼 단일 API 프로세스를 사용하세요.
- 명시적 reset: state.reset_episode=true와 새 초기 상태(전액 현금, MDD=0,
  portfolio_value=peak_value, stopped=false, safeguard_triggered=false)를 전달합니다.
  순수 함수 호출자는 반환된 safeguard 상태를 다음 state에 유지해야 합니다.

## 회계와 Reward
- 관측은 log_return, 회계는 expm1(log_return), 포트폴리오는 비용 차감 simple return.
- 누적 가치는 (1+r).cumprod() 공통 함수입니다.
- 외부 reward 이름은 return/sharpe/return_mdd입니다. portfolio_return/rolling_sharpe 별칭도 허용합니다.
- return_mdd = net return - lambda * 누적 MDD.
- Sharpe는 simple daily return, ddof=1, sqrt(252), 연 rf 기본 0입니다.
  reward는 기본 20개 수익률 미달/0분산이면 0, 성과 지표의 정의 불가 Sharpe는 null입니다.
  종전 비연율화·±5 clipping을 제거했습니다.
- 현재 시장 데이터는 Close 기반으로 배당을 추가 반영하지 않았습니다.
  총수익률 전환은 아직 결정하지 않았습니다. 프론트는 서버 지표를 표시합니다.

## 분석용 기록
- training_portfolio_history.csv, validation_history.csv, *_history.csv:
  date, portfolio_return, portfolio_value, reward, drawdown, max_drawdown,
  cash_weight, safeguard_triggered, reward_type, run_id, window_id 및 날짜 필드.
- training_asset_history.csv, validation_asset_history.csv, *_asset_history.csv:
  date, ticker, previous_weight, current_weight, target_weight, weight_change.
- previous = 거래 직전(가격 변동 반영), target = 안전장치 반영 최종 목표,
  current = 거래 완료 후, weight_change = current - previous.
- schema.json에 정확한 feature_names, asset/macro 순서와 시간축을 기록합니다.
  scaler.json은 학습 구간으로만 계산한 전처리 통계입니다. SHAP 계산은 추가하지 않았습니다.

## 검증 및 재학습
실제 입력 752일 × 15 ETF 검증 및 Gymnasium 환경 검사 통과.
계약 회귀 테스트 20개 통과(뉴스·시점·비용·MDD·reward·기록·API 핸들러).
HTTP TestClient가 이 환경에서 대기하여 중단했으며 API는 핸들러/응답 스키마로 검증했습니다.
새 모델 학습 및 전체 UI 회귀 테스트는 실행하지 않았습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 preflight
powershell -ExecutionPolicy Bypass -File .\run.ps1 train
```
기존 모델 resume 대신 새 학습으로 시작하세요. Git 커밋·푸시는 하지 않았습니다.
