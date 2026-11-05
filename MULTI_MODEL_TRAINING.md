# PPO / RecurrentPPO / SAC / TD3 실행 안내

## 현재 입력과 중간 점검 (2026-11-05)

시장 입력은 `data_process/processed/features_unscaled.parquet`, 뉴스 원본은
`runtime/data/news/raw/news_risk_nyt_2021_2025.csv`입니다.
공통 기간 전체 2021-01-04~2025-12-31 (1,255 거래일, 18,825행)을 사용합니다.
`config/investment_ai.json`, `config/multi_model.json`은 검증된 `runtime/data/prepared/`의 파일을 읽습니다.
15 ETF + 거시지표 13개 + 뉴스 risk_score를 사용하며 observation 크기는 376입니다.
매크로 순서는 다음과 같습니다.

```text
core_pce_yoy, real_pce_mom, unemp, ism_pmi, effr,
ust_2y, ust_10y, ust_30y, tips_10y, vix, usd_ret_20d, wti, yield_spread
```

`rsi_14`를 `rsi`로 연결하고, 조정종가 기준 원본 log_return을 보상 회계에 유지합니다.
상류의 `*_z`는 쓰지 않고 기존 FeatureScaler가 학습 구간에서만 정규화 통계를 계산합니다.
뉴스의 날짜·점수·위험 만료는 재계산하지 않으며, 근거와 출처 컬럼도 보존합니다.
VIX/WTI의 정보 공개 시점 근사 한계는 상류 데이터 설명을 따릅니다.
이번 중간 점검은 `single` 모드의 앞 80% 학습 / 뒤 20% 평가입니다.
약 4년 학습 + 약 1년 테스트를 한 번 수행했으며, 여러 윈도우 Walk-Forward와 구분합니다.

| 구분 | 실제 기간 | 거래일 |
| --- | --- | ---: |
| Train | 2021-01-04~2024-12-30 | 1,004 |
| Test | 2024-12-31~2025-12-31 | 251 |

현재 학습 설정은 유지합니다. Walk-Forward 반복 검증은 시장·뉴스 데이터가 더 모인 뒤 수행합니다.
이 설정 변경은 학습을 실행하지 않으며 기존 모델·서비스 선택을 변경하지 않습니다.

새 clone에서는 먼저 [README의 입력 설치 절차](README.md#투자-강화학습-모델-4종-학습은-수동-실행)를 실행하세요.
공유 패키지의 준비된 Parquet로 Dry Run할 때는 원본 CSV를 다시 준비할 필요가 없습니다.
아래 재생성은 시장 원본과 **별도 전달받은 뉴스 원본 CSV가 모두 있을 때만** 실행합니다.
원본 뉴스 CSV는 Git에 포함되지 않으며, install_trained_models.py도 원본 CSV를 설치하지 않습니다.

입력 재생성 (PowerShell, 프로젝트 루트):
```powershell
$env:INVESTMENT_AI_HOME = (Resolve-Path runtime).Path
.\.venv312\Scripts\python.exe -m backend.investment_ai.prepare_training_inputs
.\.venv312\Scripts\python.exe local_run.py preflight --config config/multi_model.json
```
기간을 지정하지 않으면 두 원본의 공통 기간 전체를 사용합니다.
뉴스가 없는 2018~2020년과 2026년은 포함하지 않습니다.
준비 명령은 모든 대상 config의 자산 목록·순서와 최종 설정을 저장 전에 검사합니다.
검증에 실패하면 Parquet, provenance, config를 변경하지 않습니다.
기존 학습된 모델, 데이터와 서비스 선택은 이 안내를 읽거나 Dry Run하는 것만으로 바뀌지 않습니다.

## 환경과 설정

프로젝트 루트 `C:\Users\user\Documents\first-class-clone`에서 실행합니다.

```powershell
.\.venv312\Scripts\python.exe -m pip install -r backend/investment_ai/requirements.txt
```

Python 3.12, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0을 사용합니다.
`config/multi_model.json`은 기존 시장·뉴스·환경 설정을 복사한 개별 모델 실험 설정입니다.
`model_type`, `reward`, `total_timesteps`, `device`, `seeds`, `models`를 수정할 수 있습니다.
`models.ppo`, `models.recurrent_ppo`, `models.sac`, `models.td3`는 각 알고리즘 전용 인자입니다.
RecurrentPPO의 `lstm_hidden_size`, `n_lstm_layers`는 factory가 policy_kwargs로 전달합니다.
PPO는 기존 n_steps/batch_size와 SB3 기본값을 유지합니다. 아래 기본값은 튜닝 결과가 아닙니다.
PPO/RecurrentPPO의 timesteps·checkpoint_steps는 유효 n_steps의 배수여야 합니다.
SAC/TD3의 timesteps는 learning_starts보다 커야 합니다. replay buffer 기본은 10,000 transitions입니다.

모든 모델은 동일한 PortfolioEnv, 관측·비중 softmax·수수료·슬리피지·MDD 안전장치·회계를 사용합니다.
raw action은 동적 자산 수의 Box(-1, 1)입니다. reward는 모델과 독립적으로 선택합니다.
새 경로는 모델별/시드별 단일 모델이며 평균·앙상블·자동 best 선택을 하지 않습니다.
기존 legacy PPO 보상 선정·앙상블 경로는 유지합니다.
legacy Recurrent PPO는 단일 seed만 지원하며, 새 포트폴리오의 명시적 LSTM session으로
최종 추론까지 성공해야 run을 완료 처리합니다. 여러 seed의 recurrent legacy 앙상블은
서비스가 지원하지 않으므로 데이터 로드·학습·결과 생성 전에 오류로 차단합니다.
Recurrent PPO의 `single` / `walk-forward`에서는 여러 seed를 각각 독립 bundle로 학습할 수 있습니다.
config의 mode와 CLI의 --mode는 `single`, `walk-forward`, `legacy`만 허용합니다.

## 학습 없이 계획 확인

```powershell
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model recurrent_ppo --reward return_mdd --timesteps 100000 --seed 42 --dry-run
```

입력 데이터·설정·분할·출력 경로만 검사하며 모델 생성/학습/결과 저장을 하지 않습니다.
단일 모드는 마지막 test_fraction(기본 20%)을 holdout으로 남깁니다.
학습에는 holdout 이전 데이터만 쓰며 scaler도 그 기간에만 fit합니다.

## 실제 학습할 때만 실행할 명령

아래 명령은 사용자가 나중에 실행합니다. 각 명령은 한 모델·한 seed를 학습합니다.

```powershell
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model ppo --reward return_mdd --timesteps 100000 --seed 42 --run-id comparison_01
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model recurrent_ppo --reward return_mdd --timesteps 100000 --seed 42 --run-id comparison_01
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model sac --reward return_mdd --timesteps 100000 --seed 42 --run-id comparison_01
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model td3 --reward return_mdd --timesteps 100000 --seed 42 --run-id comparison_01
```

같은 모델에서 이미 있는 run-id/seed 경로는 덮어쓰지 않습니다. 다시 학습할 때 run-id를 바꾸세요.
모델당 저장 위치는 아래 형식이며 `.zip` 하나만이 아닌 bundle 폴더 전체를 함께 사용합니다.

```text
runtime/models/<model>/<run_id>/<window_id>/<model>_<reward>_seed<seed>/
  model.zip
  config.json / schema.json / scaler.json
  manifest.json / run_spec.json
  training_portfolio_history.csv / training_asset_history.csv / progress.csv
  checkpoints/
  replay_buffer.pkl   # SAC/TD3 resume용, 외부의 신뢰할 수 없는 파일을 로드하지 않음
```

metadata에는 model_type, reward_type, seed, 요청/실제 timesteps, train_start/end,
학습 종료 index, 데이터/소스 hash, 완료 여부가 기록됩니다. 모델의 holdout 결과는 학습이 끝나도 자동 생성하지 않습니다.
기존 latest.json과 serving 모델도 자동 변경하지 않습니다.

## 여러 seed

```powershell
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model sac --reward return_mdd --timesteps 100000 --seeds 42 123 2026 --run-id seeds_01
```

`--seed`/`--seeds`를 생략하면 config의 seeds를 사용합니다.
Python에서는 `model_runs.run_seed_experiments(config, model_type="sac", reward_type="return_mdd", seeds=[42,123,2026])`를 호출합니다.
공통 저수준 함수 `training.train_model(data, config, train_end, seed, output, resume=None)`도 유지합니다.

## 향후 Walk-Forward: 데이터 추가 확보 후 실행

현재 중간 점검의 약 4년 Train + 약 1년 Test는 완료한 단일 구간 평가입니다.
이를 실패한 학습이나 미완료 실험으로 해석하지 않습니다. 학습 설정을 더 짧게 변경하지 않습니다.

`walk-forward` 모드는 첫 거래일을 기준으로 달력 연도를 더하고, 고정 길이 Train 구간을
`test_years`만큼 이동합니다. 각 window의 scaler/model/LSTM 메모리는 새로 시작합니다.
현재 시작일 2021-01-04에 4년 + 1년을 더하면 첫 테스트 종료 경계는 **2026-01-04**입니다.
현재 마지막 데이터 2025-12-31의 다음 평일은 2026-01-01이므로, 현 구현의
`stop_date <= last_date + BDay(1)` 완전 윈도우 조건을 충족하지 못합니다.
따라서 **현재 데이터로 `--train-years 4 --test-years 1`은 첫 윈도우부터 거부**합니다.
이 조건은 거래소 휴일 달력이 아닌 pandas BDay(주말 제외 평일)를 사용합니다.

데이터가 추가되면 아래 Dry Run으로 실제 생성되는 구간을 먼저 확인하세요.
4년/1년을 여러 번 반복하려면 각 다음 윈도우의 종료 경계까지도 시장·뉴스 공통 패널이 필요합니다.

```powershell
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model.json --model td3 --reward return_mdd --timesteps 100000 --seeds 42 123 2026 --mode walk-forward --train-years 4 --test-years 1 --run-id wf_01 --dry-run
```

데이터가 충분해 계획이 정상 출력된 뒤에만 `--dry-run`을 제거합니다.
마지막 불완전 테스트 window는 제외합니다. legacy의 expanding validation과도 다른 방식입니다.

참고로 현재 날짜 패널에 대한 분할 검증에서는 3년/1년은 1개, 2년/1년은 2개 window가 가능합니다.
이는 분할 가능성을 확인한 결과일 뿐이며 이번 중간 점검 설정을 바꾸거나 추가 학습하지 않았습니다.

| 달력 기준 설정 | Train 실제 거래일 범위 | Test 실제 거래일 범위 |
| --- | --- | --- |
| 3년/1년, window 0 | 2021-01-04~2024-01-03 | 2024-01-04~2025-01-03 |
| 2년/1년, window 0 | 2021-01-04~2023-01-03 | 2023-01-04~2024-01-03 |
| 2년/1년, window 1 | 2022-01-04~2024-01-03 | 2024-01-04~2025-01-03 |

## 저장된 모델 평가와 비교 (학습 후 명시적으로 실행)

예: SAC seed 42의 holdout 평가

```powershell
.\.venv312\Scripts\python.exe local_run.py evaluate --bundle runtime/models/sac/comparison_01/holdout/sac_return_mdd_seed42
```

각 모델·seed·window bundle마다 실행하세요. 저장된 데이터 hash가 달라지면 평가를 거부합니다.
새 모델을 학습하지 않으며 bundle/evaluation 아래에 백테스트를 저장합니다.
`ppo_history.csv`는 기존 API 호환을 위한 주 전략 파일명이고, 실제 알고리즘은 metadata.model_type입니다.

```powershell
.\.venv312\Scripts\python.exe -m backend.investment_ai.model_comparison --runs runtime/models --output results/model_comparison.csv
```

완료된 평가만 읽습니다. 결과가 없으면 `No completed model runs found.`를 출력하고 CSV를 만들지 않습니다.
출력: model, reward, seed, window, test 날짜, comparison_group, CAGR, Sharpe, Sortino, MDD, Calmar.
`model_comparison_seed_summary.csv`에 비교 그룹·모델·보상별 mean/std/count를 저장합니다.
같은 데이터·분할·보상·비용 등 환경 조건인 comparison_group끼리 비교하세요.
중복된 동일 모델/seed/holdout 재실행은 에러로 알려 평균이 중복 집계되지 않게 합니다.
CAGR은 거래일 252일 연율화, Sharpe는 ddof=1, Sortino는 모든 일자의 음수 초과수익 RMS,
MDD는 양수 낙폭, Calmar는 CAGR/MDD입니다. 분모가 0이면 미정의 값(빈 칸)입니다.
seed가 하나면 표준편차도 빈 칸입니다. 결과를 보고 사람이 최종 모델을 선택합니다.

## Serving 모델 선택

`config/serving.json`에서 직접 지정합니다. model_path는 runtime 기준 bundle 경로 또는 절대 경로입니다.

```json
{
  "model_type": "sac",
  "model_path": "models/sac/comparison_01/holdout/sac_return_mdd_seed42"
}
```

`model.zip` 경로를 넣어도 같은 폴더의 config/scaler/schema/manifest가 필요합니다.
model_type과 저장된 모델이 다르면 서버가 로드를 거부합니다.
model_path=null이면 기존 latest.json 기반 데모 동작을 유지합니다.
명시적인 run 인자 또는 INVESTMENT_AI_RUN 환경변수는 serving.json보다 우선합니다.
UI 모델 선택 기능은 추가하지 않았습니다. 변경 후 API를 재시작합니다.
포트폴리오 백테스트 화면은 해당 bundle을 evaluate한 뒤 사용할 수 있습니다.

```powershell
.\.venv312\Scripts\python.exe local_run.py api
```

## RecurrentPPO state 처리

`prediction.predict_weights(model, observation, env, model_type, state, episode_start)`는
공통 비중 변환과 다음 LSTM state를 반환합니다. 일반 모델의 state는 None입니다.
`PredictionSession`은 lstm_states와 episode_start를 저장하고 reset()에서 초기화합니다.
백테스트는 일자마다 state를 넘기고 매 백테스트 시작 시 새 session을 만듭니다.
서비스는 episode_id별로 독립 state를 유지하며 리밸런싱하지 않는 날도 관측을 전달합니다.
같은 일자의 같은 요청은 재사용하고 다른 요청/역순/날짜 누락은 거부합니다.
새 episode/reset은 현금·MDD 0 계약을 따릅니다. Safe-Guard는 기존처럼 해제되지 않습니다.
서버 재시작 시 LSTM state는 소실됩니다. 기존 episode를 조용히 이어가지 않고 명시적 reset을 요구합니다.
이 단계에서는 recurrent state 영속화/과거 실제 보유 이력 replay/여러 서버 간 공유는 구현하지 않았습니다.
일회성 infer CLI로 RecurrentPPO를 매번 새로 호출하지 말고, 연속 일별 API 또는 PredictionSession을 사용하세요.

## 검증 범위와 남은 작업

공유 완료 모델은 4종 모두 seed 42, return_mdd, 100,000 step의 단일 구간 중간 점검 결과입니다.
학습·평가 기간과 지표는 [공유 모델 안내](shared/TRAINED_MODELS.md)를 참고하세요.
코드 회귀 검사는 모델 생성·save/load·상태 전달·CLI·분할·입력 검증·완료 처리 등을 확인하며,
수정 사항 검증을 위해 실제 장시간 재학습을 시작하지 않습니다.
하이퍼파라미터 튜닝, 새로운 앙상블, 자동 모델 선정, UI 모델 선택은 구현하지 않습니다.
실제 성능·학습 수렴·GPU 속도는 사용자가 학습한 뒤 확인해야 합니다.


## 이번 작업의 파일 목록

추가:
- `backend/investment_ai/model_factory.py`: 알고리즘 생성/로드, 모델별 인자 검증
- `backend/investment_ai/prediction.py`: 공통 action→weight 및 LSTM session
- `backend/investment_ai/model_runs.py`: 개별 seed 학습·달력 Walk-Forward·명시적 평가
- `backend/investment_ai/model_comparison.py`: 저장된 평가 지표·seed 통계
- `config/multi_model.json`, `config/serving.json`
- `tests/test_model_support.py`, `MULTI_MODEL_TRAINING.md`

수정:
- `training.py`, `train.py`: factory 연결, metadata, off-policy replay 저장, CLI
- `backtest.py`, `infer.py`, `serving.py`: 공통 prediction/session, 단일 bundle serving
- `evaluate.py`, `experiment.py`, `settings.py`: 평가 진입점·호환성·검증
- `backend/investment_ai/requirements.txt`: SB3/sb3-contrib 호환 버전
- `README.md`, `.dockerignore`, `Dockerfile`, `Dockerfile.investment`: 안내와 serving 설정 포함
- `tests/test_research_agent.py`: 로컬 Gemini 설정에 영향받지 않도록 기존 Claude 테스트 격리

PortfolioEnv·portfolio accounting·reward 구현 자체는 변경하지 않았습니다.
기존 Claude 기반 뉴스 리서치 호출 방식은 이 학습 안내와 독립적입니다.
