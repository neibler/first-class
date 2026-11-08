# 완료된 투자 모델 4종 공유

`trained_models.zip`과 `trained_models_manifest.json`은 최종 모델 4개와 동일한 시장·뉴스 입력,
scaler, feature 순서, 설정, 평가 결과를 포함합니다. 재학습 없이 추론·평가할 수 있습니다.
중간 체크포인트, 학습 로그, SAC/TD3 replay buffer, API 키, Chroma 검색 DB는 포함하지 않습니다.
이 패키지는 추론·평가용이며 중단 학습 재개용이 아닙니다.

## 새 clone에서 실행 (PowerShell, 프로젝트 루트)

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r backend/requirements.txt -r frontend/requirements.txt
.\.venv312\Scripts\python.exe scripts/install_trained_models.py --activate ppo
.\.venv312\Scripts\python.exe local_run.py api
```

별도 터미널:
```powershell
.\.venv312\Scripts\python.exe local_run.py ui
```

Python 3.12 환경으로 확인했습니다. 첫 줄의 python은 3.12 실행 파일을 사용하세요.
이 공유본에는 뉴스 위험 점수 입력이 포함됩니다. 질문·답변 리서치에는 별도로
뉴스 패키지, Chroma DB, Claude 키가 필요합니다 (`RESEARCH_INTEGRATION.md`).

## 모델 선택

`--activate`에 `ppo`, `recurrent_ppo`, `sac`, `td3` 중 하나를 지정하고 API를 재시작합니다.
동일 파일 재설치는 허용하고, 내용이 다른 기존 모델·데이터는 덮어쓰지 않습니다.
활성 모델 변경은 명시적인 `--activate` 때만 적용합니다.
기존 serving 설정은 `config/serving.before_shared_install.json`에 한 번 보관합니다.
`INVESTMENT_AI_RUN`, `INVESTMENT_AI_DATA`, `INVESTMENT_AI_NEWS` 환경 변수가 있으면
공유 설정 대신 기존 경로가 적용될 수 있으므로 사용 목적에 맞게 확인하세요.

## 평가 결과

모두 seed 42, return_mdd 보상, 100,000 step입니다.
입력: 2021-01-04~2025-12-31, ETF 15개, 거시지표 13개, observation 376.
학습: 2021-01-04~2024-12-30. 평가: 2024-12-31~2025-12-31, 251 거래일.
거래비용·슬리피지 차감 후 동일 평가 구간의 누적 수익률입니다.

| 모델 | 누적 수익률 | Sharpe | 최대 낙폭 | 안전장치 |
|---|---:|---:|---:|---|
| PPO | 4.36% | 0.561 | 8.78% | 미발동 |
| Recurrent PPO | 8.83% | 0.725 | 11.88% | 미발동 |
| SAC | -14.82% | -1.779 | 16.18% | 발동 |
| TD3 | 11.66% | 0.982 | 11.37% | 미발동 |

기본 실행 예시는 PPO이며 테스트 성과로 자동 모델 선택하지 않습니다.
단일 시드 평가이고, VIX/WTI의 당시 공개 시점은 상류 데이터의 근사 한계를 따릅니다.
각 모델 폴더 `evaluation.json`, `evaluation/ppo_history.csv`에 상세 결과가 있습니다.
`ppo_history.csv`는 기존 API 호환 이름이며 SAC/TD3 폴더에서는 해당 모델의 결과입니다.

## 재평가

공유 manifest의 models 항목에 모델별 경로가 있습니다.
```powershell
.\.venv312\Scripts\python.exe local_run.py evaluate --bundle runtime/models/ppo/full_data_20261104_014257/holdout/ppo_return_mdd_seed42
```

기존 `investment_demo.zip`은 과거 3년 데이터의 PPO 데모입니다. 이번 4개 모델 공유본은
`trained_models.zip`이며 설치 명령도 `install_trained_models.py`입니다.


## 실제 전체 입력 SHAP

공유본은 각 모델의 `shap.json`을 포함합니다. 설치 후 API를 재시작하고 SHAP 메뉴에서
모델·자산·결정일을 선택하세요. 활성 투자 모델과 해석 대상 모델 선택은 독립적입니다.
376개 feature_names 전체를 사용합니다 (15자산의 20일 수익률, RSI, MACD, 현재 비중,
13개 거시지표, 자산별 뉴스 점수, drawdown, MDD, rebalance phase).

4개 모델 모두 실제 평가 경로에서 안전장치 이전에 실행된 거래 결정 중 균등 간격 10일을
골라 설명했습니다. 모든 15개 ETF 출력에 대해 8회 순열 SHAP을 계산했습니다.
기준 배경은 학습 구간의 실제 정책 거래 관측 평균이며 테스트 미래 데이터로 만들지 않습니다.
Recurrent PPO는 각 결정 직전의 LSTM 상태를 그대로 고정해 현재 입력의 기여만 설명합니다.
기억 자체의 기여도, 안전장치 이후 현금 결정, 인과 효과를 설명하는 결과는 아닙니다.
독립 입력 교란은 실제 가능한 시장·비중 조합과 다를 수 있고, 표본 순열 근사 오차가 있습니다.
화면에는 전체 feature 표와 거시지표 필터, 상위 20개 기여도 및 표본 평균 영향이 나옵니다.

재계산 (모델 학습 없음):
```powershell
$env:INVESTMENT_AI_HOME = (Resolve-Path runtime).Path
.\.venv312\Scripts\python.exe -m pip install -r backend/performance/requirements.txt
.\.venv312\Scripts\python.exe -m backend.performance.model_shap --all-shared --sample 10 --permutations 8
```
한 모델만 계산하려면 `--all-shared` 대신 `--bundle <모델 폴더>`를 사용하세요.
`--sample`은 실제 거래 결정 중 표본 수이고, `--permutations`를 늘리면 근사 정밀도와 계산량이 증가합니다.
API는 `/performance/model-shap?model=ppo`이며 모델 파일 해시가 해석 결과와 다르면 거절합니다.
팀원의 기존 `run_analysis.py`의 6개 요약 피처 분석은 이전 실행 구조용으로 남겨뒀습니다.
최신 모델 4종은 위 `model_shap` 명령과 실제 결과 전용 화면을 사용합니다.


### Model selection and portable installation

The repository default does not select a machine-specific model. After installing,
select the model explicitly and restart the API:

```powershell
python scripts/install_trained_models.py --activate ppo
```

For multiple training seeds, package one explicitly selected seed per model type:

```powershell
python scripts/build_trained_models.py --seed 42
```

If several runs have the same seed, pass `--bundle MODEL=PATH` for each ambiguous
model, using a path relative to `runtime`. Selection never uses holdout performance.
The archive contains one model per type, not a multi-seed ensemble. Packaging
rejects incomplete evaluations, differing datasets, training/evaluation periods,
comparison groups, and inconsistent evaluation history dates.

`--destination` may point to another directory. Installation adds
`installed_inputs.json` beside each model, keeping original bundle files unchanged.
Inference, evaluation and SHAP resolve bundled market/news inputs from this portable
installation root. Move the entire installation together, not individual model
folders; reactivate the model after moving it. Explicit API input overrides still
use the configured runtime root.

Legacy Recurrent PPO CLI inference supports a single model in a fresh cash episode.
Existing recurrent episodes require the persistent API so that memory is retained;
legacy recurrent multi-seed inference is rejected explicitly.
