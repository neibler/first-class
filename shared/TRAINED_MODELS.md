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
