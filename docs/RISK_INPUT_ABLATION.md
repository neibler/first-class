# risk_score 입력 포함/제외 비교

기본은 `include_risk_score: true`이며 설정이 없는 기존 모델도 true로 해석합니다.

| 설정 파일 | 입력 | 현재 입력 수 |
| --- | --- | ---: |
| `config/multi_model.json` | risk_score 포함 | 376 |
| `config/multi_model_no_risk.json` | risk_score 제외 | 361 |

두 파일은 생성 시 `include_risk_score`만 다릅니다. 이후 데이터/학습 설정을
바꿀 때는 두 파일을 동일하게 갱신하고 이 항목만 다르게 유지하세요.

제외되는 것은 자산별 risk_score 관측 15개뿐입니다. 뉴스 파일 검증, 날짜 정렬,
뉴스 메타데이터, risk_score >= 0.7 조기 리밸런싱은 그대로 유지합니다.
현재 두 설정 모두 최소 목표 3%, 전체 포트폴리오 거리 0.10, MDD 15% 초과 정책입니다.
이전 개별 ETF 10%p 실험과는 정책이 다릅니다. 입력 효과를 비교하려면 양쪽을 동일 정책으로 실행하세요.

## 실행

프로젝트 루트 PowerShell에서 확인 후 실행합니다.

```powershell
.\.venv312\Scripts\python.exe local_run.py preflight --config config/multi_model_no_risk.json
.\.venv312\Scripts\python.exe local_run.py train --config config/multi_model_no_risk.json --mode walk-forward --model ppo --reward return_mdd --timesteps 100000 --seed 42 --run-id without_risk_01 --dry-run
```

실제로 학습하려면 `--dry-run`을 제거합니다. 다른 모델은 `--model`에
`recurrent_ppo`, `sac`, `td3`를 사용합니다. 포함 실험은 `multi_model.json`과
별도 run-id를 사용합니다. 기존 완료 실험과 동일 seed/step/기간/정책인지 확인하세요.
seed 반복 실험은 양쪽에 같은 `--seeds 42 123 2026`을 지정합니다.

각 모델 학습 후 해당 bundle에 `local_run.py evaluate --bundle ...`를 실행합니다.
입력 포함 여부는 모델과 함께 config에 저장됩니다. 추론과 SHAP은 저장된 설정에서
관측 크기와 feature_names를 만들며, 제외한 점수의 SHAP 기여를 만들어 넣지 않습니다.
저장 모델의 config에서 플래그만 뒤집으면 schema 불일치로 로딩을 거부합니다.
입력 크기가 다르므로 별도로 학습한 모델을 선택해야 합니다.

## 기존 상태로 복구

다음 새 학습에 `config/multi_model.json`을 사용하면 됩니다.
기존 모델/config/schema는 수정하지 않으므로 예전 포함 모델을 다시 선택할 수도 있습니다.
현재 기본 모델 선택이나 공유 패키지를 자동으로 바꾸지 않습니다.

평가 기간은 동일하게 유지하고 비용 차감 수익률, Sharpe, MDD, 비용을 비교합니다.
이 실험은 모델 입력의 추가 효과를 보는 것이며 뉴스 기능 전체 제거 실험이 아닙니다.


세 window_0/1/2를 모두 평가하여 비교합니다. 포함/제외 설정은 동일한 prepared 입력을 사용합니다. 새 clone은 먼저 `python scripts/install_training_inputs.py`를 실행하세요.
