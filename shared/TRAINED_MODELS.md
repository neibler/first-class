# 공유 모델과 학습 입력

모델 위치는 shared/trained_models_manifest.json의 models 항목이 기준입니다. 특정 과거 run 이름을 하드코딩하지 않습니다.

```powershell
python scripts/install_trained_models.py --activate ppo
$manifest = Get-Content shared/trained_models_manifest.json -Raw | ConvertFrom-Json
$bundle = Join-Path "$PWD/runtime" $manifest.models.ppo.path
Test-Path "$bundle/model.zip"
```

사용자 지정 설치 위치는 --destination으로 지정합니다. installed_inputs.json이 bundle과 입력 루트의 관계를 보존합니다. 공유 SHAP 목록은 INVESTMENT_AI_HOME을 기준으로 읽으므로 사용자 지정 설치 시 API 시작 전에 같은 루트를 지정하세요.

```powershell
$env:INVESTMENT_AI_HOME = "D:/portfolio-runtime"
python scripts/install_trained_models.py --destination $env:INVESTMENT_AI_HOME --activate ppo
python scripts/install_training_inputs.py --destination $env:INVESTMENT_AI_HOME
```

학습용 최신 입력은 training_inputs.zip으로 따로 제공합니다. 모델 ZIP의 입력은 해당 학습 당시 데이터이며 서로 덮어쓰지 않습니다.

현재 보관 중인 이전 E 공유 ZIP에는 execution_policy 버전 기록이 없습니다. 새 코드로 학습·검증된 bundle을 패키징하기 전까지 추론은 정책 불일치 오류를 반환합니다. 안전 검사를 우회해 기존 결과를 현재 정책 결과로 표시하지 않습니다. 원본 모델/평가/SHAP은 보존됩니다.
