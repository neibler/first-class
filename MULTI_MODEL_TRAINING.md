# 투자 AI 다중 모델 학습

새 clone에서 Python 3.12 환경 및 학습 의존성을 설치한 뒤 실행하세요.

```powershell
python scripts/install_training_inputs.py
python local_run.py preflight --config config/multi_model.json
python local_run.py train --config config/multi_model.json --model ppo --seed 22 --run-id comparison_01 --dry-run
```

입력은 2019-02-05~2026-09-30, ETF 15개, macro 13개, 뉴스 점수입니다. 3% 필터와 최대 7개 보유를 적용합니다. 2021~2025년의 이전 shared 모델 입력과 구분되는 별도 검증 ZIP을 설치합니다.

config/multi_model.json은 80/20 holdout이 아니라 명시적 Walk-Forward 3개 구간입니다. 2019~2022 학습/2023 평가, 2020~2023 학습/2024 평가, 2021~2024 학습/2025 평가입니다. 첫 시작일은 2019-02-05입니다. window_0, window_1, window_2로 저장됩니다.

```powershell
$runId = "WF_" + (Get-Date -Format "yyyyMMdd_HHmmss")
foreach ($model in @("ppo", "recurrent_ppo", "sac", "td3")) {
    python local_run.py train --config config/multi_model.json --model $model --seed 22 --run-id $runId
    if ($LASTEXITCODE -ne 0) { throw "Training failed" }
    foreach ($window in @("window_0", "window_1", "window_2")) {
        $bundle = "runtime/models/$model/$runId/$window/${model}_return_mdd_seed22"
        python local_run.py evaluate --bundle $bundle
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
        python -m backend.performance.model_shap --bundle $bundle --sample 10 --permutations 8
        if ($LASTEXITCODE -ne 0) { throw "SHAP failed" }
    }
}
```

--seed 생략 시 설정의 17,22,43을 모두 학습합니다. 모델별·seed별·window별로 별도 모델을 학습합니다. --mode single로 세 구간 설정을 덮어쓰지 마세요.

최종 종류/설정 선정 후 config/final_2026.json으로 2022~2025 학습, 2026-01-02~2026-09-30 최종 평가를 실행합니다. 이 설정만 holdout 경로를 만듭니다. --train-years/--test-years는 명시적 training_windows가 없는 자동 Walk-Forward에서만 적용됩니다.

## 재개 및 호환성

`python local_run.py train --resume <bundle> --output <새 폴더>`: 저장된 seed/기간/정책과 체크포인트를 검증합니다. 추가 step 기본값은 PPO rollout 배수로 올림합니다. --device cpu를 지정할 수 있습니다. 과거 실행 정책 버전이 없는 bundle은 안전하게 거부합니다. 기존 model.zip에 정책 메타데이터를 임의로 붙이지 말고 새 정책으로 학습하세요.

새 SAC/TD3는 checkpoint.json이 가리키는 완전한 모델/버퍼 세대만 재개합니다. root model.zip은 평가·배포용 복사본입니다. 과거 off-policy bundle은 원자적 저장 여부를 증명할 수 없어 재개하지 않습니다. 실행 중 프로세스는 코드 변경을 자동 반영하지 않습니다.

원자료와 저장 모델은 수정하지 않습니다. 2026년은 독립 최종 평가로 보존하고 성과를 본 뒤 반복 튜닝하지 않습니다. VIX/WTI 가용일 근사 제한은 docs/TRAINING_2019_2026.md를 참고하세요.
