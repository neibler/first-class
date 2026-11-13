# 최신 PPO 공유 모델

현재 공유 패키지는 PPO_FINAL2026_20261113_164403의 PPO seed 22 하나만 포함합니다.
2022-01-03~2025-12-31 학습, 2026-01-02~2026-09-30 평가 결과입니다.
모델, 설정·scaler·schema, 평가 기록, SHAP, 정확히 일치하는 시장·뉴스 입력을 포함합니다.
체크포인트·학습 상세 로그·다른 모델·뉴스 검색 DB·API 키는 포함하지 않습니다.

## 팀원 설치 및 실행

프로젝트 루트의 Python 3.12 가상환경에서 실행하세요. 아래는 `.venv312` 기준입니다.

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install -r backend/investment_ai/requirements.txt
.\.venv312\Scripts\python.exe -m pip install -r backend/performance/requirements.txt
$env:INVESTMENT_AI_HOME = "$PWD/runtime"
Remove-Item Env:INVESTMENT_AI_RUN -ErrorAction SilentlyContinue
.\.venv312\Scripts\python.exe scripts/install_trained_models.py --activate ppo
.\.venv312\Scripts\python.exe local_run.py api
```

새 터미널에서:

```powershell
.\.venv312\Scripts\python.exe local_run.py ui
```

브라우저에서 http://localhost:8501 을 엽니다. 재학습이나 SHAP 재계산은 필요 없습니다.
SHAP 화면은 활성 PPO bundle의 shap.json을 사용합니다.
기존 API가 켜져 있었다면 재시작하세요. 이전 install_demo.py는 실행하지 않습니다.

공유 목록은 trained_models_manifest.json의 models가 기준이며 현재 ppo만 있습니다.
ZIP과 manifest는 반드시 함께 커밋해야 합니다. manifest만 수동 변경하지 마세요.
이전 로컬 모델 폴더가 있어도 활성 모델 선택과 공유 목록에는 포함되지 않습니다.
과거 investment_demo.zip은 이번 설치·실행 경로에서 사용하지 않습니다.

## 재생성

```powershell
$env:INVESTMENT_AI_HOME = "$PWD/runtime"
.\.venv312\Scripts\python.exe scripts/build_trained_models.py --bundle ppo=models/ppo/PPO_FINAL2026_20261113_164403/holdout/ppo_return_mdd_seed22
```

명시한 --bundle만 패키징합니다. 다른 실험을 자동으로 추가하지 않습니다.
사용자 지정 설치 위치에서는 INVESTMENT_AI_HOME을 설정한 다음 동일한 설치 명령을 사용하세요.
설치기는 전체 파일의 검증값을 확인하며 기존 파일이 다르면 덮어쓰지 않습니다.
뉴스 검색·질의응답 기능은 별도의 뉴스 DB와 API 설정이 필요합니다.
