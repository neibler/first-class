# CodeRabbit 통합 수정 결과 (2026-11-13)

첨부된 이전 14건 및 통합 31항목을 중복 병합하여 처리했다. 실제 장기 학습 및 사용자의 모델/데이터 원본은 수정하지 않았다. 모델 구조 및 관측/action 차원은 유지한다.

## 항목별 결과

| 항목 | 원인 및 처리 |
|---|---|
| 2.1 | Resume 기본 추가 step을 저장 모델의 유효 n_steps 배수로 올림. 256이면 5,120. |
| 2.2 | SAC/TD3 learning_starts/gradient_steps 정수·범위 검증, 학습 출력 폴더 생성 전 거부. |
| 2.3~2.4 | run_spec의 data_seed 및 시작/종료일로 동일 입력 subset 복원, 상대 split 검증. 중단 시에도 복원하도록 run_spec을 학습 시작 전에 기록. |
| 2.5 | 양쪽 config 키의 합집합 및 지원 기본값으로 비교. model_type/include_risk_score 누락으로 의미가 달라지면 로딩 전 거부. |
| 2.6 | 모델·버퍼·manifest를 별도 generation에 저장하고 checksum을 포함한 checkpoint.json을 원자 교체. 저장 실패 시 이전 정상 generation으로 재개. root model.zip은 배포용 복사본. |
| 2.7 | Resume --device를 적용하고 호환성 비교에서는 device 변경 허용. |
| 3.1 | schema/manifest에 execution_policy 기록. load_bundle은 data=None에서도 정책 불일치·누락 거부. |
| 3.2~3.3 | 저장된 latch 및 요청 MDD를 예측 전에 정규화. stop 경로는 LSTM 복원·예측 없이 현금 반환. 동일 stop 요청 재시도 동일 결과. |
| 3.4 | 활성 RecurrentPPO에서 None/빈 predictor 목록 모두 거부. 안전장치 경로는 예측 불필요. |
| 3.5 | bundle infer --output을 JSON/CSV로 실제 저장. |
| 3.6 | 허용 오차 내 fully-invested weights를 정규화 후 거리 산출. |
| 4.1 | 입력 준비 시 investment_ai/multi_model/multi_model_no_risk 동기화, false flag 보존. |
| 4.2 | 출력 전 모든 대상 config의 plan_runs 날짜 구간 검증. 실패 시 기존 파일 불변. |
| 4.3 | effective_mode 공통 함수 사용. preflight CLI train-years/test-years 지원. |
| 4.4 | 최신 입력 별도 training_inputs.zip/checksum manifest/설치기 제공. 이전 모델 입력과 경로 충돌 없이 설치. |
| 4.5 | 패키징 전에 실제 입력 로딩→run_spec subset→fingerprint 비교, 불일치 거부. |
| 5.1~5.3 | 거래 비중 변화와 SHAP baseline 기여를 분리. 절댓값 합 상위3, balanced 제외, 나머지/전체 raw는 상세 보존. 각 부호는 baseline 대비로 명시. |
| 5.4 | cached date의 후속 asset에도 strict zip 검증. 불완전하면 전체 문맥 unavailable. |
| 5.5 | SHAP 입력 복원에 data_seed 사용, exact_data 검사 유지. |
| 5.6 | CSV run_id를 문자열로 읽어 001234 같은 식별자 보존. |
| 5.7 | Docker image에 같은 검증 입력 설치, runtime 경로 명시. smoke에 Correlation 실제 HTTP 조회 추가. |
| 6.1 | 기존 installed_inputs.json 상대 루트 처리 검증 유지. 사용자 지정 위치에서도 bundle/data 일치, 환경 루트 사용법 문서화. |
| 6.2 | 안전 검사 후 PurePosixPath.as_posix()로 archive membership 정규화. |
| 6.3 | 쓰기 전 모든 target 및 parent 파일/디렉터리 충돌 검사. |
| 6.4 | 기본 ROOT=프로젝트/runtime은 이미 해결됨. installer 기본 destination도 명시된 INVESTMENT_AI_HOME에 맞춤. |
| 7.1~7.3 | ablation Walk-Forward 명령, 실제 window 평가 경로, manifest 기반 모델 경로로 문서 갱신. |
| 이전 문서 nitpick | SHAP table raw pattern backtick 및 VIX label은 이미 수정되어 중복 변경하지 않음. |
| 비중 표시 | 사용자 확인대로 소수 첫째 자리에서 HALF_UP 반올림하여 정수 %. 표·차트 tooltip은 동일 포맷. 원본 거래 값은 그대로. |

## 회귀 검증

- 전체 python -m pytest -q: **659 passed / 0 failed / 0 skipped**, 14 warnings, 118.35초.
- 전체 실행 시작 후 추가한 설치→Correlation 통합 검사: **1 passed**. 총 확인한 고유 테스트 660개.
- 첫 전체 실행: 로컬 통신 제한으로 TestClient 초기화 정지. 로컬 통신 허용 후 정상 끝까지 실행.
- 중간 전체 실행의 실패3개는 변경 전 SHAP 방향 필터링 기대였다. 새 요구사항에 맞게 기준값/거래 방향 분리 검증으로 수정했으며 원본 수치/정렬/상세 보존 검증을 유지했다.
- 4개 알고리즘의 작은 환경에서 실제 16-step 학습→16-step Resume 검증, 체크포인트 실패 주입, 데이터 fingerprint 변경, 기간/seed 복원, 안전장치 재시작/재시도, 경로충돌, 새 clone 입력 설치, UI 검사 포함.
- Black 전체 검사 통과, flake8 전체 검사 통과, pip check 통과, git diff --check 통과.
- 경량 API import/분석 의존성 분리 테스트는 전체 테스트에 포함하여 통과.
- 경고는 statsmodels/scipy 통계 테스트 등에서 발생한 경고이며 성공으로 숨기지 않았다. 상세는 pytest 로그 참고.
- Docker 명령이 이 환경에 없어 실제 build/compose 실행은 수행하지 못함. Dockerfile 수정, 설치 코드 및 동일 Correlation 데이터 경로는 로컬 통합 검증 완료. 실제 컨테이너는 CI docker-smoke에서 확인 필요.

## 기존 모델 및 배포 영향

- 정책 버전 없는 기존 E/기타 bundle은 새 inference/resume에서 안전 거부한다. 기존 파일을 수정하거나 정책을 임의 부착하지 않았다.
- 따라서 현재 보관된 이전 shared/trained_models.zip은 새 코드의 실행 가능한 배포본이 아니다. 새 코드·정책으로 학습한 모델을 평가·SHAP 생성 후 다시 패키징해야 한다. 기존 저장 성과 파일의 열람과 새 정책 추론은 구분해야 한다.
- 구형 off-policy checkpoint는 모델/버퍼 원자성이 입증되지 않으므로 재개를 거부한다. 새 checkpoint.json generation을 사용해야 한다.
- 실행 중이던 Python 학습 프로세스에는 수정사항이 소급 반영되지 않는다. 그 프로세스가 내놓은 구형 metadata 모델 역시 새 코드에서 거부될 수 있다.
- 재학습이 필요 없는 변경: 비중 표시, SHAP 화면 요약, 입력 설치, 문서. 새 실행 정책 인증을 받는 모델 배포에는 새 학습이 필요하다.
- 배포 전: 코드/새 입력 ZIP 함께 커밋, 새 모델 공유 패키지 갱신, Docker build 및 API 준비 상태 확인. 커밋·push는 수행하지 않음.

## 변경 파일

- `.dockerignore`
- `.gitignore`
- `Dockerfile`
- `MULTI_MODEL_TRAINING.md`
- `README.md`
- `backend/investment_ai/checkpoints.py`
- `backend/investment_ai/infer.py`
- `backend/investment_ai/model_factory.py`
- `backend/investment_ai/model_runs.py`
- `backend/investment_ai/preflight.py`
- `backend/investment_ai/prepare_training_inputs.py`
- `backend/investment_ai/serving.py`
- `backend/investment_ai/settings.py`
- `backend/investment_ai/train.py`
- `backend/investment_ai/training.py`
- `backend/performance/model_shap.py`
- `backend/performance/shap_context.py`
- `backend/performance/shap_decisions.py`
- `docs/RISK_INPUT_ABLATION.md`
- `docs/SHAP_PRESENTATION.md`
- `frontend/app_pages/portfolio.py`
- `frontend/app_pages/shap.py`
- `frontend/charts.py`
- `frontend/shap_presentation.py`
- `frontend/weight_format.py`
- `scripts/build_trained_models.py`
- `scripts/install_trained_models.py`
- `scripts/install_training_inputs.py`
- `scripts/smoke.py`
- `shared/TRAINED_MODELS.md`
- `shared/training_inputs.zip`
- `shared/training_inputs_manifest.json`
- `tests/test_model_support.py`
- `tests/test_prepare_training_inputs.py`
- `tests/test_review_model_sharing.py`
- `tests/test_review_regressions.py`
- `tests/test_shap_context.py`
- `tests/test_shap_decisions.py`
- `tests/test_shap_presentation.py`
- `tests/test_ui_pages.py`
