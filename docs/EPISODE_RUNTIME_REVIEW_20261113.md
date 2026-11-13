# 에피소드 clock 및 Resume/Runtime 리뷰 수정 — 2026-11-13

## phase 정의와 호환성

- 관측의 마지막 rebalance_phase는 `episode_elapsed_bars % rebalance_days / rebalance_days`로 통일했다.
- 학습은 기존과 동일하게 `t - episode_start`를 사용한다. 추론에서만 잘못 사용하던 days_since_rebalance를 분리했다. days_since_rebalance는 기존 API 호환 필드로 남기지만 phase 계산에 사용하지 않는다.
- 새 공통 모듈 episode_clock.py에서 bar 수와 period를 검증하고 phase를 계산한다.
- 관측 차원, 피처 순서, Action/Reward, 거래 비용 및 리밸런싱 조건은 변경하지 않았다.
- schema.json에 `phase_definition: episode_elapsed_bars_v1`을 저장한다. 정의가 없거나 다른 Bundle은 추론과 Resume 모두 명확히 거부한다. 예전 v3 표기만으로 의미 일치를 가정하지 않는다.
- 학습 환경의 숫자적 의미는 바뀌지 않았지만, 기존 Bundle에 이를 확인할 명시적 정보가 없으므로 자동 재사용/메타데이터 덧붙이기는 하지 않는다. 현재 기본 정책으로는 해당 구버전 모델을 새 코드로 재학습해야 한다. 검증된 과거 소스에 대한 별도 마이그레이션은 이번 범위에 포함하지 않았다.
- 기존 execution_policy 검증과 Resume source_hash 일치 검증도 유지한다.

## Serving 상태 계약

- episode_id별 시작일, 경과 bar 수, 마지막 관측일을 명시적으로 관리한다.
- 처음 새 현금 포트폴리오를 요청하면 현재 관측일/bar 0으로 시작한다.
- 기존 포트폴리오를 처음 복원할 때는 episode_start_date와 episode_elapsed_bars를 전달해야 한다. 날짜와 입력 거래일 수가 맞지 않으면 거부한다.
- 이후 요청은 현재 실제 보유 상태를 전달한다. clock 필드를 생략하면 서버가 거래일 기준으로 증가시킨다. 필드를 명시하면 서버 계산값과 일치해야 한다.
- 현재 입력에는 최초 복원 시 에피소드 시작일을 포함해야 한다. 계속되는 에피소드도 검증에 필요한 거래일 이력을 제공해야 한다. 누락된 거래일 수를 달력 날짜 차이로 근사하지 않는다.
- 월 정기 거래, 위험 점수 거래, 목표 거리 거래는 clock을 초기화하지 않는다. 명시적이고 유효한 reset_episode만 초기화한다.
- 같은 요청 재시도는 clock을 중복 증가시키지 않는다. RecurrentPPO는 같은 요청의 결과를 재사용해 기억 상태도 중복 전진하지 않는다.
- service_state에 clock, 안전장치 latch, RecurrentPPO의 LSTM 상태와 재시도 결과를 함께 원자적으로 저장한다. 재시작 후 복원하며 모델 파일 해시가 다르면 기존 상태를 재사용하지 않는다.
- 기존 latch 파일도 읽되, clock/기억 상태가 없는 기존 진행 중 포트폴리오는 명시적 상태 복원 또는 reset이 필요하다. 기록되지 않은 bar 수나 LSTM 상태를 임의로 생성하지 않는다.
- API 응답에 episode_start_date, episode_elapsed_bars, rebalance_phase를 제공한다. 이 정보는 추가 관측 피처가 아니라 상태·진단 정보다.

## Resume 및 입력 보존

- 신규 SAC/TD3 학습은 total_timesteps > learning_starts 검증을 유지한다.
- Resume은 검증된 체크포인트의 actual_timesteps + extra_steps로 warmup 충족 여부를 검사한다. CLI와 내부 train_model이 같은 검증 함수를 사용한다.
- 누적 1,008 step, learning_starts=1,000 모델에 500 step을 추가하는 실제 테스트를 SAC/TD3 각각 수행한다.
- 외부에 설치한 Bundle의 입력 루트를 새 Bundle 위치 기준으로 다시 계산한다. 기존 relative_root를 복사하지 않는다.
- 새 입력 바인딩에는 파일별 SHA-256을 기록한다. Evaluation/Serving/Resume에서 같은 상대 경로의 다른 데이터를 읽는 것을 막는다.
- 다른 Windows 드라이브에 저장하여 상대 경로를 만들 수 없으면 해시가 필수인 명시적 absolute_root를 사용한다. 해시 없는 절대 루트는 거부한다. 이 경우 드라이브 위치까지 바뀌면 별도 재설정이 필요하다.
- 실제 입력 바인딩 전에 학습에 전달한 구간과 원래 데이터 파일의 fingerprint도 비교한다.
- synthetic Bundle은 synthetic_market.parquet와 news_metadata.parquet를 저장한다. run_spec의 synthetic/data_seed를 보존하며 Resume 및 평가에서는 저장된 파일을 읽는다.
- 모델을 재개하면 환경 에피소드는 새로 시작한다는 기존 정책은 유지한다. 모델·optimizer·Replay Buffer의 재개와 환경 에피소드 복원을 혼동하지 않는다.

## Runtime·패키징·문서

- 모델 비교의 기본 검색 루트와 패키징 Runtime은 settings.ROOT를 사용한다. INVESTMENT_AI_HOME을 반영한다.
- 명시적 절대 Bundle 선택은 해당 Runtime 경계 안에서 지원한다. Runtime 밖 모델/입력은 오류로 거부한다.
- 비교 디렉터리가 없거나 완료 결과를 찾지 못했을 때 기존 비교 CSV를 삭제하지 않는다. 빈 결과와 안내를 반환하거나 명확한 경로 오류를 낸다.
- 패키징의 경계 검증은 resolved path로, ZIP 내부 경로는 원래 Runtime 상대 경로로 처리한다. data/market.parquet가 다른 Runtime 내부 위치를 가리켜도 config의 경로를 유지한다.
- MULTI_MODEL_TRAINING.md에 SHAP 실행 전 같은 Python 환경에서 backend/performance/requirements.txt를 설치하도록 명시했다. API Docker 의존성에는 추가하지 않았다.

## 이전 리뷰 재확인

전체 회귀 테스트에 기존 거래 정책 호환성, no-risk 설정/데이터, 기간 검증,
안전장치 latch, Docker용 Correlation 입력 설치, fresh-clone 경로,
weight_format 포함/차트 생성, SHAP 날짜 설명, 체크포인트 검색 제외,
외부 모델 설치 및 설치 실패 롤백 검증을 계속 포함한다.
기존 검증을 삭제하거나 운영 기능을 비활성화하지 않았다.
변경된 상태 파일 구조·비교 결과 보존 정책에 맞춰 테스트 기대값을 갱신했고,
모델 파일 해시 검증에 필요한 실제 임시 model.zip을 테스트 준비 단계에서 생성한다.

## 수정 파일

- backend/investment_ai/episode_clock.py (신규)
- backend/investment_ai/bundle_inputs.py (신규)
- backend/investment_ai/api.py
- backend/investment_ai/infer.py
- backend/investment_ai/portfolio_env.py
- backend/investment_ai/serving.py
- backend/investment_ai/settings.py
- backend/investment_ai/train.py
- backend/investment_ai/training.py
- backend/investment_ai/model_factory.py
- backend/investment_ai/model_runs.py
- backend/investment_ai/model_comparison.py
- scripts/build_trained_models.py
- MULTI_MODEL_TRAINING.md
- tests/test_episode_runtime_review.py (신규)
- tests/test_contract_v3.py
- tests/test_model_support.py
- tests/test_review_model_sharing.py
- tests/test_review_regressions.py
- docs/EPISODE_RUNTIME_REVIEW_20261113.md (이 문서)

## 검증 및 남은 제한

- 전체 `python -m pytest -q`: **728 passed, 1 skipped, 14 warnings**, 178.32초. 실패 없음.
- 이번 새 회귀 테스트 18개 중 17개 통과, OS 권한이 필요한 실제 symlink 통합 테스트 1개 제외. Linux CI에는 그대로 남아 있다.
- 학습/추론의 전체 observation 벡터를 period 1/5/21에서 동일 상태로 비교했다. 월별/뉴스 위험/거리 기반 거래가 포함된 경우도 일치한다.
- PPO/RecurrentPPO/SAC/TD3 임시 합성 모델의 Resume → Evaluation → CLI Infer → Serving 및 프로세스 재시작 복원을 검증했다.
- SAC/TD3 각각 warmup=1,000, 누적=1,008 상태에서 CLI extra_steps=500으로 실제 재개해 최종 1,508 step을 확인했다.
- 외부 Runtime의 실입력 형식 Parquet로 재개·평가·Serving을 검증했고, 같은 상대 경로에 다른 데이터가 있는 기본 Runtime을 사용하지 않는지 및 입력 변조를 거부하는지 확인했다.
- 모델 패키징/설치의 Runtime 경계, ZIP 입력 별칭, 비교 결과 보존 및 다른 Windows 드라이브용 해시 바인딩을 검증했다.
- 새 Python 프로세스에서 API import가 SHAP/statsmodels를 불러오지 않는 것을 확인했다.
- Black 150개 Python 파일, flake8, pip check, git diff --check 통과.
- 경고 14개는 기존 pandas/NumPy 비권장 처리와 SciPy 통계 정밀도 경고이며 숨기지 않았다.

- 사용자의 운영 모델 본학습은 실행하지 않았다. 테스트 학습/재개는 pytest 임시 폴더에서만 실행한다.
- 기존 runtime의 모델, 체크포인트, 데이터, 학습 결과 및 공유 모델 ZIP은 수정하지 않았다.
- Windows의 실제 심볼릭 링크 생성 권한이 없어 실제 링크 통합 테스트는 이 환경에서 실행할 수 없다. 원래 경로/해석된 경로를 분리한 단위 테스트는 실행하고, 실제 링크 통합 테스트는 Linux CI에서 실행되도록 유지한다.
- Docker가 설치되지 않은 로컬 환경의 제한은 그대로다. 이번 작업에서는 Docker 의존성 구성을 변경하지 않았다.
- 커밋과 푸시는 하지 않았다.
