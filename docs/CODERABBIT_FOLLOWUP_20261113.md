# 최신 CodeRabbit 추가 리뷰 수정 기록 — 2026-11-13

## 구현 내용

### 체크포인트 저장·복구·정리

- bundle별 현재 세대와 직전 검증된 정상 세대, 최대 2개를 유지한다.
- 모델·Replay Buffer·manifest 저장 및 해시 계산을 완료한 다음 checkpoint.json을 원자적으로 교체한다.
- 직전 세대 정보도 포인터에 보존한다. 현재 세대가 손상되면 직전 세대의 모델/버퍼 전체 해시를 검증한 뒤 Resume 대상으로 복구한다.
- OS 파일 잠금으로 같은 bundle에 동시 저장하는 작업을 차단한다. 프로세스 종료 시 잠금은 운영체제가 해제한다.
- 포인터 교체 전 실패하면 새 불완전 세대만 정리하고 기존 포인터/파일은 유지한다.
- 정리는 저장 성공 이후에 수행한다. 지정된 checkpoint_generations 바로 아래의 생성 규칙에 맞는 디렉터리만 대상으로 하며 심볼릭 링크·junction·활성 세대·직전 정상 세대는 제외한다.
- 모델 패키징 후보 검색에서 checkpoint_generations 및 checkpoints 아래 manifest를 제외한다.
- 기존 다른 run 폴더를 일괄 삭제하지 않는다. 다음 저장부터 해당 bundle에 정책이 적용된다.

### 학습 및 서비스 검증

- Legacy RecurrentPPO 다중 seed 검증을 공통 함수로 분리해 실제 학습과 preflight에서 동일하게 적용한다. PPO 및 단일 seed RecurrentPPO 지원은 유지한다.
- Resume에서 명시한 --train-years / --test-years를 기본값과 구분하고 지원하지 않는 조합으로 거부한다. 일반 학습의 기본 4년/1년은 유지한다.
- safeguard_triggered / stopped / reset_episode는 실제 bool만 허용한다. 문자열 "false", 숫자, null, 목록 등은 상태 변경 전에 거부한다.

### 화면 및 응답

- Correlation API 오류 시 st.stop으로 앱 전체를 중단하지 않는다. 오류·재시도 버튼을 표시하고 공통 교육용 면책 고지와 시스템 상태 패널을 계속 렌더링한다.
- 통계 기준을 ETF 로그수익률, WTI 단순수익률, 금리/VIX/금리차 일별 변화량으로 구분한다.
- 주요 SHAP 요인이 없으면 상세 테이블 안내를 표시한다.
- 평가 거래 기록과 매칭되지 않은 날짜는 Replay 기준 날짜로 명시한다.
- SHAP assets는 비어 있지 않은 mapping인지 검증하고 잘못된 응답을 사용자용 APIError로 처리한다.
- RSI 문구의 분류는 화면에 표시하는 소수 한 자리 값으로 판단해 경계값의 모순을 없앤다. 원본 입력과 SHAP 계산은 변경하지 않는다.

### 설치·Docker·문서

- 학습 입력 및 모델 설치는 모든 새 파일을 임시 경로에 완전히 쓰고 flush/fsync한 뒤 파일별 원자적 교체를 수행한다. 일반 I/O 실패 시 이번 설치가 추가한 파일을 되돌리고 기존 파일은 보존한다.
- 이 보장은 파일별 원자성과 예외 발생 시 롤백이다. 컴퓨터 강제 종료까지 여러 파일 전체의 단일 트랜잭션으로 보장하는 것은 아니다. 재실행 시 동일한 완료 파일은 재사용한다.
- .dockerignore에 weight_format.py를 포함하고 Dockerfile의 dashboard 빌드 및 smoke 검사에서 allocation 차트를 실제 생성한다.
- README의 fresh-clone 절차는 최신 학습 입력 ZIP 설치 → preflight → dry run 순서다. 과거 E 모델 ZIP과 구분한다.
- 최신 입력 2019-02-05~2026-09-30, macro 13개와 세 Walk-Forward 구간을 README 및 Correlation 문서에 반영했다. 2026 최종 평가 설정은 별도로 유지한다.

## 이전 지적 사항 재검증

| 항목 | 현재 상태 및 검증 |
|---|---|
| PPO Resume 기본 스텝 | n_steps 배수로 올림하는 기존 수정 유지, Resume 회귀 테스트 |
| 구버전 거래 정책 | execution_policy 없는 bundle 거부 유지, 우회하지 않음 |
| RecurrentPPO 안전장치 및 재시작 | 현금 latch와 반복 요청 복구 테스트 유지 |
| risk_score 제외 데이터 경로 | 최신 입력 ZIP과 일반/제외 설정 해시 일치 검사 |
| 데이터/학습 기간 | 설치 후 실제 날짜 및 명시적 3개 window 계획 테스트 |
| Docker correlation 503 | 검증 ZIP을 API 이미지에 설치하는 구조 유지, 설치 입력 기반 API 테스트 및 CI HTTP 검사 |
| 활성 모델 입력 경로 | 사용자 지정 설치 경로와 상대 루트 이동 테스트 유지 |
| SHAP 요인 선택 | 실제 보유 비중과 SHAP baseline 분리, absolute importance 순위 및 원본 보존 테스트 유지 |
| 모델 설치/부분 설치 | 경로 충돌 사전 검증, 신규 I/O 실패 롤백 테스트 |

## 변경 파일

- backend/investment_ai/checkpoints.py
- backend/investment_ai/experiment.py
- backend/investment_ai/preflight.py
- backend/investment_ai/train.py
- backend/investment_ai/serving.py
- scripts/build_trained_models.py
- scripts/install_training_inputs.py
- scripts/install_trained_models.py
- scripts/smoke.py
- frontend/app_pages/correlation.py
- frontend/app_pages/shap.py
- frontend/shap_client.py
- frontend/shap_state.py
- .dockerignore
- Dockerfile
- README.md
- docs/CORRELATION.md
- tests/test_latest_review.py
- docs/CODERABBIT_FOLLOWUP_20261113.md (이 문서)

## 모델 호환성 및 제한

- Observation, action, reward, SHAP 계산, 리밸런싱 수식과 저장된 모델/데이터/결과는 변경하지 않았다.
- 현재 거래 정책·schema·입력과 일치하는 완료 모델의 추론/평가 호환성을 유지한다.
- 이전 리뷰에서 적용된 정책 marker 없는 구버전 bundle 거부는 유지한다. 이전 E 공유 ZIP이 이 조건에 해당하면 새 정책 학습본으로 별도 갱신해야 한다. marker만 임의로 추가하지 않는다.
- Resume의 기존 source_hash 일치 조건을 유지한다. 수정 전 코드로 학습한 체크포인트는 새 코드에서 소스 불일치로 거부될 수 있다. 동일 소스 버전에서 재개하거나 새 실험으로 학습해야 한다.
- Docker 실행 파일이 없고 WSL 조회가 접근 거부되어 로컬 실제 이미지 빌드는 수행하지 못했다. 이미지 내 검증은 Dockerfile/CI에 추가했으며 push 후 Docker CI 성공 확인이 필요하다.
- 커밋/푸시는 하지 않았다.

## 검증 결과

- `python -m pytest -q`: **711 passed, 14 warnings**, 158.34초. 실패/skip 없음.
- 이번 추가 회귀 테스트: `tests/test_latest_review.py` 51개. 체크포인트 4모델 보관/복구, 동시 쓰기 차단, 포인터 교체 실패, 패키징 후보, Resume 옵션, bool 타입, UI 오류/재시도, SHAP/RSI, 설치 I/O 실패 및 기존 파일 보존, 네 설정과 설치 입력 일치를 포함한다.
- 기존 테스트 포함: 4모델 짧은 실제 학습/Resume, 모델 패키징/설치/상대 경로, API와 Correlation, SHAP 원본·선택 규칙, 구정책 거부.
- `black --check`: Python 147개 파일 통과.
- `flake8`, `pip check`, `git diff --check`: 통과.
- 경고 14개는 기존 pandas/NumPy 비권장 처리 및 거의 동일한 통계 샘플에 대한 SciPy 정밀도 경고다. 검증을 제거하거나 경고를 숨기지 않았다.
- Docker 실제 빌드/컨테이너 import는 로컬 환경 부재로 미실행. 로컬 allocation 차트 생성은 통과했고 이미지 빌드/CI 실행 검사를 추가했다.
