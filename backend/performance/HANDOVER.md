# 성능 분석 모듈 인수인계 (팀원용)

성능 분석 담당이 만든 `backend/performance/` 안내서입니다.
**기술 상세는 `README.md`, 이 문서는 "당신이 뭘 하면 되는지"만** 적었습니다.

---

## 한 줄 요약

PPO·MVO·동일가중·S&P500을 같은 조건으로 백테스트하고, 12개 지표·SHAP·ANOVA까지
계산해 CSV/JSON/PNG로 떨어뜨리는 모듈입니다. 웹에서 바로 쓸 수 있는 API도 붙어
있습니다.

**실데이터 1차 분석을 이미 돌렸습니다.** `shared/investment_demo.zip` 의 실제 데이터와
학습된 모델(`20261028_005508`)을 사용했습니다. 결과는
`verification/real/REAL_RESULTS.md` 에 있고, **PPO는 두 목표 모두 미달**했습니다.
아직 남은 것은 walk-forward 다중 구간 실험입니다.

---

## 어떻게 확인하나요?

### 1단계 — 설치 없이 (1분)

보고서용 숫자가 필요하면:
→ **`verification/real/REAL_RESULTS.md`** (실데이터 결과 + 해석 초안)

코드가 제대로 도는지만 보려면:
→ `verification/synthetic/VERIFICATION.md` (합성 데이터, 성능 근거 아님)

### 2단계 — 테스트 돌려보기 (5분)

```bash
pip install -r requirements-dev.txt
pip install -r backend/requirements.txt
python -m pytest tests/test_performance.py -q
```

80개가 통과해야 합니다. 이 테스트들은 계산이 맞는지를 봅니다. 예를 들어
"첫날 손실이 낙폭에 반영되는가", "MVO가 비중 상한 40%를 지키는가",
"rolling window가 실제로 학습 데이터를 자르는가" 같은 것들입니다.

### 3단계 — 직접 실행 (10분)

**실데이터 결과를 직접 재현** (권장, SHA256 검증 포함):

```bash
python scripts/install_demo.py
export INVESTMENT_AI_HOME="$PWD/runtime"
python -m backend.performance.run_analysis \
  --run runtime/runs/20261028_005508 --output runtime/results \
  --shap-assets SPY QQQ --shap-sample 25 \
  --mvo-strict-window --mvo-frequency month_end
```

모델 없이 배선만 확인 (가짜 데이터, `--config` 를 꼭 붙여야 합니다):

```bash
python -m backend.performance.run_analysis \
  --config config/dummy_only.json --smoke --skip-shap
```

---

## 담당자별로 필요한 것

### 🤖 투자 AI 담당 → 저에게

**이미 `shared/investment_demo.zip` 의 모델(`20261028_005508`)로 1차 분석을
끝냈습니다.** 추가로 모델을 주실 필요는 당장 없습니다.

대신 두 가지 확인 부탁드립니다.

1. **학습 스텝이 5,000입니다.** 과제 요구는 100,000이라 **1/20 수준**입니다.
   PPO가 보수적으로 학습된 주된 원인으로 의심됩니다. 재학습이 가능할까요?
2. **walk-forward 다중 구간**을 제가 돌리면 window×reward×seed 만큼 재학습합니다.
   `total_timesteps` 를 얼마로 잡는 게 현실적일까요?

새 `runs/<id>/` 를 주실 때 필요한 파일은 이렇습니다 (추가 작업 없이 그대로 읽습니다):

```
runs/<id>/
├── selection.json          ← 선택된 reward, seed 목록
├── splits.json             ← test_start_index
├── validation.csv          ← Reward ANOVA 표본 (일별 기록 없으면 이걸로 대체)
├── validation_models/<reward>/fold_*_seed_*/validation_history.csv
│                           ← 있으면 더 정밀한 Reward ANOVA 가능
└── final/seed_*/{model.zip, config.json, schema.json, scaler.json}
```

### 📊 전처리 담당 → 저에게

**지금 가장 급한 요청입니다.**

과제 예시는 "4년 학습 → 1년 테스트"인데, 현재 데이터가 2023–2025년
**752 거래일(약 3년)** 이라 4년 학습 자체가 불가능합니다.
window 3개를 만들려면 1008 + 252×3 = **1764 거래일(약 7년)** 이 필요합니다.

→ **2019년 1월부터의 가격 데이터를 받을 수 있을까요?**
   (ETF 15종 모두 2019년 이전부터 존재합니다)

못 받으면 `--auto-windows` 로 학습 1.5년 / 테스트 6개월로 줄여서 돌리고,
"4년 학습 요건을 못 맞춘 이유"를 보고서에 적어야 합니다.

### 📰 뉴스 AI 담당 → 저에게

CONTRACT_V3의 6열(`date, ticker, risk_score, risk_type, has_news, days_since_news`)만
지켜 주시면 자동으로 반영됩니다. 제 쪽에서 두 가지를 분석합니다:

- SHAP에서 `리스크 태그 점수` 피처의 중요도 순위
- 뉴스 유/무 성능 차이 (`results/metrics/risk_tag_effect.csv`)

### 🌐 웹 담당 → 제가 드리는 것

FastAPI 라우터는 **이미 등록해 뒀습니다** (`backend/investment_ai/api.py` 의
`create_app()`). 모듈이 없는 환경에서도 서버가 뜨도록 선택적 등록(try/except)으로
해 뒀습니다.

Docker도 맞춰 뒀습니다 (`Dockerfile`, `Dockerfile.investment`, `.dockerignore`):

- API 컨테이너에는 **무거운 분석 의존성(scipy/shap/matplotlib)을 설치하지 않습니다.**
  `/performance` 라우터는 결과 JSON만 읽기 때문입니다. 이미지가 커지지 않습니다.
- **분석은 컨테이너 밖에서 돌리고 `results/` 를 마운트**하는 구조입니다.
  `RESULTS_DIR` 기본값은 `Dockerfile` 이 `/results`,
  `Dockerfile.investment` 가 `/runtime/results`(이미 바인드 마운트되는 경로)입니다.

| 구성 | `RESULTS_DIR` | 분석 `--output` |
| --- | --- | --- |
| `docker-compose.yml` | `/results` | `--output results` + `./results:/results:ro` 마운트 |
| `docker-compose.investment.yml` | `/runtime/results` | `--output runtime/results` |

**`--output` 경로와 `RESULTS_DIR` 이 어긋나면** 분석이 성공해도
`/performance/health` 가 `ready: false` 로 나옵니다. 위 표대로 맞춰 주세요.

결과가 아직 없으면 `/performance/health` 가 `ready: false` 를 돌려주고
나머지 엔드포인트는 503을 냅니다. 서버가 죽지는 않습니다.

엔드포인트는 **5개**입니다.

| 엔드포인트 | 내용 |
| --- | --- |
| `GET /performance/health` | 결과 파일 준비 여부 (`ready: true/false`) |
| `GET /performance/backtest` | 전략별 12개 지표 + 일별 경로 + 국면 라벨 |
| `GET /performance/explain` | SHAP 결정 해석 |
| `GET /performance/statistics` | ANOVA·Tukey 결과 |
| `GET /performance/summary` | 한 장 요약 + 해석 문장 |

쿼리 파라미터:

- `/performance/backtest?strategy=ppo` — 한 전략만 (요약·일별 경로 둘 다 필터링)
- `/performance/backtest?include_path=false` — 요약만 (가벼움)
- `/performance/explain?ticker=QQQ&date=2025-09-30` — 특정 자산·날짜

Streamlit 쪽은 `frontend/fixtures/shap.json` 과 **같은 스키마**로 맞춰 뒀습니다.
`results/shap/shap.json` 으로 파일만 교체하면 기존 SHAP 화면이 그대로 동작합니다.

화면에 뭘 올릴지 추천:

- **성과 페이지**: `cumulative_return.png` 쪽 데이터 (PPO·MVO·동일가중·S&P500 지수)
- **위험 페이지**: `drawdown.png`, 벤치마크 2종을 포함한 `cumulative_return_all.png`
- **ANOVA 페이지**: `statistics/anova.csv` + `interpretation.txt` 의 해석 문장
  (문장은 그대로 화면에 띄울 수 있게 한국어로 써 뒀습니다)

> 벤치마크가 3개(`benchmark`, `benchmark_raw`, `market_index`)라 헷갈릴 수 있습니다.
> **첫 화면에는 `market_index`(순수 S&P500 지수)만** 쓰고, 나머지 둘은 상세
> 위험 페이지에 두는 걸 권합니다. 차이는 README에 표로 정리돼 있습니다.

---

## 지금 막힌 것 / 팀 결정이 필요한 것

| 항목 | 상태 | 필요한 결정 |
| --- | --- | --- |
| 4년 학습 데이터 | 752일(3년)뿐 | 2019년부터 받을지, 아니면 축소하고 보고서에 명시할지 |
| 실데이터 백테스트 (단일 구간) | **완료** | — |
| walk-forward 다중 구간 | 경로 검증 완료 | 정식 조건(4년 학습·100k 스텝)으로 재실행만 남음 |
| PPO가 더 나은지 | **아님** | 두 목표 모두 미달. 다만 위험지표는 시장의 절반 |
| 학습 스텝 5,000 | 과제 요구 100,000 | 투자 AI 담당과 재학습 협의 |
| MVO 252일 공분산 | 기본은 완화 | 정식 제출엔 `--mvo-strict-window` 권장 |
| MVO 리밸런싱 시점 | 현재 월말(`month_end`) | 과제 문구가 "매월 첫 거래일"이면 `month_start` 로 바꿔 재실행 필요 |
| 12개 지표 구성 | 과제 문서 기준 | 교수님 채점표와 한 번 대조 필요 |

---

## 솔직하게 말씀드리는 한계

보고서 쓸 때 반드시 들어가야 하는 내용입니다.

1. **실데이터 결과는 단일 구간(7개월 상승장)입니다.** 다중 구간 walk-forward가
   아니므로 하락장 방어력은 평가되지 않았습니다. 합성 데이터 실행 결과는
   `api/backtest.json` 의 `smoke` 플래그로 구분됩니다.
2. **벤치마크에도 Safe-Guard가 걸립니다.** S&P500이 15% 떨어지면 현금화되어
   그 뒤로 수익률이 0이 됩니다. 그래서 순수 지수(`market_index`)를 따로 만들어
   Alpha·Beta·국면 분류에 씁니다.
3. **walk-forward를 이어붙인 경로는 "한 계좌로 쭉 투자한 기록"이 아닙니다.**
   window마다 자본이 1.0으로, 보유가 전액 현금으로 초기화됩니다.
4. **같은 날 전략들의 수익률은 서로 독립이 아닙니다.** 그래서 ANOVA와 함께
   대응표본 검정과 블록 부트스트랩을 `statistics/paired.csv` 에 넣었습니다.
   **4개 쌍을 비교하므로 Holm 다중비교 보정도 함께 기록됩니다. 보고서에는
   보정 후 p값을 인용하세요** (보정 전 0.038 → 보정 후 0.152).
5. **SHAP은 인과관계가 아닙니다.** 어떤 피처가 모델의 비중 결정에 기여했는지를
   보여줄 뿐, 그게 수익을 냈다는 증거가 아닙니다.
6. **`p >= 0.05`는 "차이가 없다"는 증명이 아닙니다.** "현재 자료로는 차이를
   확인할 증거가 부족하다"는 뜻입니다.

---

질문 있으면 편하게 물어보세요. 코드 수정 없이 답할 수 있는 건
대부분 `README.md` 에 적어 뒀습니다.
