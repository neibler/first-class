# 성능 분석 (backend/performance)

투자 AI(PPO)가 실제로 좋은 전략인지 검증하고, 왜 그렇게 결정했는지 설명하고,
전략 간 차이가 통계적으로 의미 있는지 분석하는 모듈입니다.

## 설치

```bash
pip install -r backend/requirements.txt   # performance/requirements.txt 포함
```

추가로 쓰는 라이브러리: scipy, statsmodels, shap, matplotlib, scikit-learn.

## 실행

> **설정 파일은 항상 명시하세요.** 생략하면 `backend/config.json` 을 찾는데
> 레포에는 그 파일이 없어 `FileNotFoundError` 로 끝납니다.
> `--run` 을 쓸 때는 그 run 의 `config.json` 을 자동으로 읽으므로 불필요합니다.

```bash
# 1) 학습이 끝난 run 을 재사용 — 단일 홀드아웃 평가 (현재 쓰는 방식)
python scripts/install_demo.py                 # SHA256 검증 후 runtime/ 에 설치
export INVESTMENT_AI_HOME="$PWD/runtime"
python -m backend.performance.run_analysis \
  --run runtime/runs/20261028_005508 \
  --output runtime/results

# 2) Walk-forward 재학습부터 전부 (과제 요구사항. window x reward x seed 회 학습)
export INVESTMENT_AI_HOME="$PWD/runtime"
python -m backend.performance.run_analysis \
  --config config/real_market.json --walk-forward \
  --output runtime/results

# 3) 데이터가 7년 미만이면 자동 축소 + 경고
python -m backend.performance.run_analysis \
  --config config/real_market.json --walk-forward --auto-windows \
  --output runtime/results

# 4) 모델 없이 배선만 점검 (더미 데이터 + 더미 정책, 1분 내외)
python -m backend.performance.run_analysis \
  --config config/dummy_only.json --smoke --skip-shap \
  --output results-smoke
```

`--config` 로 쓸 수 있는 파일

| 파일 | 용도 |
| --- | --- |
| `config/real_market.json` | 실제 `data/market.parquet` + 뉴스 CSV |
| `config/dummy_only.json` | 합성 데이터 (`news_mode=development`) |

`INVESTMENT_AI_HOME` 은 `data_path` / `news_path` 의 기준 경로입니다.
`scripts/install_demo.py` 가 데이터와 모델을 `runtime/` 에 넣으므로 그쪽을 가리킵니다.

자주 쓰는 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--train-days` | `1008` | 학습 거래일 (과제 예시의 **4년**) |
| `--test-days` | `252` | 테스트 거래일 (**1년**) |
| `--folds` | `3` | window 개수 |
| `--rolling` | off | 확장 대신 고정 길이 롤링 윈도우 |
| `--auto-windows` | off | 데이터 부족 시 자동 축소 + 경고 기록 |
| `--main-reward` | config 값 | 최종 비교에 쓸 reward |
| `--mvo-objective` | `max_sharpe` | `min_variance` 선택 가능 |
| `--mvo-frequency` | `monthly` | `quarterly`(근사) / `month_end`·`quarter_end`(달력 말일) / `month_start`·`quarter_start`(달력 첫 거래일) |
| `--output` | `results` | 결과 폴더. Docker 로 서빙하려면 `runtime/results` 권장 |
| `--mvo-strict-window` | off | 과거 252일이 모이기 전에는 동일가중 유지 |
| `--regime-window / --regime-threshold` | `63 / 0.05` | 국면 분류 기준 |
| `--anova-frequency` | `ME` | ANOVA 표본 주기 (표본이 적으면 `W`) |
| `--shap-assets / --shap-sample` | 전체 / `40` | SHAP 대상 자산·표본 수 |
| `--shap-force-dates` | 자동 | Force Plot 날짜 (기본: 변화가 가장 큰 2일) |
| `--risk-tag` | `ablation` | `retrain`, `none` 선택 가능 |

### Walk-forward 구간

기본값은 과제 예시와 같은 **4년 학습 / 1년 테스트 / 확장 윈도우**입니다.

```
Window 1: 2019-01-02 ~ 2022-11-11 학습  ->  2022-11-14 ~ 2023-10-31 테스트
Window 2: 2019-01-02 ~ 2023-10-31 학습  ->  2023-11-01 ~ 2024-10-17 테스트
Window 3: 2019-01-02 ~ 2024-10-17 학습  ->  2024-10-18 ~ 2025-10-06 테스트
```

각 window 에서 PPO 를 **처음부터 다시 학습**합니다 (resume 아님).
3개 window 에는 1008 + 252×3 = **1764 거래일(약 7년)** 이 필요합니다.

> **우리 데이터는 2023–2025, 752 거래일(약 3년)입니다.** 그대로는 기본 구간을
> 쓸 수 없고, 실행하면 필요 일수를 알려주며 거부합니다. `--auto-windows` 를
> 붙이면 window 3개를 유지한 채 학습 1.5년 / 테스트 6개월로 줄이고, 무엇을
> 줄였는지 `results/metrics/walk_forward_plan.json` 의 `warnings` 에 남깁니다.
> **그 경고 문구는 보고서의 한계 항목에 그대로 옮겨 적어야 합니다.**
> 과제 예시대로 4년 학습을 하려면 전처리 담당에게 2019년부터의 데이터를
> 요청하세요 (ETF 15종 모두 2019년 이전부터 존재합니다).

## 산출물

```
results/
├── metrics/      strategy_comparison.csv, metrics.csv, walk_forward.csv,
│                 market_regime.csv, reward_comparison.csv, risk_tag_effect.csv
├── backtest/     전략별 *_history.csv, daily_returns.csv, portfolio_value.csv,
│                 drawdown.csv, cumulative_return.png (보고서 첫 장: 전략 3종
│                 + 지수), cumulative_return_all.png (벤치마크 2종 포함),
│                 drawdown.png, market_regime.png, reward_comparison.png,
│                 walk_forward.png
├── shap/         shap.json, summary.png, force.png, force_2.png,
│                 feature_importance.csv
├── statistics/   anova.csv, anova.json, tukey.csv, paired.csv, paired.json,
│                 interpretation.txt, risk_tag_effect.json
├── api/          backtest.json, explain.json  (웹 담당 전달용)
└── summary.json  한 장 요약
```

## 과제 산출물 ↔ 코드

| # | 산출물 | 위치 |
| --- | --- | --- |
| 1 | Walk-Forward Backtest | `walk_forward.py` |
| 2 | PPO 백테스트 | `strategies.run_strategy("ppo")` |
| 3 | MVO 전략 | `mvo.py`, `strategies.run_mvo` |
| 4 | 동일가중 전략 | `strategies.run_strategy("equal_weight")` |
| 5 | 벤치마크 비교 | `strategies.run_strategy("benchmark")` + `"benchmark_raw"` |
| 6 | 12개 성과지표 | `metrics.performance_metrics` |
| 7 | 누적수익률 그래프 | `plots.cumulative_plot` |
| 8 | Drawdown 분석 | `plots.drawdown_plot`, `strategies.drawdown_frame` |
| 9 | 시장 국면별 분석 | `market_regime.py` |
| 10 | Reward 3종 비교 | `walk_forward.aggregate`, `plots.reward_comparison_plot` |
| 11 | SHAP Summary Plot | `shap_analysis.save_summary_plot` |
| 12 | SHAP Force Plot | `shap_analysis.save_force_plot` |
| 13 | 뉴스 Risk Tag 효과 | `risk_tag.py` |
| 14 | One-way ANOVA 2개 | `anova.run_with_posthoc` (Reward 3종 / 전략 3종) |
| 15 | Two-way ANOVA 1개 | `anova.two_way` (전략 × 국면) |
| 16 | Effect Size | `anova` 의 `eta_squared`, `partial_eta_squared` |
| 17 | Tukey HSD | `anova.tukey_hsd` (p < 0.05 일 때 자동 수행) |
| 18 | 통계 결과 해석 | `anova.interpret`, `statistics/interpretation.txt` |

## 설계 결정 (보고서에 그대로 쓸 수 있는 근거)

**모든 전략을 같은 환경에서 돌립니다.** PPO·MVO·동일가중·벤치마크 전부
`investment_ai.PortfolioEnv` 를 거칩니다. 거래비용 7bps, t 관측 → t+1 종가 체결,
누적 MDD 15% Safe-Guard 가 모두 동일하게 적용되어야 비교가 공정합니다.

**Walk-forward 결과는 구간을 이어붙여 평가합니다.** 각 window 의 모델은 자기
테스트 구간만 담당하고, 그 구간들을 시간순으로 이어붙인 것이 최종 비교 대상입니다
(`results/api/backtest.json` 의 `evaluation` 이
`walk_forward_stitched_out_of_sample`). 마지막 window 의 모델을 전 기간에 적용하면
그 모델이 학습에 쓴 구간까지 평가에 섞이기 때문입니다. 이어붙일 때 일별 수익률만
연결하고 누적가치·MDD 는 전체 구간에서 다시 계산하므로, **구간별 MDD
(`walk_forward_metrics.csv`)와 전체 MDD(`metrics.csv`)는 다릅니다.** 구간을 넘는
낙폭이 반영되기 때문이며 보고서에는 둘을 구분해 적어야 합니다.

**시장 기준은 세 단계로 나눠 보고합니다.**

| 열 | 내용 | 쓰임 |
| --- | --- | --- |
| `benchmark` | SPY Buy & Hold + Safe-Guard | 전략과 완전히 같은 조건의 대조군 |
| `benchmark_raw` | SPY Buy & Hold, Safe-Guard 없음 (**비용·체결지연 포함**) | 실제로 투자 가능한 시장 수익률 |
| `market_index` | SPY 지수 수익률 자체 (비용·체결지연 없음) | **Alpha·Beta·IR·국면 분류 기준** |

`benchmark_raw` 는 안전장치만 끈 것이라 여전히 첫 진입 수수료(7bps)와
t 관측 → t+1 체결 규칙의 영향을 받습니다. 실제로 백테스트 첫날 수익률이
정확히 −0.0007(= 수수료)로 찍힙니다. 그래서 "S&P500 을 이겼는가"를 지수 기준으로
말하려면 `market_index` 를 봐야 합니다. 검증용으로 `benchmark_raw` 의 β 를 재보면
지수 대비 0.998 로 나옵니다.

국면 분류도 `market_index` 기준입니다. Safe-Guard 가 걸린 벤치마크로 분류하면
하락 후 현금 구간(수익률 0)이 "횡보장"으로 잘못 분류됩니다.

**Rolling 윈도우는 학습 데이터를 실제로 잘라서 넘깁니다.** `train_model` 은
학습 시작점을 인자로 받지 않아(`PortfolioEnv` 의 start 가 lookback-1 로 고정)
`--rolling` 만으로는 모델이 여전히 가장 이른 날짜부터 학습합니다. 그래서
`training_view()` 가 `[train_start, train_end]` 구간만 남긴 `MarketData` 사본을
만들어 전달합니다. 투자 AI 쪽 코드는 건드리지 않았습니다.

**Drawdown 의 고점에는 최초 투자원금이 포함됩니다.** 그러지 않으면 첫 거래일에
손실이 나도 그날 종가가 그때까지의 최고값이 되어 낙폭이 0 으로 계산되고,
`metrics.csv` 의 MDD 와 `drawdown.csv`·그래프가 서로 어긋납니다.

**목표 판정은 지표마다 비교 대상이 다릅니다.**

| 지표 | 비교 대상 | 목표 |
| --- | --- | --- |
| Sharpe Ratio | **동일가중** | +0.2 이상 |
| 누적수익률 | **S&P500** (`benchmark_raw`) | +10%p 이상 |

둘 다 시장과 비교하면, 동일가중보다 못한 Sharpe 도 "달성"으로 잘못 판정됩니다
(실제 실행에서 시장 기준으로는 +0.28 "달성", 동일가중 기준으로는 −0.11 "미달"로
결론이 뒤집혔습니다). `targets.json` 에는 다른 기준 대비 차이도 `difference_vs`
로 함께 기록하니 보고서에서 원하는 쪽을 인용하면 됩니다.

**ANOVA 의 독립성 한계를 대응 비교로 보완합니다.** PPO·MVO·동일가중은 같은 날
같은 시장을 겪으므로 서로 독립이 아닙니다. One-way/Two-way ANOVA 는 과제 요구대로
그대로 수행하되, `statistics/paired.csv` 에 **대응표본 t-검정 + Wilcoxon +
이동 블록 부트스트랩 95% 신뢰구간 + Holm 다중비교 보정**을 함께 저장합니다.
한 자료에서 여러 쌍을 비교하면 우연히 하나가 유의해질 확률이 올라가므로,
**보고서에는 보정 후 p값(`p_value_holm`)을 인용해야 합니다.** 각 ANOVA 결과의
`independence` 필드에는 그룹별 표본 수, 겹치는 날짜 수, 중복 행 수가 들어갑니다.
`p >= 0.05` 는 "같다"는 증명이 아니라 "차이를 확인할 증거가 부족하다"는 뜻이며,
해석 문장에도 그렇게 적혀 나옵니다.

**Alpha/Beta 기준은 전 구간에서 하나로 통일했습니다.** 전체 지표도, walk-forward
구간별 지표도 모두 `market_index`(지수 수익률)를 씁니다. 구간별 지표에는 실제로
어떤 기준을 썼는지 `market_reference` 열로 기록됩니다.

**Two-way ANOVA 의 유의한 주효과에도 Tukey HSD 를 수행합니다.** 국면 요인이
유의하면 어느 국면끼리 다른지까지 `statistics/tukey.csv` 에 남습니다
(`factor` 열로 One-way 결과와 구분됩니다).

**reward 선택은 테스트 구간으로 하지 않습니다.** 최종 비교에 쓰는 reward 는
config(또는 투자 AI 담당의 `selection.json`, 즉 validation 결과)에서 가져옵니다.
`results/metrics/reward_ranking.csv` 에 테스트 구간 기준 순위도 저장하지만, 그것으로
reward 를 고르면 홀드아웃을 선택에 쓰는 것이라 선택 편향이 생깁니다. 서술용입니다.

**12개 지표의 정의**는 `metrics.py` 의 docstring 에 적혀 있습니다. Sharpe 는 투자 AI
담당의 `accounting.sharpe_ratio` 를 그대로 호출해 두 모듈의 숫자가 갈리지 않게
했습니다 (ddof=1, √252, 연 rf → 일 rf 는 `expm1(log1p(rf)/252)`).
VaR·CVaR·MDD 는 **손실 크기를 양수로** 보고합니다.

**MVO 는 과제 제약을 그대로 구현**했습니다: 과거 252 거래일 공분산, 비중 합 1,
공매도 금지, 개별 상한 40%, SLSQP. 결정일 종가까지의 데이터만 써서 미래 정보가
들어가지 않습니다. 최적화가 실패하면 조용히 틀린 값을 내지 않고 동일가중으로
되돌립니다.

**시장 국면**은 벤치마크의 과거 63거래일 누적수익률이 +5% 이상이면 상승장,
−5% 이하면 하락장, 그 사이는 횡보장입니다. 과거만 보므로 전략 수익률과 같은
시점에서 비교할 수 있습니다.

**SHAP 은 관측 363차원을 6개 해석 피처로 묶어** 자산별로 계산합니다
(최근 5일 수익률 / RSI / MACD / 20일 변동성 / 현재 비중 / 리스크 태그 점수).
출력은 해당 자산의 목표 비중이라 SHAP +0.02 는 "비중 +2%p 기여"로 읽습니다.
결과 JSON 은 `frontend/fixtures/shap.json` 과 같은 스키마여서 프론트는 파일만
바꾸면 됩니다.

**ANOVA 표본은 겹치지 않는 월별 수익률**입니다. 일별 수익률은 자기상관이 강해
독립성 가정이 깨지기 때문입니다. 표본이 모자라면 `--anova-frequency W` 로
주별로 바꾸거나 walk-forward window 를 늘리세요.

## 보고서에 꼭 적어야 할 한계

1. **벤치마크에도 Safe-Guard 가 적용됩니다.** 벤치마크 MDD 가 15%를 넘으면 전량
   현금화되어 이후 수익률이 0 이 됩니다. 순수 Buy & Hold 지수와는 다른 값이므로,
   `results/api/backtest.json` 의 `turnover.safeguard_triggered` 를 확인하고
   발동했다면 보고서에 명시하세요.
2. **ANOVA 의 독립성·정규성 가정**은 금융 수익률에서 완전히 만족되지 않습니다.
   `assumptions` 에 Shapiro-Wilk·Levene p-value 와 비모수 Kruskal-Wallis 결과를
   함께 저장하니 같이 보고하세요.
3. **SHAP 피처 값은 전처리(scaler) 공간 기준**이며 원 단위(%, RSI 0–100)가 아닙니다.
   또한 기여도는 상관이지 인과가 아니고 투자 권고도 아닙니다.
4. **risk tag `ablation` 은 재학습이 아닙니다.** 학습된 정책이 뉴스 입력에
   반응하는지를 볼 뿐입니다. "뉴스 AI 를 붙이면 성능이 오르는가"에 답하려면
   `--risk-tag retrain` 을 쓰세요 (대신 학습 시간이 2배 듭니다).
5. **결과가 유의하지 않아도 그대로 보고합니다.** 과제는 `p < 0.05` 를 요구하지
   않습니다. `interpretation.txt` 가 효과 크기와 비유의 원인을 문장으로 만들어 둡니다.
6. **MVO 워밍업 구간.** 기본값은 과거 252일이 안 모이면 가용 기간으로 공분산을
   추정합니다. 교수님이 252일을 엄격히 요구하면 `--mvo-strict-window` 로 워밍업
   동안 동일가중을 유지하고, 그 사실을 보고서에 적으세요. **정식 제출에는 이
   옵션을 켜는 쪽을 권합니다.**
7. **MVO 리밸런싱 주기.** `monthly`/`quarterly` 는 21·63 거래일 **근사치**라
   실제 월말과 어긋납니다. 달력 기준이 필요하면 `--mvo-frequency month_end`
   (또는 `quarter_end`) 를 쓰세요. 실제 각 월의 마지막 거래일에만 거래하며,
   거래하지 않는 날에는 비용도 발생하지 않습니다.
8. **12개 지표 구성을 교수님 채점 기준과 대조하세요.** 현재는 과제 문서에 적힌
   누적수익률·CAGR·변동성·VaR·CVaR·MDD·Sharpe·Sortino·Calmar·Alpha·Beta·IR 입니다.
   승률이나 회복기간이 필요하면 `metrics.py` 에 추가하면 됩니다.
9. **window 경계마다 진입 비용이 다시 발생합니다.** 각 window 는 전액 현금에서
   시작하므로 첫 진입 수수료가 구간마다 붙습니다. 네 전략 모두 같은 조건이라
   비교는 공정하지만, 단일 Buy & Hold 보다 비용이 많습니다.
10. **Safe-Guard 는 window 마다 초기화됩니다.** 재학습 시 새 에피소드가 시작되기
    때문입니다. 한 번 발동하면 해당 구간 끝까지 현금이고, 다음 구간에서 복귀합니다.
11. **이어붙인 경로는 "한 계좌로 쭉 투자한 기록"이 아닙니다.** window 마다 자본이
    1.0, 보유가 전액 현금, Safe-Guard 상태가 0 으로 초기화되고 진입 비용이 다시
    발생합니다. 구간별로 모델을 독립 검증한 결과를 시간순으로 이은 것이며,
    연속 실계좌 운용을 주장하려면 구간 전환 시 기존 보유·비용·안전장치 상태를
    어떻게 넘길지 따로 정의해야 합니다.
12. **SHAP 은 인과가 아닙니다.** 피처가 모델의 목표 비중에 얼마나 기여했는지를
    보여줄 뿐, 그 피처가 수익을 냈다는 증거가 아닙니다. Force Plot 은 변화가 큰
    2일만 골라 생성하니, 보고서에는 그 날의 시장 상황과 함께 해석하세요.
13. **뉴스 효과는 `--risk-tag retrain` 으로 증명하세요.** 기본값 `ablation` 은
    "학습된 모델이 뉴스 입력에 반응하는가"만 봅니다. "뉴스를 넣으면 성과가
    좋아지는가"에 답하려면 같은 조건에서 뉴스 유/무 모델을 각각 재학습해
    여러 seed 로 비교해야 합니다.

## 실데이터 결과

**`verification/real/REAL_RESULTS.md` 가 보고서에 들어갈 실제 숫자입니다.**
`shared/investment_demo.zip` 의 실제 시장·뉴스 데이터와 학습된 모델
(`20261028_005508`)로 돌린 1차 결과이며, 해석 초안과 다음 실험 계획까지 들어 있습니다.

요약: **PPO는 두 목표 모두 미달**했지만 변동성·MDD·Beta가 시장의 절반 수준입니다.
다만 대응 비교에 Holm 다중비교 보정을 적용하면 어떤 쌍도 유의하지 않으므로
(PPO vs 동일가중: 보정 전 p=0.038 → 보정 후 0.152), "통계적으로 열등하다"고는
말할 수 없습니다. 학습 스텝이 과제 요구(100,000)의 1/20인 5,000이라는 점이
가장 큰 의심 요인입니다.

재현 (SHA256 검증 포함):

```bash
python scripts/install_demo.py
export INVESTMENT_AI_HOME="$PWD/runtime"
python -m backend.performance.run_analysis \
  --run runtime/runs/20261028_005508 --output results \
  --shap-assets SPY QQQ --shap-sample 25 \
  --mvo-strict-window --mvo-frequency month_end
```

## 검증 증거

검증 자료는 두 벌입니다.

- `verification/real/REAL_RESULTS.md` — **실데이터 결과**. 보고서에 쓸 숫자와
  해석 초안, 통계 해석 주의사항, 다음 실험 계획.
- `verification/synthetic/VERIFICATION.md` — 합성 데이터로 코드 동작만 확인한 로그.
  성능 근거가 아닙니다.

## 검증 상태

| 항목 | 상태 |
| --- | --- |
| 단위 테스트 80개 | 통과 |
| black / flake8 | 통과 |
| 실제 PPO 학습 연동 (`--walk-forward`) | **합성 데이터로 검증 완료** — torch + stable-baselines3 로 PPO 9회 학습 |
| 실제 PPO 모델 로딩 (`--run`) | **합성 데이터로 검증 완료** — `runs/<id>` 생성 후 전 분석 수행 |
| SHAP Summary / Force Plot | **실제 PPO 정책으로 생성 확인** |
| ANOVA 3종 (두 실행 모드 모두) | 생성 확인 |
| 팀 실데이터 백테스트 (단일 홀드아웃) | **완료** — `verification/real/REAL_RESULTS.md` |
| Walk-forward 다중 구간 실데이터 | **경로 검증 완료** (3 window, PPO 18회 재학습) — 정식 조건(4년/100k 스텝)은 미충족 |
| PPO 가 MVO·동일가중보다 낫다는 증거 | **없음** — 두 목표 모두 미달 (다중비교 보정 후 통계적 유의성은 없음) |
| 4년 학습 구간 | 데이터 길이 부족 (2019년부터의 데이터 필요) |

합성 데이터 실행은 **배선과 연동이 올바른지**를 증명할 뿐, 성능 근거가 아닙니다.
`api/backtest.json` 의 `smoke` 플래그로 더미 정책 실행과 구분됩니다.

**결과가 "PPO < MVO" 로 나와도 그대로 보고하세요.** 학습량이 적거나 데이터가
짧으면 흔한 결과이고, 원인을 분석하는 것이 과제의 요구사항(23번)입니다.
`results/metrics/targets.json` 이 Sharpe +0.2 / 누적수익률 +10%p 달성 여부를
`benchmark_raw` 대비로 판정해 둡니다.

## 다른 담당자와의 연결

- 전처리/투자 AI → 성능 분석: `runs/<id>/` 의 `selection.json`, `splits.json`,
  `final/seed_*/{model.zip, config.json, schema.json, scaler.json}` 만 있으면 됩니다.
- 뉴스 AI → 성능 분석: 계약의 뉴스 6열이 들어오면 `risk_score` 가 관측에 포함되고,
  `risk_tag.py` 와 SHAP 의 "리스크 태그 점수" 피처가 그 효과를 측정합니다.
- 성능 분석 → 웹: `backend/main.py` 에 라우터 한 줄만 추가하면 됩니다.

`backend/investment_ai/api.py` 의 `create_app()` 에 이미 등록해 두었습니다
(뉴스 라우터 바로 다음). 따로 손댈 것은 없고, 결과 폴더만 `RESULTS_DIR`
환경변수로 알려주면 됩니다 (기본 `results`).

| 엔드포인트 | 내용 |
| --- | --- |
| `GET /performance/backtest` | 전략별 12개 지표 + 일별 경로(+국면 라벨). `?strategy=ppo` 는 요약과 일별 경로를 함께 걸러 줍니다 |
| `GET /performance/explain?ticker=QQQ&date=2025-09-30` | SHAP 결정 해석 |
| `GET /performance/statistics` | ANOVA·Tukey 결과 전체 |
| `GET /performance/summary` 의 `paired_comparisons` | 대응 비교 해석 문장 |
| `GET /performance/summary` | 한 장 요약 |
| `GET /performance/health` | 결과 파일 준비 여부 |

투자 AI 의 기존 `GET /backtest` 는 모델을 직접 호출하는 라이브 엔드포인트라
이름이 겹치지 않도록 `/performance` 아래에 두었습니다.

## Docker

API 컨테이너에는 **무거운 분석 의존성(scipy/statsmodels/shap/matplotlib)을 설치하지
않습니다.** `/performance` 라우터는 결과 JSON 만 읽기 때문입니다.
분석은 컨테이너 밖에서 돌리고 `results/` 를 `RESULTS_DIR` 로 마운트하는 구조입니다.

| 구성 | `RESULTS_DIR` | 분석 실행 시 `--output` |
| --- | --- | --- |
| `docker-compose.yml` | `/results` | `--output results` + `./results:/results:ro` 마운트 |
| `docker-compose.investment.yml` | `/runtime/results` | `--output runtime/results` (추가 마운트 불필요) |

`.dockerignore` 는 `backend/performance/*.py` 만 컨텍스트에 넣습니다 (그림·로그 제외).

> **주의**: `--output` 경로와 `RESULTS_DIR` 이 가리키는 곳이 다르면, 분석은
> 성공했는데 `/performance/health` 가 계속 `ready: false` 로 나옵니다.
> `docker-compose.investment.yml` 은 `/runtime` 이 이미 바인드되므로
> `--output runtime/results` 로 저장하는 쪽이 가장 간단합니다.

라우터는 선택적으로 등록되므로, 모듈이 빠진 이미지에서도 서버는 정상 기동하고
`/performance/*` 만 사라집니다. 다만 **모듈이 있는데 그 안에서 import 가 실패하면
예외를 그대로 올립니다** — 진짜 버그가 조용히 숨지 않도록 하기 위해서입니다.
세 가지 경우 모두 테스트로 고정해 두었습니다:

- `test_performance_api_imports_without_analysis_dependencies` — 무거운 의존성 없이 import
- `test_api_starts_when_the_performance_package_is_absent` — 패키지 없어도 기동
- `test_api_surfaces_errors_from_inside_the_performance_module` — 내부 오류는 노출

CI 의 `docker-smoke` 가 컨테이너 기동 후 `GET /performance/health` 를 확인합니다.

## 테스트

```bash
pip install -r requirements-dev.txt              # pytest 설치
python -m pytest tests/test_performance.py -q    # 80개, 수 초
```
