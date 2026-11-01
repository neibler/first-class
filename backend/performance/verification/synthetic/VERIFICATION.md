# 합성 데이터 검증 로그

**코드가 돌아간다는 증거입니다. 성능 근거가 아닙니다.**
실제 숫자는 `../real/REAL_RESULTS.md` 를 보세요.

생성: 2026-11-01 06:37 UTC · Python 3.13.18
패키지: numpy==2.5.3, pandas==3.0.5, scipy==1.18.1, statsmodels==0.15.0, shap==0.54.1, matplotlib==3.11.2, torch==2.15.0, stable-baselines3==2.9.2, gymnasium==1.4.0

## 실행 명령
```bash
python -m backend.performance.run_analysis \
  --config config/dummy_only.json --smoke \
  --shap-assets SPY --shap-sample 8 --output results-smoke
```

## pytest / lint
```
80 passed, 2 warnings in 12.05s
14 files would be left unchanged.
flake8: 문제 없음
```

## 12개 지표 (합성 데이터 · 더미 정책)
```
                days  cumulative_return    cagr  annualized_volatility  var_95  cvar_95  max_drawdown  sharpe  sortino  calmar   alpha    beta  information_ratio
strategy
ppo            157.0            -0.1209 -0.1869                 0.1450  0.0173   0.0218        0.1635 -1.3537  -1.8122 -1.1429 -0.1279  0.4312            -0.2048
mvo            157.0            -0.0218 -0.0348                 0.0454  0.0052   0.0066        0.0435 -0.7579  -1.0107 -0.8005 -0.0466 -0.0768             0.3750
equal_weight   157.0            -0.1152 -0.1783                 0.1372  0.0179   0.0207        0.1515 -1.3629  -1.8318 -1.1771 -0.1217  0.4112            -0.1505
benchmark      157.0            -0.1345 -0.2069                 0.2155  0.0298   0.0408        0.1873 -0.9677  -1.2779 -1.1048 -0.1296  0.4976            -0.2306
benchmark_raw  157.0            -0.1114 -0.1727                 0.3052  0.0344   0.0445        0.3077 -0.4687  -0.6332 -0.5612  0.0153  0.9984             1.2669
market_index   157.0            -0.1200 -0.1855                 0.3055  0.0344   0.0445        0.3077 -0.5193  -0.7009 -0.6030     NaN     NaN                NaN
```

## ANOVA
```
                   label                 term  f_statistic  p_value  eta_squared  significant
검증 2: PPO vs MVO vs 동일가중 equal_weight+mvo+ppo       0.1903   0.8282       0.0178        False
        검증 3: 전략 × 시장 국면             strategy       0.1730   0.8424       0.0178        False
        검증 3: 전략 × 시장 국면               regime       0.0481   0.9531       0.0050        False
```
