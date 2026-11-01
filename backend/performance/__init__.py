"""성능 분석 담당 모듈.

과제 산출물 -> 파일 매핑
    1  Walk-Forward Backtest      walk_forward.py
    2  PPO 백테스트               strategies.py (investment_ai.backtest 재사용)
    3  MVO 전략                   mvo.py + strategies.run_mvo
    4  동일가중 전략              strategies.run_strategy("equal_weight")
    5  벤치마크 비교              strategies.run_strategy("benchmark")
    6  12개 성과지표              metrics.py
    7  누적수익률 그래프          plots.cumulative_plot
    8  Drawdown 분석              plots.drawdown_plot / strategies.drawdown_frame
    9  시장 국면별 분석           market_regime.py
    10 Reward 3종 비교            walk_forward.aggregate + plots.reward_comparison_plot
    11 SHAP Summary Plot          shap_analysis.save_summary_plot
    12 SHAP Force Plot            shap_analysis.save_force_plot
    13 뉴스 Risk Tag 효과         risk_tag.py
    14 One-way ANOVA 2개          anova.run_with_posthoc
    15 Two-way ANOVA 1개          anova.two_way
    16 Effect Size                anova.effect_label / eta_squared
    17 Tukey HSD                  anova.tukey_hsd
    18 통계 결과 해석             anova.interpret / run_analysis.two_way_interpretation
    저장·API 연결                 report.py, api.py
    전체 실행                     run_analysis.py
"""

__all__ = [
    "anova",
    "api",
    "market_regime",
    "metrics",
    "mvo",
    "plots",
    "report",
    "risk_tag",
    "shap_analysis",
    "strategies",
    "walk_forward",
]
