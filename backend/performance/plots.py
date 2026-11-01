"""Backtest figures: 누적수익률 / drawdown / 국면 / reward 비교 (과제 10·11·12·13번).

웹 담당은 Altair 로 다시 그리므로, 여기서는 보고서·발표 자료에 바로 넣을 수 있는
PNG 를 만든다. 색은 전략별로 고정해 모든 그림에서 같은 전략이 같은 색이 되게 했다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .market_regime import REGIME_LABELS  # noqa: E402
from .strategies import STRATEGY_LABELS, drawdown_frame, value_frame  # noqa: E402

#: 한글 라벨이 깨지지 않도록 설치된 한국어 폰트를 먼저 찾는다.
KOREAN_FONTS = (
    "Malgun Gothic",  # Windows
    "AppleGothic",  # macOS
    "NanumGothic",
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
)


def configure_fonts() -> str | None:
    """사용 가능한 한국어 폰트를 matplotlib 기본값으로 지정한다.

    Returns:
        지정한 폰트 이름. 하나도 없으면 None (제목의 한글이 네모로 보일 수 있으니
        폰트를 설치하거나 라벨을 영문으로 바꾸면 된다).
    """
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


configure_fonts()

COLORS = {
    "ppo": "#C8A24A",
    "mvo": "#2E6F9E",
    "equal_weight": "#6B8F71",
    "benchmark": "#8A8A8A",
    "benchmark_raw": "#4A4A4A",
    "market_index": "#2B2B2B",
}
REGIME_COLORS = {"bull": "#E8F0E4", "bear": "#F6E3E3", "sideways": "#F2F2F2"}


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


#: 보고서 첫 장에 올릴 곡선. Safe-Guard 가 걸린 벤치마크는 혼동을 주므로
#: 상세 위험분석 그림(cumulative_return_all.png)에만 넣는다.
HEADLINE_STRATEGIES = ("ppo", "mvo", "equal_weight", "market_index", "benchmark_raw")


def headline_columns(returns):
    """첫 장 그림에 쓸 열. market_index 가 있으면 benchmark_raw 는 뺀다."""
    columns = [c for c in returns.columns if c in HEADLINE_STRATEGIES]
    if "market_index" in columns and "benchmark_raw" in columns:
        columns.remove("benchmark_raw")
    return columns


def cumulative_plot(returns, path: Path, regimes=None, title="누적 수익률 비교"):
    """전략별 자산가치 곡선. regimes 를 주면 국면을 배경 음영으로 표시한다."""
    values = value_frame(returns)
    figure, ax = plt.subplots(figsize=(10, 5))
    if regimes is not None:
        for _, segment in regimes.iterrows():
            ax.axvspan(
                segment["start"],
                segment["end"],
                color=REGIME_COLORS.get(segment["regime"], "#FFFFFF"),
                zorder=0,
            )
    for column in values.columns:
        ax.plot(
            values.index,
            values[column],
            label=STRATEGY_LABELS.get(column, column),
            color=COLORS.get(column),
            linewidth=1.6,
        )
    ax.axhline(1.0, color="#BBBBBB", linewidth=0.8, linestyle="--")
    _style(ax, title, "포트폴리오 가치 (초기 자본 = 1)")
    return _save(figure, path)


def drawdown_plot(returns, path: Path, title="전략별 Drawdown"):
    drawdowns = drawdown_frame(returns)
    figure, ax = plt.subplots(figsize=(10, 4))
    for column in drawdowns.columns:
        ax.plot(
            drawdowns.index,
            -drawdowns[column],
            label=STRATEGY_LABELS.get(column, column),
            color=COLORS.get(column),
            linewidth=1.3,
        )
    _style(ax, title, "낙폭")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _save(figure, path)


def regime_bar_plot(summary, path: Path, metric="cumulative_return"):
    """국면 × 전략 막대그래프 (과제 12번)."""
    pivot = summary.pivot(index="regime", columns="strategy", values=metric)
    pivot = pivot.reindex([r for r in REGIME_LABELS if r in pivot.index])
    figure, ax = plt.subplots(figsize=(8, 4.5))
    pivot.rename(index=REGIME_LABELS).plot(
        kind="bar",
        ax=ax,
        color=[COLORS.get(c, None) for c in pivot.columns],
        width=0.75,
    )
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [STRATEGY_LABELS.get(name, name) for name in labels],
        frameon=False,
        fontsize=9,
    )
    _style(ax, f"시장 국면별 {metric}", metric)
    return _save(figure, path)


def reward_comparison_plot(metrics, path: Path, metrics_shown=None):
    """Reward 3종 비교 (과제 13번). walk-forward 지표 표를 입력으로 받는다."""
    metrics_shown = list(
        metrics_shown or ["cumulative_return", "sharpe", "max_drawdown"]
    )
    ppo = metrics[metrics.strategy == "ppo"]
    if ppo.empty:
        raise ValueError("PPO 행이 없어 reward 비교 그림을 그릴 수 없습니다")
    grouped = ppo.groupby("reward")[metrics_shown].mean()
    figure, axes = plt.subplots(
        1, len(metrics_shown), figsize=(4 * len(metrics_shown), 4)
    )
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, column in zip(axes, metrics_shown):
        grouped[column].plot(kind="bar", ax=ax, color="#C8A24A", width=0.65)
        ax.set_title(column, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.grid(alpha=0.25, axis="y", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Reward 함수 3종 성능 비교 (walk-forward 평균)", fontsize=12)
    return _save(figure, path)


def walk_forward_plot(metrics, path: Path, metric="cumulative_return"):
    """Window 별 전략 성과 추이 — 특정 구간에서만 좋았는지 확인한다."""
    table = metrics[metrics.strategy != "ppo"].copy()
    ppo = metrics[metrics.strategy == "ppo"].copy()
    figure, ax = plt.subplots(figsize=(9, 4.5))
    for strategy, block in table.groupby("strategy"):
        ax.plot(
            block.window,
            block[metric],
            marker="o",
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=COLORS.get(strategy),
        )
    for reward, block in ppo.groupby("reward"):
        ax.plot(
            block.window,
            block[metric],
            marker="s",
            linestyle="--",
            label=f"PPO ({reward})",
        )
    ax.set_xticks(sorted(metrics.window.unique()))
    ax.set_xlabel("Walk-forward window")
    _style(ax, f"Window 별 {metric}", metric)
    return _save(figure, path)
