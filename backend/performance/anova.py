"""ANOVA, effect size and Tukey HSD (과제 18~23번).

표본 설계 — 보고서에 반드시 적어야 하는 부분
- 일별 수익률을 그대로 쓰면 관측치가 겹치고 자기상관이 강해 ANOVA 의 독립성
  가정이 깨진다. 그래서 기본 표본 단위를 "겹치지 않는 월 단위 수익률"로 둔다.
- Walk-forward window × seed 단위 성과지표도 표본으로 쓸 수 있게 열어 둔다
  (표본 수는 적지만 독립성 가정이 가장 깨끗하다).
- 유의하지 않아도 효과 크기(η²)와 원인을 함께 보고한다 (과제 23번).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
#: η² 해석 기준 (Cohen).
EFFECT_BOUNDS = ((0.01, "작음"), (0.06, "중간"), (0.14, "큼"))


def period_returns(returns: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """일별 수익률 -> 겹치지 않는 기간 수익률 (기본 월말 기준)."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns 의 인덱스는 DatetimeIndex 여야 합니다")
    return (1 + returns).resample(freq).prod() - 1


def long_format(frame: pd.DataFrame, value: str = "period_return") -> pd.DataFrame:
    """wide(전략별 열) -> long(date, strategy, value[, regime]) 변환."""
    frame = frame.copy()
    key = frame.index.name or "date"
    frame.index.name = key
    extras = [c for c in frame.columns if c == "regime"]
    melted = (
        frame.drop(columns=extras)
        .reset_index()
        .melt(id_vars=key, var_name="strategy", value_name=value)
    )
    if extras:
        melted = melted.merge(frame[extras].reset_index(), on=key, how="left")
    return melted.dropna(subset=[value])


def effect_label(eta_squared: float | None) -> str:
    if eta_squared is None or not np.isfinite(eta_squared):
        return "판정 불가"
    label = "매우 작음"
    for bound, name in EFFECT_BOUNDS:
        if eta_squared >= bound:
            label = name
    return label


def one_way(groups: dict[str, np.ndarray], label: str = "") -> dict:
    """One-way ANOVA + η² + 가정 점검.

    Args:
        groups: {그룹 이름: 관측치 배열}. 그룹은 2개 이상이어야 한다.
    """
    names = [n for n, v in groups.items() if len(np.asarray(v, dtype=float)) > 0]
    if len(names) < 2:
        raise ValueError("One-way ANOVA 에는 비어 있지 않은 그룹이 2개 이상 필요합니다")
    samples = [np.asarray(groups[n], dtype=float) for n in names]
    sizes = [len(s) for s in samples]
    pooled = np.concatenate(samples)
    grand = pooled.mean()
    ss_between = float(sum(len(s) * (s.mean() - grand) ** 2 for s in samples))
    ss_within = float(sum(((s - s.mean()) ** 2).sum() for s in samples))
    ss_total = ss_between + ss_within
    df_between, df_within = len(samples) - 1, len(pooled) - len(samples)
    eta_squared = ss_between / ss_total if ss_total > 1e-18 else None
    if min(sizes) < 2 or ss_within <= 1e-18:
        f_statistic = p_value = None
    else:
        f_statistic, p_value = (float(x) for x in stats.f_oneway(*samples))
        # 그룹 평균이 사실상 같으면 F 가 -1e-17 처럼 나오고 p 가 nan 이 될 수 있다.
        if not np.isfinite(f_statistic) or not np.isfinite(p_value):
            f_statistic = 0.0 if ss_between <= 1e-18 else None
            p_value = 1.0 if f_statistic == 0.0 else None
    return {
        "test": "one_way_anova",
        "label": label,
        "groups": names,
        "n_per_group": dict(zip(names, sizes)),
        "group_mean": {n: float(s.mean()) for n, s in zip(names, samples)},
        "group_std": {
            n: float(s.std(ddof=1)) if len(s) > 1 else None
            for n, s in zip(names, samples)
        },
        "f_statistic": f_statistic,
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "eta_squared": eta_squared,
        "effect_size_label": effect_label(eta_squared),
        "significant": bool(p_value is not None and p_value < ALPHA),
        "assumptions": assumption_checks(samples, names),
    }


def assumption_checks(samples, names) -> dict:
    """정규성(Shapiro-Wilk)·등분산성(Levene) 점검. 표본이 작으면 None."""
    normality = {}
    for name, sample in zip(names, samples):
        if 3 <= len(sample) <= 5000 and sample.std(ddof=1) > 1e-18:
            normality[name] = float(stats.shapiro(sample).pvalue)
        else:
            normality[name] = None
    usable = [s for s in samples if len(s) > 1]
    equal_variance = (
        float(stats.levene(*usable, center="median").pvalue)
        if len(usable) >= 2 and all(s.std(ddof=1) > 1e-18 for s in usable)
        else None
    )
    return {
        "shapiro_p_value": normality,
        "levene_p_value": equal_variance,
        "note": (
            "p < 0.05 이면 해당 가정이 깨진 것으로 본다. "
            "정규성 위반이 크면 Kruskal-Wallis 를 보조로 함께 보고한다."
        ),
    }


def kruskal(groups: dict[str, np.ndarray]) -> dict:
    """정규성 가정이 의심될 때 쓰는 비모수 대안."""
    samples = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0]
    if len(samples) < 2 or min(len(s) for s in samples) < 2:
        return {"test": "kruskal_wallis", "h_statistic": None, "p_value": None}
    result = stats.kruskal(*samples)
    return {
        "test": "kruskal_wallis",
        "h_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < ALPHA),
    }


def two_way(
    frame: pd.DataFrame,
    value: str,
    factor_a: str,
    factor_b: str,
    label: str = "",
) -> dict:
    """Two-way ANOVA (전략 × 시장 국면, 상호작용 포함) + η², partial η²."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    data = frame[[value, factor_a, factor_b]].dropna().copy()
    for factor in (factor_a, factor_b):
        data[factor] = data[factor].astype("category")
    if data[factor_a].nunique() < 2 or data[factor_b].nunique() < 2:
        raise ValueError("Two-way ANOVA 에는 각 요인에 수준이 2개 이상 필요합니다")
    counts = data.groupby([factor_a, factor_b], observed=True).size()
    interaction = bool(counts.min() >= 2)
    formula = (
        f"Q('{value}') ~ C(Q('{factor_a}')) * C(Q('{factor_b}'))"
        if interaction
        else f"Q('{value}') ~ C(Q('{factor_a}')) + C(Q('{factor_b}'))"
    )
    model = ols(formula, data=data).fit()
    # typ=2: 불균형 설계에서 요인 순서에 영향을 받지 않는 제곱합.
    table = sm.stats.anova_lm(model, typ=2)
    ss_total = float(table["sum_sq"].sum())
    residual = float(table.loc["Residual", "sum_sq"])
    terms = {}
    rename = {
        f"C(Q('{factor_a}'))": factor_a,
        f"C(Q('{factor_b}'))": factor_b,
        f"C(Q('{factor_a}')):C(Q('{factor_b}'))": f"{factor_a}:{factor_b}",
    }
    for term, row in table.iterrows():
        if term == "Residual":
            continue
        ss = float(row["sum_sq"])
        p_value = float(row["PR(>F)"]) if np.isfinite(row["PR(>F)"]) else None
        eta_squared = ss / ss_total if ss_total > 1e-18 else None
        terms[rename.get(term, term)] = {
            "f_statistic": float(row["F"]) if np.isfinite(row["F"]) else None,
            "p_value": p_value,
            "df": float(row["df"]),
            "eta_squared": eta_squared,
            "partial_eta_squared": (
                ss / (ss + residual) if ss + residual > 1e-18 else None
            ),
            "effect_size_label": effect_label(eta_squared),
            "significant": bool(p_value is not None and p_value < ALPHA),
        }
    # 유의한 주효과에는 사후검정을 붙인다. 어떤 수준끼리 다른지는 F 검정만으로
    # 알 수 없기 때문이다 (예: 국면 요인이 유의하면 상승/횡보/하락 중 어디가
    # 다른지). 수준이 2개여도 평균차와 신뢰구간은 보고 가치가 있으므로 수행한다.
    posthoc = {}
    for factor in (factor_a, factor_b):
        values = terms.get(factor)
        if not values or not values["significant"]:
            continue
        if data[factor].nunique() < 2:
            continue
        posthoc[factor] = tukey_hsd(data, value, factor)
    return {
        "test": "two_way_anova",
        "label": label,
        "value": value,
        "factors": [factor_a, factor_b],
        "interaction_included": interaction,
        "n": int(len(data)),
        "cell_counts": {str(k): int(v) for k, v in counts.items()},
        "terms": terms,
        "posthoc": posthoc or None,
        "posthoc_note": (
            "시계열 표본은 같은 시장 구간을 공유하므로 완전히 독립이 아닙니다. "
            "Tukey 결과는 참고치로 보고, 표본 수와 구간 수를 함께 보고하세요."
        ),
        "r_squared": float(model.rsquared),
        "residual_df": float(table.loc["Residual", "df"]),
    }


def tukey_hsd(
    frame: pd.DataFrame, value: str, group: str, alpha: float = ALPHA
) -> dict:
    """ANOVA 가 유의할 때의 사후 검정 (과제 22번)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    data = frame[[value, group]].dropna()
    if data[group].nunique() < 2:
        raise ValueError("Tukey HSD 에는 그룹이 2개 이상 필요합니다")
    result = pairwise_tukeyhsd(
        data[value].to_numpy(dtype=float), data[group].astype(str), alpha=alpha
    )
    table = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
    pairs = [
        {
            "group1": str(row["group1"]),
            "group2": str(row["group2"]),
            "mean_difference": float(row["meandiff"]),
            "p_adjusted": float(row["p-adj"]),
            "lower": float(row["lower"]),
            "upper": float(row["upper"]),
            "significant": bool(row["reject"]),
        }
        for _, row in table.iterrows()
    ]
    return {"test": "tukey_hsd", "alpha": alpha, "value": value, "pairs": pairs}


def run_with_posthoc(
    frame: pd.DataFrame, value: str, group: str, label: str = ""
) -> dict:
    """One-way ANOVA 를 돌리고 p < 0.05 일 때만 Tukey HSD 를 붙인다."""
    groups = {
        str(name): block[value].to_numpy(dtype=float)
        for name, block in frame.groupby(group, observed=True)
    }
    result = one_way(groups, label=label)
    result["nonparametric"] = kruskal(groups)
    result["posthoc"] = (
        tukey_hsd(frame, value, group) if result["significant"] else None
    )
    result["independence"] = overlap_diagnostics(frame, group)
    result["groups_label"] = group
    result["interpretation"] = interpret(result)
    return result


def interpret(result: dict) -> str:
    """통계 결과를 보고서에 붙일 한국어 문장으로 바꾼다 (과제 23번)."""
    label = result.get("label") or "그룹 간"
    f_statistic, p_value = result.get("f_statistic"), result.get("p_value")
    if p_value is None:
        return (
            f"{label}: 표본이 부족하거나 그룹 내 분산이 0 이어서 F 검정을 "
            "계산할 수 없었습니다."
        )
    eta = result.get("eta_squared")
    eta_text = (
        "계산 불가" if eta is None else f"{eta:.3f} ({result['effect_size_label']})"
    )
    sizes = ", ".join(f"{k}={v}" for k, v in result.get("n_per_group", {}).items())
    if result["significant"]:
        posthoc = result.get("posthoc")
        detail = ""
        if posthoc:
            significant = [
                f"{p['group1']} vs {p['group2']}"
                for p in posthoc["pairs"]
                if p["significant"]
            ]
            detail = (
                f" Tukey HSD 결과 유의한 쌍: {', '.join(significant)}."
                if significant
                else " 다만 Tukey HSD 에서는 유의한 쌍이 없었습니다."
            )
        return (
            f"{label}: F={f_statistic:.3f}, p={p_value:.4f} < 0.05 이므로 "
            f"그룹 간 차이가 통계적으로 유의합니다. η²={eta_text}. "
            f"표본 수 {sizes}.{detail}"
        )
    if (result.get("eta_squared") or 0.0) < 1e-9:
        cause = (
            "그룹 평균이 사실상 동일합니다 "
            "(세 조건이 같은 포트폴리오를 만들었는지 먼저 확인하세요)."
        )
    else:
        cause = (
            f"표본 수({sizes})가 많지 않고 수익률의 그룹 내 분산이 평균 차이보다 "
            "크기 때문에 검정력이 낮은 것이 주된 원인으로 보입니다. "
            "표본을 늘리려면 ANOVA 주기를 월(ME) 대신 주(W)로 바꾸거나 "
            "walk-forward window 수를 늘리면 됩니다."
        )
    return (
        f"{label}: F={f_statistic:.3f}, p={p_value:.4f} >= 0.05 이므로 "
        f"그룹 간 차이가 통계적으로 유의하지 않습니다. η²={eta_text} 이며, {cause} "
        "p >= 0.05 는 그룹이 동일하다는 증명이 아니라, 현재 자료로는 차이를 "
        "확인할 증거가 부족하다는 뜻입니다."
    )


def overlap_diagnostics(frame: pd.DataFrame, group: str, date: str = "period_end"):
    """표본의 독립성을 점검한다 (보고서에 그대로 적을 진단치).

    같은 날짜의 전략 수익률은 같은 시장 충격을 공유하므로 서로 독립이 아니다.
    또 검증 fold 가 달력상 겹치면 같은 날이 여러 번 집계될 수 있다. ANOVA 를
    그대로 쓰되, 아래 수치를 함께 보고해 한계를 명시한다.
    """
    if date not in frame.columns:
        return {"available": False, "note": f"'{date}' 열이 없어 점검 불가"}
    counts = frame.groupby(group, observed=True)[date].agg(["count", "nunique"])
    duplicated = frame.duplicated([group, date]).sum()
    shared = (
        frame.groupby(date, observed=True)[group].nunique().gt(1).sum()
        if len(frame)
        else 0
    )
    return {
        "available": True,
        "rows_per_group": counts["count"].to_dict(),
        "distinct_dates_per_group": counts["nunique"].to_dict(),
        "duplicated_group_date_rows": int(duplicated),
        "dates_shared_across_groups": int(shared),
        "note": (
            "같은 날짜가 여러 그룹에 나타나면 관측치가 서로 독립이 아닙니다. "
            "ANOVA 의 p-value 는 참고치로 보고, 대응 비교(paired)와 "
            "블록 부트스트랩 신뢰구간을 함께 해석하세요."
        ),
    }


def paired_difference(
    wide: pd.DataFrame, left: str, right: str, block: int = 3, resamples: int = 2000
) -> dict:
    """같은 기간에 측정된 두 전략의 차이를 대응표본으로 검정한다.

    두 전략이 같은 시장을 겪었다는 사실을 활용하므로, 독립표본 가정을 쓰는
    One-way ANOVA 보다 이 비교가 더 적절하다. 정규성 가정이 약한 Wilcoxon 과
    자기상관을 반영하는 블록 부트스트랩 신뢰구간을 함께 돌려준다.
    """
    paired = wide[[left, right]].dropna()
    difference = (paired[left] - paired[right]).to_numpy(dtype=float)
    result = {
        "test": "paired_difference",
        "left": left,
        "right": right,
        "n": int(len(difference)),
        "mean_difference": float(difference.mean()) if len(difference) else None,
        "t_statistic": None,
        "p_value": None,
        "wilcoxon_p_value": None,
        "cohens_dz": None,
    }
    if len(difference) < 3 or difference.std(ddof=1) <= 1e-18:
        result["note"] = "표본이 부족하거나 차이의 분산이 0 이어서 검정 불가"
        result["bootstrap"] = block_bootstrap_ci(difference, block, resamples)
        return result
    statistic, p_value = stats.ttest_rel(paired[left], paired[right])
    result["t_statistic"] = float(statistic)
    result["p_value"] = float(p_value)
    result["cohens_dz"] = float(difference.mean() / difference.std(ddof=1))
    try:
        result["wilcoxon_p_value"] = float(stats.wilcoxon(difference).pvalue)
    except ValueError:
        result["wilcoxon_p_value"] = None
    result["significant"] = bool(p_value < ALPHA)
    result["bootstrap"] = block_bootstrap_ci(difference, block, resamples)
    return result


def block_bootstrap_ci(
    values, block: int = 3, resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> dict:
    """이동 블록 부트스트랩으로 평균 차이의 신뢰구간을 구한다.

    연속한 `block` 개를 한 덩어리로 뽑아 자기상관을 어느 정도 보존한다.
    """
    sample = np.asarray(values, dtype=float)
    size = len(sample)
    if size < 2:
        return {"available": False, "note": "표본 부족"}
    block = int(max(1, min(block, size)))
    starts = size - block + 1
    count = int(np.ceil(size / block))
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, starts, size=(resamples, count))
    index = (offsets[:, :, None] + np.arange(block)[None, None, :]).reshape(
        resamples, -1
    )[:, :size]
    means = sample[index].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "available": True,
        "block_size": block,
        "resamples": int(resamples),
        "level": level,
        "mean": float(sample.mean()),
        "lower": float(low),
        "upper": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def holm_adjust(p_values) -> list[float | None]:
    """Holm-Bonferroni 다중비교 보정.

    같은 자료에서 여러 쌍을 비교하면 우연히 하나가 유의해질 확률이 올라간다.
    4개 비교에서 p=0.038 은 보정 후 약 0.15 가 되어 더 이상 유의하지 않다.
    탐색적 비교라도 보정값을 함께 보고해야 결론을 과장하지 않는다.
    """
    values = list(p_values)
    usable = [(i, p) for i, p in enumerate(values) if p is not None and np.isfinite(p)]
    adjusted: list[float | None] = [None] * len(values)
    count = len(usable)
    running = 0.0
    for rank, (index, p) in enumerate(sorted(usable, key=lambda item: item[1])):
        # 단조성 유지: 보정값은 앞 순위보다 작아질 수 없다.
        running = max(running, min(1.0, p * (count - rank)))
        adjusted[index] = float(running)
    return adjusted


def apply_holm(results: list[dict], alpha: float = ALPHA) -> list[dict]:
    """대응 비교 묶음에 Holm 보정을 적용하고 결과에 기록한다."""
    adjusted = holm_adjust([result.get("p_value") for result in results])
    for result, value in zip(results, adjusted):
        result["p_value_holm"] = value
        result["significant_holm"] = bool(value is not None and value < alpha)
        result["comparisons"] = len(results)
    return results


def paired_table(results: list[dict]) -> pd.DataFrame:
    """대응 비교 결과를 CSV 로 저장할 표로."""
    rows = []
    for result in results:
        bootstrap = result.get("bootstrap") or {}
        rows.append(
            {
                "left": result["left"],
                "right": result["right"],
                "n": result["n"],
                "mean_difference": result["mean_difference"],
                "t_statistic": result["t_statistic"],
                "p_value": result["p_value"],
                "p_value_holm": result.get("p_value_holm"),
                "significant_holm": result.get("significant_holm"),
                "comparisons": result.get("comparisons"),
                "wilcoxon_p_value": result["wilcoxon_p_value"],
                "cohens_dz": result["cohens_dz"],
                "bootstrap_lower": bootstrap.get("lower"),
                "bootstrap_upper": bootstrap.get("upper"),
                "bootstrap_excludes_zero": bootstrap.get("excludes_zero"),
            }
        )
    return pd.DataFrame(rows)


def interpret_paired(result: dict) -> str:
    """대응 비교 결과를 보고서 문장으로."""
    left, right, n = result["left"], result["right"], result["n"]
    if result["p_value"] is None:
        return f"{left} vs {right}: 표본 {n}개로는 대응 검정을 할 수 없었습니다."
    bootstrap = result.get("bootstrap") or {}
    interval = (
        f" 블록 부트스트랩 95% 신뢰구간 [{bootstrap['lower']:+.4f}, "
        f"{bootstrap['upper']:+.4f}]"
        if bootstrap.get("available")
        else ""
    )
    holm = result.get("p_value_holm")
    if holm is None:
        verdict = "유의한 차이" if result.get("significant") else "유의한 차이 없음"
        correction = ""
    else:
        verdict = (
            "보정 후에도 유의"
            if result.get("significant_holm")
            else "보정 후 유의하지 않음"
        )
        correction = (
            f" {result.get('comparisons')}개 비교에 Holm 보정을 적용하면 "
            f"p={holm:.4f} 입니다."
        )
    return (
        f"{left} vs {right}: 같은 기간 대응 비교에서 평균 차이 "
        f"{result['mean_difference']:+.4f}, t={result['t_statistic']:.3f}, "
        f"보정 전 p={result['p_value']:.4f} ({verdict}), "
        f"Cohen's dz={result['cohens_dz']:.3f}, 표본 {n}개.{correction}{interval}"
    )


def summary_table(results: list[dict]) -> pd.DataFrame:
    """여러 검정 결과를 하나의 CSV 표로 (과제 21번 F/p/η² 보고)."""
    rows = []
    for result in results:
        if result["test"] == "two_way_anova":
            for term, values in result["terms"].items():
                rows.append(
                    {
                        "test": result["test"],
                        "label": result["label"],
                        "term": term,
                        "f_statistic": values["f_statistic"],
                        "p_value": values["p_value"],
                        "eta_squared": values["eta_squared"],
                        "partial_eta_squared": values["partial_eta_squared"],
                        "effect_size": values["effect_size_label"],
                        "significant": values["significant"],
                        "n": result["n"],
                    }
                )
        else:
            rows.append(
                {
                    "test": result["test"],
                    "label": result["label"],
                    "term": "+".join(result["groups"]),
                    "f_statistic": result["f_statistic"],
                    "p_value": result["p_value"],
                    "eta_squared": result["eta_squared"],
                    "partial_eta_squared": None,
                    "effect_size": result["effect_size_label"],
                    "significant": result["significant"],
                    "n": int(sum(result["n_per_group"].values())),
                }
            )
    return pd.DataFrame(rows)


def tukey_table(results: list[dict]) -> pd.DataFrame:
    """사후 검정 결과만 모아 tukey.csv 로 저장할 표를 만든다.

    One-way 는 `posthoc` 이 결과 하나, Two-way 는 {요인: 결과} 형태라 둘 다 받는다.
    """
    rows = []
    for result in results:
        posthoc = result.get("posthoc")
        if not posthoc:
            continue
        blocks = (
            posthoc.items()
            if "pairs" not in posthoc
            else [(result.get("groups_label", "group"), posthoc)]
        )
        for factor, block in blocks:
            for pair in block["pairs"]:
                rows.append(
                    {"label": result.get("label", ""), "factor": factor, **pair}
                )
    return pd.DataFrame(rows)
