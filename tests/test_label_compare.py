import pytest

from backend.news_research import label_compare as lc


def test_ranks_share_ties_and_spearman_matches_known_values():
    assert lc.ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert lc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert lc.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert lc.spearman([1, 2], [1, 2]) is None  # too few to rank


def test_kappa_corrects_for_chance():
    assert lc.cohen_kappa([True, False] * 5, [True, False] * 5) == 1.0
    # half agree, as chance predicts when both say yes half the time
    assert lc.cohen_kappa(
        [True, True, False, False], [True, False, False, True]
    ) == pytest.approx(0.0)
    assert lc.cohen_kappa([True, True, False, False], [True, True, False, True]) == (
        pytest.approx(0.5)
    )


def _label(url, relevant=True, types=("inflation",), level=0.5, direction="increase"):
    return {
        "url": url,
        "market_relevant": relevant,
        "risk_types": list(types) if relevant else [],
        "risk_level": level if relevant else 0,
        "direction": direction if relevant else "none",
        "confidence": 0.8,
        "prompt_version": "v2",
    }


def test_compare_counts_only_shared_articles_and_finds_disagreements():
    reference = [
        _label("u1", level=0.2),
        _label("u2", level=0.5),
        _label("u3", level=0.8, types=("recession", "earnings")),
        _label("u4", relevant=False),
        _label("u5"),  # candidate never labeled it
    ]
    candidate = [
        _label("u1", level=0.3),
        _label("u2", level=0.6),
        _label("u3", level=0.7, types=("market_risk", "recession")),
        _label("u4", relevant=True, types=("regulation",), level=0.2),
    ]
    m = lc.compare(reference, candidate)
    assert m["articles"] == 5 and m["compared"] == 4 and m["both_relevant"] == 3
    assert m["relevant_agreement"] == pytest.approx(0.75)
    assert m["first_type"] == pytest.approx(2 / 3)
    # types: tp = 1 + 1 + 1, fp = 1 (market_risk), fn = 1 (earnings)
    assert m["type_f1"] == pytest.approx(0.75)
    assert m["level_spearman"] == pytest.approx(1.0)
    assert m["level_mae"] == pytest.approx(0.1)
    kinds = {d["url"]: d["kind"] for d in lc.disagreements(reference, candidate)}
    assert kinds == {"u3": "첫 유형", "u4": "관련 여부"}


def test_perfect_copy_passes_every_threshold_and_report_splits_versions():
    reference = [_label(f"u{i}", level=i / 10) for i in range(1, 8)]
    reference += [{**_label("v1", relevant=False), "prompt_version": "v1"}]
    assert all(lc.passes(lc.compare(reference, reference)).values())
    text = lc.report(reference, reference, {})
    assert "기준 v1" in text and "기준 v2" in text and "미달" not in text


def test_titles_come_from_the_tagging_keys():
    keys = [{"urls": {"a01": "u1"}, "texts": {"a01": "Oil Jumps\nOPEC cut output."}}]
    assert lc.titles_from_keys(keys) == {"u1": "Oil Jumps"}
