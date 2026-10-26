"""Find config/etf_news_map.yaml terms (company aliases, keywords) in news text."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable

NYT_TAG_PREFIX = "nyt:"  # label for an exact NYT tag match, e.g. "nyt:Gold"
ACRONYM = re.compile(r"[A-Z]{2,}s?\+?")  # TIPS, CPI, REITs, OPEC+: match case


def _term_pattern(term: str, flags: int) -> re.Pattern[str]:
    # Lookarounds instead of \b so terms like "OPEC+" or "S&P 500" still match
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags)


def risk_only_terms(news_map: dict) -> frozenset[str]:
    """Risk keywords (e.g. "war", "tariff") that are not company or ETF terms."""
    topical = {a for c in news_map["companies"].values() for a in c["aliases"]}
    topical |= {k for etf in news_map["etfs"].values() for k in etf["keywords"]}
    risk = {w for words in news_map["risk_type_keywords"].values() for w in words}
    return frozenset(risk - topical)


def term_etfs(news_map: dict) -> dict[str, set[str]]:
    """Map each matcher label to the ETFs it directly concerns.

    Company aliases map to the ETFs holding that company; ETF keywords and NYT
    tags map to their own ETF. Risk-only keywords map to no ETF here: they reach
    ETFs through risk_exposure instead.
    """
    etfs_of: dict[str, set[str]] = defaultdict(set)
    for ticker, etf in news_map["etfs"].items():
        for symbol in etf["holdings"]:
            for alias in news_map["companies"][symbol]["aliases"]:
                etfs_of[alias].add(ticker)
        for keyword in etf["keywords"]:
            etfs_of[keyword].add(ticker)
        for tag in etf.get("nyt_tags", []):
            etfs_of[NYT_TAG_PREFIX + tag].add(ticker)
    return dict(etfs_of)


def build_matcher(news_map: dict) -> Callable[..., list[str]]:
    """Return ``match(text, tags=())`` listing the config terms found.

    Company aliases are case-sensitive ("Apple" the company vs "apple" the fruit)
    and ignore their ``exclude`` phrases ("Merck KGaA" is not Merck & Co).
    ETF and risk keywords are case-insensitive, except acronyms ("TIPS" the bond,
    not "tips" the advice). ``tags`` (NYT keywords) must equal an ETF's
    ``nyt_tags`` entry exactly and are labelled "nyt:<tag>".
    """
    rules: list[tuple[str, re.Pattern[str], list[re.Pattern[str]]]] = []
    for company in news_map["companies"].values():
        excludes = [_term_pattern(e, re.IGNORECASE) for e in company.get("exclude", [])]
        for alias in company["aliases"]:
            rules.append((alias, _term_pattern(alias, 0), excludes))

    keywords = [k for etf in news_map["etfs"].values() for k in etf["keywords"]]
    keywords += [k for words in news_map["risk_type_keywords"].values() for k in words]
    for keyword in dict.fromkeys(keywords):
        flags = 0 if ACRONYM.fullmatch(keyword) else re.IGNORECASE
        rules.append((keyword, _term_pattern(keyword, flags), []))

    tag_labels = {
        tag: NYT_TAG_PREFIX + tag
        for etf in news_map["etfs"].values()
        for tag in etf.get("nyt_tags", [])
    }

    def match(text: str, tags: Iterable[str] = ()) -> list[str]:
        found = []
        for label, pattern, excludes in rules:
            target = text
            for exclude in excludes:
                target = exclude.sub(" ", target)
            if pattern.search(target) and label not in found:
                found.append(label)
        for tag in tags:
            label = tag_labels.get(tag)
            if label and label not in found:
                found.append(label)
        return found

    return match
