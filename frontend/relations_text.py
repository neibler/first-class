"""Plain words for ETF relationships. Picks from server values; computes nothing."""

from frontend.shap_explanations import ASSET_NAMES

TOGETHER, APART = 0.2, -0.2  # same edges as the server's "매우 약한 관계" band


def has_final(word):
    """Whether the last Hangul syllable ends in a consonant (받침)."""
    for char in reversed(word):
        if "가" <= char <= "힣":
            return (ord(char) - ord("가")) % 28 != 0
        if char.isdigit():
            return char in "013678"
    return False


def josa(word, with_final, without_final):
    return word + (with_final if has_final(word) else without_final)


def etf_label(ticker):
    return f"{ASSET_NAMES.get(ticker, ticker)}({ticker})"


def partners(relationships):
    """Most-together and most-opposite ETF among the server's relationships."""
    etfs = [r for r in relationships if r["type"] == "ETF"]
    if not etfs:
        return None, None
    together = max(etfs, key=lambda r: r["correlation"])
    opposite = min(etfs, key=lambda r: r["correlation"])
    return together, opposite


def top_partners(relationships, count=3):
    """Up to `count` most-together and most-apart ETFs, never the same one twice."""
    etfs = sorted(
        (r for r in relationships if r["type"] == "ETF"),
        key=lambda r: r["correlation"],
        reverse=True,
    )
    # Only clearly positive partners count as "together"; a negative or weak one
    # can never fill that column, it belongs on the apart side.
    together = [r for r in etfs if r["correlation"] >= TOGETHER][:count]
    apart = [r for r in reversed(etfs) if r not in together][:count]
    return together, apart


def opposite_title(row):
    if row["correlation"] <= APART:
        return "반대로 움직여요"
    if row["correlation"] < TOGETHER:
        return "따로 움직여요"
    return "그나마 덜 같이 움직여요"


def conclusion(ticker, together, opposite):
    """One line a beginner can act on: why mixing ETFs lowers the swings."""
    me = ASSET_NAMES.get(ticker, ticker)
    if together is None:
        return None
    if together["correlation"] < TOGETHER:
        # No clearly positive partner: never call any ETF "같이".
        if opposite["correlation"] <= APART:
            other = josa(
                ASSET_NAMES.get(opposite["name"], opposite["name"]), "과", "와"
            )
            return (
                f"{josa(me, '은', '는')} 뚜렷하게 같이 움직이는 ETF가 없고, "
                f"{other} 반대로 움직여요. 다른 자산과 함께 담으면 흔들림이 줄어요."
            )
        return (
            f"{josa(me, '은', '는')} 다른 ETF들과 뚜렷하게 같이 움직이지 않아요. "
            "다른 자산과 함께 담으면 흔들림이 줄어요."
        )
    pair = josa(ASSET_NAMES.get(together["name"], together["name"]), "과", "와")
    if opposite["correlation"] <= APART:
        other = josa(ASSET_NAMES.get(opposite["name"], opposite["name"]), "과", "와")
        return (
            f"{josa(me, '은', '는')} {pair} 같이, {other} 반대로 움직이는 편이에요. "
            "반대로 움직이는 자산을 함께 담으면 서로 버텨 줘서 흔들림이 줄어요."
        )
    if opposite["correlation"] < TOGETHER:
        other = josa(ASSET_NAMES.get(opposite["name"], opposite["name"]), "은", "는")
        return (
            f"{josa(me, '은', '는')} {pair} 같이 움직이고, {other} 거의 따로 움직여요. "
            "따로 움직이는 자산을 함께 담으면 흔들림이 줄어요."
        )
    return (
        f"{josa(me, '은', '는')} 다른 ETF 대부분과 같이 움직이는 편이라, "
        "함께 담아도 흔들림이 크게 줄지는 않아요."
    )


def mix_sentence(average, strength):
    """How mixed the held ETFs are, from the server's average correlation."""
    if average >= 0.6:
        body = "담긴 ETF들이 꽤 같이 움직여서, 섞은 효과가 작은 편이에요."
    elif average >= TOGETHER:
        body = "담긴 ETF들이 어느 정도 같이 움직여서, 섞은 효과가 보통이에요."
    else:
        body = "담긴 ETF들이 서로 따로 움직여서, 섞은 효과가 큰 편이에요."
    return f"{body} (평균 관계 {average:+.2f} · {strength})"
