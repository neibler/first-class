"""Plain-language help text for general investors. Fixed copy, no calculations.

ETF 소개는 각 ETF가 무엇을 따라가는지에 대한 일반적인 설명이다. 구성 종목이나
비중처럼 자주 바뀌는 정보는 넣지 않는다. 문구를 고칠 때는 운용사 자료로 확인한다.
"""

ETF_GUIDE = {
    "SPY": "미국을 대표하는 대형 기업 500곳(S&P 500 지수)에 한 번에 투자하는 ETF예요.",
    "QQQ": "나스닥에 상장된 대형 기업 100곳(나스닥 100 지수)에 투자하는 ETF예요. 기술 기업 비중이 높아요.",
    "SMH": "반도체를 만들거나 관련 장비를 만드는 기업들에 투자하는 ETF예요.",
    "SCHD": "배당을 꾸준히 지급해 온 미국 기업들에 투자하는 ETF예요.",
    "XLE": "석유·가스 같은 미국 에너지 기업들에 투자하는 ETF예요.",
    "EFA": "미국과 캐나다를 제외한 선진국(유럽, 일본, 호주 등) 기업들에 투자하는 ETF예요.",
    "SHY": "만기가 1~3년으로 짧은 미국 국채에 투자하는 ETF예요. 가격 변동이 비교적 작아요.",
    "IEF": "만기가 7~10년인 미국 국채에 투자하는 ETF예요.",
    "TLT": "만기가 20년 이상으로 긴 미국 국채에 투자하는 ETF예요. 금리 변화에 민감해요.",
    "TIP": "물가가 오르면 원금도 함께 조정되는 미국 물가연동국채에 투자하는 ETF예요.",
    "LQD": "신용등급이 높은 미국 기업들이 발행한 회사채에 투자하는 ETF예요.",
    "GLD": "금 가격을 따라가는 ETF예요. 실제 금을 보관하는 방식으로 운용돼요.",
    "DBB": "알루미늄·아연·구리 같은 산업용 금속 가격을 따라가는 ETF예요.",
    "DBA": "옥수수·밀·대두·설탕 같은 농산물 가격을 따라가는 ETF예요.",
    "VNQ": "미국의 부동산 투자 회사(리츠)들에 투자하는 ETF예요.",
}

INDICATOR_GUIDE = [
    {
        "name": "RSI · 너무 올랐나, 너무 내렸나",
        "summary": (
            "최근 가격이 오른 날과 내린 날의 힘을 비교해 0~100 사이 숫자로 나타낸 지표예요."
        ),
        "points": [
            "70 이상이면 단기간에 많이 올라 '과매수' 상태로 봐요.",
            "30 이하면 단기간에 많이 내려 '과매도' 상태로 봐요.",
            "그 사이는 뚜렷한 쏠림이 없는 중립 구간이에요.",
        ],
    },
    {
        "name": "MACD · 추세가 바뀌는 신호",
        "summary": (
            "최근 가격 흐름(단기 평균)과 조금 더 긴 흐름(장기 평균)의 차이를 보는 지표예요."
        ),
        "points": [
            "MACD가 신호선 위로 올라가면 상승 흐름이 강해지는 신호로 봐요.",
            "MACD가 신호선 아래로 내려가면 하락 흐름이 강해지는 신호로 봐요.",
            "0보다 크면 최근 흐름이 장기 흐름보다 강하다는 뜻이에요.",
        ],
    },
]


def rsi_example(value):
    """Today's RSI in words, with the same 30/70 lines the chart draws."""
    if value >= 70:
        zone = "70 이상이라 단기간에 많이 오른 '과매수' 쪽"
    elif value <= 30:
        zone = "30 이하라 단기간에 많이 내린 '과매도' 쪽"
    else:
        zone = "30~70 사이라 한쪽으로 크게 쏠리지 않은 중립 구간"
    # One decimal so 69.9 is not shown as "70" while called neutral.
    return f"현재 {value:.1f}은(는) {zone}이에요."


def macd_example(value):
    """Today's MACD in words, against the chart's 0 line."""
    if value > 0:
        trend = "0보다 커서, 최근 흐름이 긴 흐름보다 위쪽(상승 쪽)이라는 뜻이에요."
    elif value < 0:
        trend = "0보다 작아서, 최근 흐름이 긴 흐름보다 아래쪽(하락 쪽)이라는 뜻이에요."
    else:
        trend = "0이라 최근 흐름과 긴 흐름이 같은 수준이에요."
    if value != 0 and abs(value) < 0.01:
        # "+0.00" next to "greater than zero" would contradict itself.
        trend = trend.replace("0보다 커서", "0보다 아주 조금 커서").replace(
            "0보다 작아서", "0보다 아주 조금 작아서"
        )
        return f"현재 {value:+.4f}은(는) {trend}"
    return f"현재 {value:+.2f}은(는) {trend}"


SUGGESTED_QUESTIONS = [
    "최근 QQQ에 어떤 위험이 있었나요?",
    "반도체(SMH) 관련 주요 뉴스는 무엇인가요?",
    "금리 변화가 채권 ETF에 어떤 영향을 줬나요?",
    "금(GLD) 가격에 영향을 준 사건은 무엇인가요?",
]
