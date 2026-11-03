"""ETF 거래일 16:00 ET 기준의 비정규화 데이터 결합. VIX·WTI는 당일 가격 근사값."""

import argparse
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


ROOT = Path(__file__).resolve().parents[2]
ZONE = "America/New_York"
TICKERS = "SPY QQQ SMH SCHD XLE EFA SHY IEF TLT TIP LQD GLD DBB DBA VNQ".split()
MACRO = "core_pce_yoy real_pce_mom unemp ism_pmi effr ust_2y ust_10y ust_30y tips_10y vix usd_ret_20d wti yield_spread".split()
RATES = {"ust_2y", "ust_10y", "ust_30y", "tips_10y"}
PROXIES = {"vix", "wti"}
FED_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
SOURCES = {
    "h15": "https://www.federalreserve.gov/releases/h15/",
    "h10": "https://www.federalreserve.gov/releases/h10/",
    "effr": "https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates",
    "pmi": "https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar/",
    "vix": "https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf",
}

# ISM 자체 발표문과 배포본(PR Newswire)으로 확인한 미국 발표일·당시 PMI·근거.
# 월초/주말 규칙으로 날짜를 추정하지 않으며, 원본의 월과 수치가 일치할 때만 사용.
PMI_PUBLICATIONS = """2017-01-03|54.7|https://www.prnewswire.com/news-releases/pmi-at-547-december-manufacturing-ism-report-on-business-new-orders-production-and-employment-growing-inventories-contracting-supplier-deliveries-slowing-300383456.html
2017-02-01|56.0|https://www.prnewswire.com/news-releases/pmi-at-56-january-manufacturing-ism-report-on-business-new-orders-production-and-employment-growing-inventories-contracting-supplier-deliveries-slowing-300399977.html
2017-03-01|57.7|https://www.prnewswire.com/news-releases/pmi-at-577-february-manufacturing-ism-report-on-business-new-orders-production-and-employment-growing-supplier-deliveries-slowing-inventories-growing-300415335.html
2017-04-03|57.2|https://www.prnewswire.com/news-releases/pmi-at-572-march-manufacturing-ism-report-on-business-new-orders-production-and-employment-growing-supplier-deliveries-slowing-inventories-contracting-300432790.html
2017-05-01|54.8|https://www.prnewswire.com/news-releases/pmi-at-548-april-manufacturing-ism-report-on-business-new-orders-production-and-employment-growing-supplier-deliveries-slowing-inventories-growing-300448292.html
2017-06-01|54.9|https://www.prnewswire.com/news-releases/pmi-at-549-may-manufacturing-ism-report-on-business-new-orders-inventories-and-employment-growing-supplier-deliveries-slowing-prices-increasing-300466482.html
2017-07-03|57.8|https://www.prnewswire.com/news-releases/pmi-at-578-june-manufacturing-ism-report-on-business-new-orders-production-backlog-of-orders-and-employment-growing-supplier-deliveries-slowing-inventories-contracting-prices-increasing-at-slower-rate-300482891.html
2017-08-01|56.3|https://www.prnewswire.com/news-releases/pmi-at-563-july-manufacturing-ism-report-on-business-new-orders-production-backlog-of-orders-and-employment-continue-growing-supplier-deliveries-slowing-inventories-unchanged-prices-increasing-at-faster-rate-300496966.html
2017-09-01|58.8|https://www.prnewswire.com/news-releases/pmi-at-588-august-manufacturing-ism-report-on-business-new-orders-production-backlog-of-orders-and-employment-continue-growing-supplier-deliveries-slowing-raw-materials-inventories-growing-customer-inventories-too-low-300512698.html
2017-10-02|60.8|https://www.prnewswire.com/news-releases/pmi-at-608-september-manufacturing-ism-report-on-business-300528595.html
2017-11-01|58.7|https://www.prnewswire.com/news-releases/pmi-at-587-october-manufacturing-ism-report-on-business-300546890.html
2017-12-01|58.2|https://www.prnewswire.com/news-releases/pmi-at-582-november-manufacturing-ism-report-on-business-300564806.html
2018-01-03|59.7|https://www.prnewswire.com/news-releases/pmi-at-597-december-manufacturing-ism-report-on-business-300576690.html
2018-02-01|59.1|https://www.prnewswire.com/news-releases/pmi-at-591-january-manufacturing-ism-report-on-business-300591579.html
2018-03-01|60.8|https://www.prnewswire.com/news-releases/pmi-at-608-february-manufacturing-ism-report-on-business-300606210.html
2018-04-02|59.3|https://www.prnewswire.com/news-releases/pmi-at-593-march-manufacturing-ism-report-on-business-300622353.html
2018-05-01|57.3|https://www.prnewswire.com/news-releases/pmi-at-57-3-april-manufacturing-ism-report-on-business-300638390.html
2018-06-01|58.7|https://www.ismworld.org/supply-management-news-and-reports/news-publications/inside-supply-management-magazine/blog/2018-06/rob-roundup-may-pmi/
2018-07-02|60.2|https://www.prnewswire.com/news-releases/pmi-at-60-2-june-manufacturing-ism-report-on-business-300674956.html
2018-08-01|58.1|https://www.prnewswire.com/news-releases/pmi-at-58-1-july-manufacturing-ism-report-on-business-300689662.html
2018-09-04|61.3|https://www.prnewswire.com/news-releases/pmi-at-61-3-august-manufacturing-ism-report-on-business-300704991.html
2018-10-01|59.8|https://www.prnewswire.com/news-releases/pmi-at-59-8-september-manufacturing-ism-report-on-business-300721195.html
2018-11-01|57.7|https://www.prnewswire.com/news-releases/pmi-at-57-7-october-manufacturing-ism-report-on-business-300740833.html
2018-12-03|59.3|https://www.prnewswire.com/news-releases/pmi-at-59-3-november-manufacturing-ism-report-on-business-300757695.html
2019-01-03|54.1|https://www.prnewswire.com/news-releases/pmi-at-54-1-december-manufacturing-ism-report-on-business-300770403.html
2019-02-01|56.6|https://www.prnewswire.com/news-releases/pmi-at-56-6-january-manufacturing-ism-report-on-business-300787873.html
2019-03-01|54.2|https://www.prnewswire.com/news-releases/pmi-at-54-2-february-manufacturing-ism-report-on-business-300804959.html
2019-04-01|55.3|https://www.prnewswire.com/news-releases/pmi-at-55-3-march-manufacturing-ism-report-on-business-300820736.html
2019-05-01|52.8|https://www.prnewswire.com/news-releases/pmi-at-52-8-april-manufacturing-ism-report-on-business-300841270.html
2019-06-03|52.1|https://www.prnewswire.com/news-releases/pmi-at-52-1-may-manufacturing-ism-report-on-business-300859919.html
2019-07-01|51.7|https://www.prnewswire.com/news-releases/pmi-at-51-7-june-manufacturing-ism-report-on-business-300877550.html
2019-08-01|51.2|https://www.prnewswire.com/news-releases/pmi-at-51-2-july-manufacturing-ism-report-on-business-300892730.html
2019-09-03|49.1|https://www.prnewswire.com/news-releases/pmi-at-49-1-august-manufacturing-ism-report-on-business-300909470.html
2019-10-01|47.8|https://www.prnewswire.com/news-releases/pmi-at-47-8-september-manufacturing-ism-report-on-business-300927916.html
2019-11-01|48.3|https://www.prnewswire.com/news-releases/pmi-at-48-3-gdp-growing-at-1-6-october-manufacturing-ism-report-on-business-300949478.html
2019-12-02|48.1|https://www.prnewswire.com/news-releases/pmi-at-48-1-gdp-growing-at-1-5-november-manufacturing-ism-report-on-business-300964884.html
2020-01-03|47.2|https://www.prnewswire.com/news-releases/pmi-at-47-2-gdp-growing-at-1-3-december-manufacturing-ism-report-on-business-300979292.html
2020-02-03|50.9|https://www.prnewswire.com/news-releases/pmi-at-50-9-gdp-growing-at-2-4-january-2020-manufacturing-ism-report-on-business-300997110.html
2020-03-02|50.1|https://www.prnewswire.com/news-releases/pmi-at-50-1-gdp-growing-at-2-1-february-2020-manufacturing-ism-report-on-business-301013570.html
2020-04-01|49.1|https://www.prnewswire.com/news-releases/pmi-at-49-1-march-2020-manufacturing-ism-report-on-business-301033054.html
2020-05-01|41.5|https://www.prnewswire.com/news-releases/pmi-at-41-5-april-2020-manufacturing-ism-report-on-business-301050619.html
2020-06-01|43.1|https://www.prnewswire.com/news-releases/pmi-at-43-1-may-2020-manufacturing-ism-report-on-business-301068035.html
2020-07-01|52.6|https://www.prnewswire.com/news-releases/pmi-at-52-6-june-2020-manufacturing-ism-report-on-business-301086436.html
2020-08-03|54.2|https://www.prnewswire.com/news-releases/pmi-at-54-2-july-2020-manufacturing-ism-report-on-business-301104442.html
2020-09-01|56.0|https://www.prnewswire.com/news-releases/pmi-at-56-0-august-2020-manufacturing-ism-report-on-business-301121423.html
2020-10-01|55.4|https://www.prnewswire.com/news-releases/pmi-at-55-4-september-2020-manufacturing-ism-report-on-business-301142396.html
2020-11-02|59.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-59-3-october-2020-manufacturing-ism-report-on-business-301164898.html
2020-12-01|57.5|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-57-5-november-2020-manufacturing-ism-report-on-business-301181920.html
2021-01-05|60.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-60-7-december-2020-manufacturing-ism-report-on-business-301200432.html
2021-02-01|58.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-58-7-january-2021-manufacturing-ism-report-on-business-301218544.html
2021-03-01|60.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-60-8-february-2021-manufacturing-ism-report-on-business-301236952.html
2021-04-01|64.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-64-7-march-2021-manufacturing-ism-report-on-business-301260206.html
2021-05-03|60.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-60-7-april-2021-manufacturing-ism-report-on-business-301281490.html
2021-06-01|61.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-61-2-may-2021-manufacturing-ism-report-on-business-301301816.html
2021-07-01|60.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-60-6-june-2021-manufacturing-ism-report-on-business-301323652.html
2021-08-02|59.5|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-59-5-july-2021-manufacturing-ism-report-on-business-301345349.html
2021-09-01|59.9|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-59-9-august-2021-manufacturing-ism-report-on-business-301366785.html
2021-10-01|61.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-61-1-september-2021-manufacturing-ism-report-on-business-301389267.html
2021-11-01|60.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-60-8-october-2021-manufacturing-ism-report-on-business-301412353.html
2021-12-01|61.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-61-1-november-2021-manufacturing-ism-report-on-business-301433617.html
2022-01-04|58.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-58-7-december-2021-manufacturing-ism-report-on-business-301449429.html
2022-02-01|57.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-57-6-january-2022-manufacturing-ism-report-on-business-301472006.html
2022-03-01|58.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-58-6-february-2022-manufacturing-ism-report-on-business-301492024.html
2022-04-01|57.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-57-1-march-2022-manufacturing-ism-report-on-business-301515115.html
2022-05-02|55.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-55-4-april-2022-manufacturing-ism-report-on-business-301536612.html
2022-06-01|56.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-56-1-may-2022-manufacturing-ism-report-on-business-301558163.html
2022-07-01|53.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-53-june-2022-manufacturing-ism-report-on-business-301579263.html
2022-08-01|52.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-8-july-2022-manufacturing-ism-report-on-business-301596347.html
2022-09-01|52.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-8-august-2022-manufacturing-ism-report-on-business-301615866.html
2022-10-03|50.9|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-50-9-september-2022-manufacturing-ism-report-on-business-301638361.html
2022-11-01|50.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-50-2-october-2022-manufacturing-ism-report-on-business-301663845.html
2022-12-01|49.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-november-2022-manufacturing-ism-report-on-business-301690752.html
2023-01-04|48.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-4-december-2022-manufacturing-ism-report-on-business-301712602.html
2023-02-01|47.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-4-january-2023-manufacturing-ism-report-on-business-301735466.html
2023-03-01|47.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-7-february-2023-manufacturing-ism-report-on-business-301758406.html
2023-04-03|46.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-3-march-2023-manufacturing-ism-report-on-business-301787309.html
2023-05-01|47.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-1-april-2023-manufacturing-ism-report-on-business-301811325.html
2023-06-01|46.9|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-9-may-2023-manufacturing-ism-report-on-business-301839274.html
2023-07-03|46.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-june-2023-manufacturing-ism-report-on-business-301868365.html
2023-08-01|46.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-4-july-2023-manufacturing-ism-report-on-business-301889651.html
2023-09-01|47.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-6-august-2023-manufacturing-ism-report-on-business-301915209.html
2023-10-02|49.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-september-2023-manufacturing-ism-report-on-business-301943535.html
2023-11-01|46.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-7-october-2023-manufacturing-ism-report-on-business-301973296.html
2023-12-01|46.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-7-november-2023-manufacturing-ism-report-on-business-302002512.html
2024-01-03|47.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-4-december-2023-manufacturing-ism-report-on-business-302024715.html
2024-02-01|49.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-1-january-2024-manufacturing-ism-report-on-business-302049663.html
2024-03-01|47.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-8-february-2024-manufacturing-ism-report-on-business-302076127.html
2024-04-01|50.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-50-3-march-2024-manufacturing-ism-report-on-business-302104149.html
2024-05-01|49.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-2-april-2024-manufacturing-ism-report-on-business-302132412.html
2024-06-03|48.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-7-may-2024-manufacturing-ism-report-on-business-302161049.html
2024-07-01|48.5|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-5-june-2024-manufacturing-ism-report-on-business-302186137.html
2024-08-01|46.8|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-8-july-2024-manufacturing-ism-report-on-business-302211321.html
2024-09-03|47.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-2-august-2024-manufacturing-ism-report-on-business-302235328.html
2024-10-01|47.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-2-september-2024-manufacturing-ism-report-on-business-302263093.html
2024-11-01|46.5|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-46-5-october-2024-manufacturing-ism-report-on-business-302293215.html
2024-12-02|48.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-4-november-2024-manufacturing-ism-report-on-business-302317575.html
2025-01-03|49.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-3-december-2024-manufacturing-ism-report-on-business-302341381.html
2025-02-03|50.9|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-50-9-january-2025-manufacturing-ism-report-on-business-302365753.html
2025-03-03|50.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-50-3-february-2025-manufacturing-ism-report-on-business-302388950.html
2025-04-01|49.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-march-2025-manufacturing-ism-report-on-business-302416129.html
2025-05-01|48.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-7-april-2025-manufacturing-ism-report-on-business-302443282.html
2025-06-02|48.5|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-5-may-2025-manufacturing-ism-report-on-business-302470183.html
2025-07-01|49.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-june-2025-manufacturing-ism-report-on-business-302494937.html
2025-08-01|48.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-july-2025-manufacturing-ism-report-on-business-302519014.html
2025-09-02|48.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-7-august-2025-ism-manufacturing-pmi-report-302543264.html
2025-10-01|49.1|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-49-1-september-2025-ism-manufacturing-pmi-report-302571530.html
2025-11-03|48.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-7-october-2025-ism-manufacturing-pmi-report-302601422.html
2025-12-01|48.2|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-48-2-november-2025-ism-manufacturing-pmi-report-302626979.html
2026-01-05|47.9|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-47-9-december-2025-ism-manufacturing-pmi-report-302649307.html
2026-02-02|52.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-6-january-2026-ism-manufacturing-pmi-report-302675443.html
2026-03-02|52.4|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-4-february-2026-ism-manufacturing-pmi-report-302699883.html
2026-04-01|52.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-7-march-2026-ism-manufacturing-pmi-report-302730721.html
2026-05-01|52.7|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-52-7-april-2026-ism-manufacturing-pmi-report-302759226.html
2026-06-01|54.0|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-54-may-2026-ism-manufacturing-pmi-report-302786165.html
2026-07-01|53.3|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-53-3-june-2026-ism-manufacturing-pmi-report-302814991.html
2026-08-03|55.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-55-6-july-2026-ism-manufacturing-pmi-report-302840669.html
2026-09-01|54.6|https://www.prnewswire.com/news-releases/manufacturing-pmi-at-54-6-august-2026-ism-manufacturing-pmi-report-302865127.html"""


def local_time(day, clock="00:00") -> pd.Timestamp:
    # UTC 변환 전에 현지 시각을 지정해야 EST/EDT에 관계없이 16시를 유지.
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}", tz=ZONE)


def release_time(name, row, first_version=True) -> pd.Timestamp:
    """ALFRED 날짜와 공식 발표 시각을 비교. 별도의 고정 일수 지연 없음."""
    native = row["realtime_start"]
    if name in RATES:
        # 현재 H.15의 전 영업일 자료 공표 구조로 일정 복원. 개별 과거 발표 시각은 미제공.
        scheduled = row["observation_date"] + FED_DAY
        if first_version and native > scheduled:
            return local_time(native)
        # 수정 시각의 기록은 없으므로 해당 Vintage일 16:15를 보수적으로 적용.
        return local_time(max(native, scheduled) if first_version else native, "16:15")
    if name == "usd_ret_20d":
        monday = native - timedelta(days=native.weekday())
        scheduled = FED_DAY.rollforward(monday)
        # 월요일 휴일이면 다음 연방 영업일. 이미 늦게 기록된 날짜에는 추가 지연 없음.
        return local_time(native) if native > scheduled else local_time(scheduled, "16:15")
    clock = "09:00" if name == "effr" else "08:30"
    return local_time(native, clock)


def prepare_pmi(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    calendar = {}
    for line in PMI_PUBLICATIONS.splitlines():
        day, value, url = line.split("|", 2)
        calendar[pd.Timestamp(day).to_period("M")] = (day, float(value), url)
    rows, audit = [], []
    for index, row in data.iterrows():
        original = row["release_date"]
        evidence = calendar.get(original.to_period("M"))
        verified = evidence is not None and np.isclose(row["value"], evidence[1], rtol=0, atol=1e-9)
        day = pd.Timestamp(evidence[0]) if verified else original
        # 확인 불가능한 날짜는 원본 유지. 시각 미상은 해당일 종료 후 사용하도록 명시.
        available = local_time(day, "10:00" if verified else "23:59:59")
        rows.append({**row.to_dict(), "row_id": index, "observation_date": day,
                     "event_day": day, "available_at": available})
        audit.append({
            "original_date": str(original.date()), "us_release_date": str(day.date()),
            "value": float(row["value"]), "verified": bool(verified),
            "changed": bool(original != day), "source": evidence[2] if verified else None,
            "us_release_at": available.isoformat(),
            "korea_date": str(available.tz_convert("Asia/Seoul").date()),
        })
    prepared = pd.DataFrame(rows).sort_values("available_at")
    prepared["expires_at"] = prepared["available_at"].shift(-1)
    prepared["expires_at"] = prepared["expires_at"].fillna(local_time("2026-10-01"))
    return prepared, audit


def prepare_vintages(name: str, data: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().sort_values(["observation_date", "realtime_start"])
    data["row_id"] = data.index
    data["event_day"] = data["realtime_start"]

    if name in PROXIES:
        kept, times = [], {}
        for observed, versions in data.groupby("observation_date", sort=False):
            numeric = np.flatnonzero(versions["value"].notna().to_numpy())
            # 최초 숫자만 관측일로 연결. 이후 수정·철회는 원래 Vintage일 이후에 반영.
            versions = versions.iloc[numeric[0]:] if len(numeric) else versions
            kept.extend(versions.index)
            times.update({index: local_time(day, "23:59:59") for index, day in versions["realtime_start"].items()})
            data.loc[versions.index[0], "event_day"] = observed
            times[versions.index[0]] = local_time(observed, "16:00")
        data = data.loc[kept].copy()
        data["available_at"] = pd.Series(times)
    elif name == "yield_spread":
        # 이미 계산된 금리차의 두 입력이 모두 알려진 시각만 계산. 수치 재계산 없음.
        lookup = {}
        for variable in ["ust_10y", "ust_2y"]:
            rates = raw.loc[raw["variable_name"].eq(variable)].sort_values(["observation_date", "realtime_start"])
            first = ~rates.duplicated("observation_date")
            for (_, row), initial in zip(rates.iterrows(), first):
                lookup[(variable, row["observation_date"], row["realtime_start"])] = release_time(variable, row, initial)
        times = []
        for _, row in data.iterrows():
            inputs = [local_time(row["realtime_start"])]
            for variable, observed, start in [
                ("ust_10y", row["observation_date"], row["input_realtime_start"]),
                ("ust_2y", row["base_observation_date"], row["base_realtime_start"]),
            ]:
                if pd.notna(start):
                    inputs.append(lookup[(variable, observed, start)])
            times.append(max(inputs))
        data["available_at"] = times
    else:
        initial = ~data.duplicated("observation_date")
        data["available_at"] = [release_time(name, row, first) for (_, row), first in zip(data.iterrows(), initial)]

    # PCE/달러: 04번의 계산 상태 전체가 교체됨. 그 외: 관측일별 Vintage 교체.
    grouping = "variable_name" if name in {"core_pce_yoy", "real_pce_mom", "usd_ret_20d"} else "observation_date"
    data = data.sort_values([grouping, "event_day", "available_at"])
    next_day = data.groupby(grouping)["event_day"].shift(-1)
    next_time = data.groupby(grouping)["available_at"].shift(-1)
    expiry_day = data["realtime_end"] + timedelta(days=1)
    expiry = pd.Series([local_time(day) for day in expiry_day], index=data.index)
    # 연속된 새 Vintage가 있으면 자정이 아니라 새 발표 시각에 이전 버전 교체.
    data["expires_at"] = expiry.where(next_day.ne(expiry_day), next_time)
    if (data["available_at"] >= data["expires_at"]).any():
        raise ValueError(f"{name}: 공개 시각과 유효기간의 순서가 잘못되었습니다.")
    return data.sort_values(["observation_date", "available_at"])


def select_daily(name: str, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """유효한 버전 중 최신 관측일 선택. NaN은 다른 유효 관측이 있을 때만 대체."""
    starts = pd.DatetimeIndex(data["available_at"]).asi8
    ends = pd.DatetimeIndex(data["expires_at"]).asi8
    observations = pd.DatetimeIndex(data["observation_date"]).asi8
    values = data["value"].to_numpy(dtype=float)
    selected = []
    for day in dates:
        cutoff = local_time(day, "16:00")
        active = (starts <= cutoff.value) & (cutoff.value < ends) & (observations <= day.value)
        valid = np.flatnonzero(active & np.isfinite(values))
        position = valid[-1] if len(valid) else None
        row = data.iloc[position] if position is not None else None
        latest = observations[active].max() if active.any() else None
        fallback = row is not None and row["observation_date"].value < latest
        selected.append({
            "date": day, "variable_name": name,
            "value": float(row["value"]) if row is not None else np.nan,
            "row_id": int(row["row_id"]) if row is not None else None,
            "observation_date": row["observation_date"] if row is not None else pd.NaT,
            "available_at": row["available_at"] if row is not None else pd.NaT,
            "expires_at": row["expires_at"] if row is not None else pd.NaT,
            "reason": "latest_observation_missing" if fallback else "latest_valid" if row is not None else "no_valid_value",
        })
    return pd.DataFrame(selected)


def merge_data(etf, returns, macro, raw):
    keys = ["date", "ticker"]
    dates = pd.DatetimeIndex(etf.loc[etf["ticker"].eq("SPY"), "date"].unique()).sort_values()
    if dates.min() != pd.Timestamp("2018-01-02") or dates.max() != pd.Timestamp("2026-09-30"):
        raise ValueError("SPY 기간이 2018-01-02 ~ 2026-09-30과 다릅니다.")
    if (set(etf["ticker"]) != set(TICKERS) or etf.duplicated(keys).any()
            or not etf.groupby("date").size().eq(15).all()
            or len(etf) != len(dates) * 15):
        raise ValueError("ETF 15종목·거래일 구성 또는 중복을 확인하세요.")
    if set(macro["variable_name"]) != set(MACRO):
        raise ValueError("거시경제지표 13개 구성이 일치하지 않습니다.")
    pd.testing.assert_frame_equal(
        etf[keys + ["log_return"]].sort_values(keys).reset_index(drop=True),
        returns.sort_values(keys).reset_index(drop=True), check_flags=False,
    )
    # 원지수의 발표일/수정 이력이 04번에서 변경되지 않았는지 원본과 대조.
    levels = set(MACRO) - {"core_pce_yoy", "real_pce_mom", "usd_ret_20d", "yield_spread"}
    order = ["variable_name", "observation_date", "realtime_start", "release_date"]
    pd.testing.assert_frame_equal(
        macro.loc[macro["variable_name"].isin(levels), raw.columns].sort_values(order).reset_index(drop=True),
        raw.loc[raw["variable_name"].isin(levels)].sort_values(order).reset_index(drop=True),
        check_flags=False,
    )
    audits, pmi_audit = [], []
    for name in MACRO:
        data = macro.loc[macro["variable_name"].eq(name)]
        if name == "ism_pmi":
            prepared, pmi_audit = prepare_pmi(data)
        else:
            prepared = prepare_vintages(name, data, raw)
        audits.append(select_daily(name, prepared, dates))
    audit = pd.concat(audits, ignore_index=True)
    daily = audit.pivot(index="date", columns="variable_name", values="value")[MACRO]
    result = etf.merge(daily, on="date", how="left", validate="many_to_one", sort=False)
    if len(result) != len(etf) or result.duplicated(keys).any():
        raise ValueError("결합 후 ETF 행 수 또는 고유성이 변경되었습니다.")
    pd.testing.assert_frame_equal(result[etf.columns], etf, check_flags=False)
    valid = audit["row_id"].notna()
    cutoffs = pd.Series([local_time(day, "16:00") for day in audit["date"]], index=audit.index)
    if (audit.loc[valid, "available_at"].gt(cutoffs[valid]).any()
            or audit.loc[valid, "expires_at"].le(cutoffs[valid]).any()
            or audit.loc[valid, "observation_date"].gt(audit.loc[valid, "date"]).any()):
        raise ValueError("이용 가능 시각·유효기간·관측일 검증에 실패했습니다.")
    return result, audit, pmi_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="05번 결과만 교체")
    args = parser.parse_args()
    path = ROOT / "processed/features_unscaled.parquet"
    if path.exists() and not args.overwrite:
        parser.error("05번 결과가 이미 있습니다. 교체하려면 --overwrite를 사용하세요.")
    paths = [ROOT / name for name in [
        "processed/etf_features.parquet", "processed/log_returns.parquet",
        "processed/macro_features.parquet", "raw/macro_observations.parquet",
    ]]
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    result, audit, pmi = merge_data(*[pd.read_parquet(p) for p in paths])
    result.attrs = {
        "date_range": ["2018-01-02", "2026-09-30"], "tickers": TICKERS,
        "information_cutoff": "16:00 America/New_York; DST-aware; ETF features not shifted",
        "scaling": "none; adj_close and unscaled log_return copied from 04",
        "macro_variables": MACRO, "input_sha256": hashes, "publication_sources": SOURCES,
        "timing_rules": {
            "core_pce_yoy/real_pce_mom/unemp": "native realtime_start at 08:30 ET; no extra day",
            "effr": "native realtime_start at 09:00 ET; no extra day; revisions before 16:00",
            "ust_2y/ust_10y/ust_30y/tips_10y": "max(native date, next federal business day after observation 16:15 ET); native date later than scheduled release: no extra delay; revision clock unknown: native date 16:15",
            "yield_spread": "max(04 calculation date, both original rate inputs' availability); value copied",
            "usd_ret_20d": "native date plus H.10 Monday/holiday-next-business-day 16:15 ET; already later native dates unchanged; value copied from 04",
            "ism_pmi": "verified ISM US release date 10:00 ET; unknown dates unchanged and available only after recorded day's end",
            "vix/wti": "FIRST nonmissing version at observation date 16:00 ET proxy; later revisions/withdrawals after native date's end; never backdate the latest revised price",
        },
        "timing_limitations": "Native ALFRED dates do not contain intraday timestamps. H.15/H.10 clocks use official schedules and federal holiday calendar, not a fully verified intraday archive; H.15 revisions use a conservative assumed clock.",
        "market_proxy_exceptions": {
            "vix": "User-approved same-day proxy: Cboe daily final index may include information through 16:15 ET, beyond ETF 16:00 cutoff; NOT strict intraday Point-in-Time.",
            "wti": "User-approved observation-day proxy: EIA retrospectively reported spot price, NOT a contemporaneously archived 16:00 market quote or strict Point-in-Time.",
        },
        "vintage_policy": "Choose latest observation among versions effective at cutoff; replacement shifts an adjacent native interval boundary to actual publication time. A gap/expiry remains invalid. NaN can use only a different still-valid observation; no global ffill/bfill.",
        "pmi_release_audit": pmi,
        "missing_counts": {k: int(v) for k, v in result.isna().sum().items()},
        "lineage": {name: [int(x) if pd.notna(x) else None for x in audit.loc[audit.variable_name.eq(name), "row_id"]] for name in MACRO},
        "lineage_note": "Each list follows sorted SPY dates; values are zero-based source row positions in macro_features.parquet (input SHA256 above); None means no valid value. select_daily returns detailed per-date validity/fallback audit.",
        "training_note": "Rows describe information at that date's close. PPO decision timing, initial missing rows, normalization and walk-forward splits are deferred to 06.",
    }
    if hashes != {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise ValueError("실행 중 입력 파일이 변경되었습니다.")
    temporary = path.with_suffix(".parquet.tmp")
    try:
        result.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"저장: {path} / {len(result):,}행 × {len(result.columns)}컬럼 / SPY {result.date.nunique():,}거래일")
    print("결측치:", result.attrs["missing_counts"])
    print(f"PMI 검증 {sum(x['verified'] for x in pmi)}/{len(pmi)}, 날짜 보정 {sum(x['changed'] for x in pmi)}, 미확인 {sum(not x['verified'] for x in pmi)}")
    print("기존 발표일 | 미국 발표일 | PMI | 변경 근거")
    for row in pmi:
        if row["changed"] or not row["verified"]:
            print(f"{row['original_date']} | {row['us_release_date']} | {row['value']} | {row['source'] or '미확인: 원본 날짜 유지'}")
    print("이전 유효 관측 사용(거래일):", audit.loc[audit.reason.eq("latest_observation_missing")].groupby("variable_name").size().to_dict())
    print("주의: VIX·WTI는 당일 시각/시장가격 근사값. 06번 작업 미진행.")


if __name__ == "__main__":
    main()
