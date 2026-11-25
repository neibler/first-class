"""Analysis-period choices in months and years instead of trading days."""

MONTH = 21  # trading days in about one month
YEAR = 252  # trading days in one year
ALL = "전체"


def period_options(observations):
    """Label -> trading-day count (None = whole period), only for lengths the data has.

    1, 3 and 6 months, then every whole year shorter than the data, then "전체".
    A choice as long as the data itself would just repeat "전체", so it is left out.
    """
    options = {f"{m}개월": m * MONTH for m in (1, 3, 6) if m * MONTH < observations}
    years = 1
    while years * YEAR < observations:
        options[f"{years}년"] = years * YEAR
        years += 1
    options[ALL] = None
    return options
