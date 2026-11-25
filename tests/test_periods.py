from frontend.periods import period_options


def test_choices_follow_the_data_length():
    # About 9 months of daily returns: no 1-year choice yet.
    assert period_options(187) == {"1개월": 21, "3개월": 63, "6개월": 126, "전체": None}
    # Every whole year shorter than the data appears after the months.
    assert list(period_options(600)) == [
        "1개월",
        "3개월",
        "6개월",
        "1년",
        "2년",
        "전체",
    ]


def test_a_choice_as_long_as_the_data_is_left_out():
    assert list(period_options(63)) == ["1개월", "전체"]
    assert list(period_options(252)) == ["1개월", "3개월", "6개월", "전체"]
    assert list(period_options(5)) == ["전체"]
