"""ETF and period choice shared by ETF analysis and its detail pages.

Every page reads and writes the same plain keys ("etf_choice", "window_choice"),
so a page opened from another one starts on the ETF the visitor was looking at.
Other pages set only these plain keys, never a widget key: Streamlit does not
reliably show a widget value set from a page that does not draw that widget.
"""

import streamlit as st

from frontend.shap_explanations import ASSET_NAMES
from frontend.ui import keep_choice

# Trading-day windows the correlation API computes, shown as months and years.
WINDOWS = {20: "1개월", 60: "3개월", 120: "6개월", 252: "1년"}


PAGES = ("correlation", "shap", "price", "relations")


def forget_other_pages(prefix, kind):
    """Drop other pages' widget keys so they reopen from the shared choice."""
    for page in PAGES:
        if page != prefix:
            st.session_state.pop(f"{page}_{kind}", None)


def open_on(ticker):
    """Make the next ETF page start on this ETF."""
    st.session_state["etf_choice"] = ticker
    forget_other_pages(None, "asset")


def choose_asset(prefix, options=None, where=st):
    """ETF selector; None when the shared ETF is not among this page's options.

    A page with fewer options (SHAP has only the ETFs with saved decisions) must
    not silently show its first option and then publish it as the visitor's
    choice: Streamlit resets an unavailable value to the first option. The
    selector is left empty instead, the shared choice is kept, and the page says
    the requested ETF has no record. Picking an ETF there updates the choice.
    """
    key = f"{prefix}_asset"
    names = list(options or ASSET_NAMES)
    keep_choice(key, "etf_choice")
    # A first visit with no shared choice keeps the default first option.
    if st.session_state.get(key, names[0]) not in names:
        st.session_state[key] = None

    def picked():
        if st.session_state.get(key) is not None:
            st.session_state["etf_choice"] = st.session_state[key]

    ticker = where.selectbox(
        "ETF",
        names,
        format_func=lambda t: f"{t} · {ASSET_NAMES.get(t, t)}",
        key=key,
        placeholder="ETF를 고르세요",
        on_change=picked,
    )
    if ticker is not None:
        st.session_state["etf_choice"] = ticker
    forget_other_pages(prefix, "asset")
    return ticker


def choose_window(prefix, label="분석 기간", where=st):
    key = f"{prefix}_window"
    keep_choice(key, "window_choice")
    # Default through session state only, so Streamlit never sees two defaults.
    st.session_state.setdefault(key, 60)
    window = where.selectbox(
        label,
        list(WINDOWS),
        format_func=lambda n: f"{WINDOWS[n]} ({n}거래일)",
        key=key,
    )
    st.session_state["window_choice"] = window
    forget_other_pages(prefix, "window")
    return window


def choose_etf(prefix):
    """ETF and period side by side (detail pages)."""
    a, b = st.columns(2)
    return choose_asset(prefix, where=a), choose_window(prefix, where=b)


def etf_name(name):
    return f"{ASSET_NAMES[name]}({name})" if name in ASSET_NAMES else name
