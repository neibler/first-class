"""Shared analysis cabin. Native containers own all interactive content.

The established SHAP keys and asset crops stay stable to preserve its layout.
Only the active navigation item and page.run() body vary between analysis pages.
"""

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from frontend.ui import navigation

ASSETS = Path(__file__).parent / "assets"


@lru_cache(maxsize=32)
def asset(name, modified):
    data = (ASSETS / "cabin/trimmed" / f"{name}.webp").read_bytes()
    return "url('data:image/webp;base64," + base64.b64encode(data).decode() + "')"


def decoration(name):
    st.html(f'<div class="{name}" aria-hidden="true"></div>')


def render(pages, page):
    css = (ASSETS / "shap.css").read_text(encoding="utf-8")
    css = css.replace("__CURRENT_PAGE__", page.url_path)
    names = [
        "cabin-ceiling-navy",
        "cabin-window-panel",
        "cabin-wine-panel-top",
        "cabin-wine-panel-body",
        "cabin-wine-panel-base",
        "cabin-seat-armrests",
    ]
    names += [
        f"monitor-frame-{side}-{part}"
        for side in ("top", "bottom")
        for part in ("start", "middle", "end")
    ]
    names += [f"monitor-frame-{side}-middle" for side in ("left", "right")]
    variables = []
    for name in names:
        modified = (ASSETS / "cabin/trimmed" / f"{name}.webp").stat().st_mtime_ns
        variables.append(f"--{name}:{asset(name, modified)};")
    st.html("<style>" + css + ".st-key-shap-cabin{" + "".join(variables) + "}</style>")
    with st.container(key="shap-cabin", gap=None):
        st.html(
            '<header class="shap-ceiling"><div class="shap-brand">'
            "FIRST CLASS <span>· 로보어드바이저</span></div></header>"
        )
        with st.container(key="shap-main", gap=None):
            rail, display = st.columns([1, 4.7], gap=None)
            with rail:
                with st.container(key="shap-rail", gap=None):
                    decoration("cabin-window-panel")
                    with st.container(key="shap-nav", gap="small"):
                        navigation(pages)
                    with st.container(key="shap-wine", gap=None):
                        st.html(
                            '<div class="cabin-wine-panel" aria-hidden="true">'
                            '<div class="wine-top"></div><div class="wine-body"></div>'
                            '<div class="wine-base"></div></div>'
                        )
            with display:
                with st.container(key="shap-monitor", gap=None):
                    st.html(
                        '<div class="monitor-frame" aria-hidden="true">'
                        '<div class="frame-top"><i></i><i></i><i></i></div>'
                        '<div class="frame-left"></div><div class="frame-right"></div>'
                        '<div class="frame-bottom"><i></i><i></i><i></i></div></div>'
                    )
                    with st.container(key="shap-screen", gap="small"):
                        page.run()
        st.html(
            '<div class="cabin-lower" aria-hidden="true">'
            '<div class="wine-footprint"></div>'
            '<div class="cabin-seat-armrests"></div></div>'
        )
