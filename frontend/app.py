"""First Class UI shell: brand, four-item menu, the selected page, API readiness."""

import streamlit as st

from frontend.ui import PAGE_INFO, back_link, brand, health_panel, navigation, styles
from frontend.ui import DETAILS

st.set_page_config(
    page_title="FIRST CLASS · 퍼스트 클래스",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page(
        f"app_pages/{slug}.py",
        title=title,
        icon=f":material/{icon}:",
        url_path=slug,
        default=slug == "portfolio",
    )
    for slug, title, icon in PAGE_INFO
]
page = st.navigation(pages, position="hidden")
slug = page.url_path or "portfolio"
styles(slug)

rail, content = st.columns([1, 4.6], gap="medium")
with rail:
    with st.container(key="fc-rail"):
        brand()
        navigation()
with content:
    if slug in DETAILS:
        back_link(slug)
    page.run()
    with st.container(key="fc-footer"):
        st.caption(
            "교육용 데모 · 본 시스템은 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없습니다. "
            "백테스팅 성과는 미래 수익을 보장하지 않습니다."
        )
    health_panel(slug)
