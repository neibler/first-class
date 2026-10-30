"""First Class UI shell. Analysis samples and the real API are independent."""

import streamlit as st

from frontend.ui import PAGE_INFO, header, health_panel, styles
from frontend.shap_cabin import render as analysis_cabin

st.set_page_config(
    page_title="FIRST CLASS · 로보어드바이저",
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
cockpit = page.title == "포트폴리오 현황"
if cockpit:
    styles()
    header()
    page.run()
else:
    analysis_cabin(pages, page)

st.caption(
    "교육용 데모 · 본 시스템은 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없습니다. "
    "백테스팅 성과는 미래 수익을 보장하지 않습니다."
)
health_panel(
    sample=cockpit or page.url_path == "shap", research=page.url_path == "research"
)
