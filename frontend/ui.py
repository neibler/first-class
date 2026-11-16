"""Shared light theme, four-item navigation, detail links and API readiness."""

from pathlib import Path

import streamlit as st

from frontend import api_client

ASSETS = Path(__file__).parent / "assets"

# Pages shown in the left menu, in order.
MENU = [
    ("portfolio", "포트폴리오", "flight_takeoff"),
    ("risk", "위험도·설명", "shield"),
    ("correlation", "자산 상세", "query_stats"),
    ("research", "AI 도우미", "smart_toy"),
]
# Verification and evidence pages opened by a button, with the menu they belong to.
DETAILS = {
    "performance": ("전략 성과 비교", "trending_up", "portfolio"),
    "shap": ("AI 판단 요인", "hub", "risk"),
    "anova": ("ANOVA 통계 검증", "analytics", "risk"),
}
PAGE_INFO = MENU + [(slug, title, icon) for slug, (title, icon, _) in DETAILS.items()]
HEALTH_NOTES = {
    "portfolio": "포트폴리오 화면은 저장 모델의 백테스트와 추론 API를 사용합니다.",
    "performance": (
        "이 화면은 활성 투자 모델의 백테스트 API를 사용합니다. "
        "모델이 로드되지 않으면 성과를 표시할 수 없습니다."
    ),
    "risk": (
        "위험도는 활성 투자 모델의 백테스트 API를, AI 판단 요인은 저장된 SHAP 결과를 사용합니다."
    ),
    "shap": "이 화면은 활성 투자 모델에 저장된 SHAP 결과를 조회합니다.",
    "correlation": (
        "자산 상세는 서버의 시장 데이터로 계산한 관계를 조회합니다. "
        "투자 모델 로드 여부와는 별개입니다."
    ),
    "research": (
        "/health는 서버 연결 상태입니다. 리서치에 필요한 키·뉴스 DB·임베딩 모델의 "
        "준비 여부는 분석 요청에서 확인하며, model_loaded는 리서치 준비 상태가 아닙니다."
    ),
    "anova": (
        "이 화면은 저장된 통계 결과를 조회합니다. /health의 투자 모델 상태와 "
        "SHAP 준비 여부는 통계 결과 조회 가능 여부와 별개입니다."
    ),
}


def menu_of(slug):
    """The left-menu item a page belongs to (detail pages highlight their parent)."""
    return DETAILS[slug][2] if slug in DETAILS else slug


def styles(slug):
    css = (ASSETS / "theme.css").read_text(encoding="utf-8")
    st.html("<style>" + css.replace("__CURRENT_PAGE__", menu_of(slug)) + "</style>")


def brand():
    with st.container(key="fc-brand", gap=None):
        st.markdown(":material/flight: **FIRST CLASS**")
        st.caption("퍼스트 클래스")


def navigation():
    with st.container(key="fc-nav", gap="small"):
        for slug, title, icon in MENU:
            with st.container(key=f"nav-{slug}"):
                st.page_link(
                    f"app_pages/{slug}.py",
                    label=title,
                    icon=f":material/{icon}:",
                    width="stretch",
                )


def page_heading(title, eyebrow, description):
    with st.container(key="page-heading"):
        st.caption(eyebrow)
        st.title(title)
        if description:
            st.caption(description)


def card(name):
    """A white rounded card; `name` must be unique on the page."""
    return st.container(key=f"card-{name}")


def detail_link(slug, label=None):
    """Button-like link that opens a verification or evidence page."""
    title, icon, _ = DETAILS[slug]
    with st.container(key=f"detail-{slug}"):
        st.page_link(
            f"app_pages/{slug}.py",
            label=label or f"{title} 보기",
            icon=f":material/{icon}:",
        )


def back_link(slug):
    """Return from a detail page to the menu page it belongs to."""
    parent = menu_of(slug)
    title = next(name for key, name, _ in MENU if key == parent)
    with st.container(key=f"back-{slug}"):
        st.page_link(
            f"app_pages/{parent}.py",
            label=f"{title}(으)로 돌아가기",
            icon=":material/arrow_back:",
        )


@st.fragment
def health_panel(slug):
    with st.expander("API 준비 상태"):
        st.button("연결 상태 새로고침", key="refresh_health")
        try:
            health = api_client.get_health()
        except api_client.APIError as exc:
            st.error(str(exc))
        else:
            st.success("API 연결 정상")
            if health.model_loaded:
                st.info("모델 로드됨")
            else:
                st.warning("모델이 로드되지 않았습니다.")
        st.caption(HEALTH_NOTES.get(slug, "/health 실시간 조회"))
