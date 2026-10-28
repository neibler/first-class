"""Shared cabin styling, native navigation, and explicit sample states."""

import base64
import logging
import os
from functools import lru_cache
from pathlib import Path

import streamlit as st

from frontend import api_client
from frontend.sample_data import SampleDataError

ASSETS = Path(__file__).parent / "assets"
LOGGER = logging.getLogger(__name__)
COCKPIT_ASSETS = {
    "ceiling": "trimmed/ceiling.webp",
    "pillar-left": "trimmed/pillar-left-upright.webp",
    "pillar-right": "trimmed/pillar-right-upright.webp",
    "windshield-sky": "trimmed/windshield-sky.webp",
    "control-console": "trimmed/control-console-integrated.webp",
    "instrument-top-bar": "trimmed/instrument-top-bar.webp",
    "instrument-right": "trimmed/instrument-side-right-flat.webp",
}


@lru_cache(maxsize=40)
def _asset_uri(filename, modified_ns):
    """Local assets travel over HTTP as data URIs; mtime invalidates the cache."""
    encoded = base64.b64encode((ASSETS / "cockpit" / filename).read_bytes()).decode()
    return f"url('data:image/webp;base64,{encoded}')"


PAGE_INFO = [
    ("portfolio", "포트폴리오 현황", "dashboard"),
    ("performance", "강화학습 성과", "trending_up"),
    ("shap", "SHAP 해석", "hub"),
    ("research", "에이전트 리서치", "travel_explore"),
    ("anova", "ANOVA 검증", "analytics"),
    ("risk", "리스크 모니터링", "shield"),
]


def styles():
    """Load the portfolio's scoped stylesheet and current display assets."""
    css = (ASSETS / "portfolio.css").read_text(encoding="utf-8")
    declarations = []
    for name, filename in COCKPIT_ASSETS.items():
        modified = (ASSETS / "cockpit" / filename).stat().st_mtime_ns
        declarations.append(f"--cockpit-{name}:{_asset_uri(filename, modified)};")
    variables = "".join(declarations)
    css += f"\n.stApp:has(.st-key-flight-window) {{{variables}}}"
    st.html(f"<style>{css}</style>")


def header():
    """Portfolio ceiling; the analysis cabin owns its own ceiling."""
    st.html(
        '<header class="cockpit-ceiling">'
        '<div class="cockpit-brand"><div>FIRST CLASS <span>· 로보어드바이저</span></div>'
        "<small>UI PREVIEW</small></div>"
        "</header>"
    )


def portfolio_navigation():
    """Use the same six registered pages inside the portfolio's native console."""
    with st.container(key="doors", gap="small"):
        for slug, title, icon in PAGE_INFO:
            with st.container(key=f"nav-{slug}"):
                st.page_link(
                    f"app_pages/{slug}.py",
                    label=title,
                    icon=f":material/{icon}:",
                    width="stretch",
                )


def navigation(pages):
    for page in pages:
        with st.container(key=f"nav-{page.url_path or 'portfolio'}"):
            st.page_link(page, label=page.title, icon=page.icon, width="stretch")
    # Active-page styling is owned by the shared cabin's scoped stylesheet.


def page_heading(title, eyebrow, description):
    with st.container(key="page-heading"):
        st.caption(eyebrow)
        st.title(title)
        if description:
            st.caption(description)


def sample_badge():
    st.badge("샘플 데이터", icon=":material/science:", color="blue")


def load_sample(loader, key):
    """API replacement seam: replace loader, never silently fall back on errors."""
    mode = "ready"
    if os.getenv("FIRST_CLASS_UI_DEV") == "1":
        mode = st.query_params.get("ui_state", "ready")
        if st.session_state.get(f"recovered_{key}") == mode:
            mode = "ready"
    if mode == "loading":
        st.info("샘플 데이터를 불러오는 중입니다…")
        st.skeleton(height=280)
        if st.button("불러오기 완료", key=f"finish_{key}"):
            st.session_state[f"recovered_{key}"] = mode
            st.rerun()
        return None
    try:
        with st.spinner("샘플 데이터를 불러오는 중입니다…"):
            if mode == "error":
                raise SampleDataError("샘플 데이터 로드에 실패했습니다.")
            data = None if mode == "empty" else loader()
        if not data:
            st.info("표시할 샘플 결과가 없습니다.")
        else:
            return data
    except (SampleDataError, OSError, ValueError, KeyError) as exc:
        # Never log exception text/tracebacks: they can contain paths or data.
        LOGGER.warning("SAMPLE_LOAD_FAILED: %s", type(exc).__name__)
        st.error(
            "샘플 데이터 오류: 데이터를 불러오지 못했습니다. 다시 시도해 주세요. "
            "계속되면 관리자에게 샘플 파일 확인을 요청해 주세요."
        )
    if st.button("다시 시도", key=f"retry_{key}"):
        st.session_state[f"recovered_{key}"] = mode
        st.rerun()
    return None


@st.fragment
def health_panel(sample=True, investment=False, research=False):
    with st.expander("API 연결 및 모델 준비 상태"):
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
        if investment:
            st.caption(
                "포트폴리오 화면은 저장 모델의 백테스트와 추론 API를 사용합니다."
            )
        elif research:
            st.caption(
                "리서치 준비 상태는 화면 상단에서 확인하세요. 투자 모델의 로드 여부와 별개입니다."
            )
        elif sample:
            st.caption(
                "/health 실시간 조회 · 위 분석은 UI 개발용 샘플이며 실제 모델 결과가 아닙니다."
            )
        else:
            st.caption(
                "/health 실시간 조회 · 이 화면의 분석 API와 데이터는 아직 준비되지 않았습니다. "
                "API 연결 정상은 분석 기능의 준비 완료를 뜻하지 않습니다."
            )
