from datetime import date
from urllib.parse import urlparse
import streamlit as st
from frontend import research_client
from frontend.ui import page_heading

page_heading(
    "에이전트 리서치",
    "AGENT RESEARCH / 04",
    "투자 질문을 조사하고 뉴스 근거와 출처를 확인하세요.",
)
st.caption(
    "공유 뉴스 DB는 과거 자료입니다. 데모 기준일은 2023~2025년으로 선택하세요. "
    "질문과 검색된 근거는 답변 생성을 위해 Claude API로 전송됩니다."
)
try:
    status = research_client.status()
except research_client.ResearchAPIError as exc:
    st.error(str(exc))
    status = {"configured": False, "messages": []}
for message in status["messages"]:
    st.info(message)
with st.form("research_request"):
    question = st.text_area(
        "질문", placeholder="최근 QQQ에 영향을 주는 위험 뉴스를 알려줘", max_chars=2000
    )
    as_of = st.date_input("기준일 (이 날짜까지의 뉴스 검색)", value=date(2025, 4, 10))
    ticker = st.selectbox(
        "ETF",
        [
            "전체",
            "SPY",
            "QQQ",
            "SMH",
            "SCHD",
            "XLE",
            "EFA",
            "SHY",
            "IEF",
            "TLT",
            "TIP",
            "LQD",
            "GLD",
            "DBB",
            "DBA",
            "VNQ",
        ],
    )
    submit = st.form_submit_button("리서치 시작", disabled=not status["configured"])
if submit:
    st.session_state.pop("research_result", None)
    if not question.strip():
        st.warning("질문을 입력하세요.")
    else:
        try:
            with st.spinner("뉴스 검색과 근거 검증 후 답변을 작성하고 있습니다…"):
                st.session_state["research_result"] = research_client.research(
                    question.strip(),
                    as_of.isoformat(),
                    None if ticker == "전체" else ticker,
                )
        except research_client.ResearchAPIError as exc:
            st.error(str(exc))
result = st.session_state.get("research_result")
if result:
    st.subheader("리서치 리포트와 출처")
    st.caption(f"{result['as_of']} 기준 · {result['question']}")
    st.write(result["answer"])
    for source in result["sources"]:
        st.write(
            f"[{source['n']}] {source['title']} · {source['date']} · {source['source']}"
        )
        if urlparse(source["url"]).scheme in {"http", "https"}:
            st.link_button(f"원문 [{source['n']}]", source["url"])
    st.subheader("리스크 태그")
    st.caption(
        "이번 질문의 근거에서 생성한 태그이며 PPO 학습용 CSV를 변경하지 않습니다."
    )
    if result["risk_tags"]:
        st.dataframe(result["risk_tags"], hide_index=True)
    else:
        st.info("이번 답변에서 생성한 위험 태그가 없습니다.")
    with st.expander("검색·검증 활동 기록 (완료 후 표시)"):
        for entry in result["log"]:
            st.text(entry)
    st.caption(result["disclaimer"])
