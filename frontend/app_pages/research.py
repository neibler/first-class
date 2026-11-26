"""Display the current outcome separately from one reusable session success."""

from datetime import date

import streamlit as st

from frontend import api_client
from frontend.guide import SUGGESTED_QUESTIONS
from frontend.ui import page_heading


def queue_research():
    """Snapshot form inputs once; a rerun never reconstructs a submitted request."""
    selected_date = st.session_state["research_date"]
    st.session_state["research_pending"] = {
        "question": st.session_state["research_question"].strip(),
        "as_of": selected_date.isoformat() if selected_date else None,
    }


STEP_NAMES = {
    "Planner": "① 질문 나누기",
    "Researcher": "② 뉴스 찾기",
    "Verifier": "③ 관련 있는 뉴스 고르기",
    "Rewriter": "③-1 검색어 고쳐 다시 찾기",
    "Analyst": "④ 근거로 답 쓰기",
}


def reasoning_steps(result):
    """The server's plan and step log, grouped by step. Nothing is recomputed."""
    with st.expander("AI가 답을 찾은 과정 보기"):
        st.caption(
            "AI는 질문을 검색 계획으로 나누고, 저장된 뉴스에서 찾은 뒤 관련 있는 것만 골라 "
            "답을 씁니다. 아래는 이번 답의 실제 진행 기록입니다."
        )
        if result.plan:
            # Plain text, not a table: the only tables on this page are the
            # evidence (risk tags), so a no-evidence answer shows none.
            st.markdown("**검색 계획**")
            for step in result.plan:
                etf = f" · {step.etf}" if step.etf else ""
                st.text(
                    f"· '{step.query}'{etf} · 기준일 전 {step.lookback_days}일",
                    width="stretch",
                )
        grouped = {}
        for line in result.log:
            role = line[1 : line.find("]")] if line.startswith("[") else ""
            text = line[line.find("]") + 1 :].strip() if role else line
            grouped.setdefault(STEP_NAMES.get(role, "기타"), []).append(text)
        for name, lines in grouped.items():
            st.markdown(f"**{name}**")
            for text in lines:
                st.text(f"· {text}", width="stretch")
        st.markdown(f"**찾은 뉴스 {len(result.sources)}건**")
        for source in result.sources:
            st.text(
                f"[{source.n}] {source.title} ({source.source} · {source.date})",
                width="stretch",
            )
        if not result.plan and not result.log:
            st.caption("제공된 진행 기록이 없습니다.")


@st.cache_data(ttl=300, show_spinner=False)
def news_coverage(api_base_url):
    """Server-reported news date range; the base URL keys the cache."""
    return api_client.get_research_coverage()


def out_of_coverage(as_of):
    """True only when the range is known and the date (YYYY-MM-DD) is outside."""
    known = news_coverage(api_client.get_api_base_url())
    if not known or not as_of:
        return False
    try:
        day = date.fromisoformat(as_of)
    except ValueError:
        return False
    return not known[0] <= day <= known[1]


def use_suggestion(question):
    """Fill the question box only; the user still presses 분석 실행."""
    st.session_state["research_question"] = question


st.session_state.setdefault("research_outcome", None)
st.session_state.setdefault("research_last_success", None)
api_base_url = api_client.get_api_base_url()
saved = st.session_state["research_last_success"]
if saved is not None and saved["api_base_url"] != api_base_url:
    st.session_state["research_last_success"] = None

page_heading(
    "AI 도우미",
    "AI ASSISTANT / 04",
    "궁금한 점을 물어보면 뉴스 근거와 출처를 찾아 답해 드려요.",
)
coverage = news_coverage(api_base_url)
with st.container(key="analysis-availability"):
    if coverage:
        st.caption(
            f"뉴스 DB 자료 범위는 **{coverage[0]:%Y-%m-%d} ~ {coverage[1]:%Y-%m-%d}**입니다. "
            "이 범위 밖의 기준일은 분석하지 않고 바로 알려 드립니다."
        )
    else:
        st.caption(
            "뉴스 DB 자료 범위를 확인하지 못했습니다. 범위 밖 기준일은 판단할 근거가 부족할 수 있습니다."
        )

with st.container(key="card-suggestions"):
    st.markdown("**이런 질문은 어때요?**")
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        st.button(
            question,
            key=f"research_suggestion_{index}",
            on_click=use_suggestion,
            args=(question,),
        )
    st.caption("질문을 누르면 아래 입력란에 채워집니다. 분석 실행을 눌러야 전송됩니다.")

with st.container(key="card-request"):
    st.subheader("질문하기")
    with st.form("research_form", enter_to_submit=False):
        st.text_area(
            "질문",
            placeholder="예: 최근 QQQ 위험?",
            max_chars=2000,
            key="research_question",
            persist_state="session",
            help="앞뒤 공백을 정리해 전송합니다. 최종 입력 검증은 API가 수행합니다.",
        )
        selected_date = st.date_input(
            "기준일",
            value=date(2025, 4, 10),
            # Linux strftime serializes year 1 as "1", which breaks AppTest's
            # YYYY-MM-DD parser. Keep calendar bounds portable across platforms.
            min_value=date(1000, 1, 1),
            max_value=date.max,
            format="YYYY-MM-DD",
            key="research_date",
            persist_state="session",
            help="선택한 날짜까지의 뉴스를 검색합니다.",
        )
        st.form_submit_button(
            "분석 실행",
            type="primary",
            key="research_submit",
            on_click=queue_research,
        )
    if selected_date and out_of_coverage(selected_date.isoformat()):
        st.warning(
            f"선택한 기준일이 뉴스 DB 자료 범위({coverage[0]:%Y-%m-%d} ~ "
            f"{coverage[1]:%Y-%m-%d}) 밖이라 분석하지 않습니다."
        )
    st.caption(
        "버튼을 누를 때만 처리합니다. 마지막 성공 결과와 질문·기준일이 같으면 현재 세션에서 재사용합니다. "
        "응답 대기 제한은 120초입니다. 처리 중 재클릭은 즉시 전달되지 않을 수 있으며, "
        "재클릭·대기 종료·페이지 이동은 서버 분석 취소나 재시작을 보장하지 않습니다."
    )

pending = st.session_state.pop("research_pending", None)
if pending is not None:
    st.session_state["research_outcome"] = None
    outcome = {"request": pending, "result": None, "error": None, "reused": False}
    saved = st.session_state["research_last_success"]
    # Keep existing widget constraints and API validation. Invalid requests must
    # still reach the API's 422 path, never bypass validation via a saved result.
    reusable_input = 1 <= len(pending["question"]) <= 2000 and pending["as_of"]
    try:
        if reusable_input and saved is not None and saved["request"] == pending:
            outcome["result"] = saved["result"]
            outcome["reused"] = True
        elif out_of_coverage(pending["as_of"]):
            # Refuse at once: outside the DB the server could only answer
            # "no evidence" after several paid calls. The server checks too.
            raise api_client.ResearchAPIError("research_out_of_range")
        else:
            with st.spinner("뉴스 리서치 응답을 기다리고 있습니다…"):
                result = api_client.post_research(**pending)
            outcome["result"] = result
            if result.status == "complete":
                st.session_state["research_last_success"] = {
                    "request": pending,
                    "result": result,
                    "api_base_url": api_base_url,
                }
    except api_client.ResearchAPIError as exc:
        outcome["error"] = exc.code
    st.session_state["research_outcome"] = outcome

outcome = st.session_state["research_outcome"]
if outcome is None:
    st.info("질문과 기준일을 입력한 뒤 분석 실행을 눌러 주세요.")
else:
    with st.container(key="card-chat"):
        with st.chat_message("user"):
            st.text(outcome["request"]["question"] or "(빈 질문)", width="stretch")
            st.text(f"요청 기준일: {outcome['request']['as_of'] or '미선택'}")
        st.caption(
            "위는 제출 당시의 질문입니다. 입력란을 수정해도 자동으로 다시 분석하지 않습니다."
        )
        with st.chat_message("assistant"):
            if outcome["error"]:
                st.error(str(api_client.ResearchAPIError(outcome["error"])))
            elif outcome["result"] is not None:
                result = outcome["result"]
                if outcome.get("reused"):
                    st.info("동일한 요청의 이전 분석 결과입니다.")
                if result.status == "insufficient_evidence":
                    st.warning(
                        "판단할 근거가 부족합니다. 위험이 없다는 의미는 아닙니다."
                    )
                else:
                    st.success("리서치가 완료되었습니다.")
                st.text(f"분석 기준일: {result.as_of.isoformat()}")
                st.text(result.answer, width="stretch")
                with st.expander(f"출처 보기 ({len(result.sources)}건)"):
                    for source in result.sources:
                        st.text(f"[{source.n}] {source.title}", width="stretch")
                        st.text(f"{source.source} · {source.date}", width="stretch")
                        url = api_client.safe_source_url(source.url)
                        if url:
                            st.link_button(f"원문 보기 [{source.n}]", url)
                        else:
                            st.caption(
                                "안전한 http/https URL이 아니므로 링크를 표시하지 않습니다."
                            )
                    if not result.sources:
                        st.caption("제공된 원문 출처가 없습니다.")
                with st.expander(f"위험 태그 보기 ({len(result.risk_tags)}건)"):
                    if result.risk_tags:
                        st.dataframe(
                            [
                                {
                                    "ETF": tag.ticker,
                                    "위험 유형": tag.risk_type,
                                    "뉴스 위험 지표": tag.risk_score,
                                    "위험 수준": tag.risk_level,
                                    "방향": tag.direction,
                                    "확신도": tag.confidence,
                                    "기준일": tag.date.isoformat(),
                                    "근거 번호": ", ".join(
                                        f"[{n}]" for n in tag.evidence
                                    ),
                                }
                                for tag in result.risk_tags
                            ],
                            hide_index=True,
                        )
                        st.caption(
                            "뉴스 위험 지표는 위험 수준 × 확신도이며 손실 확률이 아닙니다. "
                            "일별 CSV 위험 점수와도 다른 지표입니다."
                        )
                    else:
                        st.caption("제공된 위험 태그가 없습니다.")
                st.text(result.disclaimer, width="stretch")
                reasoning_steps(result)
                with st.expander("완료 후 실행 요약 로그"):
                    st.caption(
                        "응답에 포함된 실행 요약입니다. 실시간 진행 상황이 아닙니다."
                    )
                    st.text(
                        "\n".join(result.log) or "제공된 실행 요약이 없습니다.",
                        width="stretch",
                    )
