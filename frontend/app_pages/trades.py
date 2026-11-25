"""Trade history by date, and the reference "start today" proposal."""

from frontend.portfolio_sections import fresh_start, load_changes, why_changed
from frontend.ui import page_heading

page_heading(
    "사고판 내역",
    "REPORT / TRADES",
    "날짜를 골라 AI가 그날 무엇을 얼마나 사고팔았는지와 그 이유를 확인하세요.",
)
why_changed(load_changes())
fresh_start()
