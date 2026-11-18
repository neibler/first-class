"""Optional real-browser UI check (pip install playwright; installed Edge).

Run against a running API and dashboard:
python scripts/check_ui.py --base http://localhost:8501
Not a runtime dependency or a replacement for the existing CI smoke check.
"""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

MENU = ["포트폴리오", "위험도·설명", "자산 상세", "AI 도우미"]
# (menu to open, link text on that page, heading of the detail page)
DETAILS = [
    ("포트폴리오", "다른 전략과 성과 비교하기", "강화학습 성과"),
    ("위험도·설명", "AI 판단 요인 자세히 보기", "SHAP 해석"),
    ("위험도·설명", "전략 차이가 통계적으로 의미 있는지 보기 (ANOVA)", "ANOVA 검증"),
]


def settled(page):
    expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
        "data-test-script-state", "notRunning", timeout=30000
    )
    assert page.locator('[data-testid="stException"]').count() == 0


def open_menu(page, name):
    page.locator(".st-key-fc-nav").get_by_role("link", name=name).click()
    expect(page.get_by_role("heading", name=name, exact=True)).to_be_visible()
    settled(page)


def run(base, output):
    expect.set_options(timeout=30000)
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, frames = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "websocket",
            lambda ws: ws.on("framesent", lambda frame: frames.append(frame)),
        )
        page.goto(base)
        expect(page.get_by_text("FIRST CLASS", exact=True)).to_be_visible()
        expect(page.locator(".st-key-fc-nav").get_by_role("link")).to_have_count(4)
        settled(page)

        # The flight replay runs in the browser only; it must not rerun Python.
        replay = page.get_by_role("button", name="비행 경로 재생")
        if replay.count():
            page.wait_for_timeout(500)
            count = len(frames)
            plane = page.locator(".flight-root svg g[aria-label]")
            start = plane.get_attribute("transform")
            replay.click()
            expect(plane).not_to_have_attribute("transform", start, timeout=2000)
            assert len(frames) == count, "Animation sent a Python rerun event"
        else:
            # Without an active model the page explains instead of showing data.
            expect(page.locator('[data-testid="stAlert"]').first).to_be_visible()

        for name in MENU:
            open_menu(page, name)
        for menu, link, heading in DETAILS:
            open_menu(page, menu)
            page.get_by_role("link", name=link).click()
            expect(
                page.get_by_role("heading", name=heading, exact=True)
            ).to_be_visible()
            settled(page)
            page.get_by_role("link", name="돌아가기", exact=False).click()
            expect(page.get_by_role("heading", name=menu, exact=True)).to_be_visible()

        for width, height, label in [(1440, 1000, "desktop"), (390, 844, "mobile")]:
            page.set_viewport_size({"width": width, "height": height})
            for name in MENU:
                open_menu(page, name)
                page.wait_for_timeout(500)
                assert page.locator('[data-testid="stMain"]').evaluate(
                    "e => e.scrollWidth <= e.clientWidth + 1"
                ), f"Horizontal page overflow: {width}, {name}"
                page.screenshot(
                    path=str(output / f"{MENU.index(name) + 1}-{label}.png"),
                    full_page=True,
                )
        assert not errors, errors
        browser.close()
    print("PASS: four menus, detail pages and back links, flight replay, responsive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui"))
    args = parser.parse_args()
    run(args.base.rstrip("/"), args.output)
