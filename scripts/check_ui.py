"""Optional real-browser UI check (pip install playwright; installed Edge).

Run against normal :8501 plus FIRST_CLASS_UI_DEV=1 on :8502:
python scripts/check_ui.py --base http://localhost:8501 --dev http://localhost:8502
Not a runtime dependency or a replacement for the existing CI smoke check.
"""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def run(base, dev, output):
    # Large local image assets may still be streaming after a page link click.
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
        replay = page.get_by_role("button", name="비행 경로 재생")
        expect(replay).to_be_visible(timeout=30000)
        # Wait for the first server run (including HTTP health) to complete.
        # Testing during that initial stream could race a component remount.
        expect(page.get_by_text("API 연결 정상", exact=True)).to_be_attached(
            timeout=30000
        )
        expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
            "data-test-script-state", "notRunning", timeout=30000
        )
        page.wait_for_timeout(500)
        count = len(frames)
        plane = page.locator(".flight-root svg g[aria-label]")
        start = plane.get_attribute("transform")
        replay.click()
        expect(plane).not_to_have_attribute("transform", start, timeout=2000)
        assert len(frames) == count, "Animation sent a Python rerun event"
        # Navigation stays responsive while the replay is active.
        page.get_by_role("link", name="SHAP 해석").click()
        expect(
            page.get_by_role("heading", name="SHAP 해석", exact=True)
        ).to_be_visible()
        expect(page.get_by_role("combobox")).to_have_count(2)
        asset, date = page.get_by_role("combobox").all()
        asset.click()
        page.get_by_role("option", name="QQQ", exact=True).click()
        date.click()
        page.get_by_role("option", name="2026-04-30", exact=True).click()
        expect(page.get_by_text("QQQ · 미국 기술주", exact=False)).to_be_visible()
        expect(page.locator('[data-testid="stVegaLiteChart"]').first).to_be_visible()
        force = page.frame_locator("iframe").locator('svg[role="img"]')
        expect(force).to_be_visible()
        assert force.locator("polygon").count() == 6
        border = page.locator(".st-key-nav-shap a").evaluate(
            "e => getComputedStyle(e).borderColor"
        )
        assert border == "rgb(244, 196, 125)", border
        for title in [
            "강화학습 성과",
            "에이전트 리서치",
            "ANOVA 검증",
            "리스크 모니터링",
        ]:
            page.get_by_role("link", name=title, exact=False).click()
            expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()
            if title == "에이전트 리서치":
                # Inspect the form only; never trigger a potentially paid analysis.
                expect(
                    page.get_by_role("textbox", name="질문", exact=True)
                ).to_be_visible()
                expect(page.locator('[data-testid="stDateInput"]')).to_be_visible()
                expect(
                    page.get_by_role("button", name="분석 실행", exact=True)
                ).to_be_visible()
            else:
                expect(page.get_by_text("화면 준비 중", exact=True)).to_be_visible()
        page.get_by_role("link", name="포트폴리오 현황").click()
        expect(
            page.get_by_role("heading", name="포트폴리오 현황", exact=True)
        ).to_be_visible()
        page.get_by_role("combobox").click()
        page.get_by_role("option", name="최근 63개 관측일").click()
        expect(page.get_by_text("63개 일간 수익률", exact=False)).to_be_visible()
        page.emulate_media(reduced_motion="reduce")
        expect(
            page.get_by_role("button", name="동작 감소 · 정적 표시")
        ).to_be_disabled()
        page.emulate_media(reduced_motion="no-preference")
        for width, height, label in [(1440, 1000, "desktop"), (390, 844, "mobile")]:
            page.set_viewport_size({"width": width, "height": height})
            for route in ["", "/shap"]:
                page.goto(base + route)
                expect(page.locator("h1")).to_be_visible()
                expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
                    "data-test-script-state", "notRunning", timeout=30000
                )
                page.wait_for_timeout(1000)
                assert page.locator('[data-testid="stException"]').count() == 0
                assert page.locator('[data-testid="stMain"]').evaluate(
                    "e => e.scrollWidth <= e.clientWidth + 1"
                ), f"Horizontal page overflow: {width}, {route}"
                name = "shap" if route else "portfolio"
                page.screenshot(path=str(output / f"{name}-{label}.png"))
                target = page.locator(".st-key-force" if route else ".st-key-doors")
                target.evaluate("e => e.scrollIntoView({block:'center'})")
                page.wait_for_timeout(300)
                page.screenshot(path=str(output / f"{name}-{label}-lower.png"))
        if dev:
            for route in ["", "/shap"]:
                for state in ["loading", "empty", "error"]:
                    context = browser.new_context()
                    test = context.new_page()
                    test.goto(f"{dev}{route}?ui_state={state}")
                    label = "불러오기 완료" if state == "loading" else "다시 시도"
                    button = test.get_by_role("button", name=label, exact=True)
                    expect(button).to_be_visible(timeout=30000)
                    assert test.get_by_role("link").count() >= 5
                    button.click()
                    expect(button).not_to_be_visible()
                    expect(
                        test.locator('[data-testid="stVegaLiteChart"]').first
                    ).to_be_visible()
                    assert test.locator('[data-testid="stException"]').count() == 0
                    context.close()
        assert not errors, errors
        browser.close()
    print(
        "PASS: menus, filters, animation without reruns, reduced motion, "
        "responsive UI, states"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--dev", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui"))
    args = parser.parse_args()
    run(args.base.rstrip("/"), args.dev, args.output)
