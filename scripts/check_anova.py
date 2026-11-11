"""Check an existing ANOVA deployment in Edge using its real HTTP results."""

import argparse
import json
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def ready(page):
    expect(page.get_by_role("heading", name="ANOVA 검증", exact=True)).to_be_visible(
        timeout=60000
    )
    expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
        "data-test-script-state", "notRunning", timeout=30000
    )
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    response = httpx.get(f"{args.api}/performance/statistics", timeout=10)
    response.raise_for_status()
    results = response.json()["results"]
    assert results, "This browser check requires saved statistics results"
    expected_charts = sum(
        item["test"] == "one_way_anova"
        and any(value is not None for value in item["group_mean"].values())
        for item in results
    )
    output = Path("artifacts/ui/anova")
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    expect.set_options(timeout=30000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            page.goto(f"{args.base}/anova")
            ready(page)
            expect(page.locator('[data-testid="stPageLink"]')).to_have_count(6)
            expect(page.locator('[data-testid="stVegaLiteChart"]')).to_have_count(
                expected_charts
            )
            for title in (
                "One-way ANOVA",
                "시장 국면별 Two-way ANOVA",
                "Tukey HSD 사후검정",
            ):
                expect(
                    page.get_by_role("heading", name=title, exact=True)
                ).to_be_visible()
            expect(page.locator('[data-testid="stDataFrame"]').first).to_be_visible()
            page.get_by_role("button", name="통계 결과 새로고침", exact=True).click()
            page.wait_for_timeout(500)
            ready(page)
            overflow = page.evaluate("""() => {
                    const main = document.querySelector('[data-testid="stMain"]');
                    return main.scrollWidth - main.clientWidth;
                }""")
            assert overflow <= 1, (width, overflow)
            page.screenshot(path=str(output / f"anova-{width}.png"), full_page=True)
            page.get_by_role(
                "heading", name="Tukey HSD 사후검정", exact=True
            ).scroll_into_view_if_needed()
            page.screenshot(
                path=str(output / f"anova-posthoc-{width}.png"), full_page=True
            )
            page.get_by_role("link", name="리스크 모니터링", exact=False).click()
            expect(
                page.get_by_role("heading", name="리스크 모니터링", exact=True)
            ).to_be_visible()
            page.get_by_role("link", name="ANOVA 검증", exact=False).click()
            ready(page)
            checks.append(
                {"width": width, "overflow": overflow, "charts": expected_charts}
            )
        assert not errors, errors
        browser.close()
    (output / "verification.json").write_text(
        json.dumps({"http_status": response.status_code, "checks": checks}, indent=2),
        encoding="utf-8",
    )
    print("PASS: real statistics HTTP, desktop/mobile, charts, refresh and navigation")


if __name__ == "__main__":
    main()
