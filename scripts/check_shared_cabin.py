"""Real-browser checks for the shared cabin, navigation and short/long bodies."""

import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUTPUT = Path("artifacts/ui/shared-cabin")
PAGES = ("performance", "research", "anova", "risk", "shap")


def ready(page):
    expect(page.locator(".st-key-shap-screen h1")).to_be_visible(timeout=60000)
    expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
        "data-test-script-state", "notRunning", timeout=30000
    )
    page.wait_for_timeout(700)


def geometry(page):
    return page.evaluate("""() => {
      const box=s=>{const r=document.querySelector(s).getBoundingClientRect();
        return {x:r.x,y:r.y,width:r.width,height:r.height,
          right:r.right,bottom:r.bottom}};
      const main=document.querySelector('[data-testid="stMain"]');
      const links=[...document.querySelectorAll('.st-key-shap-nav a')];
      return {ceiling:box('.shap-ceiling'),nav:box('.st-key-shap-nav'),
        monitor:box('.st-key-shap-monitor'),screen:box('.st-key-shap-screen'),
        seat:box('.cabin-seat-armrests'),wine:box('.cabin-wine-panel'),
        frame:box('.monitor-frame'),overflow:main.scrollWidth-main.clientWidth,
        active:links.filter(a=>getComputedStyle(a).borderColor==='rgb(244, 196, 125)')
          .map(a=>new URL(a.href).pathname),
        pointer:getComputedStyle(document.querySelector('.monitor-frame')).pointerEvents};
    }""")


def assert_seams(result, width):
    assert result["overflow"] <= 1, result
    assert result["frame"] == result["monitor"], result
    assert result["pointer"] == "none"
    assert abs(result["monitor"]["bottom"] - result["seat"]["y"]) < 1, result
    assert result["screen"]["bottom"] <= result["monitor"]["bottom"] - 10, result
    assert abs(result["wine"]["y"] - result["nav"]["bottom"]) < 1, result
    if width > 900:
        assert abs(result["wine"]["bottom"] - result["seat"]["bottom"]) < 1, result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1700})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in (1440, 1100, 900, 768, 390):
            page.set_viewport_size({"width": width, "height": 1700})
            positions = None
            for slug in PAGES:
                page.goto(f"http://localhost:8501/{slug}")
                ready(page)
                expect(
                    page.locator('.st-key-shap-nav [data-testid="stPageLink"]')
                ).to_have_count(6)
                expect(page.locator('[data-testid="stException"]')).to_have_count(0)
                result = geometry(page)
                assert result["active"] == [f"/{slug}"], result
                assert_seams(result, width)
                current = {
                    key: (result[key]["x"], result[key]["y"], result[key]["width"])
                    for key in ("ceiling", "nav", "monitor")
                }
                if positions is None:
                    positions = current
                assert current == positions, (width, slug, current, positions)
                if slug != "shap":
                    expect(
                        page.locator(".st-key-analysis-availability")
                    ).to_be_visible()
                    expect(page.locator('[data-testid="stMetric"]')).to_have_count(0)
                    expect(page.locator('[data-testid="stSelectbox"]')).to_have_count(0)
                page.screenshot(
                    path=str(OUTPUT / f"{slug}-{width}.png"), full_page=True
                )
                results.append({"page": slug, "width": width, **result})
                print(f"PASS {slug} {width}", flush=True)

        # Exercise native page-link navigation in the same browser session.
        page.set_viewport_size({"width": 1440, "height": 1700})
        for slug in PAGES:
            page.locator(f".st-key-shap-nav .st-key-nav-{slug} a").click()
            expect(page).to_have_url(f"http://localhost:8501/{slug}")
            ready(page)
            assert geometry(page)["active"] == [f"/{slug}"]

        # Layout stress only: add text nodes, not fictional financial results.
        # This confirms the shared shell can grow, not real report API behavior.
        page.goto("http://localhost:8501/research")
        ready(page)
        page.evaluate("""() => {
          const screen=document.querySelector('.st-key-shap-screen');
          const block=document.createElement('div');block.id='layout-stress';
          for(let i=0;i<80;i++){const p=document.createElement('p');
            p.textContent='Layout verification text '+i;block.appendChild(p)}
          screen.appendChild(block);
        }""")
        result = geometry(page)
        assert_seams(result, 1440)
        assert result["screen"]["height"] > 2500
        page.screenshot(path=str(OUTPUT / "research-layout-stress.png"), full_page=True)
        page.goto("http://localhost:8501/shap")
        ready(page)
        page.screenshot(path=str(OUTPUT / "after-shap.png"), full_page=True)
        page.goto("http://localhost:8501/")
        expect(page.locator(".st-key-flight-window")).to_be_visible(timeout=60000)
        expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
            "data-test-script-state", "notRunning", timeout=30000
        )
        page.wait_for_timeout(1000)
        expect(page.locator(".st-key-shap-cabin")).to_have_count(0)
        page.screenshot(path=str(OUTPUT / "after-portfolio.png"), full_page=True)
        assert not errors, errors
        browser.close()
    (OUTPUT / "checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("PASS all direct routes, native navigation, seams and layout stress")


if __name__ == "__main__":
    main()
