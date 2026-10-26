"""Real Edge checks for cabin asset decoding, seams, containment and reflow."""

import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    output = Path("artifacts/ui/shap-refresh/responsive")
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in (1440, 1920, 1280, 1100, 900, 768, 390):
            page.set_viewport_size({"width": width, "height": 1100})
            page.goto("http://localhost:8501/shap")
            expect(page.locator(".st-key-force")).to_be_attached(timeout=60000)
            expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
                "data-test-script-state", "notRunning", timeout=30000
            )
            page.wait_for_timeout(800)
            result = page.evaluate("""async () => {
              const box = s => {
                const r=document.querySelector(s).getBoundingClientRect();
                return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,
                  width:r.width,height:r.height}};
              const parts=['start','middle','end'];
              const names=['cabin-ceiling-navy','cabin-window-panel',
                'cabin-wine-panel-top','cabin-wine-panel-body',
                'cabin-wine-panel-base','cabin-seat-armrests',
                ...['top','bottom'].flatMap(s=>parts.map(p=>`monitor-frame-${s}-${p}`)),
                'monitor-frame-left-middle','monitor-frame-right-middle'];
              const css=getComputedStyle(document.querySelector('.st-key-shap-cabin'));
              const assets=await Promise.all(names.map(async n=>{const im=new Image();
                im.src=css.getPropertyValue('--'+n).trim().slice(5,-2);
                await im.decode();return n}));
              const main=document.querySelector('[data-testid="stMain"]');
              return {assets,overflow:main.scrollWidth-main.clientWidth,
                monitor:box('.st-key-shap-monitor'),frame:box('.monitor-frame'),
                screen:box('.st-key-shap-screen'),
                top:box('.frame-top'),bottom:box('.frame-bottom'),
                left:box('.frame-left'),right:box('.frame-right'),
                nav:box('.st-key-shap-nav'),wine:box('.cabin-wine-panel'),
                glass:box('.wine-top'),seat:box('.cabin-seat-armrests'),
                lower:box('.cabin-lower'),
                brandColor:getComputedStyle(
                  document.querySelector('.shap-brand')).color,
                subtitleColor:getComputedStyle(
                  document.querySelector('.shap-brand span')).color,
                brand:box('.shap-brand'),ceiling:box('.shap-ceiling'),
                pointer:getComputedStyle(
                  document.querySelector('.monitor-frame')).pointerEvents};
            }""")
            assert result["overflow"] <= 1, (width, result)
            assert result["brand"]["x"] >= result["ceiling"]["x"]
            assert result["brand"]["right"] <= result["ceiling"]["right"]
            assert result["monitor"] == result["frame"], (width, result)
            assert abs(result["top"]["bottom"] - result["left"]["y"]) < 1
            assert abs(result["left"]["bottom"] - result["bottom"]["y"]) < 1
            assert result["wine"]["y"] >= result["nav"]["bottom"] - 1
            assert result["wine"]["y"] <= result["nav"]["bottom"] + 1
            assert (
                abs(
                    result["glass"]["width"] / result["glass"]["height"]
                    - 793 / (960 if width > 900 else 810)
                )
                < 0.001
            )
            assert result["brandColor"] == "rgb(252, 229, 160)"
            assert result["subtitleColor"] == "rgb(240, 245, 255)"
            assert abs(result["seat"]["x"] - result["monitor"]["x"]) < 1
            assert abs(result["seat"]["width"] - result["monitor"]["width"]) < 1
            if width > 900:
                assert abs(result["wine"]["right"] - result["monitor"]["x"]) < 1
                assert abs(result["wine"]["bottom"] - result["seat"]["bottom"]) < 1
            else:
                assert result["wine"]["bottom"] <= result["monitor"]["y"] + 1
            assert result["seat"]["y"] >= result["monitor"]["bottom"] - 1
            assert result["screen"]["x"] >= result["left"]["right"] - 1
            assert result["screen"]["right"] <= result["right"]["x"] + 1
            assert result["pointer"] == "none"
            assert page.locator('[data-testid="stPageLink"]').count() == 6
            assert page.locator('[data-testid="stException"]').count() == 0
            evidence.append({"viewport": width, **result})
            page.screenshot(path=str(output / f"shap-{width}.png"))
            page.locator(".st-key-force").scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"shap-{width}-lower.png"))
        page.set_viewport_size({"width": 1440, "height": 1700})
        page.goto("http://localhost:8501/shap")
        expect(page.locator(".st-key-force")).to_be_attached(timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(output.parent / "shap-final.png"))
        assert not errors, errors
        browser.close()
    (output / "geometry.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(
        "PASS: eight source assets, frames, controls, wine/seat containment, "
        "seven widths"
    )


if __name__ == "__main__":
    main()
