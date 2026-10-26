"""Optional Edge visual/layout check; use alongside scripts/check_ui.py."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ASSETS = [
    "ceiling",
    "pillar-left",
    "pillar-right",
    "windshield-sky",
    "instrument-top-bar",
    "control-console",
    "instrument-right",
]


def run(base, output):
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.goto(base)
        expect(page.get_by_role("button", name="비행 경로 재생")).to_be_visible(
            timeout=30000
        )
        expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
            "data-test-script-state", "notRunning", timeout=30000
        )
        # Test browser-computed variables, not just source-file existence:
        # overlarge PNG data URIs can silently invalidate a CSS property.
        decoded = page.locator(".stApp").evaluate(
            r"""async (e, names) => Promise.all(names.map(async name => {
              const css = getComputedStyle(e).getPropertyValue('--cockpit-'+name);
              const match = css.match(/url\(['"]?(.*?)['"]?\)/);
              if (!match) throw new Error('Missing asset: '+name);
              const img = new Image(); img.src = match[1]; await img.decode();
              return {name, width:img.naturalWidth, height:img.naturalHeight};
            }))""",
            ASSETS,
        )
        assert len(decoded) == len(ASSETS)
        for width in [1920, 1440, 1280, 1151, 1150, 1024, 1000, 768, 760, 390]:
            page.set_viewport_size({"width": width, "height": 1050})
            page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTop=0")
            page.wait_for_timeout(500)
            assert page.locator('[data-testid="stMain"]').evaluate(
                "e => e.scrollWidth <= e.clientWidth + 1"
            ), f"Horizontal overflow at {width}"
            window = page.locator(".st-key-flight-window")
            assert page.locator(".window-sky").evaluate(
                "e => getComputedStyle(e).backgroundImage"
                ".includes('data:image/webp')"
            )
            assert page.locator(".st-key-windshield").evaluate(
                "e => getComputedStyle(e).backgroundColor === 'rgba(0, 0, 0, 0)'"
            )
            for selector in ["h1", ".st-key-windshield", ".st-key-flight-window"]:
                assert page.locator(selector).first.evaluate(
                    "e => getComputedStyle(e).opacity === '1'"
                )
            for selector, pseudo in [
                (".cockpit-ceiling", "::before"),
                (".window-sky", None),
                (".instrument-top-bar", None),
                (".window-pillars", "::before"),
                (".window-pillars", "::after"),
                (".st-key-instruments", "::before"),
                (".aviation-right", None),
                (".lower-console", None),
            ]:
                assert page.locator(selector).evaluate(
                    "(e,p) => getComputedStyle(e,p).pointerEvents === 'none'", pseudo
                )
            chart = page.locator('.flight-root svg[role="img"]')
            assert chart.locator('path[stroke="#69dcff"]').count() == 1
            assert chart.locator('path[stroke="#ffce70"]').count() == 1
            assert chart.locator("ellipse, defs").count() == 0
            readout = page.locator(".flight-root .readout")
            chart.hover(position={"x": chart.bounding_box()["width"] / 2, "y": 150})
            chart.focus()
            before = readout.inner_text()
            chart.press("ArrowLeft")
            expect(readout).not_to_have_text(before)
            chart.press("ArrowRight")
            expect(readout).to_have_text(before)
            chart.evaluate("e => e.blur()")
            page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTop=0")
            assert page.locator(".aviation-left, .glareshield").count() == 0
            window_box = window.bounding_box()
            top_bar = page.locator(".instrument-top-bar").bounding_box()
            assert abs(window_box["y"] + window_box["height"] - top_bar["y"]) < 1
            assert (
                top_bar["y"]
                >= readout.bounding_box()["y"] + readout.bounding_box()["height"]
            )
            scene = page.locator(".window-scene").bounding_box()
            assert scene["y"] + scene["height"] >= top_bar["y"] + top_bar["height"] - 1
            assert page.locator(".window-sky").evaluate(
                "e => getComputedStyle(e).clipPath !== 'none'"
            )
            deck = page.locator(".st-key-instruments").bounding_box()
            console = page.locator(".lower-console").bounding_box()
            ceiling = page.locator(".cockpit-ceiling").bounding_box()
            for box in (console, ceiling, top_bar):
                assert abs(box["x"] - deck["x"]) < 1, (width, box, deck)
                assert abs(box["width"] - deck["width"]) < 1, (width, box, deck)
            panels = {}
            if width > 1150:
                for name in ["doors", "allocation", "gauges", "right"]:
                    selector = (
                        f".aviation-{name}" if name == "right" else f".st-key-{name}"
                    )
                    panels[name] = page.locator(selector).bounding_box()
                boxes = list(panels.values())
                assert all(
                    a["x"] + a["width"] <= b["x"] for a, b in zip(boxes, boxes[1:])
                )
                for name in ["doors", "right", "gauges"]:
                    for dimension in ["y", "height"]:
                        assert (
                            abs(
                                panels[name][dimension]
                                - panels["allocation"][dimension]
                            )
                            < 2
                        )
                # The complete flat instrument is never stretched or sliced.
                assert (
                    abs(
                        panels["right"]["width"] / panels["right"]["height"]
                        - 691 / 1801
                    )
                    < 0.001
                )
            assert abs(top_bar["y"] + top_bar["height"] - deck["y"]) < 1
            assert abs(console["y"] - deck["y"] - deck["height"]) < 1
            assert abs(console["height"] - console["width"] * 403 / 2091) < 0.1
            assert abs(top_bar["height"] - top_bar["width"] * 258 / 2065) < 0.1
            for name in ["doors", "allocation", "gauges"]:
                box = page.locator(f".st-key-{name}").bounding_box()
                assert console["y"] >= box["y"] + box["height"] + 6
            assert page.locator(".st-key-doors a").count() == 6
            evidence.append(
                {
                    "viewport": width,
                    "chart": chart.bounding_box(),
                    "panels": panels,
                    "deck": deck,
                    "console": console,
                }
            )
            page.screenshot(path=str(output / f"portfolio-{width}.png"))
            page.locator(".lower-console").scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"portfolio-{width}-lower.png"))
        # CSS must leave with the portfolio, including browser navigation.
        page.set_viewport_size({"width": 1440, "height": 1050})
        page.get_by_role("link", name="SHAP 해석").click()
        expect(
            page.get_by_role("heading", name="SHAP 해석", exact=True)
        ).to_be_visible()
        expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
            "data-test-script-state", "notRunning", timeout=30000
        )
        style_after_navigation = page.locator(".shap-brand").evaluate(
            "e => [getComputedStyle(e).backgroundImage, getComputedStyle(e).height]"
        )
        assert page.locator(".stApp").evaluate(
            "e => !getComputedStyle(e).getPropertyValue('--cockpit-windshield-sky')"
        )
        page.goto(base + "/shap")
        expect(page.locator(".shap-brand")).to_be_visible(timeout=30000)
        assert (
            page.locator(".shap-brand").evaluate(
                "e => [getComputedStyle(e).backgroundImage, getComputedStyle(e).height]"
            )
            == style_after_navigation
        )
        browser.close()
    (output / "geometry.json").write_text(
        json.dumps({"assets": decoded, "viewports": evidence}, indent=2),
        encoding="utf-8",
    )
    print(
        "PASS: decoded assets, 4-column order, aligned panels, sky/body junctions, "
        "full-width console, 10 widths, input, CSS isolation"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui/housing"))
    args = parser.parse_args()
    run(args.base.rstrip("/"), args.output)
