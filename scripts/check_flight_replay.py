"""Optional Edge regression: controlled frame clock plus real sample replays.

Install Playwright and Edge; run the UI/API before running from the repo root:
python scripts/check_flight_replay.py --repeats 10
"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check_clock(browser, source):
    """Exercise the real JS with an early first frame and a restart clock."""
    page = browser.new_page()
    page.set_content('<div class="flight-root" style="width:640px"></div>')
    page.evaluate(
        """async source => {
          const url = URL.createObjectURL(new Blob([source],
            {type: 'text/javascript'}));
          const mount = (await import(url)).default;
          URL.revokeObjectURL(url);
          window.disposeFlight = mount({parentElement: document, data: {points: [
            {date:'2026-01-01', portfolio:0, benchmark:0},
            {date:'2026-01-02', portfolio:.1, benchmark:.05},
            {date:'2026-01-03', portfolio:.2, benchmark:.1}
          ]}});
        }""",
        source,
    )
    # Let the initial ResizeObserver notification settle before clock control.
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
    result = page.evaluate("""() => {
          const pending = new Map(); let id = 0;
          window.requestAnimationFrame = callback => {
            pending.set(++id, callback); return id;
          };
          window.cancelAnimationFrame = key => pending.delete(key);
          const button = document.querySelector('button');
          const plane = document.querySelector('svg g[aria-label]');
          const at = () => plane.transform.baseVal.getItem(0).matrix.e;
          const equal = (actual, expected) => {
            if (Math.abs(actual - expected) > .001)
              throw Error(`Expected ${expected}, received ${actual}`);
          };
          const step = now => {
            if (pending.size !== 1) throw Error('Expected one pending frame');
            const [key, callback] = pending.entries().next().value;
            pending.delete(key); callback(now);
            return at();
          };
          const end = at();
          button.click();
          equal(step(-1000), 52); // Earlier than the click's performance.now().
          equal(step(-1100), 52); // Defensive lower bound.
          const middle = step(600);
          equal(middle, (52 + end) / 2);
          button.click(); // Cancel pending old frame and reset the start clock.
          equal(at(), 52);
          equal(step(0), 52); // Zero is a valid first-frame timestamp.
          equal(step(1600), middle);
          equal(step(3200), end);
          if (pending.size) throw Error('Completed replay scheduled another frame');
          button.click();
          equal(step(5000), 52);
          equal(step(15000), end); // Defensive upper bound.
          if (pending.size) throw Error('Overshoot did not finish');
          window.disposeFlight();
          return {start:52, middle, end, pending:pending.size};
        }""")
    page.close()
    return result


def check_live(browser, base, repeats):
    results = []
    for attempt in range(repeats):
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(error.message))
        page.goto(base)
        expect(page.get_by_text("API 연결 정상", exact=True)).to_be_attached(
            timeout=30000
        )
        expect(page.locator('[data-testid="stApp"]')).to_have_attribute(
            "data-test-script-state", "notRunning", timeout=30000
        )
        recent = attempt % 2 == 1
        if recent:
            page.get_by_role("combobox").click()
            page.get_by_role("option", name="최근 63개 관측일").click()
            expect(page.get_by_text("63개 일간 수익률", exact=False)).to_be_visible()
        root = page.locator(".flight-root")
        button = root.get_by_role("button")
        expect(button).to_have_text("▷ 비행 경로 재생", timeout=30000)
        plane = root.locator("svg g[aria-label]")
        endpoint = plane.get_attribute("transform")
        path = root.locator('svg > path[stroke="#69dcff"]').get_attribute("d")
        if recent:
            assert len(path.split("L")) == 64, "Old period data remained mounted"

        def moving():
            page.wait_for_function(
                """root => {
                  const svg = root.querySelector('svg');
                  const x = svg.querySelector('g[aria-label]')
                    .transform.baseVal.getItem(0).matrix.e;
                  return x > 62 && x < svg.viewBox.baseVal.width - 98;
                }""",
                arg=root.element_handle(),
                timeout=2000,
            )
            return plane.get_attribute("transform")

        button.click()
        middle = moving()
        assert middle != endpoint
        expect(plane).to_have_attribute("transform", endpoint, timeout=5000)
        expect(button).to_have_text("▷ 비행 경로 재생")
        button.click()
        moving()
        button.click()  # Restart while the second replay is still moving.
        x = plane.evaluate("e => e.transform.baseVal.getItem(0).matrix.e")
        assert x < 62, f"Restart did not reset to the start: {x}"
        moving()
        expect(plane).to_have_attribute("transform", endpoint, timeout=5000)
        expect(button).to_have_text("▷ 비행 경로 재생")
        assert not errors, errors
        results.append(
            dict(attempt=attempt + 1, recent=recent, completed=2, restarted=1)
        )
        print(f"PASS replay session {attempt + 1}/{repeats}", flush=True)
        page.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8501")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui/replay"))
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to check both periods")
    source = Path(__file__).resolve().parents[1] / "frontend/assets/flight.js"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        clock = check_clock(browser, source.read_text(encoding="utf-8"))
        live = check_live(browser, args.base, args.repeats)
        browser.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(dict(clock=clock, live=live), indent=2), encoding="utf-8"
    )
    print(f"PASS clock regression; {len(live)} sessions, 2 completions/1 restart each")


if __name__ == "__main__":
    main()
