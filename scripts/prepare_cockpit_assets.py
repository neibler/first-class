"""Rebuild display-only crops; preserve the supplied RGBA PNG originals.

Run from the repository root with the development environment (Pillow).
No financial data or text is baked into these decorative images.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "frontend" / "assets" / "cockpit"
CROPS = {
    "ceiling": (0, 45, 2172, 380),
    # Stop before the lower elbows and their blue sill lights.
    "pillar-left": (285, 200, 610, 1280),
    "pillar-right": (580, 200, 860, 1380),
    "windshield-sky": None,
    "instrument-top-bar": (17, 170, 2082, 428),
    "instrument-side-right-flat": (16, 150, 707, 1951),
    "control-console-integrated": (6, 217, 2097, 620),
}


def main():
    output = ROOT / "trimmed"
    output.mkdir(exist_ok=True)
    for name, box in CROPS.items():
        with Image.open(ROOT / f"{name}.png") as original:
            image = original.crop(box) if box else original.copy()
            # Keep alpha; avoid Chromium's ~1 MiB custom-property value limit.
            # Retain source PNGs; write only the current display assets.
            suffix = "-upright" if name.startswith("pillar-") else ""
            image.save(output / f"{name}{suffix}.webp", quality=92, method=6)


if __name__ == "__main__":
    main()
