"""Trim active cabin PNGs; preserve frame corners and the wine glass ratio.

Alpha < 8 is invisible export residue, not the metal's physical boundary.
Only straight frame middles may stretch in CSS. Originals remain untouched.
"""

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "frontend/assets/cabin"
NAMES = [
    "cabin-ceiling-navy",
    "cabin-window-panel",
    "cabin-wine-panel",
    "cabin-seat-armrests",
    "monitor-frame-top",
    "monitor-frame-bottom",
    "monitor-frame-left",
    "monitor-frame-right",
]


def main():
    destination = ROOT / "trimmed"
    destination.mkdir(exist_ok=True)
    manifest = {}
    for name in NAMES:
        source = Image.open(ROOT / f"{name}.png").convert("RGBA")
        bounds = source.getchannel("A").point(lambda a: 255 if a >= 8 else 0).getbbox()
        cropped = source.crop(bounds)
        if name == "cabin-seat-armrests":
            # Lossless base64 exceeds Chromium's ~2 MiB custom-property limit.
            # Keep source dimensions/ratio, using a high-quality display copy.
            cropped.save(destination / f"{name}.webp", quality=94, method=6)
        elif name in ("cabin-ceiling-navy", "cabin-window-panel"):
            cropped.save(destination / f"{name}.webp", lossless=True)
        # Wine and frame crops stay in memory; only their display slices ship.
        manifest[name] = {"source_size": source.size, "opaque_bounds": bounds}
        if name == "cabin-wine-panel":
            # Only the flat cabinet below the glass/table may grow vertically.
            # Keep a narrow original plinth at the common wine/seat baseline.
            slices = {"top": (0, 960), "body": (960, 1740), "base": (1740, 1983)}
            for part, (top, bottom) in slices.items():
                cropped.crop((0, top, cropped.width, bottom)).save(
                    destination / f"{name}-{part}.webp", lossless=True
                )
            manifest[name]["vertical_slices"] = slices
        if name.endswith(("top", "bottom")) and name.startswith("monitor"):
            width, height = cropped.size
            for part, box in {
                "start": (0, 0, height, height),
                "middle": (height, 0, width - height, height),
                "end": (width - height, 0, width, height),
            }.items():
                cropped.crop(box).save(
                    destination / f"{name}-{part}.webp", lossless=True
                )
        elif name.startswith("monitor"):
            width, height = cropped.size
            # Top/bottom horizontal caps own the corners; remove vertical end caps.
            cropped.crop((0, width, width, height - width)).save(
                destination / f"{name}-middle.webp", lossless=True
            )
    (destination / "bounds.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
