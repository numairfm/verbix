#!/usr/bin/env python3
"""Generates the original Verbix shell and tray icons.

Run from the repository root with the project virtualenv:
    .venv/bin/python src-tauri/icons/generate_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SCALE = 4

STATE_COLORS = {
    "idle": (107, 114, 128, 255),
    "listening": (34, 197, 94, 255),
    "speaking": (59, 130, 246, 255),
    "working": (245, 158, 11, 255),
    "error": (239, 68, 68, 255),
}
INK = (15, 23, 42, 255)
PAPER = (248, 250, 252, 255)


def _mic(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int, int]) -> None:
    """Draws a simple microphone capsule and stand inside box."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    cap_w = int(width * 0.26)
    cap_h = int(height * 0.46)
    cap_x0 = x0 + (width - cap_w) // 2
    cap_y0 = y0 + int(height * 0.12)
    draw.rounded_rectangle(
        (cap_x0, cap_y0, cap_x0 + cap_w, cap_y0 + cap_h),
        radius=cap_w // 2,
        fill=fill,
    )
    arc_box = (
        x0 + int(width * 0.20),
        cap_y0 + int(cap_h * 0.45),
        x1 - int(width * 0.20),
        cap_y0 + int(cap_h * 1.30),
    )
    draw.arc(arc_box, start=20, end=160, fill=fill, width=max(1, int(width * 0.075)))
    stem_x = x0 + width // 2
    stem_top = arc_box[3] - int(height * 0.01)
    stem_bottom = y1 - int(height * 0.14)
    draw.line((stem_x, stem_top, stem_x, stem_bottom), fill=fill, width=max(1, int(width * 0.075)))
    base_w = int(width * 0.30)
    draw.line(
        (stem_x - base_w // 2, stem_bottom, stem_x + base_w // 2, stem_bottom),
        fill=fill,
        width=max(1, int(width * 0.075)),
    )


def tray_icon(name: str, color: tuple[int, int, int, int]) -> Image.Image:
    size = 32 * SCALE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    inset = int(size * 0.04)
    draw.ellipse((inset, inset, size - inset, size - inset), fill=color, outline=INK, width=int(size * 0.035))
    _mic(draw, (int(size * 0.22), int(size * 0.16), int(size * 0.78), int(size * 0.84)), PAPER)
    return image.resize((32, 32), Image.Resampling.LANCZOS)


def app_icon(size: int) -> Image.Image:
    canvas = size * SCALE
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = int(canvas * 0.22)
    draw.rounded_rectangle((0, 0, canvas - 1, canvas - 1), radius=radius, fill=INK)
    margin = int(canvas * 0.14)
    draw.ellipse(
        (margin, margin, canvas - margin, canvas - margin),
        fill=STATE_COLORS["speaking"],
        outline=PAPER,
        width=int(canvas * 0.018),
    )
    _mic(draw, (int(canvas * 0.30), int(canvas * 0.22), int(canvas * 0.70), int(canvas * 0.78)), PAPER)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    for name, color in STATE_COLORS.items():
        tray_icon(name, color).save(HERE / f"tray-{name}.png")
    app_icon(32).save(HERE / "32x32.png")
    app_icon(128).save(HERE / "128x128.png")
    app_icon(256).save(HERE / "128x128@2x.png")
    app_icon(512).save(HERE / "icon.png")


if __name__ == "__main__":
    main()
