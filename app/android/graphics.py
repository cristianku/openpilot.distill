"""The pictures of the Google Play listing of Distill for Android, drawn from the launcher icon.

    .venv/bin/python app/android/graphics.py

Writes app/src/main/play/listings/en-US/graphics/icon/icon.png (512×512, full square: Google Play rounds
the corners) and feature-graphic/feature-graphic.png (1024×500), which helpers/build_android.sh --listing
uploads. The shapes are those of res/drawable/ic_launcher_*.xml, in the 108-unit viewport of an adaptive
icon: change both together. Pillow only, with the DejaVu fonts of the system.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / 'app' / 'src' / 'main' / 'play' / 'listings' / 'en-US' / 'graphics'
FONTS = Path('/usr/share/fonts/truetype/dejavu')
# ic_launcher_background.xml: #1D4ED8 at the bottom left to #3B82F6 at the top right.
DARK, LIGHT = (29, 78, 216), (59, 130, 246)
NAVY = (11, 15, 22)
# ic_launcher_foreground.xml: the edges of the road, its dashes (alpha 0.7) and the drop.
EDGES = [((37, 80), (49.5, 51)), ((71, 80), (58.5, 51))]
EDGE_WIDTH = 5.5
DASHES = [((54, 76), (54, 70), 3.4), ((54, 65.5), (54, 61.5), 2.8), ((54, 57.5), (54, 55.5), 2.2)]
DASH_ALPHA = 0.7
DROP_CENTER, DROP_RADIUS, DROP_TIP = (54, 38.5), 5, (54, 28.5)
SUPERSAMPLE = 4


def gradient(size: int) -> Image.Image:
    small = max(size // 8, 2)
    image = Image.new('RGB', (small, small))
    pixels = image.load()
    for y in range(small):
        for x in range(small):
            t = (x / (small - 1) + 1 - y / (small - 1)) / 2
            pixels[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(DARK, LIGHT))
    return image.resize((size, size), Image.BILINEAR)


def stroke(draw: ImageDraw.ImageDraw, start, end, width: float, fill) -> None:
    """A line with round caps (a polygon: wide PIL lines leave a step at the ends)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    draw.polygon([(start[0] + nx, start[1] + ny), (end[0] + nx, end[1] + ny),
                  (end[0] - nx, end[1] - ny), (start[0] - nx, start[1] - ny)], fill=fill)
    for x, y in (start, end):
        draw.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=fill)


def mark(size: int, unit: float, origin: float) -> Image.Image:
    """The white foreground, `unit` pixels per viewport unit, the viewport origin at (origin, origin)."""
    def at(point):
        return origin + point[0] * unit, origin + point[1] * unit

    layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for start, end in EDGES:
        stroke(draw, at(start), at(end), EDGE_WIDTH * unit, (255, 255, 255, 255))
    dashes = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    dash_draw = ImageDraw.Draw(dashes)
    for start, end, width in DASHES:
        stroke(dash_draw, at(start), at(end), width * unit, (255, 255, 255, round(255 * DASH_ALPHA)))
    layer = Image.alpha_composite(layer, dashes)
    draw = ImageDraw.Draw(layer)
    cx, cy = at(DROP_CENTER)
    tx, ty = at(DROP_TIP)
    radius = DROP_RADIUS * unit
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 255, 255, 255))
    side, up = math.sin(math.radians(60)) * radius, math.cos(math.radians(60)) * radius
    draw.polygon([(tx, ty), (cx + side, cy - up), (cx, cy), (cx - side, cy - up)], fill=(255, 255, 255, 255))
    return layer


def icon(size: int) -> Image.Image:
    """The visible 72 units of the adaptive icon (18 to 90) as a full square."""
    big = size * SUPERSAMPLE
    unit = big / 72
    image = Image.alpha_composite(gradient(big).convert('RGBA'), mark(big, unit, -18 * unit))
    return image.resize((size, size), Image.LANCZOS).convert('RGB')


def feature_graphic() -> Image.Image:
    width, height = 1024 * SUPERSAMPLE, 500 * SUPERSAMPLE
    glow = Image.new('RGB', (width, height), NAVY)
    ImageDraw.Draw(glow).ellipse([-100 * SUPERSAMPLE, -250 * SUPERSAMPLE, 900 * SUPERSAMPLE, 750 * SUPERSAMPLE],
                                 fill=(23, 45, 100))
    image = glow.filter(ImageFilter.GaussianBlur(120 * SUPERSAMPLE))
    box = 250 * SUPERSAMPLE
    tile = icon(box)
    corners = Image.new('L', (box, box), 0)
    ImageDraw.Draw(corners).rounded_rectangle([0, 0, box - 1, box - 1], radius=58 * SUPERSAMPLE, fill=255)
    image.paste(tile, (105 * SUPERSAMPLE, (height - box) // 2), corners)
    draw = ImageDraw.Draw(image)
    bold = ImageFont.truetype(str(FONTS / 'DejaVuSans-Bold.ttf'), 104 * SUPERSAMPLE)
    regular = ImageFont.truetype(str(FONTS / 'DejaVuSans.ttf'), 32 * SUPERSAMPLE)
    left = 410 * SUPERSAMPLE
    draw.text((left, 140 * SUPERSAMPLE), 'Distill', font=bold, fill=(232, 236, 243))
    draw.text((left, 288 * SUPERSAMPLE), 'Train, test and install', font=regular, fill=(154, 164, 178))
    draw.text((left, 332 * SUPERSAMPLE), 'driving models from your phone', font=regular, fill=(154, 164, 178))
    return image.resize((1024, 500), Image.LANCZOS)


def main() -> None:
    (OUT / 'icon').mkdir(parents=True, exist_ok=True)
    (OUT / 'feature-graphic').mkdir(parents=True, exist_ok=True)
    icon(512).save(OUT / 'icon' / 'icon.png', optimize=True)
    feature_graphic().save(OUT / 'feature-graphic' / 'feature-graphic.png', optimize=True)
    print('written: ' + str(OUT / 'icon' / 'icon.png') + ', ' + str(OUT / 'feature-graphic' / 'feature-graphic.png'))


if __name__ == '__main__':
    main()
