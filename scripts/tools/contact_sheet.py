#!/usr/bin/env python3
"""Stitch a directory of device screenshots into ONE labelled contact sheet.

Reviewing a UI change one screenshot at a time hides the thing you most want to catch — surfaces
drifting apart from each other. A single sheet puts every screen side by side at a glance
(operator request 2026-09-16).

Input is whatever ``xcrun xcresulttool export attachments`` produced, renamed to its logical name
(``t01-home.png``, ``t02-discover.png``, …); see the ``ios-contact-sheet`` Makefile target. Files
are ordered by filename, which is why the tour numbers its frames.

Deliberately dependency-light: Pillow only, which the repo venvs already carry.

    python scripts/tools/contact_sheet.py --in /tmp/shots/named --out /tmp/sheet.png --cols 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - environment guard, not logic
    sys.exit("FAIL: Pillow is required — pip install pillow (or use .venv/bin/python)")

# Dark, to match the app being photographed: a white mat around dark screenshots makes every tile
# look like it has a glowing border and is genuinely harder to scan.
BG = (14, 17, 24)
FG = (232, 236, 244)
LABEL_H = 34
PAD = 16


def _font(size: int) -> ImageFont.ImageFont:
    """A real TrueType face when one is available; Pillow's bitmap default otherwise.

    The default font ignores `size`, so labels would be unreadably small on a 2000px-wide sheet.
    """
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def build(paths: list[Path], out: Path, cols: int, tile_w: int) -> None:
    if not paths:
        sys.exit("FAIL: no .png files found — did the tour run and were attachments exported?")

    shots = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except (OSError, ValueError) as exc:
            # One unreadable file must not cost the whole sheet. Xcode attachments are not all
            # images despite their names — a plist or text dump saved as ``.png`` aborted the run
            # at the first bad file, after every other screen had already been processed.
            print(f"  skipped {p.name}: {exc}", file=sys.stderr)
            continue
        if tile_w <= 0:
            # Native resolution: paste the capture untouched so the sheet can be inspected at 1:1
            # and shows exactly what the device rendered. Any resampling — even a high-quality
            # downscale — softens hairline borders and antialiased text, which is precisely what a
            # visual review is trying to judge (operator 2026-09-16).
            shots.append((p.stem, img))
            continue
        # Scale every tile to the same WIDTH and keep aspect, so a landscape or differently-sized
        # capture cannot distort the grid.
        h = max(1, round(img.height * (tile_w / img.width)))
        shots.append((p.stem, img.resize((tile_w, h), Image.LANCZOS)))

    if not shots:
        sys.exit("FAIL: no readable images among the supplied files")

    # In native mode `tile_w` is 0 — the images were never resized, so the column width has to come
    # from the widest image instead. Using the flag value produced a 96px-wide sheet (just padding)
    # with every screenshot cropped away.
    col_w = tile_w if tile_w > 0 else max(img.width for _, img in shots)
    tile_h = max(img.height for _, img in shots)
    rows = (len(shots) + cols - 1) // cols
    sheet_w = cols * col_w + (cols + 1) * PAD
    sheet_h = rows * (tile_h + LABEL_H) + (rows + 1) * PAD

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    font = _font(max(13, col_w // 22))

    for i, (name, img) in enumerate(shots):
        r, c = divmod(i, cols)
        x = PAD + c * (col_w + PAD)
        y = PAD + r * (tile_h + LABEL_H + PAD)
        draw.text((x, y), name, fill=FG, font=font)
        # Top-align within the row: screens differ in height and centring them makes the eye
        # re-find the status bar on every tile.
        sheet.paste(img, (x, y + LABEL_H))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG", optimize=True)
    print(f"✓ contact sheet: {out} ({len(shots)} screens, {cols} cols, {sheet_w}x{sheet_h})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, type=Path, help="directory of .png shots")
    ap.add_argument("--out", required=True, type=Path, help="output .png path")
    ap.add_argument("--cols", type=int, default=6, help="tiles per row (default 6)")
    ap.add_argument("--tile-width", type=int, default=300, help="tile width px (default 300)")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"FAIL: not a directory: {args.src}")
    build(sorted(args.src.glob("*.png")), args.out, args.cols, args.tile_width)


if __name__ == "__main__":
    main()
