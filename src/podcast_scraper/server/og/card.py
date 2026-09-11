"""The OG card renderer — a Pillow port of ``entityShareCard.ts``'s canvas draw.

Same design as the client card (design note ``docs/wip/2026-09-11-share-card-design.md``):
near-black canvas, serif display, mono kickers/stats, ONE accent, square, lots of air. Kept in
lock-step with the TS renderer by eye — this is the server twin used for ``og:image`` so a shared
link unfurls as the card.

Fonts are bundled (``og/fonts/*.ttf``, DejaVu) rather than taken from the OS so the render is
identical on any host and needs no system fonts. Pillow is imported lazily inside the render so
importing this module never hard-requires it (the OG route degrades to a 404 if Pillow is absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import ImageDraw, ImageFont

# Design tokens — the default dark theme (web/learning-player/src/theme/directions.css), mirrored
# from the TS renderer. Literal hexes are correct here: this IS the single source that draws them.
_CANVAS = "#07090a"
_FG = "#d6e2d8"
_MUTED = "#7f958a"
_BORDER = "#1e2a28"
_DEFAULT_ACCENT = "#8ad2e5"  # --lp-topic

# Per-kind accent — only the two kinds that own a theme token get their own colour (topic cyan,
# person gold); everything else keeps the brand cyan. Matches ``accentForKind`` in the TS engine.
_KIND_ACCENT = {"topic": "#8ad2e5", "person": "#e0b354"}

_W = 1080
_H = 1440
_PAD = 96

_FONT_DIR = Path(__file__).parent / "fonts"
_SERIF = _FONT_DIR / "DejaVuSerif.ttf"
_SERIF_BOLD = _FONT_DIR / "DejaVuSerif-Bold.ttf"
_SERIF_ITALIC = _FONT_DIR / "DejaVuSerif-Italic.ttf"
_MONO = _FONT_DIR / "DejaVuSansMono.ttf"


def accent_for_kind(kind: str | None) -> str:
    """The accent hex for an entity kind; brand cyan for kinds with no theme token."""
    return _KIND_ACCENT.get(kind or "", _DEFAULT_ACCENT)


@dataclass(frozen=True)
class OgCardModel:
    """The data a card renders — assembled per entity kind by ``og/build.py`` (twin of the TS
    ``EntityCardModel``)."""

    kicker: str  # "TOPIC" / "EPISODE · CROSS-SHOW"
    title: str
    quote: str | None = None  # a signature take/insight (optional — card is clean without it)
    byline: str | None = None  # "— Dr. Elena Fischer" / "42 min · 3 insights"
    stats: str | None = None  # "28 episodes · 10 voices"
    hot: str | None = None  # the one accent-coloured stat, e.g. "↑ 2.3× rising"
    accent: str | None = None  # per-kind accent hex; falls back to the brand cyan


@lru_cache(maxsize=16)
def _font(path: str, size: int) -> "ImageFont.FreeTypeFont":
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def _wrap(
    draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont", max_w: int
) -> list[str]:
    """Greedy word-wrap by measured pixel width (mirrors the TS ``wrap``)."""
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            cand = f"{line} {word}" if line else word
            if line and draw.textlength(cand, font=font) > max_w:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return out


def render_card_png(model: OgCardModel) -> bytes:
    """Render the model to a portrait PNG (1080×1440). Raises if Pillow is unavailable."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    accent = model.accent or _DEFAULT_ACCENT
    img = Image.new("RGB", (_W, _H), _CANVAS)
    draw = ImageDraw.Draw(img)
    # Hairline frame.
    draw.rectangle((1, 1, _W - 2, _H - 2), outline=_BORDER, width=2)

    max_w = _W - _PAD * 2
    y = _PAD

    # Kicker (mono, muted, uppercase).
    kmono = _font(str(_MONO), 26)
    draw.text((_PAD, y), model.kicker.upper(), font=kmono, fill=_MUTED)
    y += 60

    # Title (bold serif, wrapped, large).
    title_font = _font(str(_SERIF_BOLD), 88)
    for line in _wrap(draw, model.title, title_font, max_w):
        draw.text((_PAD, y), line, font=title_font, fill=_FG)
        y += 104

    # The single accent: a short hairline under the title.
    y += 40
    draw.rectangle((_PAD, y, _PAD + 88, y + 4), fill=accent)
    y += 4

    # Signature quote (italic serif).
    if model.quote:
        quote_font = _font(str(_SERIF_ITALIC), 46)
        y += 40
        for line in _wrap(draw, f"“{model.quote}”", quote_font, max_w):
            draw.text((_PAD, y), line, font=quote_font, fill=_FG)
            y += 62

    # Byline (serif, muted).
    if model.byline:
        byline_font = _font(str(_SERIF), 30)
        y += 24
        draw.text((_PAD, y), model.byline, font=byline_font, fill=_MUTED)

    # Footer, anchored to the bottom: stat line (with the one hot stat in accent) then wordmark.
    stat_font = _font(str(_MONO), 26)
    stat_y = _H - _PAD - 52
    x = _PAD
    if model.stats:
        s = model.stats.upper()
        draw.text((x, stat_y), s, font=stat_font, fill=_MUTED)
        x += int(draw.textlength(s, font=stat_font))
    if model.hot:
        if model.stats:
            sep = "   ·   "
            draw.text((x, stat_y), sep, font=stat_font, fill=_MUTED)
            x += int(draw.textlength(sep, font=stat_font))
        draw.text((x, stat_y), model.hot.upper(), font=stat_font, fill=accent)

    # Wordmark with a single accent dot.
    wy = _H - _PAD - 6
    draw.ellipse((_PAD, wy + 2, _PAD + 12, wy + 14), fill=accent)
    wordmark_font = _font(str(_MONO), 24)
    draw.text((_PAD + 24, wy), "closelistening.app", font=wordmark_font, fill=_MUTED)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
