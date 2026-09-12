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
    from PIL import Image, ImageDraw, ImageFont

# Design tokens — the default dark theme (web/learning-player/src/theme/directions.css), mirrored
# from the TS renderer. Literal hexes are correct here: this IS the single source that draws them.
_CANVAS = "#07090a"
_FG = "#d6e2d8"
_MUTED = "#7f958a"
_BORDER = "#1e2a28"
_DEFAULT_ACCENT = "#8ad2e5"  # --lp-topic

# Per-kind accent so no two card kinds read alike: topic cyan + person gold (theme tokens),
# storyline violet + organization green. Show/episode keep the brand cyan (the artwork already
# differentiates them). Matches ``accentForKind`` in the TS engine — keep the two in sync.
_KIND_ACCENT = {
    "topic": "#8ad2e5",
    "person": "#e0b354",
    "storyline": "#9d8cff",
    "organization": "#5fd0a8",
}

_W = 1080
_H = 1440
_PAD = 96
_ART_BIG = 600  # the framed artwork square in the body (show / person / org)
_TREND_TILE_H = 452  # the KPI trend tile (big score + area sparkline) when the trend is the hero

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
    quote: str | None = None  # a signature spoken take — italic + quote marks
    blurb: str | None = None  # a descriptive line (org "what they do", storyline members) — roman,
    # no quote marks; rendered in the same slot as `quote` when there is no quote to show
    byline: str | None = None  # "— Dr. Elena Fischer" / "42 min · 3 insights"
    stats: str | None = None  # "28 episodes · 10 voices"
    tags: str | None = None  # a secondary footer line above the stats (episode key topics)
    credit: str | None = None  # attribution for a licensed image (person photo / org logo)
    hot: str | None = None  # the one accent-coloured stat, e.g. "↑ 2.3× rising"
    accent: str | None = None  # per-kind accent hex; falls back to the brand cyan
    artwork: bytes | None = None  # show art / person photo / org logo — a framed square in the body
    gallery: tuple[bytes, ...] = ()  # several framed squares in a row (a guest's shows' artworks)
    background: bool = False  # when True the artwork is the full-bleed backdrop (episode); else a
    # centred framed square in the lower section
    sparkline: tuple[float, ...] | None = None  # trend series (weekly) — drawn in the lower section
    trend_multiplier: float | None = None  # the single score (e.g. 2.6 → "↑2.6×") for the KPI tile


@lru_cache(maxsize=16)
def _font(path: str, size: int) -> "ImageFont.FreeTypeFont":
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def _cap_lines(lines: list[str], limit: int) -> list[str]:
    """Cap a wrapped block to ``limit`` lines, ellipsizing the last when truncated."""
    if len(lines) <= limit:
        return lines
    kept = lines[:limit]
    kept[-1] = kept[-1].rstrip(" .,;:—-") + "…"
    return kept


def _hard_break(
    draw: "ImageDraw.ImageDraw", word: str, font: "ImageFont.FreeTypeFont", max_w: int
) -> list[str]:
    """Split a single over-long, spaceless token (CJK title, URL) into pieces that each fit
    ``max_w`` — greedy word-wrap alone would render it clipped at the canvas edge."""
    pieces: list[str] = []
    cur = ""
    for ch in word:
        if cur and draw.textlength(cur + ch, font=font) > max_w:
            pieces.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        pieces.append(cur)
    return pieces


def _wrap(
    draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont", max_w: int
) -> list[str]:
    """Greedy word-wrap by measured pixel width (mirrors the TS ``wrap``); a single token wider than
    the line is hard-split by character so it never overflows the edge."""
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            for token in _hard_break(draw, word, font, max_w):
                cand = f"{line} {token}" if line else token
                if line and draw.textlength(cand, font=font) > max_w:
                    out.append(line)
                    line = token
                else:
                    line = cand
        out.append(line)
    return out


def _load_square(data: bytes, size: int) -> "Image.Image | None":
    """Decode ``data`` and center-crop-cover it to a ``size``×``size`` RGB square, or None when the
    bytes aren't a decodable image (a broken/absent asset must never break the render)."""
    from io import BytesIO

    from PIL import Image

    try:
        opened = Image.open(BytesIO(data))
        opened.load()
    except Exception:  # noqa: BLE001
        return None
    rgb = opened.convert("RGB")
    w, h = rgb.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return rgb.resize((size, size), Image.LANCZOS, box=(left, top, left + side, top + side))


def _draw_gallery(
    img: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    region_top: int,
    region_bot: int,
    imgs: "tuple[bytes, ...]",
) -> None:
    """A centred row of framed artwork squares (a guest's shows), sized to fit the content width."""
    n = len(imgs)
    if n == 0:  # nothing to lay out (the caller gates on a non-empty tuple; guard direct callers)
        return
    row_w = _W - _PAD * 2
    gap = 28
    cell = min((row_w - (n - 1) * gap) // n, region_bot - region_top)
    span = n * cell + (n - 1) * gap
    x0 = (_W - span) // 2
    cy = region_top + max(0, (region_bot - region_top - cell)) // 4  # bias UP
    for i, data in enumerate(imgs):
        sq = _load_square(data, cell)
        if sq is None:
            continue
        ax = x0 + i * (cell + gap)
        img.paste(sq, (ax, cy))
        draw.rectangle((ax, cy, ax + cell - 1, cy + cell - 1), outline=_BORDER, width=2)


def _cover(data: bytes, w: int, h: int) -> "Image.Image | None":
    """Decode ``data`` and cover-crop it to fill ``w``×``h`` (scale to the larger ratio, then
    centre-crop). None if the bytes aren't a decodable image."""
    from io import BytesIO

    from PIL import Image

    try:
        opened = Image.open(BytesIO(data))
        opened.load()
    except Exception:  # noqa: BLE001
        return None
    rgb = opened.convert("RGB")
    iw, ih = rgb.size
    scale = max(w / iw, h / ih)
    rw, rh = max(w, round(iw * scale)), max(h, round(ih * scale))
    scaled = rgb.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - w) // 2, (rh - h) // 2
    return scaled.crop((left, top, left + w, top + h))


def _paste_background(img: "Image.Image", data: bytes) -> bool:
    """Paste ``data`` as a full-bleed cover background under a canvas gradient veil (dark at the top
    and bottom where the text lives, letting the art show through the middle). Returns False when
    the bytes don't decode (caller falls back to the plain canvas). Summary/quote length no longer
    affects layout — the artwork is behind the text, not below it."""
    from PIL import Image, ImageDraw

    bg = _cover(data, _W, _H)
    if bg is None:
        return False
    veil = Image.new("RGBA", (_W, _H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(veil)
    cr, cg, cb = (int(_CANVAS.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    mid = int(_H * 0.60)
    top_a, mid_a, bot_a = 240, 120, 236  # near-opaque top/bottom, art peeks through the middle
    for yy in range(_H):
        if yy <= mid:
            a = top_a + (mid_a - top_a) * (yy / mid)
        else:
            a = mid_a + (bot_a - mid_a) * ((yy - mid) / (_H - mid))
        vd.line((0, yy, _W, yy), fill=(cr, cg, cb, int(a)))
    composed = Image.alpha_composite(bg.convert("RGBA"), veil).convert("RGB")
    img.paste(composed, (0, 0))
    return True


def _mix(a: str, b: str, t: float) -> str:
    """Blend hex ``a`` toward hex ``b`` by ``t`` ∈ [0,1] (0 → a, 1 → b)."""
    ca = tuple(int(a.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    cb = tuple(int(b.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    r, g, bl = (round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % (r, g, bl)


def _spark(
    draw: "ImageDraw.ImageDraw",
    x: int,
    y: int,
    w: int,
    h: int,
    series: "tuple[float, ...]",
    accent: str,
    *,
    width: int,
) -> None:
    """A bold area sparkline: a soft accent 'shadow' fill under an accent line + an end dot.

    Values scale to the box (max→top, 0→baseline) so the SHAPE reads, not absolute counts. The area
    fill (accent blended most of the way to canvas) is the 'shadow underneath' that makes it read as
    a data-viz, not a hairline."""
    base = y + h
    vals = [float(v) for v in series]
    n = len(vals)
    if n < 2:  # a line needs ≥2 points (the route gates at ≥4; this guards direct callers)
        draw.line((x, base, x + w, base), fill=_BORDER, width=2)
        return
    hi = max(vals) or 1.0
    pts = [(x + round(i / (n - 1) * w), base - round(v / hi * h)) for i, v in enumerate(vals)]
    # Area fill: the polyline closed down to the baseline, in a dark accent tint.
    if len(pts) >= 2:
        draw.polygon([(x, base), *pts, (x + w, base)], fill=_mix(accent, _CANVAS, 0.80))
    draw.line((x, base, x + w, base), fill=_BORDER, width=2)  # zero baseline
    if len(pts) >= 2:
        draw.line(pts, fill=accent, width=width, joint="curve")
    ex, ey = pts[-1]
    r = width + 3
    draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=accent)


def _draw_trend_tile(
    draw: "ImageDraw.ImageDraw",
    x: int,
    y: int,
    w: int,
    h: int,
    series: "tuple[float, ...]",
    multiplier: float,
    accent: str,
) -> None:
    """A dashboard KPI tile: the single score (``↑2.6×``) big + a timeframe caption, then the area
    sparkline beneath — the score IS the sparkline's current value, so the two read as one. The
    score is mono (same family as the stats/kickers) so it reads as data, not editorial display."""
    score_font = _font(str(_MONO), 118)
    # Draw the "↑2.3" and the "×" separately so the × sits on the digits' bottom edge rather than
    # floating at the mono font's math axis (a centred glyph otherwise hovers mid-height).
    main = f"↑{multiplier:.1f}"
    draw.text((x, y), main, font=score_font, fill=accent)
    mb = draw.textbbox((x, y), main, font=score_font)  # (left, top, right, bottom)
    xb = score_font.getbbox("×")  # glyph box at origin
    draw.text((mb[2] + 14, mb[3] - xb[3]), "×", font=score_font, fill=accent)
    cap_y = y + 150
    draw.text((x, cap_y), "RISING · PAST 12 MONTHS", font=_font(str(_MONO), 24), fill=_MUTED)
    spark_top = cap_y + 52
    spark_h = y + h - spark_top
    if spark_h >= 60:  # only draw the sparkline when the clamped tile left room for it
        _spark(draw, x, spark_top, w, spark_h, series, accent, width=4)


def render_card_png(model: OgCardModel) -> bytes:
    """Render the model to a portrait PNG (1080×1440). Raises if Pillow is unavailable."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    accent = model.accent or _DEFAULT_ACCENT
    img = Image.new("RGB", (_W, _H), _CANVAS)
    # Full-bleed artwork background (EPISODE only, model.background) under a gradient veil — the art
    # sits BEHIND the text, so a long summary can't push it around. Falls back to plain canvas.
    if model.artwork and model.background:
        _paste_background(img, model.artwork)
    draw = ImageDraw.Draw(img)
    # Hairline frame (drawn over the background so it always reads).
    draw.rectangle((1, 1, _W - 2, _H - 2), outline=_BORDER, width=2)

    max_w = _W - _PAD * 2
    y = _PAD

    # ── Header — IDENTICAL structure on every card: kicker → title → hairline → lede → byline. ──
    # Kicker (mono, muted, uppercase).
    kmono = _font(str(_MONO), 26)
    draw.text((_PAD, y), model.kicker.upper(), font=kmono, fill=_MUTED)
    y += 60

    # Title (bold serif, wrapped, large) — capped to 4 lines so a very long title can't swallow the
    # whole card and push the lower section off the bottom.
    title_font = _font(str(_SERIF_BOLD), 88)
    for line in _cap_lines(_wrap(draw, model.title, title_font, max_w), 4):
        draw.text((_PAD, y), line, font=title_font, fill=_FG)
        y += 104

    # The single accent: a short hairline under the title.
    y += 40
    draw.rectangle((_PAD, y, _PAD + 88, y + 4), fill=accent)
    y += 4

    # Lede: a signature quote (italic serif, quoted) OR a descriptive blurb (roman serif, unquoted).
    # Capped to 3 lines for the same reason — the lower section must always have room.
    if model.quote:
        quote_font = _font(str(_SERIF_ITALIC), 46)
        y += 40
        for line in _cap_lines(_wrap(draw, f"“{model.quote}”", quote_font, max_w), 3):
            draw.text((_PAD, y), line, font=quote_font, fill=_FG)
            y += 62
    elif model.blurb:
        blurb_font = _font(str(_SERIF), 47)
        y += 40
        for line in _cap_lines(_wrap(draw, model.blurb, blurb_font, max_w), 3):
            draw.text((_PAD, y), line, font=blurb_font, fill=_FG)
            y += 63

    # Byline (serif, muted).
    if model.byline:
        byline_font = _font(str(_SERIF), 30)
        y += 24
        draw.text((_PAD, y), model.byline, font=byline_font, fill=_MUTED)
        y += 42

    # ── Lower section. Two mutually-exclusive modes:
    #    • background art (EPISODE): the art IS the backdrop, so nothing is drawn here — the stats
    #      pin to the footer over the dark bottom veil.
    #    • framed square (SHOW / person / org): a clean centred artwork square with the card's quiet
    #      hairline, biased up; the stats float halfway to the wordmark.
    #    • trend (topic / storyline): a dashboard KPI tile — the single score (↑N×) big, then a bold
    #      area sparkline. ──
    wy = _H - _PAD - 6  # the wordmark line, pinned to the bottom margin
    region_top = y + 48
    region_bot = wy - 130
    square = model.artwork if (model.artwork and not model.background) else None
    series = (
        model.sparkline
        if (not model.artwork and model.sparkline and len(model.sparkline) >= 4)
        else None
    )
    kpi = series is not None and model.trend_multiplier is not None
    if model.gallery and not model.background:
        _draw_gallery(img, draw, region_top, region_bot, model.gallery)
    elif square is not None:
        side = min(_ART_BIG, region_bot - region_top)
        art = _load_square(square, side) if side >= 300 else None
        if art is not None:
            aw = art.height
            cy = region_top + max(0, (region_bot - region_top - aw)) // 4  # bias UP
            ax = (_W - aw) // 2
            img.paste(art, (ax, cy))
            draw.rectangle((ax, cy, ax + aw - 1, cy + aw - 1), outline=_BORDER, width=2)
    elif series is not None and model.trend_multiplier is not None:
        # KPI trend tile, clamped to the room the header left so it can NEVER draw over the pinned
        # footer. Too little room → drop it cleanly (the footer still carries "↑ N× rising").
        avail = region_bot - region_top
        h = min(_TREND_TILE_H, avail)
        if h >= 210:
            cy = region_top + max(0, (avail - h)) // 4  # bias UP, not centred
            _draw_trend_tile(draw, _PAD, cy, max_w, h, series, model.trend_multiplier, accent)

    # ── Stats line — ALWAYS pinned one row above the wordmark, so the last row lands in the same
    #    place on every card (background, framed-square, or trend-tile). ──
    stat_font = _font(str(_MONO), 26)
    stat_y = wy - 56
    # Optional secondary line (episode key topics) above the stats — spaced the SAME as the gap
    # between the stats line and the wordmark, so the three footer rows are evenly stacked.
    footer_gap = wy - stat_y
    if model.tags:
        draw.text(
            (_PAD, stat_y - footer_gap), model.tags.upper(), font=_font(str(_MONO), 24), fill=_MUTED
        )
    x = _PAD
    if model.stats:
        s = model.stats.upper()
        draw.text((x, stat_y), s, font=stat_font, fill=_MUTED)
        x += int(draw.textlength(s, font=stat_font))
    if model.hot and not kpi:
        if model.stats:
            sep = "   ·   "
            draw.text((x, stat_y), sep, font=stat_font, fill=_MUTED)
            x += int(draw.textlength(sep, font=stat_font))
        draw.text((x, stat_y), model.hot.upper(), font=stat_font, fill=accent)

    # Wordmark with a single accent dot — the dot centred on the text's vertical midline (not its
    # top), so it lines up with the letters rather than floating above them.
    wordmark_font = _font(str(_MONO), 24)
    tb = draw.textbbox((_PAD + 24, wy), "closelistening.app", font=wordmark_font)
    dot_cy = (tb[1] + tb[3]) // 2
    draw.ellipse((_PAD, dot_cy - 6, _PAD + 12, dot_cy + 6), fill=accent)
    draw.text((_PAD + 24, wy), "closelistening.app", font=wordmark_font, fill=_MUTED)

    # Attribution for a licensed image (person photo / org logo) — bottom-right on the wordmark row,
    # so a CC-BY photo/logo composited into this PUBLIC card carries its required credit.
    if model.credit:
        credit_font = _font(str(_MONO), 20)
        cw = draw.textlength(model.credit, font=credit_font)
        draw.text((_W - _PAD - cw, wy + 3), model.credit, font=credit_font, fill=_MUTED)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
