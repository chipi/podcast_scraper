"""Unit tests for the server-side OG card renderer (#2036)."""

from __future__ import annotations

from podcast_scraper.server.og.card import accent_for_kind, OgCardModel, render_card_png

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_accent_for_kind_gives_each_kind_a_distinct_colour() -> None:
    assert accent_for_kind("topic") == "#8ad2e5"
    assert accent_for_kind("person") == "#e0b354"
    assert accent_for_kind("storyline") == "#9d8cff"
    assert accent_for_kind("organization") == "#5fd0a8"
    # show/episode keep the brand cyan (their artwork differentiates them); unknown → cyan too.
    assert accent_for_kind("show") == "#8ad2e5"
    assert accent_for_kind(None) == "#8ad2e5"
    # the four coloured kinds are mutually distinct.
    assert len({accent_for_kind(k) for k in ("topic", "person", "storyline", "organization")}) == 4


def test_render_full_card_is_a_portrait_png() -> None:
    png = render_card_png(
        OgCardModel(
            kicker="Topic",
            title="Risk Is a Systems Property",
            quote="Risk is a systems property — it lives in the couplings, not the parts.",
            byline="— Dr. Elena Fischer",
            stats="28 episodes · 10 voices",
            hot="↑ 2.3× rising",
            accent=accent_for_kind("topic"),
        )
    )
    assert png[:8] == _PNG_MAGIC
    # A 1080×1440 PNG's IHDR carries the dimensions big-endian at bytes 16..24.
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    assert (width, height) == (1080, 1440)


def test_render_bare_card_without_optional_parts() -> None:
    # A card with only kicker + title must still render (person/org/show carry no quote).
    png = render_card_png(OgCardModel(kicker="Show", title="My Show"))
    assert png[:8] == _PNG_MAGIC


def test_render_wraps_a_long_title_without_error() -> None:
    png = render_card_png(
        OgCardModel(kicker="Topic", title="A " * 60 + "very long wrapping title indeed")
    )
    assert png[:8] == _PNG_MAGIC


def test_render_blurb_card_without_a_quote() -> None:
    # org / storyline / show use a descriptive blurb instead of a spoken quote.
    png = render_card_png(
        OgCardModel(
            kicker="Storyline",
            title="The Energy Transition",
            blurb="Grid · Batteries · Solar · Nuclear · Demand",
            byline="Topics discussed together",
            stats="6 topics · 34 episodes",
            hot="↑ 1.8× rising",
        )
    )
    assert png[:8] == _PNG_MAGIC


def test_render_with_artwork_square() -> None:
    # A tiny valid PNG as the identity square — must composite without error and stay 1080×1440.
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (300, 300), "#334455").save(buf, format="PNG")
    png = render_card_png(
        OgCardModel(
            kicker="Show", title="Macro Musings", stats="214 episodes", artwork=buf.getvalue()
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == (1080, 1440)


def test_render_survives_undecodable_artwork() -> None:
    # Broken art bytes must drop the square, not break the card.
    png = render_card_png(
        OgCardModel(kicker="Show", title="Macro Musings", artwork=b"not-an-image")
    )
    assert png[:8] == _PNG_MAGIC


def _png_bytes(size: int = 300, color: str = "#334455") -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def _dims(png: bytes) -> tuple[int, int]:
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _accent_in_band(png: bytes, y0: int, y1: int, x0: int, x1: int) -> bool:
    """True if any pixel in the box is near the topic-cyan accent (#8ad2e5) — used to prove the
    trend tile did NOT overflow into the footer band."""
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(png)).convert("RGB")
    ar, ag, ab = 0x8A, 0xD2, 0xE5
    for yy in range(y0, y1, 3):
        for xx in range(x0, x1, 5):
            px = im.getpixel((xx, yy))
            if not isinstance(px, tuple) or len(px) < 3:
                continue
            r, g, b = px[0], px[1], px[2]
            if abs(r - ar) < 40 and abs(g - ag) < 40 and abs(b - ab) < 40:
                return True
    return False


def test_render_full_bleed_background_episode() -> None:
    # background=True → the artwork is the backdrop under a veil; still a 1080×1440 PNG.
    png = render_card_png(
        OgCardModel(
            kicker="Episode · Show",
            title="The Grid Problem",
            blurb="A conversation about slow, correlated risk.",
            byline="Sam in conversation with Elena",
            stats="42 min · 3 insights · Jul 2026",
            tags="systems thinking · risk",
            artwork=_png_bytes(800),
            background=True,
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)


def test_render_trend_tile_and_sparkline() -> None:
    png = render_card_png(
        OgCardModel(
            kicker="Topic",
            title="Risk",
            quote="Risk is a systems property.",
            byline="— Dr. Elena Fischer",
            stats="28 episodes · 10 voices",
            sparkline=tuple(range(1, 27)),
            trend_multiplier=2.6,
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)


def test_render_trend_tile_tall_header_stays_on_card() -> None:
    # Regression: a very long title + long quote must NOT push the KPI tile off the card or over the
    # footer — the title/lede cap + the tile clamp keep everything inside 1080×1440.
    png = render_card_png(
        OgCardModel(
            kicker="Topic",
            title="Managing Systemic Risk Across Interconnected Financial Domains Worldwide",
            quote=(
                "Risk is a systems property that lives in the couplings between the parts, not in "
                "the parts themselves, and the dangerous ones are the slow correlations."
            ),
            byline="— Dr. Elena Fischer",
            stats="28 episodes · 10 voices",
            sparkline=tuple(range(1, 27)),
            trend_multiplier=2.6,
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)
    # The footer band (above the wordmark, x past the accent dot) must carry NO accent pixels — if
    # the KPI tile overflowed the header cap + clamp, the big score / sparkline would paint here.
    assert not _accent_in_band(png, 1230, 1330, 160, 1000)


def test_render_gallery_row() -> None:
    # A guest's multi-show gallery: a row of framed squares; one bad tile is skipped, not fatal.
    png = render_card_png(
        OgCardModel(
            kicker="Person",
            title="Dr. Elena Fischer",
            byline="Guest",
            stats="4 episodes · 2 shows · latest Jul 2026",
            gallery=(_png_bytes(400, "#402030"), _png_bytes(400, "#204030")),
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)


def test_cap_lines_ellipsizes_when_over_limit() -> None:
    from podcast_scraper.server.og.card import _cap_lines

    assert _cap_lines(["a", "b"], 3) == ["a", "b"]
    capped = _cap_lines(["a", "b", "c", "d", "e"], 3)
    assert len(capped) == 3
    assert capped[-1].endswith("…")


def test_render_credit_line_for_a_licensed_image() -> None:
    png = render_card_png(
        OgCardModel(
            kicker="Person",
            title="Dr. Elena Fischer",
            byline="Host of Macro Musings",
            stats="12 episodes",
            artwork=_png_bytes(400),
            credit="Photo: A. Photographer · CC BY-SA 4.0",
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)


def test_render_hard_breaks_a_spaceless_title() -> None:
    # A long unspaced token (CJK/URL) must not overflow — _hard_break splits it.
    png = render_card_png(OgCardModel(kicker="Topic", title="x" * 120))
    assert png[:8] == _PNG_MAGIC


def test_spark_and_gallery_guard_degenerate_inputs() -> None:
    # Direct-caller safety: a <2-point series and an empty gallery must not divide-by-zero.
    from PIL import Image, ImageDraw

    from podcast_scraper.server.og.card import _draw_gallery, _spark

    img = Image.new("RGB", (200, 200), "#000000")
    d = ImageDraw.Draw(img)
    _spark(d, 0, 0, 200, 80, (1.0,), "#8ad2e5", width=4)  # 1 point → baseline only, no crash
    _draw_gallery(img, d, 0, 200, ())  # empty tuple → no-op, no crash


def test_render_organization_with_its_own_accent() -> None:
    # Org has no fixture, so render the model directly — it gets the green accent + a framed logo.
    png = render_card_png(
        OgCardModel(
            kicker="Organization",
            title="The Federal Reserve",
            blurb="The central banking system of the United States.",
            stats="17 episodes · founded 1913",
            accent=accent_for_kind("organization"),
            artwork=_png_bytes(400, "#203040"),
        )
    )
    assert png[:8] == _PNG_MAGIC
    assert _dims(png) == (1080, 1440)
