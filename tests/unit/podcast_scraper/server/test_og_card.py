"""Unit tests for the server-side OG card renderer (#2036)."""

from __future__ import annotations

from podcast_scraper.server.og.card import accent_for_kind, OgCardModel, render_card_png

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_accent_for_kind_mirrors_the_theme_tokens() -> None:
    # Only topic + person own a theme token; everything else keeps the brand cyan (few-colours).
    assert accent_for_kind("topic") == "#8ad2e5"
    assert accent_for_kind("person") == "#e0b354"
    assert accent_for_kind("organization") == "#8ad2e5"
    assert accent_for_kind("show") == "#8ad2e5"
    assert accent_for_kind("storyline") == "#8ad2e5"
    assert accent_for_kind(None) == "#8ad2e5"


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
