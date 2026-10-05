"""Server-side shareable OG cards (#2036).

THE share card, for every kind: the PNG the player's Share menu shares (it fetches
``/og/{kind}/{id}.png``), and the ``og:image`` a shared LINK unfurls as. The one
renderer since 2026-10-05 — the player's own canvas twin was deleted. Bridge-only
(PRD-035 Principle 4): a card carries transcript-derived text + KG metadata
only, never source audio.
"""

from podcast_scraper.server.og.card import (
    accent_for_kind,
    OgCardModel,
    render_card_png,
)

__all__ = ["OgCardModel", "accent_for_kind", "render_card_png"]
