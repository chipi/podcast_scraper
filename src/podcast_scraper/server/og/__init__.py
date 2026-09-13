"""Server-side shareable OG cards (#2036).

Renders the same editorial "collectible" card the player draws client-side
(``composables/entityShareCard.ts``) to a PNG, so a shared LINK unfurls AS the
card (``og:image``) — not just an explicitly-shared card image. Bridge-only
(PRD-035 Principle 4): a card carries transcript-derived text + KG metadata
only, never source audio.
"""

from podcast_scraper.server.og.card import (
    accent_for_kind,
    OgCardModel,
    render_card_png,
)

__all__ = ["OgCardModel", "accent_for_kind", "render_card_png"]
