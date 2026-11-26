"""An item's ``item_cards`` row and its ``has_*`` flags, from its files (CRFC-005 §3.2).

The card replaces ``server.content_source.row_to_summary``'s metadata read and GI parse per card,
so it is computed by the same functions that build today's card: the catalog row
(``corpus_catalog.catalog_row_for_metadata_path``), ``_card_lede``, ``_card_bullets``,
``episode_list_topics``, ``_has_transcript`` and ``app_moments.episode_has_moments``. A card from
rows then says what the card from files says.

A podcast episode only, in this slice; an episode the catalog does not show (unusable, D-44) has
no card. Its artwork is null until the assets importer writes ``assets`` rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from podcast_scraper.server.app_moments import episode_has_moments
from podcast_scraper.server.content_source import _card_bullets, _card_lede, _has_transcript
from podcast_scraper.server.corpus_catalog import (
    catalog_row_for_metadata_path,
    episode_list_topics,
)

#: The ``content_items`` flags a card carries too; their writer is the item's.
FLAGS = ("has_transcript", "has_gi", "has_kg", "has_moments")


def card_row(
    corpus_root: Path, metadata_relpath: str, item: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """The ``item_cards`` columns of the episode at ``metadata_relpath`` (``item`` is its
    ``corpus_rows.item_row``), or ``None`` when the catalog shows no row for it."""
    row = catalog_row_for_metadata_path(corpus_root, metadata_relpath)
    if row is None:
        return None
    return {
        "title": row.episode_title,
        "source_title": row.feed_title or "",
        "published_at": item["published_at"],
        "duration_s": row.duration_seconds,
        "artwork_asset_id": None,
        "lede": _card_lede(row),
        "summary_bullets": _card_bullets(row),
        "top_topics": episode_list_topics(row.summary_bullets),
        "has_transcript": _has_transcript(corpus_root, metadata_relpath),
        "has_gi": row.has_gi,
        "has_kg": row.has_kg,
        "has_moments": episode_has_moments(corpus_root, row),
        "withheld": item["withheld"],
        "unusable": item["unusable"],
    }
