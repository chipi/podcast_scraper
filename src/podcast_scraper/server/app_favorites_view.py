"""Hydrate the per-user favorites store into a display-ready response.

Favorites are stored as a flat list (``{kind, ref, …}``) in the per-user overlay. For display,
``episode`` favorites re-hydrate FRESH from the catalog (so titles/artwork stay current).
Newest-first. Extend with new kinds by adding a branch + a response group. Insights are NOT
favorites — they are captures, served by the highlights path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.schemas import AppFavoritesResponse


def hydrate_favorites(root: Path, raw: Sequence[dict[str, Any]]) -> AppFavoritesResponse:
    """Group + hydrate stored favorites (newest-first) into the API response shape."""
    episodes = []
    for fav in reversed(list(raw)):  # stored newest-last → present newest-first
        if fav.get("kind") == "episode":
            slug = fav.get("ref") or fav.get("slug")
            row = resolve_slug(root, str(slug)) if slug else None
            if row is not None:
                episodes.append(row_to_summary(root, row))
    return AppFavoritesResponse(episodes=episodes)
