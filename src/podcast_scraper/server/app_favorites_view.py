"""Hydrate the per-user favorites store into a display-ready grouped response.

Favorites are stored as a flat polymorphic list (``{kind, ref, …}``) in the per-user overlay. For
display we group by kind: ``episode`` favorites re-hydrate FRESH from the catalog (so titles/artwork
stay current), while ``insight`` favorites render from the stored snapshot (insights have no global
detail route). Newest-first. Extend with new kinds by adding a branch + a response group.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.schemas import AppFavoriteInsight, AppFavoritesResponse


def hydrate_favorites(root: Path, raw: Sequence[dict[str, Any]]) -> AppFavoritesResponse:
    """Group + hydrate stored favorites (newest-first) into the API response shape."""
    episodes = []
    insights = []
    for fav in reversed(list(raw)):  # stored newest-last → present newest-first
        kind = fav.get("kind")
        if kind == "episode":
            slug = fav.get("ref") or fav.get("slug")
            row = resolve_slug(root, str(slug)) if slug else None
            if row is not None:
                episodes.append(row_to_summary(root, row))
        elif kind == "insight":
            ref = fav.get("ref")
            if isinstance(ref, str) and ref:
                start = fav.get("start_ms")
                insights.append(
                    AppFavoriteInsight(
                        ref=ref,
                        text=str(fav.get("label") or ""),
                        episode_slug=fav.get("slug") if isinstance(fav.get("slug"), str) else None,
                        podcast_title=(
                            fav.get("sublabel") if isinstance(fav.get("sublabel"), str) else None
                        ),
                        start_ms=start if isinstance(start, int) else None,
                    )
                )
    return AppFavoritesResponse(episodes=episodes, insights=insights)
