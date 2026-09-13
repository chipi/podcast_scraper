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
from podcast_scraper.server.schemas import AppFavoriteEntity, AppFavoritesResponse

_ENTITY_KINDS = {"person", "topic", "show", "storyline"}


def hydrate_favorites(root: Path, raw: Sequence[dict[str, Any]]) -> AppFavoritesResponse:
    """Group + hydrate stored favorites (newest-first) into the API response shape.

    ``episode`` favorites re-hydrate FRESH from the catalog (titles/artwork stay current); entity
    favorites (show/topic/person/storyline) render from the label snapshot taken at save time, since
    they have no per-row catalog hydration here.
    """
    episodes = []
    entities = []
    for fav in reversed(list(raw)):  # stored newest-last → present newest-first
        kind = fav.get("kind")
        # Per-user colour annotation (RFC-121 ph. 4) rides on the stored row, layered onto the
        # freshly-hydrated catalog summary; a non-string (hand-corrupt) value reads as unset.
        color = fav.get("color") if isinstance(fav.get("color"), str) else None
        if kind == "episode":
            slug = fav.get("ref") or fav.get("slug")
            row = resolve_slug(root, str(slug)) if slug else None
            if row is not None:
                summary = row_to_summary(root, row)
                summary.color = color
                episodes.append(summary)
        elif kind in _ENTITY_KINDS:
            ref = fav.get("ref")
            if isinstance(ref, str) and ref:
                entities.append(
                    AppFavoriteEntity(
                        kind=kind,  # type: ignore[arg-type]  # guarded by _ENTITY_KINDS
                        ref=ref,
                        label=str(fav.get("label") or ref),
                        sublabel=(
                            fav.get("sublabel") if isinstance(fav.get("sublabel"), str) else None
                        ),
                        color=color,
                    )
                )
    return AppFavoritesResponse(episodes=episodes, entities=entities)
