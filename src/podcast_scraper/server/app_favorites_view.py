"""Hydrate the per-user favorites store into a display-ready response.

Favorites are stored as a flat list (``{kind, ref, …}``) in the per-user overlay. For display,
``episode`` favorites re-hydrate FRESH from the catalog (so titles/artwork stay current).
Newest-first. Extend with new kinds by adding a branch + a response group. Insights are NOT
favorites — they are captures, served by the highlights path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from podcast_scraper.server.content_source import row_to_summary
from podcast_scraper.server.schemas import (
    AppFavoriteEntity,
    AppFavoriteRef,
    AppFavoriteRefsResponse,
    AppFavoritesResponse,
)
from podcast_scraper.server.slugs import resolve_slug

# A saved THEME hydrates like any other grouping. Absent from this set a theme favourite would be
# accepted by the API, stored, and then silently dropped on the way back out — present in the file,
# invisible in Library › Saved.
_ENTITY_KINDS = {"person", "topic", "show", "storyline", "theme"}


def hydrate_favorites(root: Path, raw: Sequence[dict[str, Any]]) -> AppFavoritesResponse:
    """Group + hydrate stored favorites (newest-first) into the API response shape.

    ``episode`` favorites re-hydrate FRESH from the catalog (titles/artwork stay current); entity
    favorites (show/topic/person/storyline/theme) render from the label snapshot taken at save
    time, since they have no per-row catalog hydration here.
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


FAVORITE_KINDS = ("episode", "show", "topic", "person", "theme", "storyline")


def _color(fav: dict[str, Any]) -> str | None:
    return fav.get("color") if isinstance(fav.get("color"), str) else None


def favorite_refs(raw: Sequence[dict[str, Any]]) -> AppFavoriteRefsResponse:
    """Identity of every saved item, newest first — no catalog reads at all."""
    items = []
    for fav in reversed(list(raw)):
        kind = fav.get("kind")
        ref = fav.get("ref") or (fav.get("slug") if kind == "episode" else None)
        if kind in FAVORITE_KINDS and isinstance(ref, str) and ref:
            items.append(AppFavoriteRef(kind=kind, ref=ref, color=_color(fav)))
    return AppFavoriteRefsResponse(items=items)


def query_favorites(
    root: Path,
    raw: Sequence[dict[str, Any]],
    *,
    kind: str | None,
    q: str | None,
    color: str | None,
    sort: str,
    offset: int,
    limit: int,
) -> AppFavoritesResponse:
    """One page of the favourites, filtered and sorted on the server.

    Episodes are matched on their CATALOG row (title, show) and only the page is hydrated into a
    summary — that read per saved episode is what made the whole list expensive. An episode the
    corpus no longer has is dropped, as ``hydrate_favorites`` drops it. Matching is
    case-insensitive substring, the client's ``matchesQuery``.
    """
    needle = (q or "").strip().casefold()

    def matches(*texts: str | None) -> bool:
        return not needle or any(needle in (t or "").casefold() for t in texts)

    # (kind, sort key, payload) for every favourite that passes q + colour, newest first.
    hits: list[tuple[str, str, Any]] = []
    for fav in reversed(list(raw)):
        k = fav.get("kind")
        c = _color(fav)
        if color and c != color:
            continue
        if k == "episode":
            slug = fav.get("ref") or fav.get("slug")
            row = resolve_slug(root, str(slug)) if slug else None
            if row is None or not matches(row.episode_title, row.feed_title):
                continue
            hits.append((k, row.episode_title, (row, c)))
        elif k in _ENTITY_KINDS:
            ref = fav.get("ref")
            if not isinstance(ref, str) or not ref:
                continue
            label = str(fav.get("label") or ref)
            if not matches(label):
                continue
            entity = AppFavoriteEntity(
                kind=k,  # type: ignore[arg-type]  # guarded by _ENTITY_KINDS
                ref=ref,
                label=label,
                sublabel=fav.get("sublabel") if isinstance(fav.get("sublabel"), str) else None,
                color=c,
            )
            hits.append((k, label, entity))

    counts = {k: 0 for k in FAVORITE_KINDS}
    for k, _, _ in hits:
        counts[k] += 1
    selected = [h for h in hits if not kind or h[0] == kind]
    if sort == "title":
        selected.sort(key=lambda h: h[1].casefold())
    page = selected[offset : offset + limit]

    episodes = []
    entities = []
    for k, _, payload in page:
        if k == "episode":
            row, c = payload
            summary = row_to_summary(root, row)
            summary.color = c
            episodes.append(summary)
        else:
            entities.append(payload)
    return AppFavoritesResponse(
        episodes=episodes, entities=entities, total=len(selected), counts=counts
    )
