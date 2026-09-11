"""Assemble an :class:`OgCardModel` (and light OG meta) for an entity, from the shared corpus.

Twin of the per-surface ``shareModel`` computeds in the player. Reuses the same KG-grounded card
builders the ``/api/app/*`` routes use, so the unfurled card says exactly what the in-app card says.
Bridge-only: transcript-derived text + KG metadata only, never audio.

Every builder is best-effort: an unknown id or a thin/broken corpus returns ``None`` (the route
404s; the HTML injector falls back to the generic site OG), never an exception to the request path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from podcast_scraper.server.og.card import accent_for_kind, OgCardModel

# The shareable entity kinds — mirrors the player's share surfaces + router names.
OG_KINDS = frozenset({"topic", "person", "organization", "episode", "show", "storyline"})


@dataclass(frozen=True)
class OgMeta:
    """The light meta injected into the SPA document head for a shared link."""

    title: str
    description: str


def _eps(n: int) -> str:
    return f"{n} episode" if n == 1 else f"{n} episodes"


def build_og_model(root: Path, kind: str, ident: str) -> OgCardModel | None:
    """Build the card model for ``(kind, ident)``, or ``None`` when it can't be resolved."""
    ident = ident.strip()
    if not ident or kind not in OG_KINDS:
        return None
    try:
        if kind == "topic":
            return _topic(root, ident)
        if kind == "person":
            return _person(root, ident)
        if kind == "organization":
            return _org(root, ident)
        if kind == "storyline":
            return _storyline(root, ident)
        if kind == "episode":
            return _episode(root, ident)
        if kind == "show":
            return _show(root, ident)
    except Exception:  # noqa: BLE001 - never let a corpus quirk break the request path
        return None
    return None


def build_og_meta(root: Path, kind: str, ident: str) -> OgMeta | None:
    """The document-head meta for a shared link — title + a one-line description drawn from the
    card's strongest text (its quote, else its stat/byline)."""
    model = build_og_model(root, kind, ident)
    if model is None:
        return None
    description = model.quote or model.stats or model.byline or "Close Listening"
    return OgMeta(title=model.title, description=description)


def _topic(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import (
        build_topic_card,
        build_topic_perspectives,
    )

    card = build_topic_card(root, ident)
    if card is None:
        return None
    quote: str | None = None
    persp = build_topic_perspectives(root, ident)
    if persp and persp.perspectives and persp.perspectives[0].insights:
        # Perspectives are salience-sorted, so [0].insights[0] is the leading voice's top take.
        quote = persp.perspectives[0].insights[0].text
    return OgCardModel(
        kicker="Topic",
        title=card.label,
        quote=quote,
        stats=_eps(card.episode_count),
        accent=accent_for_kind("topic"),
    )


def _person(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import build_person_card

    card = build_person_card(root, ident)
    if card is None:
        return None
    role = (card.role or "").strip()
    return OgCardModel(
        kicker="Person",
        title=card.label,
        byline=role.title() if role else None,
        stats=_eps(card.episode_count),
        accent=accent_for_kind("person"),
    )


def _org(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import build_org_card

    card = build_org_card(root, ident)
    if card is None:
        return None
    return OgCardModel(
        kicker="Organization",
        title=card.label,
        stats=_eps(card.episode_count),
        accent=accent_for_kind("organization"),
    )


def _storyline(root: Path, ident: str) -> OgCardModel | None:
    # A storyline IS its anchor topic's theme cluster (same as StorylineView) — the route param is
    # the anchor topic id and everything derives from the topic card.
    from podcast_scraper.server.app_relational_view import build_topic_card

    card = build_topic_card(root, ident)
    if card is None:
        return None
    n_topics = 1 + len(card.theme_sibling_topics or [])
    n_eps = len(card.episodes or [])
    parts = [f"{n_topics} {'topic' if n_topics == 1 else 'topics'}"]
    if n_eps:
        parts.append(_eps(n_eps))
    return OgCardModel(
        kicker="Storyline",
        title=card.theme_cluster_label or card.label,
        stats=" · ".join(parts),
        accent=accent_for_kind("storyline"),
    )


def _episode(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_slugs import resolve_slug

    row = resolve_slug(root, ident)
    if row is None:
        return None
    quote = row.summary_bullets[0] if row.summary_bullets else None
    mins = round(row.duration_seconds / 60) if row.duration_seconds else None
    return OgCardModel(
        kicker=f"Episode · {row.feed_title}" if row.feed_title else "Episode",
        title=row.episode_title,
        quote=quote,
        byline=f"{mins} min" if mins else None,
        accent=accent_for_kind("episode"),
    )


def _show(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.corpus_catalog import aggregate_feeds

    feed = next(
        (f for f in aggregate_feeds(cached_catalog(root)) if f.get("feed_id") == ident), None
    )
    if feed is None:
        return None
    return OgCardModel(
        kicker="Show",
        title=feed.get("display_title") or ident,
        stats=_eps(int(feed.get("episode_count", 0))),
        accent=accent_for_kind("show"),
    )
