"""Assemble an :class:`OgCardModel` (and light OG meta) for an entity, from the shared corpus.

Twin of the per-surface ``shareModel`` computeds in the player. Reuses the same KG-grounded card
builders the ``/api/app/*`` routes use, so the unfurled card says exactly what the in-app card says.
Bridge-only: transcript-derived text + KG metadata only, never audio.

Every builder is best-effort: an unknown id or a thin/broken corpus returns ``None`` (the route
404s; the HTML injector falls back to the generic site OG), never an exception to the request path.
Each enrichment (quote, blurb, artwork, trend) is independently guarded — a missing piece just drops
that piece, it never fails the card.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from podcast_scraper.server.og.card import accent_for_kind, OgCardModel

# The shareable entity kinds — mirrors the player's share surfaces + router names.
OG_KINDS = frozenset({"topic", "person", "organization", "episode", "show", "storyline"})

_MAX_QUOTE = 200  # trim a runaway insight so the card stays a card, not a paragraph
_MAX_BLURB = 160


@dataclass(frozen=True)
class OgMeta:
    """The light meta injected into the SPA document head for a shared link."""

    title: str
    description: str


def _eps(n: int) -> str:
    return f"{n} episode" if n == 1 else f"{n} episodes"


def _clip(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    t = " ".join(text.split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def build_og_model(root: Path, kind: str, ident: str) -> OgCardModel | None:
    """Build the card model for ``(kind, ident)``, or ``None`` when it can't be resolved."""
    ident = ident.strip()
    if not ident or kind not in OG_KINDS:
        return None
    try:
        builder = {
            "topic": _topic,
            "person": _person,
            "organization": _org,
            "storyline": _storyline,
            "episode": _episode,
            "show": _show,
        }[kind]
        return builder(root, ident)
    except Exception:  # noqa: BLE001 - never let a corpus quirk break the request path
        return None


def build_og_meta(root: Path, kind: str, ident: str) -> OgMeta | None:
    """The document-head meta for a shared link — title + a one-line description drawn from the
    card's strongest text (its quote/blurb, else its stat/byline)."""
    model = build_og_model(root, kind, ident)
    if model is None:
        return None
    description = model.quote or model.blurb or model.stats or model.byline or "Close Listening"
    return OgMeta(title=model.title, description=description)


# --------------------------------------------------------------------------- #
# Shared enrichment helpers — each best-effort (None on any miss).
# --------------------------------------------------------------------------- #
def _asset_bytes(path: str | Path) -> bytes | None:
    try:
        return Path(path).read_bytes()
    except Exception:  # noqa: BLE001
        return None


def _artwork_bytes(root: Path, relpath: str | None) -> bytes | None:
    """Local corpus-art file bytes for a catalog image relpath (episode/show art)."""
    if not relpath:
        return None
    from podcast_scraper.server.app_artwork import safe_artwork_target

    target = safe_artwork_target(root, relpath)
    return _asset_bytes(target) if target else None


@lru_cache(maxsize=8)
def _velocity_map(root_str: str, kind: str) -> dict[str, float]:
    """entity_id → velocity for a kind, from the same ``trending`` computation the app uses. Cached
    per (corpus, kind); empty when the corpus has no temporal-velocity artifact."""
    from podcast_scraper.server.app_momentum import trending

    try:
        rows = trending(Path(root_str), None, kind=kind, scope="corpus", limit=50)
        return {r.entity_id: float(r.velocity) for r in rows}
    except Exception:  # noqa: BLE001
        return {}


def _hot(root: Path, kind: str, entity_id: str) -> str | None:
    """The one accent stat: "↑ N× rising", only when the entity is genuinely rising (≥1.5×)."""
    v = _velocity_map(str(root), kind).get(entity_id)
    return f"↑ {v:.1f}× rising" if v and v >= 1.5 else None


# --------------------------------------------------------------------------- #
# Per-kind builders.
# --------------------------------------------------------------------------- #
def _topic(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import (
        build_topic_card,
        build_topic_perspectives,
    )

    card = build_topic_card(root, ident)
    if card is None:
        return None
    quote: str | None = None
    byline: str | None = None
    voices = len(card.related_people or [])
    persp = build_topic_perspectives(root, ident)
    if persp and persp.perspectives:
        voices = persp.perspective_count or voices
        lead = persp.perspectives[0]
        if lead.insights:
            # Salience-sorted → [0].insights[0] is the leading voice's top take; attribute it.
            quote = _clip(lead.insights[0].text, _MAX_QUOTE)
            byline = f"— {lead.person_name}" if lead.person_name else None
    stats = _eps(card.episode_count)
    if voices:
        stats += f" · {voices} {'voice' if voices == 1 else 'voices'}"
    return OgCardModel(
        kicker="Topic",
        title=card.label,
        quote=quote,
        byline=byline,
        stats=stats,
        hot=_hot(root, "topic", card.id),
        accent=accent_for_kind("topic"),
    )


def _person(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import build_person_card

    card = build_person_card(root, ident)
    if card is None:
        return None
    # Blurb: the external one-line descriptor (person_web) — "American economist and Fed historian";
    # else what they talk about, from their top co-occurring topics.
    blurb = card.web.description if card.web else None
    if not blurb and card.related_topics:
        names = [t.label for t in card.related_topics[:3] if t.label]
        blurb = f"On {', '.join(names)}." if names else None
    # Byline: the show they HOST (their primary identity), else the headline role.
    byline: str | None = None
    host_show = next((s for s in (card.shows or []) if (s.role or "").lower() == "host"), None)
    if host_show and host_show.title:
        byline = f"Host of {host_show.title}"
    elif card.role:
        byline = card.role.strip().title()
    photo = _person_photo(root, card.id)
    return OgCardModel(
        kicker="Person",
        title=card.label,
        blurb=_clip(blurb, _MAX_BLURB),
        byline=byline,
        stats=_eps(card.episode_count),
        accent=accent_for_kind("person"),
        artwork=photo,
    )


def _person_photo(root: Path, person_id: str) -> bytes | None:
    from podcast_scraper.enrichment.enrichers.person_web import person_image_path

    found = person_image_path(root, person_id)
    return _asset_bytes(found[0]) if found else None


def _org(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_relational_view import build_org_card

    card = build_org_card(root, ident)
    if card is None:
        return None
    web = card.web
    blurb = (web.description or web.summary) if web else None
    stats = _eps(card.episode_count)
    if web and web.founded:
        stats += f" · founded {web.founded}"
    elif web and web.industry:
        stats += f" · {web.industry}"
    return OgCardModel(
        kicker="Organization",
        title=card.label,
        blurb=_clip(blurb, _MAX_BLURB),
        stats=stats,
        accent=accent_for_kind("organization"),
        artwork=_org_logo(root, card.id),
    )


def _org_logo(root: Path, org_id: str) -> bytes | None:
    from podcast_scraper.enrichment.enrichers.org_web import org_logo_path

    found = org_logo_path(root, org_id)
    return _asset_bytes(found[0]) if found else None


def _storyline(root: Path, ident: str) -> OgCardModel | None:
    # A storyline IS its anchor topic's theme cluster (same as StorylineView): topics discussed
    # together. The card explains that (byline), shows WHICH topics (blurb) + how big + the trend.
    from podcast_scraper.server.app_relational_view import build_topic_card

    card = build_topic_card(root, ident)
    if card is None:
        return None
    members = [card.label] + [t.label for t in (card.theme_sibling_topics or []) if t.label]
    n_topics = len(members)
    n_eps = len(card.episodes or [])
    stats = f"{n_topics} {'topic' if n_topics == 1 else 'topics'}"
    if n_eps:
        stats += f" · {_eps(n_eps)}"
    hot = _hot(root, "storyline", card.theme_cluster_id) if card.theme_cluster_id else None
    return OgCardModel(
        kicker="Storyline",
        title=card.theme_cluster_label or card.label,
        blurb=_clip(" · ".join(members), _MAX_BLURB),
        byline="Topics discussed together",
        stats=stats,
        hot=hot,
        accent=accent_for_kind("storyline"),
    )


def _episode(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_slugs import resolve_slug

    row = resolve_slug(root, ident)
    if row is None:
        return None
    # Quote: the strongest salience-ranked GI insight (matches the in-app episode share), else the
    # lead summary bullet.
    quote: str | None = None
    n_insights = 0
    if row.gi_relative_path:
        gi = load_json_artifact(root, row.gi_relative_path)
        if gi:
            insights = insights_from_gi(gi)
            n_insights = len(insights)
            if insights:
                quote = _clip(insights[0].text, _MAX_QUOTE)
    if not quote and row.summary_bullets:
        quote = _clip(row.summary_bullets[0], _MAX_QUOTE)
    mins = round(row.duration_seconds / 60) if row.duration_seconds else None
    byline_parts = [f"{mins} min"] if mins else []
    if n_insights:
        byline_parts.append(
            f"{n_insights} insight" if n_insights == 1 else f"{n_insights} insights"
        )
    art = _artwork_bytes(root, row.episode_image_local_relpath or row.feed_image_local_relpath)
    return OgCardModel(
        kicker=f"Episode · {row.feed_title}" if row.feed_title else "Episode",
        title=row.episode_title,
        quote=quote,
        byline=" · ".join(byline_parts) or None,
        accent=accent_for_kind("episode"),
        artwork=art,
    )


def _show(root: Path, ident: str) -> OgCardModel | None:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.corpus_catalog import aggregate_feeds

    rows = cached_catalog(root)
    feed = next((f for f in aggregate_feeds(rows) if f.get("feed_id") == ident), None)
    if feed is None:
        return None
    stats = _eps(int(feed.get("episode_count", 0)))
    cadence, typical = _cadence_and_length(rows, ident)
    if cadence:
        stats += f" · {cadence}"
    if typical:
        stats += f" · ~{typical} min"
    return OgCardModel(
        kicker="Show",
        title=feed.get("display_title") or ident,
        blurb=_clip(feed.get("description"), _MAX_BLURB),
        stats=stats,
        hot=_hot(root, "show", ident),
        accent=accent_for_kind("show"),
        artwork=_artwork_bytes(root, feed.get("image_local_relpath")),
    )


def _cadence_and_length(rows: list, feed_id: str) -> tuple[str | None, int | None]:
    """One-word publishing rhythm + median episode minutes for a show (as the show page does)."""
    dates: list[float] = []
    mins: list[int] = []
    for r in rows:
        if (getattr(r, "feed_id", None) or "") != feed_id:
            continue
        pub = (getattr(r, "publish_date", "") or "").strip()
        if len(pub) >= 10:
            try:
                dates.append(datetime.fromisoformat(pub[:10]).timestamp())
            except ValueError:
                pass
        dur = getattr(r, "duration_seconds", None)
        if dur:
            mins.append(int(dur // 60))
    cadence: str | None = None
    if len(dates) >= 3:
        dates.sort(reverse=True)
        gaps = [(dates[i] - dates[i + 1]) / 86_400 for i in range(len(dates) - 1)]
        gaps.sort()
        median_gap = gaps[len(gaps) // 2]
        if median_gap > 0:
            cadence = (
                "daily"
                if median_gap < 2
                else (
                    "weekly"
                    if median_gap < 10
                    else "biweekly" if median_gap < 20 else "monthly" if median_gap < 45 else None
                )
            )
    typical = int(statistics.median(mins)) if len(mins) >= 3 else None
    return cadence, typical
