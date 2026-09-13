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

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from podcast_scraper.server.og.card import accent_for_kind, OgCardModel

logger = logging.getLogger(__name__)

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


def build_og_model(
    root: Path, kind: str, ident: str, *, with_art: bool = True
) -> OgCardModel | None:
    """Build the card model for ``(kind, ident)``, or ``None`` when it can't be resolved.

    ``with_art=False`` skips loading the artwork/photo/logo/gallery bytes (MB-scale file reads) —
    used by :func:`build_og_meta`, which only needs the text fields for the document head."""
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
        return builder(root, ident, with_art)
    except Exception:  # noqa: BLE001 - never let a corpus quirk break the request path
        # A legitimate "not found" returns None WITHOUT raising, so reaching here means a real bug
        # (bad schema field, unexpected type) — surface it in logs instead of a silent 404.
        logger.warning("OG card build failed for %s/%s", kind, ident, exc_info=True)
        return None


def build_og_meta(root: Path, kind: str, ident: str) -> OgMeta | None:
    """The document-head meta for a shared link — title + a one-line description drawn from the
    card's strongest text. Skips artwork loading (text-only) so the SPA document path is cheap."""
    model = build_og_model(root, kind, ident, with_art=False)
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


def _velocity_mtime(root: Path) -> float:
    """mtime of the temporal-velocity artifact that feeds corpus trending, or 0.0 when absent. Used
    as a cache key so ``_trend_map`` self-invalidates when the corpus is re-enriched."""
    try:
        return (root / "enrichments" / "temporal_velocity.json").stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=16)
def _trend_map_cached(
    root_str: str, kind: str, _mtime: float
) -> dict[str, tuple[float, tuple[float, ...]]]:
    from podcast_scraper.server.app_momentum import trending

    try:
        # window="1y" so the "↑N×" score matches the card's "PAST 12 MONTHS" caption + the 52-week
        # sparkline (trending's default window is 3m, which would mislabel the score).
        rows = trending(Path(root_str), None, kind=kind, scope="corpus", limit=50, window="1y")
        return {r.entity_id: (float(r.velocity), tuple(r.series or ())) for r in rows}
    except Exception:  # noqa: BLE001
        logger.warning("OG trend lookup failed for kind=%s", kind, exc_info=True)
        return {}


def _trend_map(root: Path, kind: str) -> dict[str, tuple[float, tuple[float, ...]]]:
    """entity_id → (velocity, weekly-series), cached per (corpus, kind, artifact-mtime) so a
    re-enrich busts the cache rather than serving a stale trend for the process lifetime."""
    return _trend_map_cached(str(root), kind, _velocity_mtime(root))


def _people(names: list[str], cap: int = 2) -> str:
    """Join up to ``cap`` names, then ``+N`` for the rest ("Elena Fischer, Sam +2")."""
    if len(names) <= cap:
        return ", ".join(names)
    return ", ".join(names[:cap]) + f" +{len(names) - cap}"


def _episode_roster(root: Path, row: object) -> tuple[list[str], list[str], list[str]]:
    """(hosts, guests, topic-labels) named on an episode, from its KG."""
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_kg_view import entities_from_kg

    rel = getattr(row, "kg_relative_path", None)
    if not rel:
        return [], [], []
    persons, _orgs, topics = entities_from_kg(load_json_artifact(root, rel))
    hosts = [p.name for p in persons if (p.role or "") == "host"]
    guests = [p.name for p in persons if (p.role or "") == "guest"]
    return hosts, guests, [t.label for t in topics if t.label]


def _show_signals(root: Path, feed_id: str, rows: list) -> tuple[str | None, list[str]]:
    """(host, top-topics) for a show, from ONE bounded KG scan over the feed's episodes. The host is
    the person most often in the ``host`` role (metadata-grounded, not inferred from ranking); the
    top topics are what the show is most ABOUT per our KG — the value-add over the feed's blurb."""
    from collections import Counter

    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_kg_view import entities_from_kg

    hosts: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    seen = 0
    for r in rows:
        if (getattr(r, "feed_id", None) or "") != feed_id or not r.kg_relative_path:
            continue
        persons, _orgs, tps = entities_from_kg(load_json_artifact(root, r.kg_relative_path))
        for p in persons:
            if (p.role or "") == "host":
                hosts[p.name] += 1
        for t in tps:
            if t.label:
                topics[t.label] += 1
        seen += 1
        if seen >= 10:
            break
    host = hosts.most_common(1)[0][0] if hosts else None
    return host, [label for label, _ in topics.most_common(4)]


def _trend(
    root: Path, kind: str, entity_id: str
) -> tuple[str | None, tuple[float, ...] | None, float | None]:
    """(hot-stat, sparkline, multiplier) for an entity: the "↑ N×" score + weekly series, present
    only when genuinely rising (≥1.5×) and there's enough series to draw. Any may be None."""
    v, series = _trend_map(root, kind).get(entity_id, (0.0, ()))
    if not (v and v >= 1.5):
        return None, None, None
    spark = series if series and len(series) >= 4 else None
    return f"↑ {v:.1f}× rising", spark, round(v, 1)


# --------------------------------------------------------------------------- #
# Per-kind builders.
# --------------------------------------------------------------------------- #
def _topic(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
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
    hot, spark, mult = _trend(root, "topic", card.id)
    return OgCardModel(
        kicker="Topic",
        title=card.label,
        quote=quote,
        byline=byline,
        stats=stats,
        hot=hot,
        sparkline=spark,
        trend_multiplier=mult,
        accent=accent_for_kind("topic"),
    )


def _person(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.app_relational_view import build_person_card

    card = build_person_card(root, ident)
    if card is None:
        return None
    web = card.web
    # Lede: the external one-line descriptor / bio from the enricher; else what they talk about.
    blurb = (web.description or web.bio) if web else None
    topics = [t.label for t in (card.related_topics or []) if t.label]
    if not blurb and topics:
        blurb = f"On {', '.join(topics[:3])}."
    role = (card.role or "").lower()
    host_show = next((s for s in (card.shows or []) if (s.role or "").lower() == "host"), None)
    latest = max(
        (e for e in (card.episodes or []) if e.publish_date),
        key=lambda e: e.publish_date or "",
        default=None,
    )
    stats_parts = [_eps(card.episode_count)]
    byline: str | None = None
    tags: str | None = None

    if role == "host" and host_show:
        # Host: name the show + its key topics (from our KG of that show) in the footer.
        byline = f"Host of {host_show.title}"
        _h, show_topics = _show_signals(root, host_show.feed_id, cached_catalog(root))
        tags = " · ".join(show_topics[:3]) or None
    elif role == "guest":
        # Guest: role + how many shows / which show + when, and the topics. No single episode title
        # — naming one of several would be arbitrary (the shows are already the gallery).
        byline = "Guest"
        n_shows = len(card.shows or [])
        if n_shows > 1:
            stats_parts.append(f"{n_shows} shows")
        elif latest and latest.podcast_title:
            stats_parts.append(latest.podcast_title)
        last = _published(latest.publish_date) if latest else None
        if last:
            stats_parts.append(f"latest {last}")
        tags = " · ".join(topics[:3]) or None
    else:
        if role:
            byline = role.title()
        tags = " · ".join(topics[:3]) or None

    # Visual: the person's own photo when the enricher has one. Otherwise, for a GUEST, a row of the
    # shows they've appeared on — each tile the episode's own art, falling back to the show art.
    artwork: bytes | None = None
    gallery: tuple[bytes, ...] = ()
    credit: str | None = None
    if with_art:
        photo = _person_photo(root, card.id)
        if photo:
            artwork = photo
            # The photo is a licensed third-party image (person_web / CC-BY); this card is PUBLIC
            # and unauthenticated, so it must carry attribution.
            credit = _image_credit(
                "Photo", getattr(web, "image_artist", None), getattr(web, "image_license", None)
            )
        elif role == "guest":
            gallery = _guest_show_tiles(root, card)
        elif role == "host" and host_show:
            artwork = _feed_artwork(root, host_show.feed_id)  # show cover art — promotional

    return OgCardModel(
        kicker="Person",
        title=card.label,
        blurb=_clip(blurb, _MAX_BLURB),
        byline=byline,
        stats=" · ".join(stats_parts),
        tags=tags,
        accent=accent_for_kind("person"),
        artwork=artwork,
        gallery=gallery,
        credit=credit,
    )


def _image_credit(kind: str, artist: str | None, license_: str | None) -> str | None:
    """A compact attribution line for a licensed image, e.g. "Photo: A. Smith · CC BY-SA 4.0"."""
    parts = [p.strip() for p in (artist, license_) if p and p.strip()]
    return f"{kind}: {' · '.join(parts)}" if parts else None


def _feed_artwork(root: Path, feed_id: str) -> bytes | None:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.corpus_catalog import aggregate_feeds

    feeds = aggregate_feeds(cached_catalog(root))
    feed = next((f for f in feeds if f.get("feed_id") == feed_id), None)
    return _artwork_bytes(root, feed.get("image_local_relpath")) if feed else None


def _guest_show_tiles(root: Path, card: object) -> tuple[bytes, ...]:
    """One artwork tile per distinct show a guest appears on (latest episode first), each the
    episode's OWN art with a fall-back to the show art. Capped at four."""
    from typing import Any

    from podcast_scraper.server.app_slugs import resolve_slug

    by_feed: dict[str, Any] = {}
    for e in getattr(card, "episodes", None) or []:
        row = resolve_slug(root, e.slug)
        if row is None:
            continue
        fid = row.feed_id or ""
        cur = by_feed.get(fid)
        if cur is None or (row.publish_date or "") > (getattr(cur, "publish_date", "") or ""):
            by_feed[fid] = row
    tiles = []
    for row in list(by_feed.values())[:4]:
        art = _artwork_bytes(root, row.episode_image_local_relpath or row.feed_image_local_relpath)
        if art:
            tiles.append(art)
    return tuple(tiles)


def _person_photo(root: Path, person_id: str) -> bytes | None:
    from podcast_scraper.enrichment.enrichers.person_web import person_image_path

    found = person_image_path(root, person_id)
    return _asset_bytes(found[0]) if found else None


def _org(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
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
    logo = _org_logo(root, card.id) if with_art else None
    # The logo is a licensed image (org_web); this PUBLIC card must attribute it.
    credit = _image_credit("Logo", None, getattr(web, "logo_license", None)) if logo else None
    return OgCardModel(
        kicker="Organization",
        title=card.label,
        blurb=_clip(blurb, _MAX_BLURB),
        stats=stats,
        accent=accent_for_kind("organization"),
        artwork=logo,
        credit=credit,
    )


def _org_logo(root: Path, org_id: str) -> bytes | None:
    from podcast_scraper.enrichment.enrichers.org_web import org_logo_path

    found = org_logo_path(root, org_id)
    return _asset_bytes(found[0]) if found else None


def _storyline(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
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
    tcid = card.theme_cluster_id
    hot, spark, mult = _trend(root, "storyline", tcid) if tcid else (None, None, None)
    return OgCardModel(
        kicker="Storyline",
        title=card.theme_cluster_label or card.label,
        blurb=_clip(" · ".join(members), _MAX_BLURB),
        byline="Topics discussed together",
        stats=stats,
        hot=hot,
        sparkline=spark,
        trend_multiplier=mult,
        accent=accent_for_kind("storyline"),
    )


def _episode(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_slugs import resolve_slug

    row = resolve_slug(root, ident)
    if row is None:
        return None
    # Lede: the episode SUMMARY — a reliable "what this is about" (the surfaced GI insight is often
    # filler, so the summary sells the listen better + frees the byline to name the guest).
    blurb = None
    if row.summary_bullets:
        blurb = row.summary_bullets[0]
    elif row.summary_text:
        blurb = row.summary_text
    # Byline names both voices as one natural phrase: "Sam in conversation with Dr. Elena Fischer"
    # (host + guest), degrading to just the guest or just the host when only one is named.
    hosts, guests, topics = _episode_roster(root, row)
    if hosts and guests:
        byline: str | None = f"{_people(hosts)} in conversation with {_people(guests)}"
    elif guests:
        byline = f"With {_people(guests)}"
    elif hosts:
        byline = f"Hosted by {_people(hosts)}"
    else:
        byline = None
    # Insight count for the bottom meta line (loaded only for the count).
    n_insights = 0
    if row.gi_relative_path:
        gi = load_json_artifact(root, row.gi_relative_path)
        if gi:
            n_insights = len(insights_from_gi(gi))
    mins = round(row.duration_seconds / 60) if row.duration_seconds else None
    # Footer stats: duration · insights · when it published.
    meta = [f"{mins} min"] if mins else []
    if n_insights:
        meta.append(f"{n_insights} insight" if n_insights == 1 else f"{n_insights} insights")
    published = _published(row.publish_date)
    if published:
        meta.append(published)
    art = (
        _artwork_bytes(root, row.episode_image_local_relpath or row.feed_image_local_relpath)
        if with_art
        else None
    )
    return OgCardModel(
        kicker=f"Episode · {row.feed_title}" if row.feed_title else "Episode",
        title=row.episode_title or "Untitled episode",  # title is str-typed but corpora can miss it
        blurb=_clip(blurb, _MAX_BLURB),
        byline=byline,
        stats=" · ".join(meta) or None,
        tags=" · ".join(topics[:3]) or None,  # key topics, a secondary footer line
        accent=accent_for_kind("episode"),
        artwork=art,
        background=True,  # episode uses its art as the full-bleed backdrop (summary-length-proof)
    )


def _published(publish_date: str | None) -> str | None:
    """A compact 'Mon YYYY' for the footer, or None when the date isn't parseable."""
    if not publish_date or len(publish_date) < 7:
        return None
    try:
        return datetime.fromisoformat(publish_date[:10]).strftime("%b %Y")
    except ValueError:
        return None


def _show(root: Path, ident: str, with_art: bool = True) -> OgCardModel | None:
    from podcast_scraper.server.app_catalog_cache import cached_catalog
    from podcast_scraper.server.corpus_catalog import aggregate_feeds

    rows = cached_catalog(root)
    feed = next((f for f in aggregate_feeds(rows) if f.get("feed_id") == ident), None)
    if feed is None:
        return None
    # Footer stats: episode count · cadence · when the newest episode dropped (no duration).
    stats = _eps(int(feed.get("episode_count", 0)))
    cadence, _typical = _cadence_and_length(rows, ident)
    if cadence:
        stats += f" · {cadence}"
    last = _last_published(rows, ident)
    if last:
        stats += f" · latest {last}"
    host, about = _show_signals(root, ident, rows)
    return OgCardModel(
        kicker="Show",
        title=feed.get("display_title") or ident,
        # Lede = a SHORT version of the show's own description (under the title).
        blurb=_first_sentence(feed.get("description"), 150),
        byline=f"Hosted by {host}" if host else None,
        stats=stats,
        tags=" · ".join(about[:3]) or None,  # key topics moved to the footer (as with episode)
        accent=accent_for_kind("show"),
        artwork=_artwork_bytes(root, feed.get("image_local_relpath")) if with_art else None,
    )


def _first_sentence(text: str | None, cap: int) -> str | None:
    """A short lede: the first sentence, or a clip at ``cap`` chars — whichever is shorter."""
    if not text:
        return None
    t = " ".join(text.split())
    for sep in (". ", "! ", "? "):
        i = t.find(sep)
        if 0 < i < cap:
            return t[: i + 1]
    return t if len(t) <= cap else t[: cap - 1].rstrip() + "…"


def _last_published(rows: list, feed_id: str) -> str | None:
    """'Mon YYYY' of the feed's most recent episode (ISO dates sort lexicographically)."""
    dates = [
        r.publish_date
        for r in rows
        if (getattr(r, "feed_id", None) or "") == feed_id and getattr(r, "publish_date", None)
    ]
    return _published(max(dates)) if dates else None


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
