"""Assemble the post-episode recap model for ONE episode (RFC-122 #2038/#2039).

The single source both recap surfaces render: the in-app end-card (`GET /episodes/{slug}/recap`)
and the daily digest email (`app_digest_daily_recap`). Extracted here so the two cannot drift and
so the email job can build a recap WITHOUT importing the HTTP route module.

Pure read over one episode's own artifacts — summary key points, top salience-ranked insights, the
single strongest attributed quote, key topics, and the storylines it belongs to. Best-effort per
field: a thin corpus drops a field, never fails.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.search.theme_clusters import (
    consumer_theme_cluster_map,
    top_theme_clusters_by_member_count,
)
from podcast_scraper.server.app_artwork import artwork_url
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import (
    AppEpisodeRecap,
    AppInsight,
    AppQuote,
    AppStorylineRef,
    AppTopic,
)


def signature_quote(insights: list[AppInsight]) -> AppQuote | None:
    """The strongest attributed quote for the recap's anchor (RFC-122).

    Walk the salience-ranked insights (highest first) and take the first supporting quote that names
    a speaker — an attributed line is the memorable anchor. Fall back to the first quote of any kind
    when none is attributed, and to nothing when there are no quotes. Same selection intent as the
    #2036 share card's signature quote.
    """
    fallback: AppQuote | None = None
    for ins in insights:
        for q in ins.quotes:
            if fallback is None:
                fallback = q
            if q.speaker:
                return q
    return fallback


def episode_storylines(
    root: Path, topics: list[AppTopic], *, limit: int = 3
) -> list[AppStorylineRef]:
    """The distinct storylines (theme clusters) the episode's topics belong to (RFC-122).

    Each is a navigable reference addressed by its anchor topic id — the param the client storyline
    route takes. Only clusters that clear the corpus surfacing floor (and therefore have an anchor)
    are included, so every chip opens a real storyline; empty when the corpus has no theme clusters.
    """
    if not topics:
        return []
    theme_map = consumer_theme_cluster_map(root)  # topic_id -> {theme_cluster_id, ...}
    if not theme_map:
        return []
    # thc id -> {id, label, size, anchor_topic_id}; the floor + anchor are enforced here.
    summaries = {c["id"]: c for c in top_theme_clusters_by_member_count(root, top_n=1000)}
    out: list[AppStorylineRef] = []
    seen: set[str] = set()
    for topic in topics:
        info = theme_map.get(topic.id)
        thc = info.get("theme_cluster_id") if info else None
        summary = summaries.get(thc) if thc else None
        if not summary:
            continue
        anchor = str(summary["anchor_topic_id"])
        if anchor in seen:
            continue
        seen.add(anchor)
        out.append(AppStorylineRef(id=anchor, label=str(summary["label"])))
        if len(out) >= limit:
            break
    return out


def build_episode_recap(
    root: Path, row: CatalogEpisodeRow, slug: str, *, limit: int = 3
) -> AppEpisodeRecap:
    """Assemble the recap model for one catalog row (RFC-122).

    ``limit`` caps the top insights by salience. Degrades gracefully: no GI yields empty insights +
    a null quote; no KG yields no topics/storylines. Never raises on a thin corpus.
    """
    insights: list[AppInsight] = []
    if row.has_gi:
        insights = insights_from_gi(load_json_artifact(root, row.gi_relative_path), limit=limit)
    all_topics: list[AppTopic] = []
    if row.has_kg:
        _p, _o, all_topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
    storylines = episode_storylines(root, all_topics)
    local_art = row.episode_image_local_relpath or row.feed_image_local_relpath
    return AppEpisodeRecap(
        slug=slug,
        title=row.episode_title,
        podcast_title=row.feed_title,
        artwork_url=artwork_url(local_art, "large"),
        key_points=list(row.summary_bullets),
        summary_text=row.summary_text,
        insights=insights,
        signature_quote=signature_quote(insights),
        topics=all_topics[:6],
        storylines=storylines,
        has_gi=row.has_gi,
    )
