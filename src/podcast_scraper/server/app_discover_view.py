"""Consumer personalized-discovery ranking (PRD-043 FR4 / 3.5) — flag-gated, recency fallback.

When personalization is OFF (default) or the user has no interests, the order is **recency**
(newest-first — the catalog default, unchanged). When personalization is ON *and* the user has
interests, episodes rank by a provisional **significance × interest-cluster-affinity** score —
gated behind ``APP_PERSONALIZED_RANKING`` until the weights are tuned on real engagement.

No new persistence: interests are per-user files; this only re-orders the shared catalog. The
ranking reuses the same KG view as the entity endpoints; the candidate pool is bounded by the
caller so the per-episode KG loads stay cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import AppEpisodeSummary

# Interest affinity is weighted relative to the depth baseline; a fully on-interest episode gets a
# ``1 + AFFINITY_WEIGHT`` multiplier. Provisional — the whole path is flag-gated until tuned.
_AFFINITY_WEIGHT = 2.0


def _significance(row: CatalogEpisodeRow) -> float:
    """Provisional content-depth signal: grounded insights > KG > summary richness."""
    score = 1.0
    if row.has_gi:
        score += 2.0
    if row.has_kg:
        score += 1.0
    score += min(len(row.summary_bullets), 5) * 0.2
    return score


def _episode_features(
    root: Path, row: CatalogEpisodeRow, cluster_map: dict[str, dict[str, object]]
) -> tuple[set[str], set[str], set[str]]:
    """Interest-matchable ids this episode touches: (topic-cluster ids, topic ids, person ids).

    One KG load per episode. An interest token matches whichever set its prefix belongs to —
    ``tc:`` → cluster, ``topic:`` → topic, ``person:`` → person — so a follow on any of those
    (clusters from the picker; topics/people from entity cards) re-ranks discovery.
    """
    if not row.has_kg:
        return set(), set(), set()
    artifact = load_json_artifact(root, row.kg_relative_path)
    if artifact is None:
        return set(), set(), set()
    persons, _orgs, topics = entities_from_kg(artifact)
    clusters: set[str] = set()
    topic_ids: set[str] = set()
    for topic in topics:
        topic_ids.add(topic.id)
        info = cluster_map.get(topic.id)
        cid = info.get("cluster_id") if info else None
        if isinstance(cid, str):
            clusters.add(cid)
    return clusters, topic_ids, {p.id for p in persons}


def rank_discover(
    root: Path,
    interests: Iterable[str],
    rows: Sequence[CatalogEpisodeRow],
    *,
    limit: int,
) -> list[AppEpisodeSummary]:
    """Rank ``rows`` by significance × interest affinity; recency when interests are empty.

    ``rows`` is the candidate pool, already in recency order (newest-first). With no interests
    we simply take the first ``limit`` (recency). With interests we re-score the pool and keep
    the original order as a stable tie-break (so equal-score episodes stay newest-first).

    Interests are a mixed token list — topic-cluster (``tc:``) from the picker, plus topics
    (``topic:``) and people (``person:``) followed from entity cards. Affinity is the fraction of
    all followed tokens the episode matches (clusters, topics or people).
    """
    interest_set = {str(i) for i in interests if str(i)}
    if not interest_set:
        return [row_to_summary(root, r) for r in rows[:limit]]
    # Only `tc:` / `topic:` / `person:` tokens are honored; any other prefix lands in
    # `cluster_interests`, never matches an episode, and just dilutes the affinity denominator.
    person_interests = {t for t in interest_set if t.startswith("person:")}
    topic_interests = {t for t in interest_set if t.startswith("topic:")}
    cluster_interests = interest_set - person_interests - topic_interests
    cluster_map = consumer_topic_cluster_map(root)
    scored: list[tuple[float, int, CatalogEpisodeRow]] = []
    for idx, row in enumerate(rows):
        clusters, topics, persons = _episode_features(root, row, cluster_map)
        matched = (
            len(clusters & cluster_interests)
            + len(topics & topic_interests)
            + len(persons & person_interests)
        )
        affinity = matched / len(interest_set)
        score = _significance(row) * (1.0 + _AFFINITY_WEIGHT * affinity)
        scored.append((score, -idx, row))  # -idx → earlier (newer) wins score ties
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [row_to_summary(root, r) for _score, _neg_idx, r in scored[:limit]]
