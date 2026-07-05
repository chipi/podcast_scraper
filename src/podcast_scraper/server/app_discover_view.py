"""Consumer personalized-discovery ranking (PRD-043 FR4 / 3.5) — flag-gated, recency fallback.

When personalization is OFF (default) or the user has no interests, the order is **recency**
(newest-first — the catalog default, unchanged). When personalization is ON *and* the user has
interests, episodes rank by the enabled signals in the tunable **ranking-signal registry**
(``app_ranking_config`` — significance, interest affinity, trend velocity, …), gated behind
``APP_PERSONALIZED_RANKING``. Signals are on/off + weight-tunable so ranking can be A/B'd from
one place; the default config reproduces the prior significance × affinity behaviour.

No new persistence: interests are per-user files; this only re-orders the shared catalog. The
ranking reuses the same KG view as the entity endpoints; the candidate pool is bounded by the
caller so the per-episode KG loads stay cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    RankingConfig,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_SIGNIFICANCE,
    SIGNAL_TREND_VELOCITY,
)
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import AppEpisodeSummary

# The temporal_velocity enricher envelope (corpus scope) — topic momentum for the trend signal.
_VELOCITY_REL = "enrichments/temporal_velocity.json"


def _significance(row: CatalogEpisodeRow, params: dict[str, Any] | None = None) -> float:
    """Content-depth signal: grounded insights > KG > summary richness. Weights from config."""
    params = params or {}
    score = 1.0
    if row.has_gi:
        score += float(params.get("gi_bonus", 2.0))
    if row.has_kg:
        score += float(params.get("kg_bonus", 1.0))
    step = float(params.get("bullet_step", 0.2))
    cap = int(params.get("bullet_cap", 5))
    score += min(len(row.summary_bullets), cap) * step
    return score


def _topic_velocities(root: Path) -> dict[str, float]:
    """``topic_id`` → ``velocity_last_over_6mo`` from the temporal_velocity envelope.

    Empty when the envelope is absent or malformed, so a missing enrichment just leaves the
    trend signal contributing nothing rather than erroring the ranking.
    """
    env = load_json_artifact(root, _VELOCITY_REL)
    data = env.get("data", env) if isinstance(env, dict) else None
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        return {}
    out: dict[str, float] = {}
    for t in topics:
        if not isinstance(t, dict):
            continue
        tid = t.get("topic_id")
        vel = t.get("velocity_last_over_6mo")
        if isinstance(tid, str) and isinstance(vel, (int, float)):
            out[tid] = float(vel)
    return out


def _trend_boost(topic_ids: set[str], velocities: dict[str, float], cap: float) -> float:
    """0 for a flat/cooling episode, up to ``cap - 1`` for a hot one.

    Uses the episode's hottest topic velocity above the 1.0 flat line, capped so a single
    spiking topic can't dominate the whole feed.
    """
    if not topic_ids or not velocities:
        return 0.0
    best = max((velocities.get(t, 1.0) for t in topic_ids), default=1.0)
    return max(0.0, min(best, cap) - 1.0)


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
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
) -> list[AppEpisodeSummary]:
    """Rank ``rows`` by the enabled ranking signals; recency when interests are empty.

    ``rows`` is the candidate pool, already in recency order (newest-first). With no interests
    we simply take the first ``limit`` (recency). With interests we re-score the pool and keep
    the original order as a stable tie-break (so equal-score episodes stay newest-first).

    Signals come from ``config`` (the operator-tunable registry, one source of truth): a base
    ``significance`` depth score, multiplied by ``1 + Σ weightᵢ · signalᵢ`` over the enabled
    boosts. ``interest_affinity`` is the fraction of followed tokens the episode matches
    (topic-cluster ``tc:`` / ``topic:`` / ``person:``); ``trend_velocity`` (off by default) adds
    the episode's hottest topic momentum. A disabled signal has weight 0 → no effect, so the
    default config reproduces the prior significance × affinity behaviour exactly.
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
    sig_params = config.params_of(SIGNAL_SIGNIFICANCE)
    affinity_weight = config.weight_of(SIGNAL_INTEREST_AFFINITY)
    trend_weight = config.weight_of(SIGNAL_TREND_VELOCITY)
    trend_cap = float(config.params_of(SIGNAL_TREND_VELOCITY).get("cap", 1.5))
    velocities = _topic_velocities(root) if trend_weight > 0 else {}
    scored: list[tuple[float, int, CatalogEpisodeRow]] = []
    for idx, row in enumerate(rows):
        clusters, topics, persons = _episode_features(root, row, cluster_map)
        matched = (
            len(clusters & cluster_interests)
            + len(topics & topic_interests)
            + len(persons & person_interests)
        )
        multiplier = 1.0 + affinity_weight * (matched / len(interest_set))
        if trend_weight > 0:
            multiplier += trend_weight * _trend_boost(topics, velocities, trend_cap)
        score = _significance(row, sig_params) * multiplier
        scored.append((score, -idx, row))  # -idx → earlier (newer) wins score ties
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [row_to_summary(root, r) for _score, _neg_idx, r in scored[:limit]]
