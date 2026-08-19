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

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map
from podcast_scraper.search.topic_clusters import consumer_topic_cluster_map
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    RankingConfig,
    SIGNAL_DISCOVER_POOL,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_RECENCY,
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


def _affinity_boost(
    matched_explicit: int,
    matched_derived: int,
    *,
    weight: float,
    derived_ratio: float,
    cap: float,
) -> float:
    """How much a matching episode is lifted — SATURATING, and never divided by how much you follow.

    It used to be ``weight * (matched / len(interests))``, and that denominator punished engagement:
    with two follows a single match was worth x2.0, with twenty it was worth x1.1. Personalisation
    faded precisely for the users who had told the product the most about themselves. Following one
    more show is not a statement that everything else matters less.

    Now each MATCH contributes, and the sum saturates at ``cap`` so a broad-interest episode cannot
    run away with the feed. Matching two of your interests is worth more than matching one; matching
    six is not worth six times as much.

    ``derived_ratio`` is what an INFERRED interest is worth against a stated one. Derived tokens
    used to share the explicit denominator, so turning them on actively weakened explicit follows;
    counted separately at a fraction of the weight, they can only ever add.
    """
    if weight <= 0:
        return 0.0
    contribution = matched_explicit + derived_ratio * matched_derived
    if contribution <= 0:
        return 0.0
    # 1 - 0.5**n : one match gives half the cap, two gives three quarters, and it never exceeds it.
    saturated = 1.0 - 0.5**contribution
    # float(): weight/cap come out of the ranking config as Any, so returning the product
    # unconverted made the declared -> float unenforced.
    return float(weight * min(saturated, cap))


def _feed_significance_means(
    rows: Sequence[CatalogEpisodeRow], params: dict[str, Any] | None
) -> dict[str, float]:
    """Mean raw significance per feed — the denominator that stops COVERAGE outranking INTEREST.

    ``_significance`` scores ``has_gi`` / ``has_kg`` / bullet count: whether ENRICHMENT RAN, not how
    good the episode is. On a uniformly-enriched corpus that is harmless, and ours is uniformly
    enriched — which is exactly why the fixture cannot reveal the problem. The moment coverage is
    uneven, a well-processed show outranks every episode of a sparsely-processed one in EVERY user's
    feed, regardless of what they actually follow. "More enriched" would beat "this is what you
    asked for".

    Normalising each episode against its own feed's mean removes that: a show is compared with its
    own siblings, so depth still orders episodes WITHIN a show while a pipeline artefact cannot
    reorder shows against each other.
    """
    totals: dict[str, list[float]] = {}
    for row in rows:
        totals.setdefault(row.feed_id or "", []).append(_significance(row, params))
    return {feed: (sum(vals) / len(vals)) for feed, vals in totals.items() if vals}


def _topic_velocities(root: Path) -> dict[str, float]:
    """``topic_id`` → trend velocity for the discover trend boost.

    RFC-103: prefer read-time content momentum (today-relative) from the enricher's
    ``content_series``; fall back to the pre-baked ``velocity_last_over_6mo`` when a corpus has no
    ``content_series`` yet. Empty when neither is available, so a missing enrichment just leaves the
    trend signal contributing nothing rather than erroring the ranking.
    """
    from podcast_scraper.server.app_momentum import content_topic_velocities

    momentum_vel = content_topic_velocities(root)
    if momentum_vel is not None:
        return momentum_vel
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
    root: Path,
    row: CatalogEpisodeRow,
    cluster_map: dict[str, dict[str, object]],
    theme_map: dict[str, dict[str, object]],
) -> tuple[set[str], set[str], set[str]]:
    """Interest-matchable ids this episode touches: (cluster ids, topic ids, person ids).

    One KG load per episode. An interest token matches whichever set its prefix belongs to —
    ``tc:`` (semantic cluster) / ``thc:`` (theme cluster / "storyline") → cluster, ``topic:`` →
    topic, ``person:`` → person — so a follow on any of those (semantic clusters + storylines from
    the picker; topics/people from entity cards) re-ranks discovery. Both cluster kinds share the
    ``clusters`` set: their ids are prefix-disjoint, so a followed ``thc:`` only matches its own.
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
        tinfo = theme_map.get(topic.id)
        tcid = tinfo.get("theme_cluster_id") if tinfo else None
        if isinstance(tcid, str):
            clusters.add(tcid)
    return clusters, topic_ids, {p.id for p in persons}


#: Sidecar doc types that carry an interest token (`source_id`) for an episode.
_KG_DOC_TYPES = ("kg_topic", "kg_entity")

#: How many candidates enter ranking, as a multiple of the requested page size.
DISCOVER_POOL_MULTIPLE = 4

#: Floor on the recency leg, as a SHARE of the corpus, independent of page size.
#:
#: `DISCOVER_POOL_MULTIPLE` alone makes the window a fixed 48 at the default page size, whatever
#: the corpus is. Measured on production 2026-08-19 (678 episodes, 14 feeds): the recency leg
#: reached 48/678 = **7.1%**, so 630 episodes could not reach the ranker at all unless they
#: happened to match a followed interest. That share shrinks as the corpus grows — at 1,500
#: episodes the same 48 is 3% — so discovery gets narrower precisely as there is more to discover.
#:
#: The fixture could never show this: 4 * 12 = 48 EXCEEDS its 36 episodes, so the pool was the
#: whole corpus and the window was never a constraint in any test we own.
#:
#: 15% of 678 is ~102 candidates. The cost is one KG artifact load per candidate in
#: `_episode_features`, so this roughly doubles the ranking walk rather than multiplying it, and
#: `DISCOVER_POOL_MAX` bounds the absolute worst case for a corpus that keeps growing.
DISCOVER_POOL_CORPUS_SHARE = 0.15

#: Hard ceiling on either leg. Ranking is O(candidates) in KG loads, so an unbounded share would
#: turn a large corpus into a slow endpoint. 400 is ~8x the old fixed window and still bounded.
DISCOVER_POOL_MAX = 400

#: Below this page size the corpus share does not apply — see `_pool_window`. A request for one or
#: two episodes is a probe or a widget, not a discovery feed, and loading a corpus-proportional
#: number of KG artifacts to answer it would be pure waste.
DISCOVER_POOL_MIN_LIMIT_FOR_SHARE = 5

#: Cache for :func:`interest_episode_index`, keyed by the sidecar's (path, mtime, size).
_INTEREST_INDEX_CACHE: dict[str, tuple[tuple[float, int], dict[str, set[str]]]] = {}


def interest_episode_index(root: Path) -> dict[str, set[str]]:
    """``interest token -> {metadata_relative_path}`` for every episode carrying it.

    Read from ``search/metadata.json``, the sidecar the two-tier indexer already writes: its
    ``kg_topic`` / ``kg_entity`` rows carry ``source_id`` (``topic:…`` / ``person:…`` — the SAME id
    space interests live in) alongside ``source_metadata_relative_path``. So the mapping the pool
    needs is one JSON parse, not a KG load per episode.

    Cached on the file's (mtime, size): the corpus is rewritten out-of-band by the pipeline rather
    than mutated in place, and everything else in this path re-reads per request, so a stale index
    would be a real bug rather than a stale cache. Returns ``{}`` when the sidecar is missing —
    a corpus without a search index simply falls back to the recency window.

    Only ``topic:`` and ``person:`` tokens appear. Cluster tokens (``tc:`` / ``thc:``) are absent
    by design: they are corpus-wide umbrellas (#1669), so they match essentially everything and
    would make the interest leg of the pool meaningless.
    """
    path = root / "search" / "metadata.json"
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = str(path)
    stamp = (stat.st_mtime, stat.st_size)
    cached = _INTEREST_INDEX_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    index: dict[str, set[str]] = {}
    if isinstance(raw, dict):
        for entry in raw.values():
            if not isinstance(entry, dict) or entry.get("doc_type") not in _KG_DOC_TYPES:
                continue
            token = entry.get("source_id")
            relpath = entry.get("source_metadata_relative_path")
            if isinstance(token, str) and isinstance(relpath, str):
                index.setdefault(token, set()).add(relpath)
    _INTEREST_INDEX_CACHE[key] = (stamp, index)
    return index


def _pos_int(value: object, fallback: int) -> int:
    """A positive int from an override, else *fallback*. A bad value must not empty the pool."""
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return fallback
    try:
        out = int(value)
    except (TypeError, ValueError):
        return fallback
    return out if out > 0 else fallback


def _pos_float(value: object, fallback: float) -> float:
    """A positive finite float from an override, else *fallback*."""
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return fallback
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if out > 0 and math.isfinite(out) else fallback


def _pool_window(limit: int, corpus_size: int, config: RankingConfig | None = None) -> int:
    """How many episodes each leg of the pool may hold.

    The larger of a page-size multiple and a share of the corpus, capped. The page-size term keeps
    a small corpus behaving exactly as before; the corpus-share term is what stops the window
    becoming a keyhole as the corpus grows (see `DISCOVER_POOL_CORPUS_SHARE`).

    Reads `discover_pool` from *config* when given, so admission is tunable exactly like every
    scoring weight — the module constants are the fallback, not the source of truth. That matters
    for #1795: a sweep cannot search over a parameter it has no way to set.
    """
    params = config.params_of(SIGNAL_DISCOVER_POOL) if config is not None else {}
    multiple = _pos_int(params.get("page_multiple"), DISCOVER_POOL_MULTIPLE)
    share = _pos_float(params.get("corpus_share"), DISCOVER_POOL_CORPUS_SHARE)
    ceiling = _pos_int(params.get("max_candidates"), DISCOVER_POOL_MAX)
    min_limit = _pos_int(params.get("min_limit_for_share"), DISCOVER_POOL_MIN_LIMIT_FOR_SHARE)

    by_page = max(limit * multiple, limit)
    if limit < min_limit:
        # A deliberately tiny page is a probe, not a feed. `TestPoolPolicyIsExplicit` uses
        # limit=1 and limit=2 precisely to force the bound the 36-episode fixture otherwise
        # hides, and widening those calls would destroy the only tests that demonstrate
        # truncation at all. The corpus share exists to stop a REAL page size becoming a
        # keyhole on a large corpus; it has no business rewriting a probe.
        return by_page
    by_corpus = int(corpus_size * share)
    return min(max(by_page, by_corpus), ceiling)


def build_discover_pool(
    rows: Sequence[CatalogEpisodeRow],
    *,
    limit: int,
    interests: Iterable[str] = (),
    root: Path | None = None,
    config: RankingConfig | None = None,
) -> Sequence[CatalogEpisodeRow]:
    """The candidate set ``rank_discover`` scores: the newest ``4 * limit`` episodes, UNION the
    newest ``4 * limit`` that match an interest.

    Ranking is not free — ``_episode_features`` loads one KG artifact per candidate — so the pool
    must be bounded. What that bound *excludes* is a product decision, not an implementation
    detail, and a pure recency window got it wrong: recency is a proxy for relevance that fails
    exactly where personalisation matters most. On a large corpus a user who follows scuba, whose
    best episodes are four years old on a feed that stopped publishing, would match none of the
    newest 32 — so discovery would silently return no personalisation at all while the telemetry
    still recorded the feed as ``personalized``.

    So the pool is a union of two bounded legs:

    * **recency** — the newest ``4 * limit``, so a brand-new or interest-less user still gets a
      sensible feed and fresh episodes are never starved out;
    * **relevance** — the newest ``4 * limit`` episodes carrying a followed ``topic:`` / ``person:``
      token, found through :func:`interest_episode_index` (one cached JSON read, no KG loads), so
      an old-but-matching episode can reach the ranker at all.

    Ranking still decides the final order; this only decides what is allowed to compete. Falls back
    to the recency leg alone when there are no interests or no search sidecar, which is exactly the
    previous behaviour.

    Both the route and the offline eval must call this. They diverged once — the eval ranked the
    FULL catalog while production ranked a slice — so the eval scored a system that never ran.
    """
    window_size = _pool_window(limit, len(rows), config)
    window = list(rows[:window_size])
    tokens = {str(t) for t in interests if str(t)}
    if not tokens or root is None:
        return window

    index = interest_episode_index(root)
    if not index:
        return window
    matching: set[str] = set()
    for token in tokens:
        matching |= index.get(token, set())
    if not matching:
        return window

    in_window = {r.metadata_relative_path for r in window}
    extra = [
        r
        for r in rows[window_size:]
        if r.metadata_relative_path in matching and r.metadata_relative_path not in in_window
    ][:window_size]
    return [*window, *extra]


def _recency_boost(publish_date: str | None, newest: date | None, half_life_days: float) -> float:
    """How fresh this episode is RELATIVE TO THE FRESHEST in the pool — 1.0 down towards 0.0.

    Decay is measured against the newest candidate rather than wall-clock ``now``, deliberately:

    * it is deterministic, so the eval and the tests do not drift as the calendar moves;
    * it says the useful thing. Against wall-clock, a corpus that stopped updating a year ago has
      every episode decayed to ~0 and the signal silently stops discriminating — exactly when a
      user most needs "what is newest here". Relative decay keeps ranking the shelf you have.

    ``half_life_days`` is the age at which the boost halves: 0 days -> 1.0, one half-life -> 0.5.
    """
    if newest is None or not publish_date or half_life_days <= 0:
        return 0.0
    try:
        published = date.fromisoformat(str(publish_date)[:10])
    except ValueError:
        return 0.0
    age_days = max(0, (newest - published).days)
    return float(2.0 ** (-age_days / half_life_days))


def _newest_publish_date(rows: Sequence[CatalogEpisodeRow]) -> date | None:
    """The most recent parseable publish date in the pool — the origin recency decays from."""
    best: date | None = None
    for row in rows:
        if not row.publish_date:
            continue
        try:
            d = date.fromisoformat(str(row.publish_date)[:10])
        except ValueError:
            continue
        if best is None or d > best:
            best = d
    return best


def rank_discover(
    root: Path,
    interests: Iterable[str],
    rows: Sequence[CatalogEpisodeRow],
    *,
    limit: int,
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
    derived_interests: Iterable[str] = (),
) -> list[AppEpisodeSummary]:
    """Rank ``rows`` by the enabled ranking signals; recency when interests are empty.

    ``rows`` is the candidate pool, already in recency order (newest-first). With no interests
    we simply take the first ``limit`` (recency). With interests we re-score the pool and keep
    the original order as a stable tie-break (so equal-score episodes stay newest-first).

    Signals come from ``config`` (the operator-tunable registry, one source of truth): a base
    ``significance`` depth score, multiplied by ``1 + Σ weightᵢ · signalᵢ`` over the enabled
    boosts. ``interest_affinity`` is the fraction of followed tokens the episode matches (semantic
    cluster ``tc:`` / theme cluster ``thc:`` / ``topic:`` / ``person:``); ``trend_velocity`` (off
    by default) adds the episode's hottest topic momentum. A disabled signal has weight 0 → no
    effect, so the
    default config reproduces the prior significance × affinity behaviour exactly.
    """
    explicit_set = {str(i) for i in interests if str(i)}
    # Derived tokens are kept SEPARATE, not merged into one set (#19). Pooled, they shared the
    # `matched / len(interests)` denominator, so switching APP_DERIVED_INTERESTS on dropped a
    # 2-follow user's per-match affinity from 0.5 to 0.2 — enabling implicit personalisation
    # WEAKENED the follows the user had explicitly chosen. An explicit follow is a stated
    # preference; a derived token is an inference, and the two do not deserve the same vote.
    derived_set = {str(i) for i in derived_interests if str(i)} - explicit_set
    interest_set = explicit_set | derived_set
    if not interest_set:
        return [row_to_summary(root, r) for r in rows[:limit]]

    # Only `tc:` / `thc:` / `topic:` / `person:` tokens are honored; any other prefix lands in
    # `cluster_interests`, never matches an episode, and just dilutes the affinity denominator.
    def _split(tokens: set[str]) -> tuple[set[str], set[str], set[str]]:
        persons = {t for t in tokens if t.startswith("person:")}
        topics = {t for t in tokens if t.startswith("topic:")}
        return persons, topics, tokens - persons - topics

    explicit_persons, explicit_topics, explicit_clusters = _split(explicit_set)
    derived_persons, derived_topics, derived_clusters = _split(derived_set)
    cluster_map = consumer_topic_cluster_map(root)
    theme_map = consumer_theme_cluster_map(root)
    sig_params = config.params_of(SIGNAL_SIGNIFICANCE)
    affinity_weight = config.weight_of(SIGNAL_INTEREST_AFFINITY)
    affinity_params = config.params_of(SIGNAL_INTEREST_AFFINITY)
    derived_ratio = float(affinity_params.get("derived_ratio", 0.5))
    affinity_cap = float(affinity_params.get("cap", 1.0))
    trend_weight = config.weight_of(SIGNAL_TREND_VELOCITY)
    trend_cap = float(config.params_of(SIGNAL_TREND_VELOCITY).get("cap", 1.5))
    velocities = _topic_velocities(root) if trend_weight > 0 else {}
    feed_means = _feed_significance_means(rows, sig_params)
    # Recency as a GRADED signal, not just the tie-break below. Before this, any non-empty interest
    # set sorted the whole pool by score and recency survived only as `-idx` — so following one
    # topic reshuffled even the 90% of the feed unrelated to it, newest-first becoming
    # enrichment-depth-first. A decaying boost lets a follow re-rank without the feed losing its
    # sense of time, and smooths the hard pool-boundary cliff from #17 at the same time.
    recency_weight = config.weight_of(SIGNAL_RECENCY)
    recency_half_life = float(config.params_of(SIGNAL_RECENCY).get("half_life_days", 30.0))
    newest = _newest_publish_date(rows) if recency_weight > 0 else None
    scored: list[tuple[float, int, CatalogEpisodeRow]] = []
    for idx, row in enumerate(rows):
        clusters, topics, persons = _episode_features(root, row, cluster_map, theme_map)
        matched_explicit = (
            len(clusters & explicit_clusters)
            + len(topics & explicit_topics)
            + len(persons & explicit_persons)
        )
        matched_derived = (
            len(clusters & derived_clusters)
            + len(topics & derived_topics)
            + len(persons & derived_persons)
        )
        multiplier = 1.0 + _affinity_boost(
            matched_explicit,
            matched_derived,
            weight=affinity_weight,
            derived_ratio=derived_ratio,
            cap=affinity_cap,
        )
        if trend_weight > 0:
            multiplier += trend_weight * _trend_boost(topics, velocities, trend_cap)
        if recency_weight > 0:
            multiplier += recency_weight * _recency_boost(
                row.publish_date, newest, recency_half_life
            )
        # Normalised against the episode's OWN FEED, so enrichment coverage cannot outrank
        # interest across shows (see _feed_significance_means). Falls back to the raw score when a
        # feed somehow has no mean, which keeps behaviour defined rather than dividing by zero.
        feed_mean = feed_means.get(row.feed_id or "", 0.0)
        base = _significance(row, sig_params)
        score = (base / feed_mean if feed_mean > 0 else base) * multiplier
        scored.append((score, -idx, row))  # -idx → earlier (newer) wins score ties
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [row_to_summary(root, r) for _score, _neg_idx, r in scored[:limit]]
