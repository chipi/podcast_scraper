"""``topic_theme_clusters`` — corpus-wide THEME clusters (deterministic).

Groups Topics that are *discussed together* (co-occurrence), as opposed to
``topic_clusters.json`` which groups Topics that *mean the same thing*
(embedding cosine similarity). A **theme** is a set of topics an editor keeps
returning to in the same conversations — e.g. ``{shadow fleet, oil prices,
sanctions}`` — even when they are not semantically alike. Semantic clusters
answer "what is like this topic"; theme clusters answer "what storyline does
this topic belong to". They are complementary, shipped side by side, and
themed apart in the UI (this enricher uses the ``thc:`` graph compound-node
prefix vs the semantic ``tc:``).

Method (deterministic, no models):
  1. Per episode, read the KG Topic nodes → topic sets + per-topic episode ids.
  2. Aggregate pairwise co-occurrence counts + per-topic document frequency.
  3. Weight each co-occurring pair by lift = P(a,b)/(P(a)·P(b)) = cnt·N/(df_a·df_b)
     — "how much more than chance". Only pairs co-occurring in ``>= min_pair``
     episodes become edges (drops singleton flukes whose lift is spuriously
     huge on tiny corpora).
  4. Greedy average-linkage on the lift graph: repeatedly merge the two clusters
     with the highest mean inter-cluster lift while that mean stays
     ``>= merge_threshold`` (same shape as the semantic clusterer, so behaviour
     is familiar and chaining is avoided).

Output mirrors ``topic_clusters.json`` (``clusters[].members[]`` etc.) so the
same consumer / graph code can read it, but tags each cluster
``cluster_type="theme"`` and uses the ``thc:`` compound-node prefix.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import load_kg, node_label, nodes_of_type
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)

_DEFAULT_MIN_PAIR_EPISODES = 2
_DEFAULT_MERGE_THRESHOLD = 2.0  # mean lift required to merge (>= 2× chance)

# graph-v3 tier 7 — super-theme rollup.
#
# Once theme clusters are computed we run a *second* average-linkage pass on
# the clusters themselves, weighted by mean cross-cluster topic-lift (i.e. how
# strongly the topics of cluster A co-occur with the topics of cluster B in
# real episodes). The rollup gives the viewer a legend that scales — 100 flat
# clusters is unbrowsable, ~6 super-themes is.
#
# Bounded to [5, 8] regardless of corpus size: below 5 the rollup adds no
# structure, above 8 the legend stops working as a browse surface (7±2 rule).
# When the cluster count is already ≤ SUPER_THEME_MIN, each cluster is its own
# super-theme (rollup no-op).
#
# Signal choice — shared-bridge / cross-cluster co-occurrence (operator picked
# 2026-07-17). Alternatives to explore + compare later live in
# ``docs/wip/super-theme-signal-comparison.md``.
_SUPER_THEME_TARGET = 6  # target count when N > _SUPER_THEME_MIN
_SUPER_THEME_MIN = 5  # no rollup when cluster_count ≤ this
_SUPER_THEME_MAX = 8  # never emit more super-themes than this


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "cluster"


# Above this many topics, skip average-linkage clustering (see the guard below).
_MAX_LINKAGE_TOPICS = 400


def _average_linkage_to_target(
    n: int,
    weight: "Any",
    target: int,
) -> list[set[int]]:
    """Greedy average-linkage that merges the top-scoring pair each round until
    the cluster count reaches ``target`` (or fewer). Distinct from
    :func:`_average_linkage`: no threshold gate — the loop is driven purely by
    the target N, so it always converges regardless of edge sparsity.

    Ties broken by the natural iteration order (deterministic for a fixed n).
    """
    clusters: list[set[int]] = [{i} for i in range(n)]
    if target <= 0 or n <= target:
        return clusters

    def mean_inter(ci: set[int], cj: set[int]) -> float:
        tot = 0.0
        for a in ci:
            for b in cj:
                tot += weight(a, b)
        return tot / (len(ci) * len(cj))

    while len(clusters) > target:
        best = -1.0
        bi, bj = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = mean_inter(clusters[i], clusters[j])
                if s > best:
                    best = s
                    bi, bj = i, j
        if bi < 0:
            break
        clusters[bi] |= clusters[bj]
        clusters.pop(bj)
    return clusters


def _average_linkage(
    n: int,
    weight: "Any",
    threshold: float,
) -> list[set[int]]:
    """Greedy average-linkage merge on an (implicit) weighted graph.

    ``weight(i, j)`` returns the edge weight between members i and j (0 when no
    edge). Repeatedly merges the two clusters with the highest **mean** pairwise
    weight while that mean stays ``>= threshold``. Mirrors the semantic
    clusterer's average-linkage to avoid single-linkage chaining.
    """
    # Bound the O(n^4)-worst-case average-linkage (per-merge pair-scan × pairwise
    # mean) on pathologically large topic sets: above the cap, degrade to
    # no-clustering rather than burning a CPU core past the timeout in a sync
    # thread that can't be cancelled (review 2026-07-17 low/theme-linkage).
    if n > _MAX_LINKAGE_TOPICS:
        return [{i} for i in range(n)]

    clusters: list[set[int]] = [{i} for i in range(n)]

    def mean_inter(ci: set[int], cj: set[int]) -> float:
        tot = 0.0
        for a in ci:
            for b in cj:
                tot += weight(a, b)
        return tot / (len(ci) * len(cj))

    while len(clusters) > 1:
        best = -1.0
        bi, bj = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = mean_inter(clusters[i], clusters[j])
                if s > best:
                    best = s
                    bi, bj = i, j
        if bi < 0 or best < threshold:
            break
        clusters[bi] |= clusters[bj]
        clusters.pop(bj)
    return clusters


def _clamp_super_target(raw: int) -> int:
    """Clamp the requested super-theme target to the [MIN, MAX] band."""
    return max(_SUPER_THEME_MIN, min(_SUPER_THEME_MAX, raw))


def _assign_super_themes(
    clusters_out: list[dict[str, Any]],
    cluster_member_sets: list[set[int]],
    weight: "Any",
    super_theme_target: int,
) -> None:
    """graph-v3 tier 7 — super-theme rollup (shared-bridge / cross-cluster lift).

    Computes mean pairwise topic-lift BETWEEN clusters. Two clusters merge if
    their topics tend to be discussed in the same episodes — the "shared
    bridge" signal at the enricher layer. Mutates ``clusters_out`` in place,
    adding ``super_theme_id`` + ``super_theme_label`` to every entry. When the
    cluster count is at or below ``_SUPER_THEME_MIN`` each cluster is its own
    super-theme (rollup is a no-op).

    Additive schema change — consumers reading v1 without the ``super_theme_*``
    fields keep working (see viewer TopicClustersCluster type).
    """
    n_clusters = len(clusters_out)

    def _cluster_mean_lift(ci: set[int], cj: set[int]) -> float:
        tot = 0.0
        cnt = 0
        for a in ci:
            for b in cj:
                if a == b:
                    continue
                tot += weight(a, b)
                cnt += 1
        return tot / cnt if cnt else 0.0

    if n_clusters <= _SUPER_THEME_MIN:
        super_groups = [{i} for i in range(n_clusters)]
    else:
        super_groups = _average_linkage_to_target(
            n_clusters,
            lambda i, j: (
                _cluster_mean_lift(cluster_member_sets[i], cluster_member_sets[j])
                if i != j
                else 0.0
            ),
            super_theme_target,
        )

    def _super_sort_key(members: set[int]) -> tuple[int, str]:
        largest_ci = max(
            members,
            key=lambda ci: (clusters_out[ci]["member_count"], -ci),
        )
        return (-len(members), clusters_out[largest_ci]["canonical_label"])

    used_super_slugs: set[str] = set()
    super_theme_of_cluster: dict[int, tuple[str, str]] = {}
    for members in sorted(super_groups, key=_super_sort_key):
        # Super-theme label = canonical_label of the largest child cluster;
        # ties broken by lowest cluster index for determinism.
        anchor = max(
            members,
            key=lambda ci: (clusters_out[ci]["member_count"], -ci),
        )
        super_label = clusters_out[anchor]["canonical_label"]
        base = _slugify(super_label)
        slug = base
        k = 0
        while slug in used_super_slugs:
            k += 1
            slug = f"{base}-{k}"
        used_super_slugs.add(slug)
        super_id = f"sth:{slug}"
        for ci in members:
            super_theme_of_cluster[ci] = (super_id, super_label)

    for ci, cl in enumerate(clusters_out):
        sid, slabel = super_theme_of_cluster.get(ci, ("", ""))
        cl["super_theme_id"] = sid
        cl["super_theme_label"] = slabel


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    bundles = all_bundles or []
    n_eps = len(bundles)

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    topic_df: dict[str, int] = defaultdict(int)
    topic_label: dict[str, str] = {}
    topic_eps: dict[str, list[str]] = defaultdict(list)

    for b in bundles:
        kg = load_kg(b)
        topics = nodes_of_type(kg, "Topic")
        for t in topics:
            tid = t.get("id")
            if tid:
                topic_label.setdefault(str(tid), node_label(t))
        ids = sorted({str(t.get("id")) for t in topics if t.get("id")})
        ep_id = str(getattr(b, "episode_id", "") or "")
        for tid in ids:
            topic_df[tid] += 1
            if ep_id:
                topic_eps[tid].append(ep_id)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_count[(ids[i], ids[j])] += 1

    min_pair = int(config.get("min_pair_episode_count", _DEFAULT_MIN_PAIR_EPISODES))
    threshold = float(config.get("merge_threshold", _DEFAULT_MERGE_THRESHOLD))

    # Build lift edges over the co-occurring subgraph only (pairs seen in
    # >= min_pair episodes). Topics with no such edge are singletons and are
    # excluded from clustering entirely — keeps the merge loop small and
    # meaningful on real corpora.
    edge_lift: dict[tuple[str, str], float] = {}
    involved: set[str] = set()
    for (a, b_), cnt in pair_count.items():
        if cnt < min_pair:
            continue
        da, db = topic_df[a], topic_df[b_]
        lift = (cnt * n_eps / (da * db)) if (n_eps and da and db) else 0.0
        if lift <= 0:
            continue
        edge_lift[(a, b_)] = lift
        involved.add(a)
        involved.add(b_)

    idx_topics = sorted(involved)
    index = {t: i for i, t in enumerate(idx_topics)}
    weights: dict[tuple[int, int], float] = {}
    for (a, b_), lift in edge_lift.items():
        i, j = index[a], index[b_]
        weights[(min(i, j), max(i, j))] = lift

    def weight(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return weights.get((min(i, j), max(i, j)), 0.0)

    clusters = _average_linkage(len(idx_topics), weight, threshold)

    def degree(member_idx: int, members: set[int]) -> float:
        return sum(weight(member_idx, other) for other in members)

    used_slugs: set[str] = set()
    clusters_out: list[dict[str, Any]] = []
    singletons = 0
    # Keep the underlying member index alongside each output row so the
    # super-theme pass below can operate on the *original* topic-lift
    # graph without recomputing per-topic weights.
    cluster_member_sets: list[set[int]] = []
    for members in sorted(clusters, key=lambda c: (-len(c), min(c) if c else 0)):
        if len(members) < 2:
            singletons += len(members)
            continue
        member_ids = sorted(idx_topics[i] for i in members)
        # Canonical = the topic most strongly tied to the rest of the cluster
        # (highest summed lift), tie-broken by frequency then id for determinism.
        canonical = max(
            member_ids,
            key=lambda t: (degree(index[t], members), topic_df[t], t),
        )
        canonical_label = topic_label.get(canonical, canonical)
        base = _slugify(canonical_label)
        slug = base
        k = 0
        while slug in used_slugs:
            k += 1
            slug = f"{base}-{k}"
        used_slugs.add(slug)
        members_json = [
            {
                "topic_id": tid,
                "label": topic_label.get(tid, tid),
                "episode_ids": sorted(set(topic_eps.get(tid, []))),
                "lift_to_cluster": round(degree(index[tid], members), 4),
            }
            for tid in member_ids
        ]
        clusters_out.append(
            {
                "cluster_type": "theme",
                "canonical_label": canonical_label,
                "graph_compound_parent_id": f"thc:{slug}",
                "member_count": len(member_ids),
                "members": members_json,
            }
        )
        cluster_member_sets.append(members)

    super_theme_target = _clamp_super_target(
        int(config.get("super_theme_target", _SUPER_THEME_TARGET))
    )
    _assign_super_themes(clusters_out, cluster_member_sets, weight, super_theme_target)

    return {
        "schema_version": "1",
        "method": "cooccurrence_lift",
        "episode_count": n_eps,
        "min_pair_episode_count": min_pair,
        "merge_threshold": threshold,
        "topic_count": len(topic_df),
        "cluster_count": len(clusters_out),
        "singletons": singletons,
        "clusters": clusters_out,
        # graph-v3 tier 7 — super-theme rollup summary.
        "super_theme_target": super_theme_target,
        "super_theme_count": len(
            {c["super_theme_id"] for c in clusters_out if c.get("super_theme_id")}
        ),
        "super_theme_method": "cross_cluster_lift_avg_linkage",
    }


_enrich_async = sync_enricher(_compute)


class TopicThemeClustersEnricher:
    """Corpus-scope THEME clustering (topics discussed together, via co-occurrence lift)."""

    manifest = EnricherManifest(
        id="topic_theme_clusters",
        version="1.1.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="topic_theme_clusters.json",
        description=(
            "Corpus-wide THEME clusters — topics discussed together (co-occurrence "
            "lift, average-linkage). Complements the semantic topic_clusters."
        ),
        expected_duration_s=30,
    )

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Enricher.enrich impl — delegates to the sync body via @sync_enricher."""
        return await _enrich_async(bundle, corpus_root, all_bundles, config, ctx)


__all__ = ["TopicThemeClustersEnricher"]
