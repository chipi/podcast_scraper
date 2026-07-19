"""``guest_coappearance`` — Person pairs by shared episodes (deterministic).

For each unordered pair of Persons (P1, P2), counts the episodes where
both appear as Quote speakers. Output is ranked by episode_count.

Reads ``*.gi.json`` (SPOKEN_BY edges + Person nodes).

Filters out unresolved diarization placeholders (``SPEAKER_NN`` /
``person:speaker-NN``) before pair-counting — these IDs are
episode-local, so two episodes' ``SPEAKER_03`` are unrelated and
counting them as a corpus-wide pair pollutes the leaderboard. See
``is_unresolved_speaker_placeholder`` in ``_loaders``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
    is_unresolved_speaker_placeholder,
    load_gi,
    nodes_of_type,
)
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)

# graph-v3 tier 7-4 — person-community rollup.
#
# On top of the pairwise co-appearance graph we run connected-components
# on the sub-graph of edges whose ``episode_count`` >= ``community_min_pair``
# (default 2 — one co-appearance is chance; two is a pattern). Each
# component becomes a "person community" that the viewer paints as a soft
# underlay region behind the Person nodes (parallel to theme regions for
# topics, but on the Person axis).
#
# Simpler than MCL and deterministic: threshold + union-find. Enough
# structure to see "the crypto crowd" / "the FT columnists" clusters on
# real corpora; MCL / label-propagation are queued in
# ``docs/wip/super-theme-signal-comparison.md`` as future signal experiments.
#
# Singleton persons (no qualifying co-appearance edge) don't get a
# community_id — they render at the default plumbing tint.
_DEFAULT_COMMUNITY_MIN_PAIR = 2


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    labels: dict[str, str] = {}

    bundles = all_bundles or []
    placeholder_ids: set[str] = set()
    for b in bundles:
        gi = load_gi(b)
        for node in nodes_of_type(gi, "Person"):
            pid = str(node.get("id") or "")
            if not pid:
                continue
            name = str((node.get("properties") or {}).get("name") or pid)
            labels[pid] = name
            if is_unresolved_speaker_placeholder(pid, name):
                placeholder_ids.add(pid)
        person_ids: set[str] = set()
        for edge in edges_of_type(gi, "SPOKEN_BY"):
            pid = str(edge.get("to") or "")
            if not pid or pid in placeholder_ids:
                continue
            if is_unresolved_speaker_placeholder(pid, labels.get(pid)):
                placeholder_ids.add(pid)
                continue
            person_ids.add(pid)
        sorted_ids = sorted(person_ids)
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                a, c = sorted_ids[i], sorted_ids[j]
                pair_count[(a, c)] += 1

    pairs_out: list[dict[str, Any]] = []
    for (a, c), cnt in sorted(pair_count.items(), key=lambda x: (-x[1], x[0])):
        pairs_out.append(
            {
                "person_a_id": a,
                "person_b_id": c,
                "person_a_name": labels.get(a, a),
                "person_b_name": labels.get(c, c),
                "episode_count": cnt,
            }
        )

    community_min_pair = int(config.get("community_min_pair", _DEFAULT_COMMUNITY_MIN_PAIR))
    communities = _detect_person_communities(pair_count, labels, community_min_pair)

    # #1208 — no-silent-fail contract; see temporal_velocity for rationale.
    partial_reason: str | None = None
    if len(bundles) == 0:
        partial_reason = "no_bundles"
    elif not pairs_out:
        partial_reason = "no_co_appearing_persons"
    if partial_reason is not None:
        _logger.warning(
            "guest_coappearance produced empty output run_id=%s enricher=%s reason=%s bundles=%d",
            ctx.run_id,
            ctx.enricher_id,
            partial_reason,
            len(bundles),
        )

    return {
        "pairs": pairs_out,
        "episode_count": len(bundles),
        # graph-v3 tier 7-4 — person community rollup.
        "community_method": "connected_components_threshold",
        "community_min_pair": community_min_pair,
        "community_count": len(communities),
        "communities": communities,
        "partial_reason": partial_reason,
    }


def _slugify_person_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug or "community"


def _detect_person_communities(
    pair_count: dict[tuple[str, str], int],
    labels: dict[str, str],
    min_pair: int,
) -> list[dict[str, Any]]:
    """Connected-components on the thresholded person-pair graph.

    Edges kept only when ``episode_count >= min_pair``. Union-find yields
    community components; each community is labelled by its highest-degree
    member (most co-appearances within the component). Singletons — persons
    with no qualifying edge — are dropped: they don't need a community_id
    and rendering them as their own group pollutes the palette.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    degree: dict[str, int] = defaultdict(int)
    involved: set[str] = set()
    for (a, b), cnt in pair_count.items():
        if cnt < min_pair:
            continue
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
        degree[a] += cnt
        degree[b] += cnt
        involved.add(a)
        involved.add(b)

    groups: dict[str, list[str]] = defaultdict(list)
    for pid in involved:
        groups[find(pid)].append(pid)

    used_slugs: set[str] = set()
    out: list[dict[str, Any]] = []
    for member_ids in sorted(groups.values(), key=lambda g: (-len(g), min(g))):
        # Anchor = highest-degree member, ties broken by name for determinism.
        anchor = max(member_ids, key=lambda pid: (degree[pid], labels.get(pid, pid)))
        anchor_label = labels.get(anchor, anchor)
        base = _slugify_person_label(anchor_label)
        slug = base
        k = 0
        while slug in used_slugs:
            k += 1
            slug = f"{base}-{k}"
        used_slugs.add(slug)
        out.append(
            {
                "community_id": f"pco:{slug}",
                "community_label": anchor_label,
                "member_ids": sorted(member_ids),
                "member_count": len(member_ids),
            }
        )
    return out


_enrich_async = sync_enricher(_compute)


class GuestCoappearanceEnricher:
    """Corpus-scope Person-pair shared-episode counts."""

    manifest = EnricherManifest(
        id="guest_coappearance",
        version="1.1.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".gi.json"],
        writes="guest_coappearance.json",
        description="Pairs of Persons appearing in the same episode, ranked by episode_count.",
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


__all__ = ["GuestCoappearanceEnricher"]
