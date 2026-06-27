"""``guest_coappearance`` — Person pairs by shared episodes (deterministic).

For each unordered pair of Persons (P1, P2), counts the episodes where
both appear as Quote speakers. Output is ranked by episode_count.

Reads ``*.gi.json`` (SPOKEN_BY edges + Person nodes).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import edges_of_type, load_gi, nodes_of_type
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)


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
    for b in bundles:
        gi = load_gi(b)
        person_ids: set[str] = set()
        for edge in edges_of_type(gi, "SPOKEN_BY"):
            pid = str(edge.get("to") or "")
            if pid:
                person_ids.add(pid)
        for node in nodes_of_type(gi, "Person"):
            pid = str(node.get("id") or "")
            if not pid:
                continue
            labels[pid] = str((node.get("properties") or {}).get("name") or pid)
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
    return {"pairs": pairs_out, "episode_count": len(bundles)}


_enrich_async = sync_enricher(_compute)


class GuestCoappearanceEnricher:
    """Corpus-scope Person-pair shared-episode counts."""

    manifest = EnricherManifest(
        id="guest_coappearance",
        version="1.0.0",
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
        return await _enrich_async(bundle, corpus_root, all_bundles, config, ctx)


__all__ = ["GuestCoappearanceEnricher"]
