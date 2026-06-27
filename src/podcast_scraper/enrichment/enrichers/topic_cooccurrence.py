"""``topic_cooccurrence`` — per-episode Topic-pair co-occurrence (deterministic).

Reads the episode KG, lists all Topic nodes, emits unordered pairs.
Each pair is one row with the two ids + labels and ``episode_count=1``
(per-episode scope: only this episode contributes).

The corpus-scope ``topic_cooccurrence_corpus`` enricher aggregates
these per-episode rows into a single ranked corpus-wide file.
"""

from __future__ import annotations

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


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    assert bundle is not None  # EPISODE scope guarantees bundle
    kg = load_kg(bundle)
    topics = nodes_of_type(kg, "Topic")
    ids = sorted({str(t.get("id")) for t in topics if t.get("id")})
    labels = {str(t.get("id")): node_label(t) for t in topics if t.get("id")}
    pairs: list[dict[str, Any]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            pairs.append(
                {
                    "topic_a_id": a,
                    "topic_b_id": b,
                    "topic_a_label": labels.get(a, a),
                    "topic_b_label": labels.get(b, b),
                    "episode_count": 1,
                }
            )
    return {"episode_id": bundle.episode_id, "pairs": pairs}


_enrich_async = sync_enricher(_compute)


class TopicCooccurrenceEnricher:
    """Episode-scope Topic-pair co-occurrence."""

    manifest = EnricherManifest(
        id="topic_cooccurrence",
        version="1.0.0",
        scope=EnricherScope.EPISODE,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="topic_cooccurrence.json",
        description="Per-episode Topic-pair co-occurrence (unordered pairs).",
        expected_duration_s=5,
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


__all__ = ["TopicCooccurrenceEnricher"]
