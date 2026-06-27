"""``topic_cooccurrence_corpus`` — corpus-wide Topic-pair counts (deterministic).

Aggregates per-episode Topic-pair occurrences across the corpus and
ranks pairs by ``episode_count`` (number of episodes the pair
co-occurs in). The output drives Topic-Entity view edges, autoresearch
hypothesis generation, and dashboard "co-mentioned topics".

Reuses the same algorithm shape as the existing
``podcast_scraper.kg.corpus.topic_cooccurrence`` (corpus-scope KG
aggregator), but reads directly from the per-episode bundles instead
of from a separate ``loaded`` list — saves one IO pass when the
``topic_cooccurrence`` enricher has already run, and tolerates being
run standalone.
"""

from __future__ import annotations

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


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    pair_labels: dict[tuple[str, str], tuple[str, str]] = {}
    bundles = all_bundles or []
    for b in bundles:
        kg = load_kg(b)
        topics = nodes_of_type(kg, "Topic")
        ids = sorted({str(t.get("id")) for t in topics if t.get("id")})
        labels = {str(t.get("id")): node_label(t) for t in topics if t.get("id")}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b_ = ids[i], ids[j]
                key = (a, b_)
                pair_count[key] += 1
                pair_labels[key] = (labels.get(a, a), labels.get(b_, b_))
    pairs: list[dict[str, Any]] = []
    for (a, b_), cnt in sorted(pair_count.items(), key=lambda x: (-x[1], x[0])):
        la, lb = pair_labels[(a, b_)]
        pairs.append(
            {
                "topic_a_id": a,
                "topic_b_id": b_,
                "topic_a_label": la,
                "topic_b_label": lb,
                "episode_count": cnt,
            }
        )
    return {"episode_count": len(bundles), "pairs": pairs}


_enrich_async = sync_enricher(_compute)


class TopicCooccurrenceCorpusEnricher:
    """Corpus-scope Topic-pair co-occurrence aggregator (ranked)."""

    manifest = EnricherManifest(
        id="topic_cooccurrence_corpus",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="topic_cooccurrence_corpus.json",
        description="Corpus-wide Topic-pair co-occurrence (ranked by episode_count).",
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


__all__ = ["TopicCooccurrenceCorpusEnricher"]
