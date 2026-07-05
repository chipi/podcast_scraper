"""``topic_cooccurrence_corpus`` — corpus-wide Topic-pair counts (deterministic).

Aggregates per-episode Topic-pair occurrences across the corpus. Each
pair carries two signals: **A** = ``episode_count`` (raw frequency — how
many episodes the pair co-occurs in) and **B** = ``lift`` / ``pmi``
(association strength — does the pair co-occur *more than chance*, given
each topic's own frequency). The default ordering is by ``episode_count``;
the Topic card ranks the same pairs both ways so A and B can be compared
live. The output also drives autoresearch hypothesis generation and the
dashboard "co-mentioned topics".

Reuses the same algorithm shape as the existing
``podcast_scraper.kg.corpus.topic_cooccurrence`` (corpus-scope KG
aggregator), but reads directly from the per-episode bundles instead
of from a separate ``loaded`` list — saves one IO pass when the
``topic_cooccurrence`` enricher has already run, and tolerates being
run standalone.
"""

from __future__ import annotations

import math
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
    topic_df: dict[str, int] = defaultdict(int)  # episodes each topic appears in
    bundles = all_bundles or []
    for b in bundles:
        kg = load_kg(b)
        topics = nodes_of_type(kg, "Topic")
        ids = sorted({str(t.get("id")) for t in topics if t.get("id")})
        labels = {str(t.get("id")): node_label(t) for t in topics if t.get("id")}
        for tid in ids:
            topic_df[tid] += 1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b_ = ids[i], ids[j]
                key = (a, b_)
                pair_count[key] += 1
                pair_labels[key] = (labels.get(a, a), labels.get(b_, b_))
    n = len(bundles)
    pairs: list[dict[str, Any]] = []
    for (a, b_), cnt in sorted(pair_count.items(), key=lambda x: (-x[1], x[0])):
        la, lb = pair_labels[(a, b_)]
        da, db = topic_df[a], topic_df[b_]
        # A = raw ``episode_count`` (how often the pair co-occurs). B = lift/PMI
        # (does the pair co-occur *more than chance*?). lift = P(a,b)/(P(a)·P(b))
        # = cnt·N / (df_a·df_b); >1 ⇒ more than independence predicts. PMI =
        # log2(lift). Both 0.0 when a frequency is missing. Ranking A vs B is a
        # UI concern — we emit the raw signals per pair and let the card sort.
        lift = (cnt * n / (da * db)) if (n and da and db) else 0.0
        pmi = math.log2(lift) if lift > 0 else 0.0
        pairs.append(
            {
                "topic_a_id": a,
                "topic_b_id": b_,
                "topic_a_label": la,
                "topic_b_label": lb,
                "episode_count": cnt,
                "topic_a_episode_count": da,
                "topic_b_episode_count": db,
                "lift": round(lift, 4),
                "pmi": round(pmi, 4),
            }
        )
    return {"episode_count": n, "pairs": pairs}


_enrich_async = sync_enricher(_compute)


class TopicCooccurrenceCorpusEnricher:
    """Corpus-scope Topic-pair co-occurrence aggregator (ranked)."""

    manifest = EnricherManifest(
        id="topic_cooccurrence_corpus",
        version="1.1.0",  # +lift/pmi + per-topic episode counts per pair
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="topic_cooccurrence_corpus.json",
        description="Corpus-wide Topic-pair co-occurrence (episode_count + lift/PMI per pair).",
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


__all__ = ["TopicCooccurrenceCorpusEnricher"]
