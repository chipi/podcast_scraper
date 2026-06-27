"""``topic_similarity`` — corpus-scope cosine-similarity Top-K per topic (embedding tier).

For every Topic in the corpus KG, calls the injected
``EmbeddingProvider.topic_vector(topic_id)`` and computes cosine
similarity against every other topic's vector. Emits the top-K
neighbours per topic (descending similarity), tie-broken by topic_id.

The injected provider is built from
``podcast_scraper.enrichment.scorers.embedding.TopicEmbeddingProvider``
in production (wrap your real ``sentence-transformers`` model) and
from the chunk-1 ``MockEmbeddingProvider`` / ``HashEmbedder`` in tests.

Resilience inherited from the EMBEDDING tier policy: 3 retries, 30s
max backoff, circuit at 5 consecutive failures.
"""

from __future__ import annotations

import math
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
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.protocol import EmbeddingProvider


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _gather_topics(
    all_bundles: list[EpisodeArtifactBundle] | None,
) -> tuple[list[str], dict[str, str]]:
    """Return (sorted unique topic_ids, label map) seen anywhere in the corpus."""
    ids: set[str] = set()
    labels: dict[str, str] = {}
    for b in all_bundles or []:
        kg = load_kg(b)
        for node in nodes_of_type(kg, "Topic"):
            tid = str(node.get("id") or "")
            if not tid:
                continue
            ids.add(tid)
            labels[tid] = node_label(node)
    return sorted(ids), labels


class TopicSimilarityEnricher:
    """Corpus-scope cosine-similarity Top-K per topic.

    Construction takes an injected ``EmbeddingProvider``. The executor
    treats this as one enricher; the provider's per-call resilience
    (timeout, retry) flows through the standard EMBEDDING-tier policy.
    """

    manifest = EnricherManifest(
        id="topic_similarity",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.EMBEDDING,
        reads=[".kg.json"],
        writes="topic_similarity.json",
        description="Per-Topic top-K cosine-similar neighbours via injected EmbeddingProvider.",
        expected_duration_s=120,
    )

    def __init__(self, provider: EmbeddingProvider, *, top_k: int = 10) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._provider = provider
        self._top_k = top_k

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        # Backend exceptions (DependencyAccessError / ScorerTimeoutError /
        # ModelLoadError) BUBBLE so the executor's retry classifier can apply
        # the embedding-tier policy. Domain results (cancel / empty corpus)
        # return an EnricherResult directly.
        top_k = int(config.get("top_k", self._top_k))
        if top_k < 1:
            top_k = self._top_k
        ids, labels = _gather_topics(all_bundles)
        if not ids:
            return EnricherResult(
                status=STATUS_OK,
                data={"topics": [], "top_k": top_k, "topic_count": 0, "missing_topic_ids": []},
            )
        vectors: dict[str, list[float]] = {}
        missing: list[str] = []
        for tid in ids:
            if ctx.cancel_event.is_set():
                from podcast_scraper.enrichment.protocol import STATUS_CANCELLED

                return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
            vec = await self._provider.topic_vector(tid)
            if vec is None:
                missing.append(tid)
                continue
            vectors[tid] = vec

        topics_out: list[dict[str, Any]] = []
        ranked_ids = sorted(vectors.keys())
        for tid in ranked_ids:
            base = vectors[tid]
            scored: list[tuple[float, str]] = []
            for other in ranked_ids:
                if other == tid:
                    continue
                scored.append((_cosine(base, vectors[other]), other))
            scored.sort(key=lambda x: (-x[0], x[1]))
            neighbours = [
                {
                    "topic_id": other,
                    "topic_label": labels.get(other, other),
                    "similarity": round(score, 6),
                }
                for score, other in scored[:top_k]
            ]
            topics_out.append(
                {
                    "topic_id": tid,
                    "topic_label": labels.get(tid, tid),
                    "top_k": neighbours,
                }
            )
        return EnricherResult(
            status=STATUS_OK,
            data={
                "topic_count": len(ranked_ids),
                "top_k": top_k,
                "missing_topic_ids": missing,
                "topics": topics_out,
            },
        )


__all__ = ["TopicSimilarityEnricher"]
