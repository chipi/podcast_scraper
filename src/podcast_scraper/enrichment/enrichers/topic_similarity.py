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

**Incremental (RFC-118 PR2):** the wall-clock is the per-topic embedding calls; the
O(topics²) cosine over in-memory vectors is cheap and always runs full. Vectors are
cached per topic in ``enrichments/topic_similarity.vectors_cache.json`` keyed by the
topic's LABEL and the provider's ``model_marker`` — label equality is the exact
invalidation for a label-embedding, so ``enrich_incremental`` re-embeds only new or
relabelled topics (an unmarked provider fail-safes to re-embed everything). Full and
incremental share one ``_compute`` kernel; only which topics hit the provider differs.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from podcast_scraper.enrichment.enrichers._loaders import load_kg, node_label, nodes_of_type
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    ProviderRequirement,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.protocol import EmbeddingProvider

if TYPE_CHECKING:
    from podcast_scraper.corpus_delta import CorpusDelta

_logger = logging.getLogger(__name__)

VECTOR_CACHE_SCHEMA_VERSION = 1
VECTOR_CACHE_FILENAME = "topic_similarity.vectors_cache.json"


def vector_cache_path(corpus_root: Path) -> Path:
    """RFC-118 vector cache sidecar, next to the enricher's output."""
    return Path(corpus_root) / "enrichments" / VECTOR_CACHE_FILENAME


def _load_vector_cache(corpus_root: Path, *, model_marker: str) -> dict[str, Any]:
    """``{topic_id: {label, vector}}``; ``{}`` on absence, corruption, or marker
    mismatch — an empty/unknown marker never reuses (fail-safe re-embed)."""
    if not model_marker:
        return {}
    try:
        raw = json.loads(vector_cache_path(corpus_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != VECTOR_CACHE_SCHEMA_VERSION
        or raw.get("model_marker") != model_marker
    ):
        return {}
    vectors = raw.get("vectors")
    return vectors if isinstance(vectors, dict) else {}


def _write_vector_cache(corpus_root: Path, vectors: dict[str, Any], *, model_marker: str) -> None:
    """Atomically persist the vector cache. Skipped for unmarked providers; non-fatal."""
    if not model_marker:
        return
    p = vector_cache_path(corpus_root)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "schema": VECTOR_CACHE_SCHEMA_VERSION,
                    "model_marker": model_marker,
                    "vectors": vectors,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError as exc:
        _logger.warning("topic_similarity: could not write vector cache %s: %s", p, exc)


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
        # Corpus-scope: embeds every episode's topic text in ONE call, plus a cold
        # sentence-transformers download on the first run of a fresh HF cache (E3).
        # ``expected_duration_s`` is primarily the hard ``asyncio.wait_for`` cap — 120s
        # killed a legit run over the ~678-episode prod corpus. It also doubles as the
        # heartbeat stall-warning threshold (executor.py: ``is_stalled(factor=1.2)`` is
        # evaluated post-completion; enrichers never call ``record_heartbeat``), so raising
        # it also raises that warning threshold — acceptable, it's a post-hoc log line, not
        # a live watchdog. Sized for compute + a run-1 cold MiniLM download; the HF cache
        # volume makes runs 2+ warm. Advisor 2026-08-23.
        expected_duration_s=300,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 7,
                    "description": "Number of nearest-neighbour topics to emit per topic.",
                },
            },
        },
        provider_requirement=ProviderRequirement(
            protocol="EmbeddingProvider",
            description="Embedding source (sentence-transformers checkpoint, embeddings API, …).",
        ),
        supports_incremental=True,
    )

    # top_k default tuned 10 -> 7 (#1105): the prod-v2 Opus-silver eval measured
    # precision/recall of 80%/80% at K=7 vs 71%/99% at K=10 — 7 trades already-saturated
    # recall for a cleaner "related topics" surface (eval: enrichment_topic_similarity_*).
    def __init__(self, provider: EmbeddingProvider, *, top_k: int = 7) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._provider = provider
        self._top_k = top_k

    @property
    def _model_marker(self) -> str:
        return str(getattr(self._provider, "model_marker", "") or "")

    async def _compute(
        self,
        *,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle],
        config: dict[str, Any],
        ctx: RunContext,
        reusable_vectors: dict[str, list[float]],
    ) -> EnricherResult:
        # Backend exceptions (DependencyAccessError / ScorerTimeoutError /
        # ModelLoadError) BUBBLE so the executor's retry classifier can apply
        # the embedding-tier policy. Domain results (cancel / empty corpus)
        # return an EnricherResult directly.
        """The shared full/incremental kernel (RFC-118 §4.1).

        ``reusable_vectors`` maps ``topic_id → vector`` the caller validated (label +
        model marker). Full == this kernel with an empty map; the O(topics²) cosine
        always runs over the complete vector set, so only which topics hit the
        provider can differ between the paths — never ranking or output shape (§7).
        """
        top_k = int(config.get("top_k", self._top_k))
        if top_k < 1:
            top_k = self._top_k
        ids, labels = _gather_topics(all_bundles)
        if not ids:
            return EnricherResult(
                status=STATUS_OK,
                data={"topics": [], "top_k": top_k, "topic_count": 0, "missing_topic_ids": []},
            )
        # Feed the corpus id→label map to the provider so embeddings use the human topic
        # label ("AI development"), not the id slug ("ai-development"). The provider is built
        # from the profile (model / device) with no corpus access, so the enricher — which
        # has the KG labels — must supply them, or every vector is a slug embedding.
        if hasattr(self._provider, "labels"):
            self._provider.labels = dict(labels)  # type: ignore[attr-defined]
        vectors: dict[str, list[float]] = {}
        missing: list[str] = []
        embedded = 0
        for tid in ids:
            if ctx.cancel_event.is_set():
                from podcast_scraper.enrichment.protocol import STATUS_CANCELLED

                return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
            cached_vec = reusable_vectors.get(tid)
            if cached_vec is not None:
                vectors[tid] = cached_vec
                continue
            vec = await self._provider.topic_vector(tid)
            embedded += 1
            if vec is None:
                missing.append(tid)
                continue
            vectors[tid] = vec
        if reusable_vectors:
            _logger.info(
                "topic_similarity incremental: %d/%d topics re-embedded (%d reused) run_id=%s",
                embedded,
                len(ids),
                len(ids) - embedded,
                ctx.run_id,
            )
        # Persist for the next incremental pass — only topics present in the CURRENT
        # corpus, each stamped with the label it was embedded from.
        _write_vector_cache(
            corpus_root,
            {tid: {"label": labels.get(tid, ""), "vector": v} for tid, v in vectors.items()},
            model_marker=self._model_marker,
        )

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
            # Async enrichers return EnricherResult directly (no @sync_enricher wrapper), so they
            # must set records_written themselves — one record per topic with computed neighbours.
            records_written=len(topics_out),
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
        """Full pass: every topic hits the provider; the vector cache is rebuilt."""
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=all_bundles or [],
            config=config,
            ctx=ctx,
            reusable_vectors={},
        )

    async def enrich_incremental(
        self,
        *,
        delta: "CorpusDelta",
        prior_output: dict[str, Any] | None,
        corpus_root: Path,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Delta pass: reuse cached vectors for topics whose label is unchanged.

        Label equality (plus the provider's model marker, checked at cache load) IS
        the invalidation for a label-embedding — the episode delta itself only
        matters through the labels it adds or changes. ``prior_output`` is unused:
        the ranking is recomputed over the full vector set every run.
        """
        reusable: dict[str, list[float]] = {}
        if not delta.forced:
            _, current_labels = _gather_topics(list(delta.all_bundles))
            cached = _load_vector_cache(corpus_root, model_marker=self._model_marker)
            for tid, entry in cached.items():
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                vector = entry.get("vector")
                if (
                    isinstance(label, str)
                    and label
                    and isinstance(vector, list)
                    and current_labels.get(tid) == label
                ):
                    reusable[tid] = [float(x) for x in vector]
        return await self._compute(
            corpus_root=corpus_root,
            all_bundles=list(delta.all_bundles),
            config=config,
            ctx=ctx,
            reusable_vectors=reusable,
        )


__all__ = ["TopicSimilarityEnricher"]
